# -*- coding: utf-8 -*-
"""
U1 多因子 + ML Walk-Forward 回测脚本
==================================

作用：
- 在 ultrashort_main 这样的截面因子数据上，做逐日 Walk-Forward 回测；
- 每个交易日 t：
    - 用过去 N 个交易日的样本训练模型（回归，预测 ret_col）；
    - 在 t 当日截面上预测各股票的期望收益；
    - 过滤价格 / 成交额后，按预测收益排序选出 top_k 只股票等权买入；
    - 用 ret_col 作为组合当日收益，叠加成资金曲线。

输出：
- reports/u1_walkforward_equity_<job-id>_<tag>.csv   （每日权益曲线）
- reports/u1_walkforward_yearly_<job-id>_<tag>.csv   （按年拆分统计）
- reports/u1_walkforward_summary_<job-id>_<tag>.txt  （文本 summary）

推荐先用：
- 特征：log_amount, log_volume, log_amt_mean_20, amt_to_mean_20,
        vol_20, ret_1, ret_5, ret_20, rev_1
- 模型：rf
- 训练窗口：train_days=240（大概 1 年）
- warmup：min_train_days=120

命令示例（Windows，按你现在的习惯）：

python -m tools.u1_walkforward_ml_backtest ^
    --job-id ultrashort_main ^
    --start-date 2020-01-01 ^
    --end-date 2025-10-31 ^
    --top-k 3 ^
    --min-price 3 ^
    --max-price 80 ^
    --min-amount 20000000 ^
    --ret-col ret_1 ^
    --features log_amount,log_volume,log_amt_mean_20,amt_to_mean_20,vol_20,ret_1,ret_5,ret_20,rev_1 ^
    --model rf ^
    --train-days 240 ^
    --min-train-days 120 ^
    --tag u1_v1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config

# 尝试导入 sklearn（如果环境里没有，会抛 RuntimeError）
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    SKLEARN_AVAILABLE = True
except Exception as e:  # pragma: no cover
    RandomForestRegressor = GradientBoostingRegressor = Ridge = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = e  # 仅用于报错信息

log = get_logger("U1WalkForward")


# ===== 通用工具函数 =====

def _calc_basic_stats(equity_df: pd.DataFrame) -> Dict[str, float]:
    """给定带有 equity / daily_return 的 DataFrame，计算一组通用回测指标。"""
    if equity_df.empty:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_ratio": 0.0,
        }

    equity = equity_df["equity"].astype(float)
    daily_ret = equity_df["daily_return"].astype(float)

    n_days = len(equity_df)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if n_days > 0:
        ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    else:
        ann_return = 0.0

    if n_days > 1:
        ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    else:
        ann_vol = 0.0

    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    win_ratio = float((daily_ret > 0).mean()) if n_days > 0 else 0.0

    return {
        "n_days": int(n_days),
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_ratio": float(win_ratio),
    }


def _resolve_paths(sys_cfg: Dict[str, Any], job_id: str, tag: str) -> Dict[str, Path]:
    """根据系统配置解析数据路径和输出路径。"""
    paths_cfg = sys_cfg.get("paths", {}) or {}
    project_root = paths_cfg.get("project_root", ".") or "."
    dataset_dir = paths_cfg.get("dataset_dir", os.path.join("data", "datasets")) or os.path.join(
        "data", "datasets"
    )
    reports_dir = paths_cfg.get("reports_dir", "reports") or "reports"

    # 进入项目根目录（跟你现在其他脚本行为保持一致）
    try:
        os.chdir(project_root)
    except Exception:
        pass

    dataset_path = Path(dataset_dir) / f"{job_id}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到数据集文件: {dataset_path}")

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    equity_file = reports_path / f"u1_walkforward_equity_{job_id}_{tag}.csv"
    yearly_file = reports_path / f"u1_walkforward_yearly_{job_id}_{tag}.csv"
    summary_file = reports_path / f"u1_walkforward_summary_{job_id}_{tag}.txt"

    return {
        "dataset_path": dataset_path,
        "equity_file": equity_file,
        "yearly_file": yearly_file,
        "summary_file": summary_file,
    }


def make_model(name: str):
    """根据字符串创建一个回归模型，用于预测 ret_col。"""
    name = name.lower()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn 未安装，无法创建模型: {SKLEARN_IMPORT_ERROR}")

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=42,
        )
    if name == "gb":
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    if name == "ridge":
        return Ridge(alpha=1.0)

    raise ValueError(f"未知模型: {name}")


def _apply_static_filters(df: pd.DataFrame, min_price: float, max_price: float, min_amount: float) -> pd.DataFrame:
    """按价格 / 成交额做一次全局静态过滤。"""
    out = df.copy()

    if "close" in out.columns:
        out = out[(out["close"] >= min_price) & (out["close"] <= max_price)]

    if "amount" in out.columns:
        out = out[out["amount"] >= min_amount]

    return out


# ===== Walk-Forward 回测主逻辑 =====

def run_walkforward_backtest(
    df: pd.DataFrame,
    features: List[str],
    ret_col: str,
    top_k: int,
    train_days: int,
    min_train_days: int,
    min_price: float,
    max_price: float,
    min_amount: float,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    在截面多因子数据上做逐日 Walk-Forward 回测。

    参数
    ----
    df          : 输入数据（需包含 trade_date / trading_date，features，ret_col）
    features    : 特征列名列表
    ret_col     : 收益列名（如 ret_1）
    top_k       : 每日持有的股票数量
    train_days  : 训练窗口长度（按交易日数）
    min_train_days : 开始交易前最少需要的历史交易日数
    min_price, max_price, min_amount : 静态过滤条件
    model_name  : 模型名称（rf / gb / ridge）

    返回
    ----
    equity_df   : 每日资金曲线 DataFrame
    yearly_df   : 按年统计 DataFrame
    stats       : 总体指标字典
    """
    df = df.copy()

    # 处理日期列
    if "trade_date" in df.columns:
        date_col = "trade_date"
    elif "trading_date" in df.columns:
        date_col = "trading_date"
    else:
        raise RuntimeError("数据集中既没有 trade_date 也没有 trading_date 列，无法回测。")

    df[date_col] = pd.to_datetime(df[date_col])
    if date_col != "trade_date":
        df = df.rename(columns={date_col: "trade_date"})
        date_col = "trade_date"

    df = df.sort_values(["trade_date"] + [c for c in ["symbol", "code"] if c in df.columns])

    # 只保留实际存在的特征列
    use_cols = [c for c in features if c in df.columns]
    missing = sorted(set(features) - set(use_cols))
    if missing:
        log.warning("以下特征在数据集中不存在，将被忽略: %s", missing)

    if not use_cols:
        raise ValueError("有效特征列为空，请检查 --features 或数据列名。")

    if ret_col not in df.columns:
        raise ValueError(f"数据集中缺少收益列 {ret_col!r} 。")

    df = df.dropna(subset=use_cols + [ret_col])
    if df.empty:
        raise RuntimeError("清洗后的数据为空。")

    # 静态过滤（价格 / 成交额）
    df = _apply_static_filters(df, min_price, max_price, min_amount)
    if df.empty:
        raise RuntimeError("静态过滤后数据为空，请检查价格 / 成交额条件。")

    dates = np.sort(df["trade_date"].unique())
    equity_records: List[Dict[str, Any]] = []

    equity = 1.0
    n_total_trades = 0
    n_trade_days = 0

    for i, cur_date in enumerate(dates):
        # 至少要有 min_train_days 个历史交易日才开始交易
        if i < min_train_days:
            continue

        # 训练窗口：过去 train_days 个交易日
        start_idx = max(0, i - train_days)
        train_dates = dates[start_idx:i]

        train_df = df[df["trade_date"].isin(train_dates)]
        test_df = df[df["trade_date"] == cur_date]

        if train_df.empty or test_df.empty:
            continue

        train_df = train_df.dropna(subset=use_cols + [ret_col])
        test_df = test_df.dropna(subset=use_cols + [ret_col])
        if train_df.empty or test_df.empty:
            continue

        # 训练模型
        model = make_model(model_name)
        X_train = train_df[use_cols].values
        y_train = train_df[ret_col].values.astype(float)

        try:
            model.fit(X_train, y_train)
        except Exception as e:  # pragma: no cover - 仅防御
            log.warning("训练模型失败，date=%s, err=%s，跳过该日。", cur_date, e)
            continue

        # 当日预测 + 选股
        X_test = test_df[use_cols].values
        try:
            score = model.predict(X_test)
        except Exception as e:  # pragma: no cover
            log.warning("预测失败，date=%s, err=%s，跳过该日。", cur_date, e)
            continue

        df_day = test_df.copy()
        df_day["score"] = score

        # 选出得分最高的 top_k 只股票
        df_day = df_day.sort_values("score", ascending=False)
        df_sel = df_day.head(top_k)

        if df_sel.empty:
            daily_ret = 0.0
            n_sel = 0
        else:
            daily_ret = float(df_sel[ret_col].mean())  # 简单等权
            n_sel = len(df_sel)

        equity *= (1.0 + daily_ret)

        equity_records.append(
            {
                "trade_date": cur_date,
                "daily_return": daily_ret,
                "equity": equity,
                "n_selected": n_sel,
            }
        )

        if n_sel > 0:
            n_trade_days += 1
            n_total_trades += n_sel

    equity_df = pd.DataFrame(equity_records)
    if not equity_df.empty:
        equity_df = equity_df.sort_values("trade_date")
        equity_df["year"] = equity_df["trade_date"].dt.year

    # 总体指标
    stats = _calc_basic_stats(equity_df)

    # 按年统计
    yearly_records: List[Dict[str, Any]] = []
    if not equity_df.empty:
        for year, grp in equity_df.groupby("year"):
            s = _calc_basic_stats(grp)
            yearly_records.append({"year": int(year), **s})

    yearly_df = pd.DataFrame(yearly_records)

    # 额外信息
    stats["n_trade_days"] = int(n_trade_days)
    stats["n_total_trades"] = int(n_total_trades)

    return equity_df, yearly_df, stats


# ===== 打印 & 主程序 =====

def format_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_stats_table(title: str, stats: Dict[str, float]) -> None:
    print(f"==== {title} ====")
    print(f"回测交易日数 : {stats.get('n_days', 0):5d}")
    print(f"累计收益     : {format_pct(stats.get('total_return', 0.0))}")
    print(f"年化收益     : {format_pct(stats.get('ann_return', 0.0))}")
    print(f"年化波动     : {format_pct(stats.get('ann_vol', 0.0))}")
    print(f"Sharpe       : {stats.get('sharpe', 0.0):6.2f}")
    print(f"最大回撤     : {format_pct(stats.get('max_drawdown', 0.0))}")
    print(f"胜率(按日)   : {format_pct(stats.get('win_ratio', 0.0))}")
    print(f"有交易的天数 : {stats.get('n_trade_days', 0):5d}")
    print(f"总持仓数     : {stats.get('n_total_trades', 0):5d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U1 多因子 + ML Walk-Forward 回测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--job-id", required=True, help="数据集 ID，比如 ultrashort_main")
    parser.add_argument("--start-date", required=True, help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="回测结束日期 (YYYY-MM-DD)")

    parser.add_argument("--top-k", type=int, default=3, help="每天持有的股票数量")
    parser.add_argument("--min-price", type=float, default=3.0, help="最低股价过滤")
    parser.add_argument("--max-price", type=float, default=80.0, help="最高股价过滤")
    parser.add_argument("--min-amount", type=float, default=20_000_000.0, help="最低成交额过滤")

    parser.add_argument("--ret-col", default="ret_1", help="用于回测的收益列名")

    parser.add_argument(
        "--features",
        default="log_amount,log_volume,log_amt_mean_20,amt_to_mean_20,vol_20,ret_1,ret_5,ret_20,rev_1",
        help="逗号分隔的特征列名列表",
    )

    parser.add_argument(
        "--model",
        default="rf",
        choices=["rf", "gb", "ridge"],
        help="使用的模型类型",
    )

    parser.add_argument(
        "--train-days",
        type=int,
        default=240,
        help="训练窗口长度（按交易日数，大致≈一年）",
    )

    parser.add_argument(
        "--min-train-days",
        type=int,
        default=120,
        help="开始交易前最少需要的历史交易日数（warmup）",
    )

    parser.add_argument(
        "--tag",
        default="u1_v1",
        help="结果输出文件名后缀 tag，方便区分不同实验",
    )

    args = parser.parse_args()

    sys_cfg = get_system_config()
    paths = _resolve_paths(sys_cfg, args.job_id, args.tag)

    print(f"[U1WF] 读取数据集: {paths['dataset_path']}")
    df = pd.read_parquet(paths["dataset_path"])

    # 按日期做一次粗过滤，减少不必要的数据
    if "trade_date" in df.columns:
        date_col = "trade_date"
    elif "trading_date" in df.columns:
        date_col = "trading_date"
    else:
        raise RuntimeError("数据集中既没有 trade_date 也没有 trading_date 列。")

    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)

    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    feature_list = [c.strip() for c in args.features.split(",") if c.strip()]

    equity_df, yearly_df, stats = run_walkforward_backtest(
        df=df,
        features=feature_list,
        ret_col=args.ret_col,
        top_k=args.top_k,
        train_days=args.train_days,
        min_train_days=args.min_train_days,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        model_name=args.model,
    )

    # 保存结果
    equity_df.to_csv(paths["equity_file"], index=False)
    yearly_df.to_csv(paths["yearly_file"], index=False)

    with open(paths["summary_file"], "w", encoding="utf-8") as f:
        f.write("==== U1 Walk-Forward 回测统计 ====\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print_stats_table("U1 Walk-Forward 回测统计 (整体)", stats)

    print("==== 按年份拆分统计（基于日度收益） ====")
    if yearly_df.empty:
        print("无有效交易日")
    else:
        print(
            f"{'year':>6s} {'n_days':>6s} {'total_ret':>10s} "
            f"{'ann_ret':>9s} {'ann_vol':>9s} {'sharpe':>8s} "
            f"{'max_dd':>10s} {'win_ratio':>10s}"
        )
        for _, row in yearly_df.iterrows():
            print(
                f"{int(row['year']):6d} "
                f"{int(row['n_days']):6d} "
                f"{format_pct(row['total_return']):>10s} "
                f"{format_pct(row['ann_return']):>9s} "
                f"{format_pct(row['ann_vol']):>9s} "
                f"{row['sharpe']:8.2f} "
                f"{format_pct(row['max_drawdown']):>10s} "
                f"{format_pct(row['win_ratio']):>10s}"
            )

    print()
    print(f"[U1WF] 已保存每日权益曲线到: {paths['equity_file']}")
    print(f"[U1WF] 已保存按年份统计到: {paths['yearly_file']}")
    print(f"[U1WF] 已保存 summary 到: {paths['summary_file']}")


if __name__ == "__main__":
    main()
