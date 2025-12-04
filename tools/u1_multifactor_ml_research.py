# -*- coding: utf-8 -*-
"""
U1 多因子 + 机器学习研究脚本

用法示例（和你现在用的一致）：

python -m tools.u1_multifactor_ml_research ^
    --job-id ultrashort_main ^
    --start-date 2020-01-01 ^
    --end-date 2025-10-31 ^
    --top-k 30 ^
    --min-price 3 ^
    --max-price 80 ^
    --min-amount 20000000 ^
    --ret-col ret_1 ^
    --train-end-date 2022-12-31 ^
    --model rf ^
    --features log_amount,log_volume,log_amt_mean_20,amt_to_mean_20,vol_20,ret_1,ret_5,ret_20,rev_1
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# 日志工具
# ----------------------------------------------------------------------


def _get_logger(name: str = "U1") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = _get_logger("U1")


# ----------------------------------------------------------------------
# 通用统计函数
# ----------------------------------------------------------------------


def _compute_stats(daily_ret: pd.Series) -> Dict[str, float]:
    """根据日度收益序列计算统计指标。"""
    daily_ret = daily_ret.dropna().sort_index()
    n_days = int(daily_ret.shape[0])
    if n_days == 0:
        return dict(
            n_days=0,
            total_return=0.0,
            ann_return=0.0,
            ann_vol=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_ratio=0.0,
        )

    equity = (1.0 + daily_ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    ann_factor = 252.0 / n_days
    ann_return = float((1.0 + total_return) ** ann_factor - 1.0)
    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_drawdown = float(dd.min())

    win_ratio = float((daily_ret > 0).mean())

    return dict(
        n_days=n_days,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_ratio=win_ratio,
    )


# ----------------------------------------------------------------------
# 核心逻辑：单次实验
# ----------------------------------------------------------------------


def _make_model(model_name: str) -> Pipeline:
    """根据名字创建一个 sklearn Pipeline 模型。"""
    model_name = model_name.lower()
    if model_name == "rf":
        base = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gb":
        base = GradientBoostingRegressor(random_state=42)
    elif model_name == "ridge":
        base = Ridge(alpha=1.0)
    else:
        raise ValueError(f"未知模型：{model_name}，目前支持 rf / gb / ridge")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", base),
        ]
    )
    return pipe


def run_experiment(
    job_id: str,
    start_date: str,
    end_date: str,
    train_end_date: str,
    top_k: int,
    min_price: float,
    max_price: float,
    min_amount: float,
    ret_col: str,
    feature_cols: Sequence[str],
    model_name: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    供脚本内部和 sweep 调用的核心函数。

    返回：
    - stats_full: 全样本统计 dict
    - stats_train: 训练段统计 dict
    - stats_test: 测试段统计 dict
    - equity_df: 全样本每日权益曲线 DataFrame
    - yearly_df: 全样本按年份统计 DataFrame
    - summary_row: 用于 sweep 的摘要 dict
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "datasets" / f"{job_id}.parquet"
    out_dir = project_root / "reports" / "u1_multifactor"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("读取数据集: %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集文件: {data_path}")

    df = pd.read_parquet(data_path)

    # 统一日期列名 trade_date
    if "trade_date" not in df.columns and "trading_date" in df.columns:
        logger.info("数据集中没有 trade_date 列，发现 trading_date 列，已自动重命名为 trade_date 用于分析……")
        df = df.rename(columns={"trading_date": "trade_date"})

    if "trade_date" not in df.columns:
        raise KeyError("数据集中必须包含 trade_date 或 trading_date 列。")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    train_end_ts = pd.to_datetime(train_end_date)

    # 基本过滤
    mask_date = (df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)
    df = df.loc[mask_date].copy()

    # 价格与成交额过滤（列名根据你的数据结构调整，这里假定 close / amount）
    for col in ["close", "amount"]:
        if col not in df.columns:
            raise KeyError(f"数据集中缺少必要列: {col}")

    mask_price = (df["close"] >= min_price) & (df["close"] <= max_price)
    mask_amt = df["amount"] >= min_amount

    df = df.loc[mask_price & mask_amt].copy()
    df = df.dropna(subset=[ret_col])

    logger.info("过滤后样本数: %d", df.shape[0])

    # 处理特征列：忽略不存在的列
    requested = list(feature_cols)
    valid_features = [c for c in requested if c in df.columns]
    missing = [c for c in requested if c not in df.columns]

    if missing:
        logger.info("以下特征在数据集中不存在，已忽略: %s", missing)

    if not valid_features:
        # 这里的报错信息会被 sweep 用来识别“这一组特征完全失效”
        raise ValueError("有效特征列为空，请检查 --features 或数据列名。")

    logger.info("使用特征字段 %d 个: %s", len(valid_features), valid_features)

    # 按日期划分训练 / 测试
    df_train = df[df["trade_date"] <= train_end_ts].copy()
    df_test = df[df["trade_date"] > train_end_ts].copy()
    n_train, n_test = df_train.shape[0], df_test.shape[0]

    logger.info("有效样本数: %d", df.shape[0])
    logger.info("训练样本: %d, 测试样本: %d", n_train, n_test)

    X_train = df_train[valid_features].values
    y_train = df_train[ret_col].values

    X_full = df[valid_features].values
    y_full = df[ret_col].values

    # 建模
    logger.info("开始训练模型: %s ……", model_name)
    model = _make_model(model_name)
    model.fit(X_train, y_train)
    logger.info("模型训练完成。")

    # 预测 & 日度组合收益
    df["pred"] = model.predict(X_full)

    df_sorted = df.sort_values(["trade_date", "pred"], ascending=[True, False])
    top = df_sorted.groupby("trade_date").head(top_k)
    daily_ret_full = top.groupby("trade_date")[ret_col].mean().sort_index()

    # 全样本权益曲线
    equity = (1.0 + daily_ret_full).cumprod()
    equity_df = pd.DataFrame(
        {
            "trade_date": daily_ret_full.index,
            "daily_return": daily_ret_full.values,
            "equity": equity.values,
        }
    )

    # 训练 / 测试切分的日度收益
    daily_ret_train = daily_ret_full[daily_ret_full.index <= train_end_ts]
    daily_ret_test = daily_ret_full[daily_ret_full.index > train_end_ts]

    stats_full = _compute_stats(daily_ret_full)
    stats_train = _compute_stats(daily_ret_train)
    stats_test = _compute_stats(daily_ret_test)

    # 按年份统计（基于日度收益）
    year_group = daily_ret_full.groupby(daily_ret_full.index.year)
    yearly_rows = []
    for year, ret in year_group:
        s = _compute_stats(ret)
        yearly_rows.append(
            dict(
                year=int(year),
                n_days=s["n_days"],
                total_return=s["total_return"],
                ann_return=s["ann_return"],
                ann_vol=s["ann_vol"],
                sharpe=s["sharpe"],
                max_drawdown=s["max_drawdown"],
                win_ratio=s["win_ratio"],
            )
        )
    yearly_df = pd.DataFrame(yearly_rows).sort_values("year")

    # 保存文件
    equity_path = out_dir / f"u1_equity_{job_id}.csv"
    yearly_path = out_dir / f"u1_yearly_{job_id}.csv"

    equity_df.to_csv(equity_path, index=False, encoding="utf-8")
    yearly_df.to_csv(yearly_path, index=False, encoding="utf-8")

    logger.info("已保存每日权益曲线到: %s", equity_path)
    logger.info("已保存按年份统计到: %s", yearly_path)

    # 用于 sweep 的摘要
    summary_row = dict(
        n_features=len(valid_features),
        n_train=n_train,
        n_test=n_test,
        test_ann_return=stats_test["ann_return"],
        test_sharpe=stats_test["sharpe"],
        test_max_dd=stats_test["max_drawdown"],
        full_ann_return=stats_full["ann_return"],
        full_sharpe=stats_full["sharpe"],
        full_max_dd=stats_full["max_drawdown"],
        train_sharpe=stats_train["sharpe"],
    )

    return stats_full, stats_train, stats_test, equity_df, yearly_df, summary_row


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("u1_multifactor_ml_research.py")

    p.add_argument("--job-id", required=True, help="数据集 ID（对应 data/datasets/{job_id}.parquet）")
    p.add_argument("--start-date", required=True, help="回测开始日期 YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="回测结束日期 YYYY-MM-DD")
    p.add_argument("--train-end-date", required=True, help="训练集截止日期 YYYY-MM-DD")

    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--max-price", type=float, default=80.0)
    p.add_argument("--min-amount", type=float, default=20_000_000.0)

    p.add_argument("--ret-col", default="ret_1", help="目标收益列，默认 ret_1")
    p.add_argument(
        "--features",
        required=True,
        help="逗号分隔的特征列名，如: log_amount,log_volume,vol_20,ret_1,ret_5,ret_20,rev_1",
    )
    p.add_argument(
        "--model",
        default="rf",
        help="模型名称: rf / gb / ridge",
    )

    return p.parse_args()


def _print_stats_block(title: str, stats: Dict[str, float]) -> None:
    print(f"==== {title} 回测统计 ====")
    print(f"回测交易日数 : {stats['n_days']}")
    print(f"累计收益     : {stats['total_return'] * 100:.2f}%")
    print(f"年化收益     : {stats['ann_return'] * 100:.2f}%")
    print(f"年化波动     : {stats['ann_vol'] * 100:.2f}%")
    print(f"Sharpe       : {stats['sharpe']:.2f}")
    print(f"最大回撤     : {stats['max_drawdown'] * 100:.2f}%")
    print(f"胜率(按日)   : {stats['win_ratio'] * 100:.2f}%")
    print()


def main() -> None:
    args = parse_args()

    feature_cols: List[str] = [s.strip() for s in args.features.split(",") if s.strip()]

    stats_full, stats_train, stats_test, equity_df, yearly_df, summary = run_experiment(
        job_id=args.job_id,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        top_k=args.top_k,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        ret_col=args.ret_col,
        feature_cols=feature_cols,
        model_name=args.model,
    )

    print()
    print("==== U1 多因子+ML 回测统计（整体） ====")
    _print_stats_block("整体", stats_full)
    _print_stats_block("训练集 (Train)", stats_train)
    _print_stats_block("测试集 (Test)", stats_test)

    # 年度拆分统计
    print("==== 按年份拆分统计（基于日度收益） ====")
    print(yearly_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


if __name__ == "__main__":
    main()
