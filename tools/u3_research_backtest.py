"""
U3 Research Backtest (单文件研究版)

思路：
- 直接使用 data/datasets/ultrashort_main.parquet
- 训练一个滚动的 LogisticRegression，用过去 N 天数据预测二分类标签
- 每个交易日按预测概率排序，买入 Top-K 只股，持有 1 天
- 用 ret_1 作为当日收益，生成权益曲线 + 年度统计

注意：
- 默认用 label_u2 作为标签列；如果数据里没有，就用 ret_5 >= 2% 作为正样本。
- 默认特征列：open/high/low/close/volume/amount/vol_20/amt_mean_20
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


TRADING_DAYS_PER_YEAR = 240


# ----------------------------------------------------------------------
# 数据加载与预处理
# ----------------------------------------------------------------------


def load_dataset(
    path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在：{p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "trading_date" not in df.columns:
        raise ValueError("数据集中找不到 trading_date 列，请确认使用的是 ultrashort_main 数据集。")

    # 日期标准化
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    if "code" not in df.columns:
        raise ValueError("数据集中找不到 code 列。")

    # 过滤日期区间
    if start_date is not None:
        s = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        df = df[df["trading_date"] >= s]
    if end_date is not None:
        e = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df[df["trading_date"] <= e]

    df = df.sort_values(["trading_date", "code"]).reset_index(drop=True)
    return df


def prepare_label(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, str]:
    """
    确定用于训练的标签列。如果 label_col 不存在，则退化为：
    label_tmp = (ret_5 >= 0.02) ? 1 : 0
    """
    if label_col in df.columns:
        return df.copy(), label_col

    if "ret_5" not in df.columns:
        raise ValueError(
            f"数据中既没有 {label_col}，也没有 ret_5，无法构造标签。"
        )

    df = df.copy()
    df["label_tmp"] = (df["ret_5"].astype(float) >= 0.02).astype(int)
    print(
        f"[U3] 警告：找不到标签列 {label_col}，自动使用 ret_5>=2% 构造 label_tmp 作为标签。"
    )
    return df, "label_tmp"


def choose_feature_cols(df: pd.DataFrame, label_col: str, ret_col: str) -> List[str]:
    """
    默认特征：open/high/low/close/volume/amount/vol_20/amt_mean_20 中存在的列；
    若这些列都不存在，则使用所有数值列（排除 label_col 和 ret_col）。
    """
    preferred = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vol_20",
        "amt_mean_20",
    ]
    feat = [c for c in preferred if c in df.columns]

    if not feat:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat = [c for c in numeric_cols if c not in {label_col, ret_col}]
        print(
            "[U3] 警告：默认特征列都不存在，退化为“所有数值列-标签-收益”。"
        )

    print(f"[U3] 使用特征列：{feat}")
    return feat


# ----------------------------------------------------------------------
# 回测核心逻辑
# ----------------------------------------------------------------------


def _train_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Optional[LogisticRegression]:
    if LogisticRegression is None:
        raise RuntimeError(
            "未安装 scikit-learn，无法运行 U3 回测。请先执行：pip install scikit-learn"
        )

    train_df = train_df.dropna(subset=feature_cols + [label_col])
    if train_df.empty:
        return None

    if train_df[label_col].nunique() < 2:
        # 全是 0 或全是 1，模型训练没有意义
        return None

    X = train_df[feature_cols].astype(float).values
    y = train_df[label_col].astype(int).values

    model = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def _daily_summary(
    daily_ret: pd.Series, equity: pd.Series
) -> Dict[str, float]:
    daily_ret = daily_ret.fillna(0.0)
    n_days = len(daily_ret)
    if n_days == 0:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_ratio": 0.0,
        }

    total_return = float((1.0 + daily_ret).prod() - 1.0)
    ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    win_ratio = float((daily_ret > 0).sum() / n_days)

    return {
        "n_days": n_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_ratio": win_ratio,
    }


def run_u3_backtest(
    df: pd.DataFrame,
    train_days: int = 120,
    top_k: int = 3,
    min_price: float = 3.0,
    max_price: float = 80.0,
    min_amount: float = 2e7,
    label_col: str = "label_u2",
    ret_col: str = "ret_1",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    返回：
        equity_df: 每日权益曲线
        yearly_df: 按年份汇总指标
        summary : 全区间总体指标
    """
    df, label_col = prepare_label(df, label_col)

    if ret_col not in df.columns:
        raise ValueError(f"数据集中找不到收益列 {ret_col}。")

    feature_cols = choose_feature_cols(df, label_col, ret_col)

    trade_dates = sorted(df["trading_date"].unique())
    if len(trade_dates) <= train_days + 1:
        raise ValueError(
            f"可用交易日太少（{len(trade_dates)}），不足以完成 train_days={train_days} 的滚动回测。"
        )

    records: List[Dict[str, Any]] = []
    equity = 1.0

    for i, cur_date in enumerate(trade_dates):
        # 前 train_days 天作为训练窗口
        if i == 0:
            continue
        start_idx = max(0, i - train_days)
        train_dates = trade_dates[start_idx:i]
        if len(train_dates) < max(40, int(train_days * 0.5)):
            # 训练数据太少，跳过
            continue

        train_df = df[df["trading_date"].isin(train_dates)]
        model = _train_model(train_df, feature_cols, label_col)
        if model is None:
            # 无法训练出有效模型，当天不交易
            records.append(
                {
                    "trading_date": cur_date,
                    "n_candidates": 0,
                    "n_trades": 0,
                    "daily_return": 0.0,
                    "equity": equity,
                }
            )
            continue

        day_df = df[df["trading_date"] == cur_date].copy()
        day_df = day_df.dropna(subset=feature_cols)

        # 基本过滤：价格 & 成交额
        if "close" in day_df.columns:
            day_df = day_df[
                (day_df["close"].astype(float) >= min_price)
                & (day_df["close"].astype(float) <= max_price)
            ]
        if "amount" in day_df.columns:
            day_df = day_df[day_df["amount"].astype(float) >= min_amount]

        n_candidates = len(day_df)
        if n_candidates == 0:
            daily_ret = 0.0
            n_trades = 0
        else:
            X_day = day_df[feature_cols].astype(float).values
            prob = model.predict_proba(X_day)[:, 1]
            day_df["prob_u3"] = prob
            day_df = day_df.sort_values("prob_u3", ascending=False)

            selected = day_df.head(top_k)
            n_trades = len(selected)

            if n_trades > 0:
                daily_ret = float(selected[ret_col].astype(float).mean())
            else:
                daily_ret = 0.0

        equity *= 1.0 + daily_ret

        records.append(
            {
                "trading_date": cur_date,
                "n_candidates": int(n_candidates),
                "n_trades": int(n_trades),
                "daily_return": daily_ret,
                "equity": equity,
            }
        )

    equity_df = pd.DataFrame(records)
    if equity_df.empty:
        raise RuntimeError("回测没有产生任何记录，请检查参数和数据。")

    # 总体统计
    summary = _daily_summary(equity_df["daily_return"], equity_df["equity"])

    # 按年份统计
    equity_df["year"] = pd.to_datetime(equity_df["trading_date"]).dt.year
    yearly_stats: List[Dict[str, Any]] = []
    for year, sub in equity_df.groupby("year"):
        s = _daily_summary(sub["daily_return"], sub["equity"])
        s["year"] = int(year)
        yearly_stats.append(s)

    cols = [
        "year",
        "n_days",
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "win_ratio",
    ]
    yearly_df = pd.DataFrame(yearly_stats)[cols].sort_values("year").reset_index(drop=True)

    return equity_df, yearly_df, summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.u3_research_backtest",
        description="U3 Research Backtest：在 ultrashort_main 数据集上做滚动回测。",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="u3_research_main",
        help="回测任务 ID，仅用于日志和输出文件命名。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/ultrashort_main.parquet",
        help="输入数据集路径（默认：data/datasets/ultrashort_main.parquet）。",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="回测起始日期（YYYY-MM-DD，默认 2020-01-01）。",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="回测结束日期（YYYY-MM-DD，默认用数据集最大日期）。",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=120,
        help="每次滚动训练使用的历史交易日数量，默认 120。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每日持仓股票数量（最多 Top-K），默认 3。",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=3.0,
        help="最低股价过滤，默认 3 元。",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=80.0,
        help="最高股价过滤，默认 80 元。",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=2e7,
        help="最低成交额过滤，默认 2e7（两千万）。",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label_u2",
        help="标签列名，默认 label_u2；若不存在将自动用 ret_5>=2% 构造。",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default="ret_1",
        help="用作收益的列名，默认 ret_1。",
    )
    parser.add_argument(
        "--output-equity",
        type=str,
        default=None,
        help="保存每日权益曲线到 CSV 的路径（默认：reports/u3_equity_{job_id}.csv）。",
    )
    parser.add_argument(
        "--output-yearly",
        type=str,
        default=None,
        help="保存年度统计到 CSV 的路径（默认：reports/u3_yearly_{job_id}.csv）。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    df = load_dataset(args.dataset, args.start_date, args.end_date)

    equity_df, yearly_df, summary = run_u3_backtest(
        df=df,
        train_days=args.train_days,
        top_k=args.top_k,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        label_col=args.label_col,
        ret_col=args.ret_col,
    )

    # 输出文件路径
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.output_equity is None:
        equity_path = reports_dir / f"u3_equity_{args.job_id}.csv"
    else:
        equity_path = Path(args.output_equity)

    if args.output_yearly is None:
        yearly_path = reports_dir / f"u3_yearly_{args.job_id}.csv"
    else:
        yearly_path = Path(args.output_yearly)

    equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    print("\n==== U3 Research 回测统计（整体） ====")
    print(f"交易日数 : {summary['n_days']}")
    print(f"累计收益 : {summary['total_return'] * 100:8.2f}%")
    print(f"年化收益 : {summary['ann_return'] * 100:8.2f}%")
    print(f"年化波动 : {summary['ann_vol'] * 100:8.2f}%")
    print(f"Sharpe  : {summary['sharpe']:8.2f}")
    print(f"最大回撤 : {summary['max_drawdown'] * 100:8.2f}%")
    print(f"胜率(按日): {summary['win_ratio'] * 100:8.2f}%")

    print("\n==== 按年份拆分统计（基于日度收益） ====")
    print(yearly_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    print(f"\n[U3] 已保存每日权益曲线到: {equity_path}")
    print(f"[U3] 已保存按年份统计到  : {yearly_path}")


if __name__ == "__main__":
    main()
