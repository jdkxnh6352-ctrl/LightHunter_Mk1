# tools/u0_baseline_research.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

DATE_COL = "trade_date"
CODE_COL = "code"


@dataclass
class U0Config:
    job_id: str
    start_date: str
    end_date: str
    top_k: int = 30
    min_price: float = 3.0
    max_price: float = 80.0
    min_amount: float = 20_000_000.0
    feature_col: str = "amount"   # 用来排序选股的特征
    ret_col: str = "ret_1"        # 用来算收益的列
    feature_dir: str = "desc"     # "desc" = 特征越大越好, "asc" = 越小越好


def parse_args(argv: Optional[List[str]] = None) -> U0Config:
    parser = argparse.ArgumentParser(
        prog="u0_baseline_research",
        description="U0 Baseline：每天按指定特征排序选 Top-K，用指定收益列做回测。",
    )
    parser.add_argument("--job-id", required=True, help="数据集 id，例如 ultrashort_main")
    parser.add_argument("--start-date", required=True, help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=30, help="每天最多持仓数量")
    parser.add_argument("--min-price", type=float, default=3.0, help="最低股价过滤")
    parser.add_argument("--max-price", type=float, default=80.0, help="最高股价过滤")
    parser.add_argument("--min-amount", type=float, default=20_000_000.0, help="最低成交额过滤")
    parser.add_argument("--feature-col", type=str, default="amount", help="排序用特征列名")
    parser.add_argument("--ret-col", type=str, default="ret_1", help="用来计算收益的列名")
    parser.add_argument(
        "--feature-dir",
        type=str,
        choices=["asc", "desc"],
        default="desc",
        help="特征排序方向: desc=越大越好, asc=越小越好",
    )

    args = parser.parse_args(argv)

    return U0Config(
        job_id=args.job_id,
        start_date=args.start_date,
        end_date=args.end_date,
        top_k=args.top_k,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        feature_col=args.feature_col,
        ret_col=args.ret_col,
        feature_dir=args.feature_dir,
    )


def load_dataset(cfg: U0Config) -> pd.DataFrame:
    dataset_path = Path("data") / "datasets" / f"{cfg.job_id}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到数据集文件: {dataset_path}")

    print(f"[U0] 读取数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # trade_date / trading_date 兼容
    if DATE_COL not in df.columns:
        if "trading_date" in df.columns:
            print(
                f"[U0] 数据集中没有 {DATE_COL} 列, 发现 trading_date 列, "
                f"已自动重命名为 {DATE_COL} 用于回测……"
            )
            df = df.rename(columns={"trading_date": DATE_COL})
        else:
            raise KeyError(f"数据集中缺少交易日列: {DATE_COL} / trading_date")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df


def filter_universe(df: pd.DataFrame, cfg: U0Config) -> pd.DataFrame:
    # 日期过滤
    start = pd.to_datetime(cfg.start_date).date()
    end = pd.to_datetime(cfg.end_date).date()
    df = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)].copy()

    # 价格过滤
    if "close" in df.columns:
        if cfg.min_price is not None:
            df = df[df["close"] >= cfg.min_price]
        if cfg.max_price is not None:
            df = df[df["close"] <= cfg.max_price]

    # 成交额过滤
    if cfg.min_amount is not None and "amount" in df.columns:
        df = df[df["amount"] >= cfg.min_amount]

    # 检查必要列
    missing = [c for c in [cfg.feature_col, cfg.ret_col] if c not in df.columns]
    if missing:
        raise KeyError(f"数据集中缺少必要列: {missing}")

    # 类型清洗
    df[cfg.feature_col] = pd.to_numeric(df[cfg.feature_col], errors="coerce")
    df[cfg.ret_col] = pd.to_numeric(df[cfg.ret_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[cfg.feature_col, cfg.ret_col, DATE_COL])

    return df


def calc_equity_curve(df: pd.DataFrame, cfg: U0Config) -> pd.DataFrame:
    """
    已经过滤好的数据：每天按 feature_col 排序，选 top_k，平均持仓，用 ret_col 算每天收益。
    """
    if df.empty:
        raise RuntimeError("过滤后数据为空，无法回测。")

    ascending = cfg.feature_dir == "asc"
    records = []
    equity = 1.0

    grouped = df.groupby(DATE_COL)
    print(f"[U0] 回测交易日样本数: {len(grouped)}")

    for trade_date, g in grouped:
        # 用特征排序 —— 这里就是和你之前脚本不一样的地方
        day = g.sort_values(cfg.feature_col, ascending=ascending)

        if cfg.top_k is not None and cfg.top_k > 0:
            day = day.head(cfg.top_k)

        if day.empty:
            continue

        # 等权持仓的日收益
        daily_ret = float(day[cfg.ret_col].mean())
        equity *= (1.0 + daily_ret)

        records.append(
            {
                DATE_COL: trade_date,
                "daily_return": daily_ret,
                "equity": equity,
                "n_positions": len(day),
            }
        )

    equity_df = pd.DataFrame(records).sort_values(DATE_COL).reset_index(drop=True)
    return equity_df


def calc_stats(equity_df: pd.DataFrame) -> Dict[str, float]:
    if equity_df.empty:
        raise RuntimeError("equity_df 为空，无法统计。")

    rets = equity_df["daily_return"].values.astype(float)
    n_days = len(rets)
    eq = equity_df["equity"].values.astype(float)

    total_return = float(eq[-1] / eq[0] - 1.0)
    ann_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))
    sharpe = float(ann_return / (ann_vol + 1e-8))

    # 最大回撤
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    max_drawdown = float(drawdown.min())

    win_ratio = float((rets > 0).mean())

    return dict(
        n_days=n_days,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_ratio=win_ratio,
    )


def calc_yearly_stats(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year"] = df[DATE_COL].dt.year

    rows = []
    for year, g in df.groupby("year"):
        sub = g.sort_values(DATE_COL)
        stats = calc_stats(sub)
        row = dict(year=year, **stats)
        rows.append(row)

    yearly = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return yearly


def print_report(overall: Dict[str, float], yearly: pd.DataFrame):
    print("==== U0 Baseline 回测统计（整体） ====")
    print(f"交易日数 : {overall['n_days']}")
    print(f"累计收益 : {overall['total_return'] * 100:6.2f}%")
    print(f"年化收益 : {overall['ann_return'] * 100:6.2f}%")
    print(f"年化波动 : {overall['ann_vol'] * 100:6.2f}%")
    print(f"Sharpe  : {overall['sharpe']:6.2f}")
    print(f"最大回撤: {overall['max_drawdown'] * 100:6.2f}%")
    print(f"胜率(按日): {overall['win_ratio'] * 100:6.2f}%")
    print()
    print("==== 按年份拆分统计（基于日度收益） ====")
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
    print(
        yearly[cols].to_string(
            index=False,
            formatters={
                "total_return": lambda x: f"{x:7.4f}",
                "ann_return": lambda x: f"{x:7.4f}",
                "ann_vol": lambda x: f"{x:7.4f}",
                "sharpe": lambda x: f"{x:7.4f}",
                "max_drawdown": lambda x: f"{x:7.4f}",
                "win_ratio": lambda x: f"{x:7.4f}",
            },
        )
    )


def save_reports(cfg: U0Config, equity_df: pd.DataFrame, yearly_df: pd.DataFrame):
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    eq_path = reports_dir / f"u0_baseline_equity_{cfg.job_id}.csv"
    yearly_path = reports_dir / f"u0_baseline_yearly_{cfg.job_id}.csv"

    equity_df.to_csv(eq_path, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    print(f"[U0] 已保存每日权益曲线到: {eq_path}")
    print(f"[U0] 已保存按年份统计到: {yearly_path}")


def main(argv: Optional[List[str]] = None):
    cfg = parse_args(argv)
    df = load_dataset(cfg)
    df = filter_universe(df, cfg)
    equity_df = calc_equity_curve(df, cfg)
    overall = calc_stats(equity_df)
    yearly = calc_yearly_stats(equity_df)
    print_report(overall, yearly)
    save_reports(cfg, equity_df, yearly)


if __name__ == "__main__":
    main()
