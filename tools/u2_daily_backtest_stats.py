import argparse
import math
import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def _calc_max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负数）"""
    if equity.empty:
        return float("nan")
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())


def _summary_from_daily(daily: pd.DataFrame) -> Dict[str, Any]:
    """根据按交易日聚合好的 daily 表，算一组回测指标。"""
    if daily.empty:
        return {
            "n_days": 0,
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "win_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "avg_positions": float("nan"),
        }

    daily = daily.copy()
    # 累乘得到净值曲线
    daily["equity"] = (1.0 + daily["daily_ret"]).cumprod()

    n_days = len(daily)
    total_return = float(daily["equity"].iloc[-1] - 1.0)
    ann_return = (1.0 + total_return) ** (250.0 / n_days) - 1.0

    vol_daily = float(daily["daily_ret"].std(ddof=0))
    ann_vol = vol_daily * math.sqrt(250.0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    win_ratio = float((daily["daily_ret"] > 0).mean())
    max_dd = _calc_max_drawdown(daily["equity"])
    avg_n = float(daily["n"].mean())

    return {
        "n_days": n_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "win_ratio": win_ratio,
        "max_drawdown": max_dd,
        "avg_positions": avg_n,
    }


def run_stats(
    input_csv: str,
    ret_col: str = "ret_1",
    top_k: int = 3,
    min_prob: float = 0.6,
    min_price: Optional[float] = 5.0,
    max_price: Optional[float] = 80.0,
    min_amount: Optional[float] = 2e7,
):
    if not os.path.exists(input_csv):
        raise SystemExit(f"[STATS][ERROR] 找不到回测结果文件: {input_csv}")

    df = pd.read_csv(input_csv)
    if "trade_date" not in df.columns:
        raise SystemExit("[STATS][ERROR] CSV 里缺少 trade_date 列。")

    if ret_col not in df.columns:
        raise SystemExit(f"[STATS][ERROR] CSV 里找不到收益列 {ret_col}。")

    # 统一处理一下日期和排序
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["trade_date", "u2_prob"], ascending=[True, False])

    # 过滤条件
    mask = pd.Series(True, index=df.index)

    if min_prob is not None:
        mask &= df["u2_prob"] >= float(min_prob)
    if min_price is not None and "close" in df.columns:
        mask &= df["close"] >= float(min_price)
    if max_price is not None and "close" in df.columns:
        mask &= df["close"] <= float(max_price)
    if min_amount is not None and "amount" in df.columns:
        mask &= df["amount"] >= float(min_amount)

    df = df[mask].copy()

    if df.empty:
        raise SystemExit("[STATS][ERROR] 过滤条件过严，数据为空。")

    print(f"[STATS] 过滤后样本数: {len(df)}")

    # 如果没有 rank 列，就按 u2_prob 自动排个名
    if "rank" not in df.columns:
        df["rank"] = df.groupby("trade_date")["u2_prob"].rank(
            method="first", ascending=False
        )

    top = df[df["rank"] <= top_k].copy()
    if top.empty:
        raise SystemExit("[STATS][ERROR] top_k 过滤后没有任何样本。")

    # 按交易日聚合：等权持有 top_k，收益取平均
    daily = (
        top.groupby("trade_date")
        .agg(
            daily_ret=(ret_col, "mean"),
            avg_prob=("u2_prob", "mean"),
            n=("code", "count"),
            label_hit_any=("label_u2", "max"),
        )
        .sort_index()
    )

    # 汇总整体指标
    summary = _summary_from_daily(daily)
    per_stock_hit = float(top["label_u2"].mean())
    per_day_hit_any = float(daily["label_hit_any"].mean())

    print("\n==== U2 日常打分回测统计（整体）====")
    print(f"  交易日数       : {summary['n_days']}")
    print(f"  累计收益       : {summary['total_return']:.2%}")
    print(f"  年化收益       : {summary['ann_return']:.2%}")
    print(f"  年化波动率     : {summary['ann_vol']:.2%}")
    print(f"  Sharpe        : {summary['sharpe']:.2f}")
    print(f"  单日胜率       : {summary['win_ratio']:.2%}")
    print(f"  最大回撤       : {summary['max_drawdown']:.2%}")
    print(f"  日均持仓数量   : {summary['avg_positions']:.2f}")
    print(f"  单票命中率     : {per_stock_hit:.2%}")
    print(f"  每日至少中一票 : {per_day_hit_any:.2%}")

    # 按年份再做一份拆分
    daily_with_year = daily.copy()
    daily_with_year["year"] = daily_with_year.index.year

    yearly = (
        daily_with_year.groupby("year")
        .apply(lambda g: pd.Series(_summary_from_daily(g)))
        .reset_index()
    )

    print("\n==== 按年份拆分的统计（基于日度收益）====")
    print(
        yearly[
            ["year", "n_days", "total_return", "ann_return", "ann_vol",
             "sharpe", "win_ratio", "max_drawdown"]
        ].to_string(index=False, justify="center",
                    float_format=lambda x: f"{x: .4f}")
    )

    return daily, yearly


def main():
    parser = argparse.ArgumentParser(
        description="从 u2_daily_scoring_backtest 生成的 CSV 里，做一个简单的选股 + 回测统计。"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="reports/u2_daily_backtest_demo.csv",
        help="u2_daily_scoring_backtest 生成的 CSV 路径（默认: reports/u2_daily_backtest_demo.csv）",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default="ret_1",
        choices=["ret_1", "ret_5", "ret_20"],
        help="用哪一列收益做回测：ret_1 / ret_5 / ret_20（默认 ret_1）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每天取前 top_k 只股票（按 u2_prob 从高到低排序，默认 3）",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.6,
        help="过滤条件：u2_prob 不低于该阈值（默认 0.6）",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="过滤条件：收盘价不低于该值（默认 5 元）",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=80.0,
        help="过滤条件：收盘价不高于该值（默认 80 元）",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=2e7,
        help="过滤条件：成交额不低于该值（默认 2e7，单位：元）",
    )
    parser.add_argument(
        "--output-equity",
        type=str,
        default=None,
        help="如果指定，则把按日收益 + 净值曲线另存为这个 CSV。",
    )
    parser.add_argument(
        "--output-yearly",
        type=str,
        default=None,
        help="如果指定，则把按年份的统计结果另存为这个 CSV。",
    )

    args = parser.parse_args()

    daily, yearly = run_stats(
        input_csv=args.input,
        ret_col=args.ret_col,
        top_k=args.top_k,
        min_prob=args.min_prob,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
    )

    if args.output_equity:
        daily.to_csv(args.output_equity, index=True)
        print(f"[STATS] 已保存日度收益 & 净值曲线到: {args.output_equity}")

    if args.output_yearly:
        yearly.to_csv(args.output_yearly, index=False)
        print(f"[STATS] 已保存按年份统计到: {args.output_yearly}")


if __name__ == "__main__":
    main()
