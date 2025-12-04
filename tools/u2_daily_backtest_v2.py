"""
U2 日常打分：改持仓规则 + 止盈止损 版本回测脚本

用法（示例，在项目根目录下）::

    python -m tools.u2_daily_backtest_v2 ^
        --input reports/u2_daily_backtest_demo.csv ^
        --ret-col ret_1 ^
        --top-k 3 ^
        --min-prob 0.6 ^
        --min-price 5 ^
        --max-price 80 ^
        --min-amount 2e7 ^
        --position-weight 0.3 ^
        --stop-loss 0.03 ^
        --take-profit 0.06 ^
        --output-equity reports/u2_backtest_equity_v2.csv ^
        --output-yearly reports/u2_backtest_yearly_v2.csv

说明：
- 仍然使用 u2_daily_scoring_backtest 生成的 CSV 作为输入（每行是一只股票在某天的打分+真实未来收益）。
- 先按你设定的过滤条件/top-k 选出每天要交易的股票；
- 然后用新的持仓规则 + 止盈止损，生成每日组合收益、整体/按年份统计，以及 NAV 曲线。

注意：这里的止盈止损是用 ret_1（1 日收益）做“截断”，
      相当于假设未来 1 日价格运动不会超过 ret_1 之外更多细节，
      所以这是一个 *近似* 的止盈/止损回测，而不是精确的日内路径模拟。
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_and_filter(
    path: str,
    ret_col: str,
    top_k: int,
    min_prob: Optional[float],
    min_price: Optional[float],
    max_price: Optional[float],
    min_amount: Optional[float],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    required_cols = {"trade_date", "code", ret_col, "u2_prob", "close", "amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少这些字段：{missing}")

    mask = pd.Series(True, index=df.index)
    if min_prob is not None:
        mask &= df["u2_prob"] >= float(min_prob)
    if min_price is not None:
        mask &= df["close"] >= float(min_price)
    if max_price is not None:
        mask &= df["close"] <= float(max_price)
    if min_amount is not None:
        mask &= df["amount"] >= float(min_amount)

    df = df[mask].copy()
    if df.empty:
        raise RuntimeError("过滤之后数据为空，请检查 min_prob / 价格 / 成交额 条件。")

    print(f"[STATS] 过滤后样本数：{len(df)}")

    # 按日期+概率排序，取每天 top-k
    df = df.sort_values(["trade_date", "u2_prob"], ascending=[True, False])
    df["rank_in_day"] = df.groupby("trade_date").cumcount() + 1
    df = df[df["rank_in_day"] <= top_k].copy()

    if df.empty:
        raise RuntimeError("top-k 过滤之后没有任何样本，请检查 top_k / min_prob 等参数。")

    print(f"[STATS] 经过 top_k={top_k} 后样本数：{len(df)}")

    return df


def apply_stop_and_position(
    df: pd.DataFrame,
    ret_col: str,
    position_weight: float,
    max_positions: Optional[int],
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 用 ret_col 做止盈/止损截断
    ret = df[ret_col].astype(float)
    eff_ret = ret.copy()

    if stop_loss is not None:
        sl = abs(float(stop_loss))
        eff_ret = np.maximum(eff_ret, -sl)
    if take_profit is not None:
        tp = float(take_profit)
        eff_ret = np.minimum(eff_ret, tp)

    df = df.copy()
    df["eff_ret"] = eff_ret

    # 持仓规则：每只 position_weight，最多 max_positions 只；不满仓部分视为持有现金（收益为 0）
    if position_weight <= 0:
        raise ValueError("position_weight 必须 > 0")

    if max_positions is None:
        max_positions = max(1, int(math.floor(1.0 / position_weight)))

    print(
        f"[STATS] 持仓规则：每只 {position_weight:.2%} 仓，"
        f"最多 {max_positions} 只，满仓 = {min(1.0, position_weight * max_positions):.0%}"
    )

    daily_rows = []

    for dt, g in df.groupby("trade_date"):
        g = g.sort_values("u2_prob", ascending=False)
        n_signals = len(g)
        n_trades = min(n_signals, max_positions)
        if n_trades == 0:
            continue

        rets = g["eff_ret"].values[:n_trades]
        weights = np.full(n_trades, position_weight, dtype=float)
        total_w = weights.sum()

        if total_w > 1.0:
            # 超过 100% 仓位的话等比缩放到满仓，现金权重为 0
            weights *= 1.0 / total_w
            cash_w = 0.0
        else:
            cash_w = 1.0 - total_w

        day_ret = float(np.dot(rets, weights))  # 现金收益为 0

        daily_rows.append((dt, day_ret, n_trades, cash_w, n_signals))

    daily = pd.DataFrame(
        daily_rows,
        columns=["date", "ret", "n_trades", "cash_weight", "n_signals"],
    ).sort_values("date").reset_index(drop=True)

    if daily.empty:
        raise RuntimeError("没有生成任何每日收益（daily_ret），请检查参数。")

    daily["equity"] = (1.0 + daily["ret"]).cumprod()

    return df, daily


def summarize_overall(daily: pd.DataFrame) -> dict:
    n_days = len(daily)
    total_return = float(daily["equity"].iloc[-1] - 1.0)

    ann_factor = 252 / n_days
    ann_return = (1.0 + total_return) ** ann_factor - 1.0
    ann_vol = float(daily["ret"].std(ddof=1) * math.sqrt(252)) if n_days > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    roll_max = daily["equity"].cummax()
    drawdown = daily["equity"] / roll_max - 1.0
    max_dd = float(drawdown.min())

    win_ratio = float((daily["ret"] > 0).mean())

    stats = {
        "n_days": n_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_ratio": win_ratio,
    }

    print("\n==== U2 日常打分改持仓+止盈止损 回测统计（整体）====")
    print(f"交易日数量      : {n_days}")
    print(f"累计收益        : {total_return:8.2%}")
    print(f"年化收益        : {ann_return:8.2%}")
    print(f"年化波动        : {ann_vol:8.2%}")
    print(f"Sharpe          : {sharpe:8.2f}")
    print(f"最大回撤        : {max_dd:8.2%}")
    print(f"胜率（按日）    : {win_ratio:8.2%}")

    return stats


def summarize_yearly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["year"] = df["date"].dt.year

    rows = []
    for year, g in df.groupby("year"):
        n_days = len(g)
        total_return = float((1.0 + g["ret"]).prod() - 1.0)

        ann_factor = 252 / n_days
        ann_return = (1.0 + total_return) ** ann_factor - 1.0
        ann_vol = float(g["ret"].std(ddof=1) * math.sqrt(252)) if n_days > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

        roll_max = g["equity"].cummax()
        drawdown = g["equity"] / roll_max - 1.0
        max_dd = float(drawdown.min())

        win_ratio = float((g["ret"] > 0).mean())

        rows.append(
            {
                "year": int(year),
                "n_days": n_days,
                "total_return": total_return,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_ratio": win_ratio,
            }
        )

    yearly = pd.DataFrame(rows).sort_values("year")

    print("\n==== 按年份拆分的统计（基于日度收益）====")
    print(
        yearly.to_string(
            index=False,
            formatters={
                "total_return": "{:.2%}".format,
                "ann_return": "{:.2%}".format,
                "ann_vol": "{:.2%}".format,
                "sharpe": "{:.2f}".format,
                "max_drawdown": "{:.2%}".format,
                "win_ratio": "{:.2%}".format,
            },
        )
    )

    return yearly


def main():
    parser = argparse.ArgumentParser(
        description="U2 日常打分：改持仓规则 + 止盈止损 版本回测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="reports/u2_daily_backtest_demo.csv",
        help="u2_daily_scoring_backtest 生成的 CSV 文件路径",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default="ret_1",
        help="用作收益的列名（例如 ret_1 / ret_5 / ret_20）",
    )

    parser.add_argument(
        "--top-k", type=int, default=3, help="每天最多入选的股票数量（按 u2_prob 从高到低排序）"
    )
    parser.add_argument("--min-prob", type=float, default=0.6, help="U2 预测概率下限")
    parser.add_argument("--min-price", type=float, default=5.0, help="收盘价下限")
    parser.add_argument("--max-price", type=float, default=80.0, help="收盘价上限")
    parser.add_argument("--min-amount", type=float, default=2e7, help="当日成交额下限")

    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.3,
        help="单只股票的目标仓位，例如 0.3 表示 30% 仓",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="当日最多持仓数量（默认 floor(1/position_weight)）",
    )

    parser.add_argument(
        "--stop-loss", type=float, default=0.03, help="单笔止损线（例如 0.03 表示 -3%）"
    )
    parser.add_argument(
        "--take-profit", type=float, default=0.06, help="单笔止盈线（例如 0.06 表示 +6%）"
    )

    parser.add_argument(
        "--output-equity",
        type=str,
        default="reports/u2_backtest_equity_v2.csv",
        help="保存日度收益 & 净值曲线的 CSV 路径",
    )
    parser.add_argument(
        "--output-yearly",
        type=str,
        default="reports/u2_backtest_yearly_v2.csv",
        help="保存按年份拆分统计的 CSV 路径",
    )

    args = parser.parse_args()

    # 确保输出目录存在
    for out_path in [args.output_equity, args.output_yearly]:
        out_dir = Path(out_path).parent
        if out_dir and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(
        path=args.input,
        ret_col=args.ret_col,
        top_k=args.top_k,
        min_prob=args.min_prob,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
    )

    df_trades, daily = apply_stop_and_position(
        df=df,
        ret_col=args.ret_col,
        position_weight=args.position_weight,
        max_positions=args.max_positions,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )

    stats = summarize_overall(daily)
    yearly = summarize_yearly(daily)

    # 输出 CSV
    daily.to_csv(args.output_equity, index=False)
    yearly.to_csv(args.output_yearly, index=False)

    print(f"\n[STATS] 已保存日度收益 & 净值曲线到: {args.output_equity}")
    print(f"[STATS] 已保存按年份统计到      : {args.output_yearly}")


if __name__ == "__main__":
    main()
