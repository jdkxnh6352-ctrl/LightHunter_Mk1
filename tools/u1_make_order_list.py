# -*- coding: utf-8 -*-
"""
tools/u1_make_order_list.py

功能：
- 读取 U1 日打分结果（默认：reports/u1_scores_<job_id>_<tag>_<yyyymmdd>_top3.csv）；
- 按给定资金、单票最大仓位、手数等规则，生成一份“下单列表”CSV；
- 默认等权分配权重，或复用打分文件里的 weight 列（如果有的话）。

输出：
- reports/u1_orders_<job_id>_<tag>_<yyyymmdd>.csv

用法示例（Windows，和其它 tools 一样在项目根目录运行）：

    python -m tools.u1_make_order_list ^
        --job-id ultrashort_main ^
        --as-of 2025-10-31 ^
        --tag u1_v1_base_rf ^
        --capital 3000000 ^
        --max-weight 0.25 ^
        --min-notional 300000 ^
        --lot-size 100 ^
        --top-k 3

说明：
- capital        ：本次计划投入的总资金（现金），单位和 close*股数一致；
- max-weight     ：单票最大权重（占 capital 的比例），例如 0.25 表示最多 25%；
- min-notional   ：若某票分配到的资金 < 这个阈值，则自动跳过；
- lot-size       ：每手股数（A 股默认 100）；
- top-k          ：可选，只对打分前 top-k 的股票生成下单；0 表示全用；
- tag            ：和日打分脚本 `--tag` 一致，例如 u1_v1_base_rf；
- as-of          ：交易日，必须与你跑 `u1_daily_scoring_ml` 时的一致。
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def _now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _infer_scores_path(job_id: str, tag: str, as_of: str, use_full: bool = False) -> Path:
    """
    根据命名规范自动推断日打分结果路径。

    默认优先找 *_top3.csv，找不到再退回 *_full.csv。
    """
    date_str = as_of.replace("-", "")
    reports_dir = Path("reports")

    if use_full:
        cand = reports_dir / f"u1_scores_{job_id}_{tag}_{date_str}_full.csv"
        return cand

    top3 = reports_dir / f"u1_scores_{job_id}_{tag}_{date_str}_top3.csv"
    if top3.exists():
        return top3

    full = reports_dir / f"u1_scores_{job_id}_{tag}_{date_str}_full.csv"
    return full


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到日打分结果文件：{path}")
    print(f"{_now()} [U1Order] 读取打分文件：{path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"打分文件为空：{path}")
    return df


def _prepare_weights(
    df: pd.DataFrame,
    max_weight: float,
) -> pd.Series:
    """
    返回每只股票的目标权重（Series，索引与 df 一致）。

    规则：
    - 若存在 'weight' 列，则以此为“原始权重”，做非负 & 归一化；
    - 否则按等权分配；
    - 每只股票最终权重 <= max_weight；
    - 总权重 <= 1，多余资金留作现金垫底。
    """
    n = len(df)
    if n == 0:
        raise ValueError("没有任何候选股票，无法生成权重。")

    if "weight" in df.columns:
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if w.sum() <= 0:
            w = pd.Series([1.0 / n] * n, index=df.index)
        else:
            w = w / w.sum()
    else:
        w = pd.Series([1.0 / n] * n, index=df.index)

    if max_weight is not None and max_weight > 0.0:
        w = w.clip(upper=float(max_weight))

    # 可选：再做一次归一化，避免 sum > 1（这里宁可 <1，留点现金）
    total = float(w.sum())
    if total > 1.0:
        w = w / total

    return w


def generate_orders(
    df_scores: pd.DataFrame,
    trade_date: str,
    capital: float,
    max_weight: float,
    min_notional: float,
    lot_size: int,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    核心逻辑：从日打分结果生成下单列表（买入）。

    返回包含以下列的 DataFrame：
        - trade_date
        - code
        - side
        - weight
        - shares
        - notional
        - price
        - u1_score
        - rank
        - close
        - amount
    """
    if capital <= 0:
        raise ValueError("capital 必须 > 0")

    # 按打分从高到低排一下，方便 rank
    if "u1_score" not in df_scores.columns:
        raise KeyError("打分文件缺少 'u1_score' 列。")
    df = df_scores.sort_values("u1_score", ascending=False).reset_index(drop=True)

    # 生成权重
    weights = _prepare_weights(df, max_weight=max_weight)

    orders: List[Dict[str, Any]] = []
    used_capital = 0.0

    for idx, row in df.iterrows():
        code = str(row.get("code"))
        try:
            price = float(row.get(price_col, float("nan")))
        except Exception:
            price = float("nan")

        if not math.isfinite(price) or price <= 0:
            print(f"{_now()} [U1Order] 代码 {code} 缺少有效价格 {price_col}，跳过。")
            continue

        w = float(weights.loc[idx])
        if w <= 0:
            continue

        target_notional = capital * w
        if min_notional > 0 and target_notional < min_notional:
            print(
                f"{_now()} [U1Order] 代码 {code} 分配到 {target_notional:.0f} "
                f"小于 min_notional={min_notional:.0f}，跳过。"
            )
            continue

        # 计算股数并按手数向下取整
        raw_shares = target_notional / price
        if lot_size <= 0:
            lot_size = 1
        lots = math.floor(raw_shares / lot_size)
        shares = lots * lot_size

        if shares <= 0:
            print(
                f"{_now()} [U1Order] 代码 {code} 按资金 {target_notional:.0f} "
                f"& 价格 {price:.2f} 算出的股数不足 1 手，跳过。"
            )
            continue

        notional = shares * price
        used_capital += notional

        orders.append(
            {
                "trade_date": trade_date,
                "code": code,
                "side": "BUY",
                "weight": w,
                "shares": int(shares),
                "notional": float(notional),
                "price": float(price),
                "u1_score": float(row.get("u1_score", float("nan"))),
                "rank": int(idx + 1),
                "close": float(row.get("close", price)),
                "amount": float(row.get("amount", float("nan"))),
            }
        )

    if not orders:
        print(f"{_now()} [U1Order] 没有生成任何有效订单。")
        return pd.DataFrame(columns=[
            "trade_date",
            "code",
            "side",
            "weight",
            "shares",
            "notional",
            "price",
            "u1_score",
            "rank",
            "close",
            "amount",
        ])

    df_orders = pd.DataFrame(orders)
    print(
        f"{_now()} [U1Order] 共生成 {len(df_orders)} 条订单，"
        f"计划使用资金约 {used_capital:.0f}，"
        f"剩余现金约 {capital - used_capital:.0f}。"
    )
    return df_orders


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.u1_make_order_list",
        description="U1 日打分结果 → 下单列表 生成脚本",
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="数据集 / 任务 ID，例如 ultrashort_main。",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="打分对应的交易日，例如 2025-10-31。",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="策略标签，例如 u1_v1_base_rf（需与日打分脚本使用的一致）。",
    )

    parser.add_argument(
        "--capital",
        type=float,
        required=True,
        help="本次计划投入的总资金，单位与价格 * 股数一致。",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.25,
        help="单票最大权重（占 capital 的比例，默认 0.25）。",
    )
    parser.add_argument(
        "--min-notional",
        type=float,
        default=0.0,
        help="单票最小资金（若分配金额低于此值则跳过，默认不限制）。",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=100,
        help="每手股数（A 股默认 100）。",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="close",
        help="用于计算下单价格的列名（默认 close）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="仅使用打分前 top-k 的股票（0 表示全用当前打分文件中的全部）。",
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default="",
        help="可选：直接指定打分结果 CSV 路径；不指定则按约定命名自动推断。",
    )
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="优先使用 *_full.csv 而不是 *_top3.csv。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出下单列表 CSV 路径；默认写到 reports/u1_orders_<job_id>_<tag>_<yyyymmdd>.csv",
    )

    args = parser.parse_args(argv)

    # 解析日期字符串
    try:
        dt.datetime.strptime(args.as_of, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("as-of 必须是 YYYY-MM-DD 格式，例如 2025-10-31。")

    # 决定打分文件路径
    if args.scores_file:
        scores_path = Path(args.scores_file)
    else:
        scores_path = _infer_scores_path(args.job_id, args.tag, args.as_of, use_full=args.use_full)

    df_scores = _load_scores(scores_path)

    # 可选：仅用 top-k
    if args.top_k and args.top_k > 0:
        df_scores = (
            df_scores.sort_values("u1_score", ascending=False)
            .head(args.top_k)
            .reset_index(drop=True)
        )

    df_orders = generate_orders(
        df_scores=df_scores,
        trade_date=args.as_of,
        capital=args.capital,
        max_weight=float(args.max_weight),
        min_notional=float(args.min_notional),
        lot_size=int(args.lot_size),
        price_col=args.price_col,
    )

    # 写出 CSV
    if args.output:
        out_path = Path(args.output)
    else:
        date_str = args.as_of.replace("-", "")
        out_name = f"u1_orders_{args.job_id}_{args.tag}_{date_str}.csv"
        out_path = Path("reports") / out_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_orders.to_csv(out_path, index=False)
    print(f"{_now()} [U1Order] 下单列表已保存到：{out_path}")


if __name__ == "__main__":
    main()
