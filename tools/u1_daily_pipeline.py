# tools/u1_daily_pipeline.py
# -*- coding: utf-8 -*-

"""
U1 日打分流水线（Pipeline）

功能：
1. 调用 U1 日打分引擎（u1_daily_scoring_ml），生成 full / topK 打分结果；
2. 在 full 打分结果上做二次过滤：
   - 剔除 科创板 / 创业板 / 北交所 股票
   - 剔除 名称中包含 ST / *ST 的股票（如果有 name 列）
3. 重新按 u1_score 排名，重写 full / topK 打分文件；
4. 根据过滤后的 topK 结果生成当日下单列表 u1_orders_*.csv，
   并在其中加入「建议买入时间 / 建议卖出时间」两个字段。

当前版本只给出“推荐标的 + 推荐时间点”，
不做自动资金/仓位控制，方便你手工下单。
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


# ----------------------------------------------------------------------
# 一些小工具
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 日打分流水线（带板块/ST 过滤 + 下单列表）"
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="任务 ID，例如 ultrashort_main",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="打分日期，格式 YYYY-MM-DD，例如 2025-10-31",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="模型 / 配置标签，例如 u1_v1_base_rf",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每天选出的股票数量（默认 3）",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=3.0,
        help="候选股票最低价格（会透传给 u1_daily_scoring_ml）",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=80.0,
        help="候选股票最高价格（会透传给 u1_daily_scoring_ml）",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=20_000_000.0,
        help="候选股票当日成交额下限（会透传给 u1_daily_scoring_ml）",
    )
    parser.add_argument(
        "--positions-csv",
        default=None,
        help=(
            "可选：昨日持仓文件路径（目前版本只占个位，"
            "以后做自动仓位管理时会用到）"
        ),
    )
    parser.add_argument(
        "--target-weight",
        type=float,
        default=0.03,
        help=(
            "建议单票目标权重，用于下单列表展示（默认 0.03=3%%，"
            "目前只做参考，不做强仓位约束）"
        ),
    )
    return parser.parse_args()


def run_scoring_engine(args: argparse.Namespace) -> None:
    """调用 U1 日打分引擎 u1_daily_scoring_ml。"""

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_daily_scoring_ml",
        "--job-id",
        args.job_id,
        "--as-of",
        args.as_of,
        "--top-k",
        str(args.top_k),
        "--min-price",
        str(args.min_price),
        "--max-price",
        str(args.max_price),
        "--min-amount",
        str(args.min_amount),
        "--tag",
        args.tag,
    ]

    print("[PIPELINE] 调用日打分引擎: {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=True)
    print("[PIPELINE] 日打分完成。")


def code_is_allowed(code) -> bool:
    """
    代码过滤规则（只用 code，本身跟名字无关）：

    - 剔除创业板： 300xxx, 301xxx
    - 剔除科创板： 688xxx, 689xxx
    - 大致剔除北交所： 430000~499999, 800000 以上
    """
    if pd.isna(code):
        return False

    s = str(code)
    # 去掉可能的 .SZ / .SH 后缀
    if "." in s:
        s = s.split(".", 1)[0]

    # 全部按 6 位补零
    if len(s) < 6:
        s = s.zfill(6)

    # 创业板
    if s.startswith(("300", "301")):
        return False

    # 科创板
    if s.startswith(("688", "689")):
        return False

    # 粗略北交所过滤
    try:
        n = int(s)
    except ValueError:
        return False

    # 430000~499999：老三板/北交所一大坨
    if 430000 <= n < 500000:
        return False

    # 800000 以上：基本也不是主板 A 股
    if n >= 800000:
        return False

    return True


def apply_security_filters(df: pd.DataFrame) -> pd.DataFrame:
    """在 full 打分结果上做板块 + ST 过滤。"""

    total_before = len(df)
    if "code" not in df.columns:
        raise ValueError("full 打分文件中缺少 code 列，无法做板块过滤。")

    # 代码过滤（科创 / 创业 / 北交等）
    mask_code = df["code"].apply(code_is_allowed)
    df_filtered = df[mask_code].copy()
    after_code = len(df_filtered)
    removed_code = total_before - after_code
    print(
        f"[PIPELINE] 代码过滤后剩余 {after_code} / {total_before} 条 "
        f"(剔除 {removed_code} 条科创/创业/北交股票)。"
    )

    # 名称过滤（ST / *ST）
    if "name" in df_filtered.columns:
        mask_st = ~df_filtered["name"].astype(str).str.contains(
            "ST", case=False, na=False
        )
        after_st = mask_st.sum()
        removed_st = after_code - after_st
        df_filtered = df_filtered[mask_st].copy()
        print(
            f"[PIPELINE] ST 名称过滤后剩余 {after_st} 条，"
            f"剔除 {removed_st} 条包含 ST 的股票。"
        )
    else:
        print("[PIPELINE] WARNING: full 打分文件中没有 name 列，无法按 ST 名称过滤。")

    return df_filtered


def parse_as_of_date(as_of: str) -> dt.date:
    """把 as_of 解析成 date 对象，支持 YYYY-MM-DD 或 YYYYMMDD。"""
    if "-" in as_of:
        return dt.datetime.strptime(as_of, "%Y-%m-%d").date()
    return dt.datetime.strptime(as_of, "%Y%m%d").date()


def next_weekday(d: dt.date) -> dt.date:
    """返回下一个工作日（简单跳过周六日，不处理法定节假日）。"""
    d = d + dt.timedelta(days=1)
    while d.weekday() >= 5:  # 5=周六, 6=周日
        d = d + dt.timedelta(days=1)
    return d


def compute_recommend_times(as_of_str: str) -> Tuple[str, str]:
    """
    给定 as_of（信号日），返回：
    - 建议买入时间：下一个工作日 09:31:00
    - 建议卖出时间：下一个工作日 14:55:00
    （返回字符串格式 YYYY-MM-DD HH:MM:SS）
    """
    as_of_date = parse_as_of_date(as_of_str)
    trade_date = next_weekday(as_of_date)

    buy_dt = dt.datetime.combine(trade_date, dt.time(9, 31, 0))
    sell_dt = dt.datetime.combine(trade_date, dt.time(14, 55, 0))

    return (
        buy_dt.strftime("%Y-%m-%d %H:%M:%S"),
        sell_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )


def build_orders_from_scores(args: argparse.Namespace) -> None:
    """
    从 full/topK 打分文件生成下单列表：
    - 先在 full 打分上做板块/ST 过滤 + 按 u1_score 重新排序；
    - 覆盖写回 full/topK 文件；
    - 生成 u1_orders_*.csv，并写入建议买入/卖出时间。
    """

    base_dir = Path(".")
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 文件名里的 as_of 一律用 YYYYMMDD 形式
    as_of_tag = args.as_of.replace("-", "")

    full_path = reports_dir / f"u1_scores_{args.job_id}_{args.tag}_{as_of_tag}_full.csv"
    topk_path = (
        reports_dir
        / f"u1_scores_{args.job_id}_{args.tag}_{as_of_tag}_top{args.top_k}.csv"
    )
    orders_path = (
        reports_dir / f"u1_orders_{args.job_id}_{args.tag}_{as_of_tag}.csv"
    )

    print(f"[PIPELINE] 预期 full 打分文件: {full_path}")
    print(f"[PIPELINE] 预期 topK 打分文件: {topk_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"找不到 full 打分文件: {full_path}")

    full_df = pd.read_csv(full_path)
    print(f"[PIPELINE] 已读取 full 打分记录 {len(full_df)} 条。")

    # 先做板块 + ST 过滤
    full_df = apply_security_filters(full_df)

    if "u1_score" not in full_df.columns:
        raise ValueError("full 打分文件中缺少 u1_score 列，无法排序。")

    # 按 u1_score 重新排序 + 打 rank
    full_df = full_df.sort_values("u1_score", ascending=False).reset_index(
        drop=True
    )
    full_df["rank"] = full_df.index + 1

    # 覆盖回写 full 文件（让后续看报告时看到的就是过滤后的结果）
    full_df.to_csv(full_path, index=False)
    print(f"[PIPELINE] 已覆盖保存过滤后的 full 打分文件: {full_path}")

    # 生成 topK 结果并覆盖 topK 文件
    topk_df = full_df.head(args.top_k).copy()
    print(
        f"[PIPELINE] 过滤并重排后 topK 记录数 {len(topk_df)} 条 "
        f"(top_k={args.top_k})。"
    )
    topk_df.to_csv(topk_path, index=False)
    print(f"[PIPELINE] 已覆盖保存过滤后的 topK 文件: {topk_path}")

    # 根据 topK 生成下单列表
    rec_buy_time, rec_sell_time = compute_recommend_times(args.as_of)
    print(
        f"[PIPELINE] 统一建议买入时间: {rec_buy_time}，"
        f"建议卖出时间: {rec_sell_time}"
    )

    orders_rows = []
    for _, row in topk_df.iterrows():
        orders_rows.append(
            {
                "job_id": args.job_id,
                "tag": args.tag,
                "as_of": as_of_tag,  # 下单表里 as_of 仍然用 YYYYMMDD 方便对齐
                "code": row["code"],
                "rank": int(row["rank"]),
                "u1_score": float(row["u1_score"]),
                "close": float(row.get("close", 0.0)),
                "amount": float(row.get("amount", 0.0)),
                # 目前版本只做多头买入
                "action": "BUY",
                # 目标权重只是一个建议参考值
                "target_weight": float(args.target_weight),
                # 新增：建议买入/卖出时间
                "rec_buy_time": rec_buy_time,
                "rec_sell_time": rec_sell_time,
            }
        )

    orders_df = pd.DataFrame(orders_rows)
    orders_df.to_csv(orders_path, index=False)
    print(f"[PIPELINE] 下单列表已保存: {orders_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(
        "===================================================================="
    )
    print(
        f"[PIPELINE] U1 日打分流水线启动，job_id={args.job_id}, tag={args.tag}, "
        f"as_of={args.as_of}, top_k={args.top_k}"
    )
    print(
        "===================================================================="
    )

    # 1) 调用日打分引擎
    run_scoring_engine(args)

    # 2) 基于打分结果 + 过滤规则 生成下单列表
    build_orders_from_scores(args)

    print(
        "===================================================================="
    )
    print("[PIPELINE] 全流程完成。")
    print(
        "===================================================================="
    )


if __name__ == "__main__":
    main()
