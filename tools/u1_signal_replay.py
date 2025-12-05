# tools/u1_signal_replay.py
#
# U1 历史信号回放：
# 1. 读取 ultrashort_main 数据集，获取指定区间内的所有交易日；
# 2. 对每个交易日调用 tools.u1_daily_scoring_ml 做一次“事后打分”；
# 3. 全部打分完成后，调用 tools.u1_signal_report 做一份“大号信号体检报告”。
#
# 用法示例：
#   python -m tools.u1_signal_replay \
#       --job-id ultrashort_main \
#       --tag u1_v1_base_rf \
#       --start-date 2020-01-01 \
#       --end-date 2025-10-31 \
#       --ret-col ret_1 \
#       --top-k 3 5 10 \
#       --n-buckets 10 \
#       --min-price 3 \
#       --max-price 80 \
#       --min-amount 20000000 \
#       --resume
#
# 说明：
# - 实际打分仍然由 tools.u1_daily_scoring_ml 完成，这里只是批量调度；
# - 信号评估仍然由 tools.u1_signal_report 完成；
# - 本脚本不会动你的 paper_trades / 实盘日志，只做“离线回放”。

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import subprocess
from pathlib import Path
from typing import List

import pandas as pd


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------


def detect_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 DataFrame 中自动识别日期列，并统一命名为 trade_date。

    支持的候选列名：
        - "trade_date"
        - "trading_date"
        - "date"
        - "as_of"
    """
    candidates = ["trade_date", "trading_date", "date", "as_of"]
    for col in candidates:
        if col in df.columns:
            if col != "trade_date":
                df = df.rename(columns={col: "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            return df

    raise ValueError("未在数据集中找到日期列（期望列名之一: trade_date / trading_date / date / as_of）")


def get_trading_days(dataset_path: Path, start_date: dt.date, end_date: dt.date) -> List[dt.date]:
    """从 parquet 数据集中提取 [start_date, end_date] 内的所有交易日（去重排序）。"""
    print(f"[Replay] 读取数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = detect_date_col(df)

    all_days = sorted(df["trade_date"].unique())
    days = [d for d in all_days if start_date <= d <= end_date]

    if not days:
        raise ValueError(f"在 {start_date} ~ {end_date} 区间内没有找到交易日，请检查数据集。")

    print(f"[Replay] 交易日总数: {len(days)} （{days[0]} ~ {days[-1]}）")
    return days


def build_scores_filename(job_id: str, tag: str, as_of: dt.date) -> Path:
    """根据 job_id / tag / 日期构造 full 打分文件名。"""
    ymd = as_of.strftime("%Y%m%d")
    fname = f"u1_scores_{job_id}_{tag}_{ymd}_full.csv"
    return Path("reports") / fname


def run_daily_scoring(
    job_id: str,
    tag: str,
    as_of: dt.date,
    top_k: int,
    min_price: float,
    max_price: float,
    min_amount: float,
) -> None:
    """
    调用已有的 tools.u1_daily_scoring_ml，对某个 as_of 日期打分。

    相当于：
        python -m tools.u1_daily_scoring_ml \
            --job-id JOB_ID \
            --as-of YYYY-MM-DD \
            --tag TAG \
            --top-k TOP_K \
            --min-price MIN_PRICE \
            --max-price MAX_PRICE \
            --min-amount MIN_AMOUNT
    """
    as_of_str = as_of.isoformat()
    print(f"[Replay] >>> 开始回放打分：as_of={as_of_str}")

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_daily_scoring_ml",
        "--job-id",
        job_id,
        "--as-of",
        as_of_str,
        "--tag",
        tag,
        "--top-k",
        str(top_k),
        "--min-price",
        str(min_price),
        "--max-price",
        str(max_price),
        "--min-amount",
        str(min_amount),
    ]

    # 这里让原脚本自己打印详细日志；如果报错会抛 CalledProcessError
    subprocess.run(cmd, check=True)
    print(f"[Replay] <<< 本日打分完成：as_of={as_of_str}")


def run_signal_report(
    job_id: str,
    tag: str,
    ret_col: str,
    top_k_list: List[int],
    n_buckets: int,
) -> None:
    """
    调用已有的 tools.u1_signal_report，基于所有 u1_scores_*_full.csv 做综合体检。
    """
    print("\n[Replay] 开始生成“大号信号体检报告”……")

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_signal_report",
        "--job-id",
        job_id,
        "--tag",
        tag,
        "--ret-col",
        ret_col,
        "--n-buckets",
        str(n_buckets),
        "--top-k",
    ] + [str(k) for k in top_k_list]

    subprocess.run(cmd, check=True)

    report_name = f"u1_signal_report_{job_id}_{tag}_{ret_col}.md"
    report_path = Path("reports") / report_name
    print(f"[Replay] 信号体检报告已生成：{report_path}")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 历史信号回放：批量调用 u1_daily_scoring_ml + u1_signal_report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--job-id", required=True, help="数据集 ID，例如 ultrashort_main")
    parser.add_argument("--tag", required=True, help="模型 / 配置标签，例如 u1_v1_base_rf")

    parser.add_argument("--start-date", required=True, help="回放起始日期，例如 2020-01-01")
    parser.add_argument("--end-date", required=True, help="回放结束日期，例如 2025-10-31")

    parser.add_argument("--ret-col", default="ret_1", help="用于评价信号的未来收益列，比如 ret_1 / ret_5 / ret_20")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="回放时主要关注的 top-k 列表，既会传给体检报告，也会影响 daily_scoring 的 top-k 上限",
    )
    parser.add_argument("--n-buckets", type=int, default=10, help="信号分位桶的个数（传给 u1_signal_report）")

    parser.add_argument("--min-price", type=float, default=3.0, help="价格过滤：最低股价")
    parser.add_argument("--max-price", type=float, default=80.0, help="价格过滤：最高股价")
    parser.add_argument("--min-amount", type=float, default=20_000_000.0, help="成交额过滤：最低日成交额")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="若目标打分文件已存在，则跳过该交易日（方便中断后续跑）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    job_id = args.job_id
    tag = args.tag

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    dataset_path = Path("data") / "datasets" / f"{job_id}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")

    # 1) 获取回放区间内所有交易日
    trading_days = get_trading_days(dataset_path, start_date, end_date)

    # daily_scoring 的 top-k 上限，只取列表中的最大值
    max_top_k = max(args.top_k)

    # 2) 按天回放打分
    print("\n[Replay] =============================================================")
    print(f"[Replay] 开始历史信号回放：job_id={job_id}, tag={tag}")
    print(f"[Replay] 回放区间：{start_date} ~ {end_date}，共 {len(trading_days)} 个交易日")
    print("[Replay] =============================================================\n")

    for i, as_of in enumerate(trading_days, start=1):
        scores_path = build_scores_filename(job_id, tag, as_of)

        if args.resume and scores_path.exists():
            print(f"[Replay] ({i}/{len(trading_days)}) {as_of} 已存在打分文件，跳过：{scores_path}")
            continue

        print(f"[Replay] ({i}/{len(trading_days)}) 处理日期：{as_of} ……")
        try:
            run_daily_scoring(
                job_id=job_id,
                tag=tag,
                as_of=as_of,
                top_k=max_top_k,
                min_price=args.min_price,
                max_price=args.max_price,
                min_amount=args.min_amount,
            )
        except subprocess.CalledProcessError as e:
            # 出错时给出提示，但不中断全流程（可以按需改成 raise 直接停掉）
            print(f"[Replay][WARN] 日期 {as_of} 打分失败，错误码 {e.returncode}，命令：{' '.join(e.cmd)}")
            print("               为了不中断整个回放流程，本日将被跳过。")
            continue

    print("\n[Replay] 所有日期的打分阶段已结束。")
    print("         下面进入“信号体检报告”汇总阶段。\n")

    # 3) 生成“大号信号体检报告”
    run_signal_report(
        job_id=job_id,
        tag=tag,
        ret_col=args.ret_col,
        top_k_list=args.top_k,
        n_buckets=args.n_buckets,
    )

    print("\n[Replay] 历史信号回放流程全部完成。")


if __name__ == "__main__":
    main()
