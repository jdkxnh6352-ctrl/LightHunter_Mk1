# -*- coding: utf-8 -*-
"""
U1 日常调度脚本（单资产/多资产通用）

功能：
- 自动从 ultrashort_main.parquet 里识别最后一个交易日；
- 如果今天 > 数据集最后交易日，则自动把 as-of 调整为最后交易日；
- 自动推断“昨日持仓文件”路径；
- 调用 U1 日度流水线：tools.u1_daily_pipeline
"""

import sys
import subprocess
from pathlib import Path
from datetime import date, timedelta

import pandas as pd


# ------------------------
# 固定参数（按 U1 v1 base+rf）
# ------------------------
JOB_ID = "ultrashort_main"
TAG = "u1_v1_base_rf"

TOP_K = 3
MIN_PRICE = 3.0
MAX_PRICE = 80.0
MIN_AMOUNT = 20_000_000
TARGET_WEIGHT = 0.03

DATASET_PATH = Path("data") / "datasets" / f"{JOB_ID}.parquet"
REPORT_DIR = Path("reports")
LIVE_DIR = Path("data") / "live"


def detect_last_trade_date(dataset_path: Path) -> date:
    """
    从 parquet 里识别最后一个交易日。

    兼容两种列名：
    - trade_date
    - trading_date
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在：{dataset_path}")

    print(f"[JOB] 正在读取数据集：{dataset_path}")
    df = pd.read_parquet(dataset_path)

    if "trade_date" in df.columns:
        s = pd.to_datetime(df["trade_date"])
        col_name = "trade_date"
    elif "trading_date" in df.columns:
        s = pd.to_datetime(df["trading_date"])
        col_name = "trading_date"
    else:
        raise RuntimeError(
            f"数据集中没有 trade_date / trading_date 列，当前列为：{list(df.columns)}"
        )

    last_dt = s.max()
    last_date = last_dt.date()
    print(f"[JOB] 使用列 {col_name}，数据集最后交易日：{last_date}")
    return last_date


def build_positions_path(as_of: date) -> Path:
    """
    生成“昨日持仓文件”路径。
    这里只是按日期减一天；如果没有文件，会在 pipeline 里被视为当前空仓。
    """
    prev_date = as_of - timedelta(days=1)
    fname = f"positions_{prev_date.strftime('%Y%m%d')}.csv"
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_DIR / fname
    return path


def main() -> None:
    # 1. 目标 as-of（默认用“今天”）
    today = date.today()
    print(f"[JOB] 脚本运行日期（today）: {today}")

    # 2. 从数据集中识别最后交易日
    last_trade_date = detect_last_trade_date(DATASET_PATH)

    # 3. 如果 today 晚于数据集最后交易日，则自动往回“夹紧”
    if today > last_trade_date:
        effective_as_of = last_trade_date
        print(
            f"[JOB][WARN] 今天 {today} 晚于数据集最后交易日 {last_trade_date}，"
            f"本次任务 as-of 自动使用 {effective_as_of}"
        )
    else:
        effective_as_of = today
        print(f"[JOB] 使用今天作为 as-of：{effective_as_of}")

    as_of_str = effective_as_of.strftime("%Y-%m-%d")

    # 4. 推断昨日持仓文件路径
    positions_csv = build_positions_path(effective_as_of)
    if not positions_csv.exists():
        print(
            f"[JOB][WARN] 找不到昨日持仓文件：{positions_csv}，视为当前空仓。"
        )

    # 5. 组装并调用 U1 日度流水线
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_daily_pipeline",
        "--job-id",
        JOB_ID,
        "--as-of",
        as_of_str,
        "--tag",
        TAG,
        "--top-k",
        str(TOP_K),
        "--min-price",
        str(MIN_PRICE),
        "--max-price",
        str(MAX_PRICE),
        "--min-amount",
        str(MIN_AMOUNT),
        "--positions-csv",
        str(positions_csv),
        "--target-weight",
        str(TARGET_WEIGHT),
    ]

    print("\n[RUN] 即将执行：")
    print(" ", " ".join(cmd), "\n")

    subprocess.run(cmd, check=True)
    print("[JOB] 日常 U1 任务完成。")


if __name__ == "__main__":
    main()
