# -*- coding: utf-8 -*-
"""
run_u1_v2_plus_replay_job.py

功能：
1）为 U1 v2_plus 模型补齐 2020–2025 全部历史打分（u1_scores_*.csv）
2）在打分齐全后，调用 tools.u1_strategy_replay 做策略级回放（top3，无风控）

使用方法：
    python run_u1_v2_plus_replay_job.py
"""

import sys
import subprocess
from pathlib import Path

import pandas as pd

# ===== 全局配置 =====
JOB_ID = "ultrashort_main"
TAG = "u1_v2_plus_rf"
RET_COL = "ret_1"
TOP_K = 3

# 与 v1 一致的价格 / 成交额过滤
MIN_PRICE = 3.0
MAX_PRICE = 80.0
MIN_AMOUNT = 20_000_000

# 使用哪一组特征（在 tools.u1_daily_scoring_ml 里定义）
FEATURES = "v2_plus"


def detect_date_col(df: pd.DataFrame) -> str:
    """自动识别日期列名。"""
    for col in ("trade_date", "date", "as_of"):
        if col in df.columns:
            return col
    raise ValueError("未找到日期列（期待列名之一：trade_date / date / as_of）。")


def main() -> None:
    root = Path(__file__).resolve().parent

    print("============================================================")
    print("[FLOW] run_u1_v2_plus_replay_job 启动")
    print(f"[FLOW] 任务 ID: {JOB_ID}, 模型 tag: {TAG}, 收益列: {RET_COL}, top_k: {TOP_K}")
    print("============================================================")

    # ---------- 读取基础数据集，确定交易日 ----------
    data_path = root / "data" / "datasets" / f"{JOB_ID}.parquet"
    if not data_path.exists():
        raise SystemExit(f"[ERROR] 找不到数据集: {data_path}")

    print(f"[Replay] 读取基础数据集: {data_path}")
    df = pd.read_parquet(data_path)

    date_col = detect_date_col(df)
    print(f"[Replay] 使用日期列: {date_col}")

    # 取去重后的交易日列表，并按时间排序
    trade_dates = (
        pd.to_datetime(df[date_col].dropna())
        .sort_values()
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # ---------- 第一步：批量生成历史打分 ----------
    print("============================================================")
    print(f"[Replay] 开始批量生成历史打分（模型: {TAG}, top_k={TOP_K}）……")

    for as_of in trade_dates:
        as_of_key = as_of.replace("-", "")
        score_csv = reports_dir / f"u1_scores_{JOB_ID}_{TAG}_{as_of_key}_full.csv"

        if score_csv.exists():
            print(f"[Replay] as_of={as_of} 已存在打分文件，跳过: {score_csv.name}")
            continue

        cmd = [
            py,
            "-m",
            "tools.u1_daily_scoring_ml",
            "--job-id",
            JOB_ID,
            "--as-of",
            as_of,
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
            "--features",
            FEATURES,
        ]
        print(f"[Replay] 生成打分: as_of={as_of} ……")
        subprocess.run(cmd, check=True)

    print("[Replay] 历史打分阶段已结束。")
    print("============================================================")

    # ---------- 第二步：调用策略回放 ----------
    print("[Replay] 开始执行策略级历史回放……")

    cmd = [
        py,
        "-m",
        "tools.u1_strategy_replay",
        "--job-id",
        JOB_ID,
        "--tag",
        TAG,
        "--ret-col",
        RET_COL,
        "--top-k",
        str(TOP_K),
    ]
    subprocess.run(cmd, check=True)

    print("[Replay] DONE: 策略级历史回放完成。")
    print("============================================================")


if __name__ == "__main__":
    main()
