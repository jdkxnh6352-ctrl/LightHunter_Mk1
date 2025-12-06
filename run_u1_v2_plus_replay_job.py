# -*- coding: utf-8 -*-
"""
run_u1_v2_plus_replay_job.py

功能：
1）为 U1 v2_plus 模型补齐 2020–2025 的全部历史打分（reports/u1_scores_*.csv）
2）在打分齐全后，调用 tools.u1_strategy_replay 做策略级回放（top3，无风控）

使用方法（在项目根目录）：
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

# 与 tools.u1_daily_scoring_ml 默认保持一致
MIN_TRAIN_DAYS = 80

# 如果你想人为截断起止日期，可以在这里改；
# 写 None 就表示“不限制，以数据 + min_train_days 为准”
START_DATE = None         # 例如 "2020-03-20"
END_DATE = None           # 例如 "2025-10-31"


def detect_date_col(df: pd.DataFrame) -> str:
    """自动识别日期列名。"""
    for col in ("trade_date", "trading_date", "date", "as_of"):
        if col in df.columns:
            return col
    raise ValueError("未找到日期列（期待列名之一：trade_date / trading_date / date / as_of）。")


def parse_date(s: str):
    """把 'YYYY-MM-DD' 字符串转成 pandas 时间戳."""
    return pd.to_datetime(s).normalize()


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

    # 规范成 DatetimeIndex，去重、排序
    all_dates = (
        pd.to_datetime(df[date_col].dropna())
        .dt.normalize()
        .sort_values()
        .unique()
    )
    all_dates_idx = pd.DatetimeIndex(all_dates)

    if all_dates_idx.size <= MIN_TRAIN_DAYS:
        raise SystemExit(
            f"[ERROR] 数据集中可用交易日只有 {all_dates_idx.size} 天，"
            f"少于 min_train_days={MIN_TRAIN_DAYS}，无法回放。"
        )

    first_date = all_dates_idx[0]
    last_date = all_dates_idx[-1]
    safe_start_date = all_dates_idx[MIN_TRAIN_DAYS]

    print(
        f"[Replay] 全部交易日范围: {first_date:%Y-%m-%d} ~ {last_date:%Y-%m-%d}，"
        f"共 {all_dates_idx.size} 天"
    )
    print(
        f"[Replay] 按 min_train_days={MIN_TRAIN_DAYS}，"
        f"首个可安全回放的交易日: {safe_start_date:%Y-%m-%d}"
    )

    # 应用可选的 START_DATE / END_DATE 限制，但不会早于 safe_start_date
    effective_start = safe_start_date
    if START_DATE:
        user_start = parse_date(START_DATE)
        if user_start < safe_start_date:
            print(
                f"[Replay] 提示：用户指定 start_date={user_start:%Y-%m-%d} "
                f"历史数据不足 {MIN_TRAIN_DAYS} 天，自动调整为 "
                f"{safe_start_date:%Y-%m-%d}。"
            )
        else:
            effective_start = user_start

    effective_end = None
    if END_DATE:
        effective_end = parse_date(END_DATE)

    # 选出实际用于回放的 as_of 交易日
    mask = all_dates_idx >= effective_start
    if effective_end is not None:
        mask &= all_dates_idx <= effective_end

    selected_dates = all_dates_idx[mask]

    # 再保险：如果用户给的 END_DATE 太早，可能没有任何可用日期
    if selected_dates.size == 0:
        raise SystemExit(
            "[ERROR] 过滤 START_DATE / END_DATE 后没有任何可用交易日，"
            "请检查配置。"
        )

    trade_dates = selected_dates.strftime("%Y-%m-%d").tolist()

    print(
        f"[Replay] 实际用于回放的 as_of 交易日数: {len(trade_dates)} 个，"
        f"范围 {trade_dates[0]} ~ {trade_dates[-1]}"
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
