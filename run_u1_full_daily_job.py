#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_u1_full_daily_job.py

日常全流程：
1. 更新 ultrashort_main 数据集（从原始行情 / 分钟线等数据生成 ultrashort_main.parquet）
2. 做一次数据缺口检查（按股票逐日统计缺失情况，输出 markdown 报告）
3. 跑 U1 日打分流水线（u1_daily_pipeline / run_u1_daily_job）
4. 追加纸上实盘记录（把当日下单列表写进 paper_trades_ultrashort_main.csv）
5. 刷新纸上实盘绩效评估报告（权益曲线 + 年度统计 + markdown）

你可以直接用 Windows 计划任务每天晚上跑一遍：
    python run_u1_full_daily_job.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import date

import pandas as pd


# ========= 全局配置 =========

# 研究任务 ID
JOB_ID = "ultrashort_main"

# 当前上线的 U1 模型 tag（我们之前定好的 U1 v1 base+rf）
U1_TAG = "u1_v1_base_rf"

# 数据集 & 日志路径（按你项目现在的约定来）
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "datasets" / f"{JOB_ID}.parquet"
LIVE_DIR = BASE_DIR / "data" / "live"
REPORTS_DIR = BASE_DIR / "reports"


# ========= 工具函数 =========

def run_step(name: str, cmd: list[str]) -> None:
    """包装一下 subprocess.run，统一打印日志。"""
    print("\n" + "=" * 80)
    print(f"[FLOW] 开始步骤：{name}")
    print("[FLOW] 命令：", " ".join(map(str, cmd)))
    print("=" * 80 + "\n")

    subprocess.run(cmd, check=True)
    print(f"[FLOW] 步骤完成：{name}\n")


def detect_latest_trading_date() -> str:
    """
    从 ultrashort_main.parquet 里推断最新交易日（字符串格式：YYYYMMDD）

    会自动适配 trade_date / trading_date / date 三种列名中的任意一个。
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"找不到数据集文件：{DATASET_PATH}")

    print(f"[FLOW] 读取数据集以检测最新交易日：{DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)

    for col in ("trade_date", "trading_date", "date"):
        if col in df.columns:
            date_col = col
            break
    else:
        raise ValueError(
            f"在数据集中未找到日期列（期望列名之一：trade_date / trading_date / date），"
            f"实际列名：{list(df.columns)}"
        )

    # 转成日期类型再取最大
    s = pd.to_datetime(df[date_col])
    latest = s.max().date()
    print(f"[FLOW] 检测到数据集中最新交易日：{latest.isoformat()}")

    return latest.strftime("%Y%m%d")


# ========= 主流程 =========

def main() -> None:
    today = date.today()
    print("=" * 80)
    print(f"[FLOW] run_u1_full_daily_job 启动，今天日期：{today.isoformat()}")
    print(f"[FLOW] 任务 ID: {JOB_ID}, U1 模型 tag: {U1_TAG}")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 1. 更新 ultrashort_main 数据集
    # ------------------------------------------------------------------
    #
    # 默认调用根目录下的 build_ultrashort_main_dataset.py。
    # 这个脚本内部最好自己做好：
    #   - 从最新的原始行情 / 分钟线 / tick 库里，把到“昨天收盘”的数据都加工好
    #   - 输出到 data/datasets/ultrashort_main.parquet
    #
    # 如果你后面把数据更新逻辑拆成多步（ts_collector / ts_data_pipeline 等），
    # 在这里把命令改成一串即可。
    run_step(
        "1) 更新 ultrashort_main 数据集",
        [sys.executable, "build_ultrashort_main_dataset.py", "--job-id", JOB_ID],
    )

    # ------------------------------------------------------------------
    # 2. 数据缺口检查（按股票逐日统计缺失情况）
    # ------------------------------------------------------------------
    run_step(
        "2) 数据缺口检查 (tools.u0_data_gap_check)",
        [
            sys.executable,
            "tools/u0_data_gap_check.py",
            "--job-id",
            JOB_ID,
        ],
    )

    # ------------------------------------------------------------------
    # 3. U1 日打分 + 下单列表生成
    # ------------------------------------------------------------------
    #
    # run_u1_daily_job.py 里已经封装好了：
    #   - 自动从 ultrashort_main.parquet 里找最后一个交易日当 as-of
    #   - 按我们定好的 U1 v1 参数调用 tools.u1_daily_pipeline
    #   - 读取前一交易日的持仓 CSV（找不到就当作空仓）
    #   - 输出 full scores 和 topK 两个 CSV
    run_step(
        "3) U1 日打分流水线 (run_u1_daily_job)",
        [sys.executable, "run_u1_daily_job.py"],
    )

    # ------------------------------------------------------------------
    # 4. 追加纸上实盘记录
    # ------------------------------------------------------------------
    #
    # 这里我们不手动填 as-of，而是直接从最新数据集里推断，
    # 保证和日打分流水线用的是同一个交易日。
    as_of = detect_latest_trading_date()

    run_step(
        f"4) 纸上实盘记录追加 (tools.u1_paper_recorder, as_of={as_of})",
        [
            sys.executable,
            "-m",
            "tools.u1_paper_recorder",
            "--job-id",
            JOB_ID,
            "--tag",
            U1_TAG,
            "--as-of",
            as_of,
        ],
    )

    # ------------------------------------------------------------------
    # 5. 刷新纸上实盘绩效评估
    # ------------------------------------------------------------------
    #
    # 会读取 data/live/paper_trades_ultrashort_main.csv，
    # 生成：
    #   - 折线权益：reports/u1_paper_equity_*.csv
    #   - 按年份统计：reports/u1_paper_yearly_*.csv
    #   - markdown 分析报告：reports/u1_paper_*.md
    run_step(
        "5) 纸上实盘绩效评估 (tools.u1_paper_perf_lab)",
        [
            sys.executable,
            "-m",
            "tools.u1_paper_perf_lab",
            "--job-id",
            JOB_ID,
            "--tag",
            U1_TAG,
        ],
    )

    print("\n" + "=" * 80)
    print("[FLOW] 全流程完成 ✅")
    print(f"[FLOW] 你可以查看：")
    print(f"  - 数据集：{DATASET_PATH}")
    print(f"  - 纸上实盘日志：{LIVE_DIR / f'paper_trades_{JOB_ID}.csv'}")
    print(f"  - 日打分结果：{REPORTS_DIR}/u1_scores_*")
    print(f"  - 下单列表：{REPORTS_DIR}/u1_orders_*")
    print(f"  - 数据体检 / 缺口 & 纸上实盘 markdown 报告：{REPORTS_DIR}/*.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
