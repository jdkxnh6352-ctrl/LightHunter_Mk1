# -*- coding: utf-8 -*-
"""
run_u1_v2_plus_full_job.py

U1 v2_plus 因子版：一键跑完
1) 在 ultrashort_main 数据集中写入 v2_plus 因子
2) 用 v2_plus 因子做 Walk-forward 训练 + 回测

使用方式：
    在项目根目录下执行：
        python run_u1_v2_plus_full_job.py
"""

import datetime as dt
import subprocess
import sys
from pathlib import Path


# ========= 全局参数 =========

# 任务 ID / 数据集 ID
JOB_ID = "ultrashort_main"

# v2_plus 模型的 tag（模型文件、打分文件都会用到）
TAG_V2 = "u1_v2_plus_rf"

# 收益列（和你现在流水线保持一致）
RET_COL = "ret_1"

# 基础数据集路径
DATASET_PATH = Path("data/datasets/ultrashort_main.parquet")

# Walk-forward 训练区间（按你现在 2020–2025 的历史做法来定）
START_DATE = "2020-01-01"
END_DATE = "2025-10-31"

# 训练 / 打分用的选股过滤参数（保持和 U1 v1 一致）
TOP_K = 3
MIN_PRICE = 3.0
MAX_PRICE = 80.0
MIN_AMOUNT = 20_000_000.0


# ========= 小工具 =========

def run_step(title: str, cmd: list[str]) -> None:
    """打印命令并执行一个子步骤。"""
    print("\n" + "=" * 70)
    print(f"[FLOW] 开始步骤: {title}")
    print("[FLOW] 命令:", " ".join(str(c) for c in cmd))
    print("=" * 70)
    subprocess.run(cmd, check=True)
    print(f"[FLOW] 步骤完成: {title}")


# ========= 主流程 =========

def main() -> None:
    today = dt.date.today().strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"[FLOW] run_u1_v2_plus_full_job 启动, 今日日期: {today}")
    print(
        f"[FLOW] 任务 ID: {JOB_ID}, "
        f"U1 v2 模型 tag: {TAG_V2}, "
        f"收益列: {RET_COL}"
    )
    print("=" * 70)

    # ---------- 步骤 1：写入 v2_plus 因子 ----------
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"找不到基础数据集: {DATASET_PATH}，请确认路径是否正确。"
        )

    cmd_feat = [
        sys.executable,
        "-m",
        "tools.u1_features_v2_plus",
        "--input",
        str(DATASET_PATH),
    ]
    run_step("1) 在数据集中增加 u1_v2_plus 因子", cmd_feat)

    # ---------- 步骤 2：Walk-forward 训练 + 回测 ----------
    cmd_wf = [
        sys.executable,
        "-m",
        "tools.u1_walkforward_ml_backtest",
        "--job-id",
        JOB_ID,
        "--start-date",
        START_DATE,
        "--end-date",
        END_DATE,
        "--tag",
        TAG_V2,
        "--ret-col",
        RET_COL,
        "--top-k",
        str(TOP_K),
        "--min-price",
        str(MIN_PRICE),
        "--max-price",
        str(MAX_PRICE),
        "--min-amount",
        str(MIN_AMOUNT),
        # 如果 u1_walkforward_ml_backtest 里支持按 features 组名切换，
        # 可以打开下面这行，让它显式用 v2_plus 因子组：
        # "--features", "u1_v2_plus",
    ]
    run_step("2) Walk-forward 训练 + 回测 (v2_plus)", cmd_wf)

    # ---------- 预留：后续可以在这里继续接「日常打分 / 策略回放」 ----------
    # 比如：
    #   - 调用 tools.u1_daily_pipeline 做指定 as-of 的日内打分
    #   - 调用 tools.u1_strategy_replay / u1_strategy_replay_compare 做策略回放
    #
    # 先把训练这块跑稳定，等你这一步 OK 了，我们再一起把下面这部分补完整。

    print("\n" + "=" * 70)
    print("[FLOW] U1 v2_plus 全流程完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()
