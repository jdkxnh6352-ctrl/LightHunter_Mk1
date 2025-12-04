from __future__ import annotations

import argparse
import logging
import os
from typing import List

import pandas as pd

from config.config_center import get_system_config
from alpha.training_pipelines import TrainingPipelines, WalkForwardConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_walkforward_backtests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter Walk‑Forward 滚动回测启动脚本（U1/U2/U3 等策略）",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="ultrashort_main",
        help="训练/回测任务的 job_id（通常与 alpha 任务配置对应）",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="U1,U2,U3",
        help="要进行滚动回测的策略 ID 列表，逗号分隔，如：U1,U2,U3",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Walk‑Forward 总体起始日期（含），格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Walk‑Forward 总体结束日期（含），格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=252 * 2,
        help="训练窗口长度（按自然日估算，实际以交易日过滤为准）",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=21,
        help="测试窗口长度（按自然日估算）",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="窗口滑动步长（默认与 test_days 相同）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sliding",
        choices=["sliding", "expanding"],
        help="窗口模式，sliding/expanding（目前主要影响 DatasetBuilder 的训练集构造逻辑）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每天按预测得分选取的 Top‑K 股票数",
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        default=None,
        help="ExperimentLab 中的 group 名称（不填则使用 job_id_wf）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_system_config()
    reports_dir = cfg.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    strategies: List[str] = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategies:
        raise SystemExit("必须指定至少一个策略 ID，例如：--strategies U1,U2,U3")

    wf_conf = WalkForwardConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        mode=args.mode,
    )

    logger.info(
        "启动 Walk‑Forward 回测：job_id=%s, strategies=%s, wf_conf=%s",
        args.job_id,
        strategies,
        wf_conf,
    )

    tp = TrainingPipelines(system_config=cfg)
    result = tp.run_walkforward_for_strategies(
        job_id=args.job_id,
        strategy_ids=strategies,
        wf_conf=wf_conf,
        experiment_group=args.experiment_group,
        top_k=args.top_k,
    )

    by_year = result["by_year"]
    by_fold = result["by_fold"]

    # 输出到控制台
    if isinstance(by_year, pd.DataFrame) and not by_year.empty:
        logger.info("==== 按年份汇总绩效 ====")
        logger.info("\n%s", by_year.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if isinstance(by_fold, pd.DataFrame) and not by_fold.empty:
        logger.info("==== 按阶段（Fold）汇总绩效 ====")
        logger.info("\n%s", by_fold.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    # 输出到文件（CSV）
    year_path = os.path.join(reports_dir, "walkforward_by_year.csv")
    fold_path = os.path.join(reports_dir, "walkforward_by_fold.csv")

    if isinstance(by_year, pd.DataFrame) and not by_year.empty:
        by_year.to_csv(year_path, index=False)
        logger.info("按年份汇总结果已写入：%s", year_path)

    if isinstance(by_fold, pd.DataFrame) and not by_fold.empty:
        by_fold.to_csv(fold_path, index=False)
        logger.info("按阶段汇总结果已写入：%s", fold_path)


if __name__ == "__main__":
    main()
