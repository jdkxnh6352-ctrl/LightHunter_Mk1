# -*- coding: utf-8 -*-
"""
examples/run_end_to_end_backtest.py

LightHunter Mk2 - 端到端回测示例脚本

目标：
------
从一个 job_id 出发，一次性跑完：
    1）构建训练/回测数据集
    2）训练 / 加载模型
    3）执行回测
    4）将结果记录到 ExperimentLab / PerformanceLab（由各模块自身负责）

注意：
------
- 本脚本主要演示“编排流程”的骨架，具体的命令行参数可能需要与你本地
  的 alpha.training_pipelines / backtest_core / performance_lab 中的 CLI 定义
  做微调。
- 建议从项目根目录运行：

    python -m examples.run_end_to_end_backtest \
        --job-id ultrashort_main \
        --model-id ultrashort_xgb \
        --retrain

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from config.config_center import get_system_config
from core.logging_utils import get_logger

# ExperimentLab 是整体科研记录中枢
try:
    from lab.experiment_lab import get_experiment_lab  # type: ignore
except Exception:
    get_experiment_lab = None  # type: ignore


log = get_logger(__name__)


def _run_subprocess(cmd: list[str], cwd: Optional[str] = None) -> int:
    """
    统一封装子进程调用，方便日志和错误处理。
    """
    cmd_display = " ".join(cmd)
    log.info("启动子进程：%s", cmd_display)
    proc = subprocess.Popen(cmd, cwd=cwd)
    proc.wait()
    rc = proc.returncode
    if rc != 0:
        log.error("子进程失败：%s (exit=%s)", cmd_display, rc)
    else:
        log.info("子进程完成：%s (exit=%s)", cmd_display, rc)
    return rc


def run_end_to_end(
    job_id: str,
    model_id: str,
    retrain: bool,
    dry_run: bool = False,
    extra_train_args: Optional[list[str]] = None,
    extra_backtest_args: Optional[list[str]] = None,
) -> int:
    """
    编排一个“端到端回测”流程。

    Args:
        job_id:     数据集 / 策略配置 ID（例如 ultrashort_main）
        model_id:   模型 ID（例如 ultrashort_xgb）
        retrain:    是否重新训练模型（否则假定已有模型可直接加载）
        dry_run:    若为 True，则只打印流程不实际执行子进程
        extra_train_args:  传给训练管线的额外参数
        extra_backtest_args: 传给回测核心的额外参数
    """
    cfg = get_system_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1) 启动 ExperimentLab run（可选）
    run_id: Optional[str] = None
    lab = None
    if get_experiment_lab is not None:
        try:
            lab = get_experiment_lab(cfg)
            run_id = lab.start_run(
                name=f"e2e_backtest_{job_id}_{model_id}",
                run_type="e2e_backtest",
                config={
                    "job_id": job_id,
                    "model_id": model_id,
                    "retrain": retrain,
                },
            )
            log.info("ExperimentLab run 创建成功: run_id=%s", run_id)
        except Exception:
            log.exception("ExperimentLab run 创建失败（忽略继续）。")
            lab = None
            run_id = None

    # 2) Step A: 训练 / 微调模型
    if retrain:
        train_cmd = [
            sys.executable,
            "-m",
            "alpha.training_pipelines",
            "--job",
            job_id,
            "--model",
            model_id,
            "--mode",
            "train",
        ]
        if extra_train_args:
            train_cmd.extend(extra_train_args)

        log.info("准备执行训练管线：job=%s model=%s", job_id, model_id)
        if dry_run:
            log.info("[DRY RUN] 训练命令：%s", " ".join(train_cmd))
        else:
            rc = _run_subprocess(train_cmd, cwd=project_root)
            if rc != 0:
                log.error("训练阶段失败，终止端到端流程。")
                _safe_end_run(lab, run_id, status="failed_train")
                return rc

    # 3) Step B: 回测
    backtest_cmd = [
        sys.executable,
        "-m",
        "backtest_core",
        "--job",
        job_id,
        "--model",
        model_id,
    ]
    if extra_backtest_args:
        backtest_cmd.extend(extra_backtest_args)

    log.info("准备执行回测：job=%s model=%s", job_id, model_id)
    if dry_run:
        log.info("[DRY RUN] 回测命令：%s", " ".join(backtest_cmd))
        final_rc = 0
    else:
        final_rc = _run_subprocess(backtest_cmd, cwd=project_root)

    # 4) 结束 ExperimentLab run（由 backtest_core 自己记录回测细节，这里只记录整体状态）
    if final_rc == 0:
        _safe_end_run(lab, run_id, status="ok")
    else:
        _safe_end_run(lab, run_id, status="failed_backtest")

    return final_rc


def _safe_end_run(lab: Any, run_id: Optional[str], status: str) -> None:
    if lab is None or not run_id:
        return
    try:
        lab.end_run(run_id, status=status, finished_at=datetime.utcnow().isoformat())
    except Exception:
        log.exception("ExperimentLab end_run 失败（忽略）。")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter Mk2 - 端到端回测示例 (examples/run_end_to_end_backtest.py)"
    )
    parser.add_argument(
        "--job-id",
        default="ultrashort_main",
        help="数据/策略 job ID（默认：ultrashort_main）",
    )
    parser.add_argument(
        "--model-id",
        default="ultrashort_xgb",
        help="模型 ID（默认：ultrashort_xgb）",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="是否重新训练模型（默认不重训，直接假定已有模型）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的命令，不真正运行子进程。",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        help="额外传给训练管线的参数（在 --train-args 之后写，例如：--train-args --epochs 20）",
    )
    # 简化：回测额外参数也复用 train-args 的写法，如需区分你可以自己扩展
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    extra_train_args = args.train_args or []
    # argparse.REMAINDER 会包含开头的 "--"，这里过滤掉
    extra_train_args = [a for a in extra_train_args if a != "--"]

    return run_end_to_end(
        job_id=args.job_id,
        model_id=args.model_id,
        retrain=args.retrain,
        dry_run=args.dry_run,
        extra_train_args=extra_train_args,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
