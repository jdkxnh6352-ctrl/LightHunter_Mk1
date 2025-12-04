# -*- coding: utf-8 -*-
"""
ops/night_ops.py

LightHunter Mk4 - NightOps 夜间科研流水线
========================================

职责：
- 串联整条夜间科研线：
  1）TS 数据修复（ts_data_repair）
  2）特征 & 标签流水线（ts_data_pipeline）
  3）训练（alpha.training_pipelines）
  4）回测（backtest_core）

运行方式：
    python -m ops.night_ops --mode full

mode 说明：
- full          : 1 -> 2 -> 3 -> 4
- data_only     : 1 -> 2
- train_only    : 3
- backtest_only : 4

依赖：
- config/system_config.json 中：
    paths.project_root
    alpha.default_job
    ops.night_ops.enabled / mode（可选）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str):
        return logging.getLogger(name)


try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        return {}


log = get_logger(__name__)


class NightOps:
    """
    NightOps 流水线总控。
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)
        self.system_config = system_config or {}

        paths_cfg = self.system_config.get("paths", {}) or {}
        self.project_root = paths_cfg.get("project_root", ".") or "."

        alpha_cfg = self.system_config.get("alpha", {}) or {}
        self.default_job_id = alpha_cfg.get("default_job", "ultrashort_main")

        ops_cfg = self.system_config.get("ops", {}) or {}
        self.night_ops_cfg = ops_cfg.get("night_ops", {}) or {}

        # 确保工作目录一致
        try:
            os.chdir(self.project_root)
        except Exception as e:
            log.warning("切换到 project_root=%s 失败：%s", self.project_root, e)

    # ------------------------------------------------------------------
    # 构建流水线
    # ------------------------------------------------------------------

    def build_pipeline(self, mode: str) -> List[Dict[str, Any]]:
        """
        根据 mode 构造待执行步骤列表。
        步骤为 dict：{name, command, fatal}
        """
        job_id = self.default_job_id

        # 默认流水线定义
        full_steps: List[Dict[str, Any]] = [
            {
                "name": "ts_repair",
                "desc": "TS 数据修复（缺失/异常修补）",
                "command": ["python", "-m", "ts_data_repair", "--mode", "auto"],
                "fatal": True,
            },
            {
                "name": "ts_data_pipeline",
                "desc": "特征 & 标签流水线重建",
                "command": ["python", "-m", "ts_data_pipeline", "--mode", "full"],
                "fatal": True,
            },
            {
                "name": "train_alpha",
                "desc": f"训练超短线模型（job={job_id}）",
                "command": ["python", "-m", "alpha.training_pipelines", "--job", job_id],
                "fatal": True,
            },
            {
                "name": "backtest_alpha",
                "desc": f"超短线策略回测（job={job_id}）",
                "command": ["python", "-m", "backtest_core", "--job", job_id],
                "fatal": False,
            },
        ]

        mode = mode.lower()
        if mode == "full":
            return full_steps
        if mode == "data_only":
            return full_steps[0:2]
        if mode == "train_only":
            return full_steps[2:3]
        if mode == "backtest_only":
            return full_steps[3:4]

        log.warning("未知 NightOps 模式：%s，将退回 full。", mode)
        return full_steps

    # ------------------------------------------------------------------
    # 执行逻辑
    # ------------------------------------------------------------------

    def run(self, mode: str = "full") -> int:
        """
        按给定 mode 运行 NightOps。
        """
        enabled = bool(self.night_ops_cfg.get("enabled", True))
        if not enabled:
            log.warning("ops.night_ops.enabled = false，NightOps 被显式关闭，直接退出。")
            return 0

        log.info("NightOps 启动，mode=%s，default_job_id=%s", mode, self.default_job_id)

        steps = self.build_pipeline(mode)
        for step in steps:
            ok = self._run_step(step)
            if not ok and step.get("fatal", True):
                log.error("NightOps 在步骤 '%s' 失败（fatal），中止流水线。", step.get("name"))
                return 1

        log.info("NightOps 流水线（mode=%s）全部完成。", mode)
        return 0

    def _run_step(self, step: Dict[str, Any]) -> bool:
        """
        执行单个步骤。
        """
        name = step.get("name", "unnamed")
        desc = step.get("desc", "")
        cmd = step.get("command") or []
        fatal = bool(step.get("fatal", True))

        if not cmd:
            log.error("NightOps 步骤 '%s' 未配置 command，跳过。", name)
            return not fatal

        log.info("NightOps 开始步骤：%s - %s", name, desc)
        log.info("命令：%s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=False,
            )
            if result.returncode == 0:
                log.info("NightOps 步骤 '%s' 完成，返回码=0。", name)
                return True
            else:
                log.warning(
                    "NightOps 步骤 '%s' 结束，返回码=%d。",
                    name,
                    result.returncode,
                )
                return False
        except FileNotFoundError as e:
            log.error("NightOps 步骤 '%s' 启动失败，可执行文件未找到：%s", name, e)
            return False
        except Exception as e:
            log.error("NightOps 步骤 '%s' 运行异常：%s", name, e)
            return False


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LightHunter NightOps 夜间科研流水线")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="NightOps 模式：full / data_only / train_only / backtest_only（默认读取 system_config.ops.night_ops.mode 或 full）",
    )
    args = parser.parse_args(argv)

    cfg = get_system_config(refresh=False)
    ops_cfg = (cfg.get("ops") or {}).get("night_ops") or {}
    default_mode = ops_cfg.get("mode", "full")

    mode = args.mode or default_mode

    night_ops = NightOps(system_config=cfg)
    exit_code = night_ops.run(mode=mode)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
