# -*- coding: utf-8 -*-
"""
lab/experiment_lab.py

LightHunter Mk4 - 实验记录中心 ExperimentLab

功能：
- 为每个 experiment_id 管理多个 run（带时间戳）
- 记录：配置、参数、指标、备注、artifact（模型等）
- 提供上下文管理器，方便与训练/回测管线集成
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from config.config_center import get_system_config  # type: ignore

log = logging.getLogger("ExperimentLab")


@dataclass
class ExperimentRun:
    experiment_id: str
    run_id: str
    run_dir: str
    matrix_id: Optional[str] = None
    created_at: str = ""
    status: str = "running"  # running / success / failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentLab:
    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        root_dir: Optional[str] = None,
    ) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)
        self.cfg = system_config or {}
        paths_cfg = self.cfg.get("paths", {}) or {}

        project_root = paths_cfg.get("project_root", ".") or "."
        try:
            os.chdir(project_root)
        except Exception as e:  # pragma: no cover
            log.warning("切换到 project_root=%s 失败：%s", project_root, e)

        if root_dir is None:
            exp_dir_rel = paths_cfg.get("experiments_dir", "experiments")
            if os.path.isabs(exp_dir_rel):
                root_dir = exp_dir_rel
            else:
                root_dir = os.path.join(project_root, exp_dir_rel)

        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _write_json(self, path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _append_jsonl(self, path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    # ------------------------------------------------------------------
    # 运行生命周期
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        experiment_id: str,
        matrix_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> ExperimentRun:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_id}_{timestamp}"
        run_dir = os.path.join(self.root_dir, experiment_id, run_id)
        os.makedirs(run_dir, exist_ok=True)

        run = ExperimentRun(
            experiment_id=experiment_id,
            run_id=run_id,
            run_dir=run_dir,
            matrix_id=matrix_id,
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            status="running",
        )

        # 初始 run_info
        run_info_path = os.path.join(run_dir, "run_info.json")
        self._write_json(run_info_path, run.to_dict())

        # 保存配置
        if config is not None:
            cfg_path = os.path.join(run_dir, "config.json")
            self._write_json(cfg_path, config)

        log.info("启动实验运行：%s | run_dir=%s", run_id, run_dir)
        return run

    def end_experiment(
        self,
        run: ExperimentRun,
        status: str = "success",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        run.status = status
        run_info_path = os.path.join(run.run_dir, "run_info.json")
        info = run.to_dict()
        info["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        if extra:
            info.setdefault("extra", {})
            info["extra"] = {**info["extra"], **extra}
        self._write_json(run_info_path, info)
        log.info("实验运行结束：%s | status=%s", run.run_id, status)

    # ------------------------------------------------------------------
    # 日志接口
    # ------------------------------------------------------------------

    def log_params(
        self,
        run: ExperimentRun,
        params: Mapping[str, Any],
    ) -> None:
        path = os.path.join(run.run_dir, "params.json")
        existing: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        merged = {**existing, **params}
        self._write_json(path, merged)

    def log_metrics(
        self,
        run: ExperimentRun,
        metrics: Mapping[str, Any],
        split: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "metrics": dict(metrics),
        }
        if split:
            rec["split"] = split
        if step is not None:
            rec["step"] = step
        path = os.path.join(run.run_dir, "metrics.jsonl")
        self._append_jsonl(path, rec)

    def append_note(
        self,
        run: ExperimentRun,
        key: str,
        value: Any,
    ) -> None:
        """
        写入/更新 notes.json，用于存放 summary 等附加信息。
        """
        path = os.path.join(run.run_dir, "notes.json")
        existing: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[key] = value
        self._write_json(path, existing)

    def log_artifact(
        self,
        run: ExperimentRun,
        local_path: str,
        artifact_subdir: str = "artifacts",
    ) -> None:
        if not os.path.exists(local_path):
            log.warning("log_artifact: 文件不存在：%s", local_path)
            return
        file_name = os.path.basename(local_path)
        dest_dir = os.path.join(run.run_dir, artifact_subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_name)
        try:
            shutil.copy2(local_path, dest_path)
        except Exception as e:  # pragma: no cover
            log.warning("复制 artifact 失败：%s -> %s, err=%s", local_path, dest_path, e)
        else:
            log.info("artifact 已记录：%s -> %s", local_path, dest_path)

    # ------------------------------------------------------------------
    # 上下文管理器：推荐在外层使用，自动收尾
    # ------------------------------------------------------------------
    from contextlib import contextmanager

    @contextmanager
    def experiment_context(
        self,
        experiment_id: str,
        matrix_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        run = self.start_experiment(experiment_id=experiment_id, matrix_id=matrix_id, config=config)
        try:
            yield run
            self.end_experiment(run, status="success")
        except Exception as e:  # pragma: no cover
            log.exception("experiment_context 中发生异常：%s", e)
            self.end_experiment(run, status="failed", extra={"error": str(e)})
            raise


# 可选：提供一个默认的 lab 实例
_default_lab: Optional[ExperimentLab] = None


def get_experiment_lab() -> ExperimentLab:
    global _default_lab
    if _default_lab is None:
        _default_lab = ExperimentLab()
    return _default_lab
