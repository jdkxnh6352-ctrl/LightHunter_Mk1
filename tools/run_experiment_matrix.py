# -*- coding: utf-8 -*-
"""
tools/run_experiment_matrix.py

LightHunter Mk4 - 实验矩阵批量运行器

功能：
- 读取 config/experiment_matrix.yaml
- 合并 defaults + 单个 experiment 配置（+ sweeps 展开）
- 逐个调用 alpha.training_pipelines.run_training_for_experiment
- 通过 lab.ExperimentLab 记录参数、指标、模型文件等信息

用法示例：
    python -m tools.run_experiment_matrix \
        --matrix config/experiment_matrix.yaml \
        --only US_CORE_V1,US_OF_ENHANCED_V1

"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("需要安装 pyyaml：pip install pyyaml") from e

from config.config_center import get_system_config  # type: ignore
from lab.experiment_lab import ExperimentLab  # type: ignore
from alpha.training_pipelines import run_training_for_experiment  # type: ignore

log = logging.getLogger("run_experiment_matrix")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归字典合并：override 覆盖 base，返回新 dict，不修改原对象。
    """
    result: Dict[str, Any] = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _apply_path_override(target: Dict[str, Any], path: str, value: Any) -> None:
    """
    根据类似 'dataset.label_col' 的路径在 target 中设置值。
    """
    keys = path.split(".")
    d: Dict[str, Any] = target
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]  # type: ignore
    d[keys[-1]] = value


def _expand_sweeps(
    experiments_map: Dict[str, Dict[str, Any]],
    sweeps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将 sweeps 定义展开为一组新的实验配置。
    sweeps 规范示例：
      - id: SWEEP_LABEL_HORIZON
        base_experiment: US_CORE_V1
        vary:
          dataset.label_col:
            - lbl_target_main_reg
            - lbl_ret_next_open
          filters.min_turnover:
            - 0.02
            - 0.05
    """
    if not sweeps:
        return []

    extra_experiments: List[Dict[str, Any]] = []

    for sweep in sweeps:
        sweep_id = sweep.get("id") or "SWEEP"
        base_id = sweep.get("base_experiment")
        vary = sweep.get("vary", {}) or {}
        if not base_id or base_id not in experiments_map:
            log.warning("Sweep %s 的 base_experiment=%s 未找到，跳过", sweep_id, base_id)
            continue

        base_exp = experiments_map[base_id]
        if not vary:
            continue

        keys = list(vary.keys())
        value_lists: List[List[Any]] = []
        for k in keys:
            vs = vary.get(k)
            if not isinstance(vs, list):
                log.warning("Sweep %s 中路径 %s 的 vary 值不是 list，跳过该路径", sweep_id, k)
                continue
            value_lists.append(vs)

        if not keys or not value_lists:
            continue

        # 构造笛卡尔积
        combos: List[Dict[str, Any]] = []

        def _recurse(idx: int, current: Dict[str, Any]) -> None:
            if idx == len(keys):
                combos.append(copy.deepcopy(current))
                return
            path = keys[idx]
            for val in value_lists[idx]:
                current[path] = val
                _recurse(idx + 1, current)
            current.pop(path, None)

        _recurse(0, {})

        for i, combo in enumerate(combos):
            new_exp = copy.deepcopy(base_exp)
            for path, val in combo.items():
                _apply_path_override(new_exp, path, val)

            new_id = f"{sweep_id}__{base_id}__{i+1}"
            new_exp["id"] = new_id
            # 标记来源
            new_exp.setdefault("tags", [])
            if isinstance(new_exp["tags"], list):
                if f"sweep:{sweep_id}" not in new_exp["tags"]:
                    new_exp["tags"].append(f"sweep:{sweep_id}")
            new_exp.setdefault("description", "")
            new_exp["description"] = (
                new_exp["description"]
                + f" [SWEEP {sweep_id} from {base_id} combo={json.dumps(combo, ensure_ascii=False)}]"
            )
            new_exp.setdefault("enabled", True)
            extra_experiments.append(new_exp)

    return extra_experiments


class ExperimentMatrixRunner:
    def __init__(self, system_config: Dict[str, Any], matrix_path: str) -> None:
        self.cfg = system_config
        self.matrix_path = matrix_path
        self.lab = ExperimentLab(system_config=self.cfg)

        paths_cfg = self.cfg.get("paths", {}) or {}
        project_root = paths_cfg.get("project_root", ".") or "."
        try:
            os.chdir(project_root)
        except Exception as e:  # pragma: no cover
            log.warning("切换到 project_root=%s 失败：%s", project_root, e)

        self._matrix: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.experiments_raw: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def load_matrix(self) -> None:
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"找不到 experiment_matrix 配置文件：{self.matrix_path}")
        with open(self.matrix_path, "r", encoding="utf-8") as f:
            self._matrix = yaml.safe_load(f) or {}

        self.defaults = self._matrix.get("defaults", {}) or {}
        exps = self._matrix.get("experiments", []) or []
        if not isinstance(exps, list):
            raise ValueError("experiment_matrix.yaml 中的 experiments 字段必须是 list。")
        self.experiments_raw = exps

        sweeps = self._matrix.get("sweeps", []) or []
        exp_map = {e.get("id"): e for e in exps if isinstance(e, dict) and e.get("id")}
        extra = _expand_sweeps(exp_map, sweeps)
        if extra:
            log.info("从 sweeps 展开出 %d 个额外实验。", len(extra))
            self.experiments_raw = exps + extra

    # ------------------------------------------------------------------

    def _iter_experiments(
        self,
        only_ids: Optional[Iterable[str]] = None,
        tag: Optional[str] = None,
    ) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        迭代有效实验：
        返回 (experiment_id, effective_cfg, raw_exp) 三元组。
        """
        only_set = set(only_ids) if only_ids else None

        for raw in self.experiments_raw:
            exp_id = raw.get("id")
            if not exp_id:
                continue
            if not raw.get("enabled", True):
                continue
            if only_set is not None and exp_id not in only_set:
                continue
            if tag:
                tags = raw.get("tags") or []
                if tag not in tags:
                    continue

            eff = _deep_merge(self.defaults, raw)
            eff["id"] = exp_id
            yield exp_id, eff, raw

    # ------------------------------------------------------------------

    def run(
        self,
        only_ids: Optional[Iterable[str]] = None,
        tag: Optional[str] = None,
        dry_run: bool = False,
        max_experiments: Optional[int] = None,
    ) -> None:
        self.load_matrix()

        to_run: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = list(
            self._iter_experiments(only_ids=only_ids, tag=tag)
        )
        if max_experiments is not None and max_experiments > 0:
            to_run = to_run[: max_experiments]

        if dry_run:
            print("将要运行的实验列表：")
            for exp_id, eff, _ in to_run:
                name = eff.get("name", "")
                print(f"- {exp_id}: {name}")
            return

        if not to_run:
            log.warning("没有匹配的实验需要运行（可能都被禁用或过滤掉）。")
            return

        matrix_id = os.path.basename(self.matrix_path)
        log.info(
            "开始运行实验矩阵：%s，共 %d 个实验。",
            matrix_id,
            len(to_run),
        )

        for idx, (exp_id, eff, raw) in enumerate(to_run, start=1):
            log.info("【%d/%d】开始实验 %s - %s", idx, len(to_run), exp_id, eff.get("name", ""))

            with self.lab.experiment_context(
                experiment_id=exp_id,
                matrix_id=matrix_id,
                config=eff,
            ) as run:
                try:
                    result = run_training_for_experiment(
                        experiment_id=exp_id,
                        exp_cfg=eff,
                        system_config=self.cfg,
                        experiment_lab=self.lab,
                        run=run,
                    )
                    # training_pipelines 内部已经调用 log_metrics/log_params/log_artifact
                    # 这里可以额外写一个 summary
                    summary = {
                        "train_metrics": result.get("train_metrics"),
                        "val_metrics": result.get("val_metrics"),
                        "test_metrics": result.get("test_metrics"),
                        "model_path": result.get("model_path"),
                    }
                    self.lab.append_note(run, "result_summary", summary)
                    log.info("实验 %s 完成。", exp_id)
                except Exception as e:
                    log.exception("实验 %s 执行失败：%s", exp_id, e)
                    # context manager 会自动标记为 failed

        log.info("所有实验运行完毕。")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter 实验矩阵批量运行器"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="experiment_matrix.yaml 路径（默认：config/experiment_matrix.yaml）",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只运行指定实验ID，逗号分隔，例如：US_CORE_V1,US_OF_ENHANCED_V1",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="按 tag 过滤实验（只运行包含该 tag 的实验）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要运行的实验列表，不真正执行",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="最多运行多少个实验（按顺序截断）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    cfg = get_system_config(refresh=False)
    paths_cfg = cfg.get("paths", {}) or {}
    project_root = paths_cfg.get("project_root", ".") or "."
    try:
        os.chdir(project_root)
    except Exception as e:  # pragma: no cover
        log.warning("切换到 project_root=%s 失败：%s", project_root, e)

    matrix_path = args.matrix
    if not matrix_path:
        config_dir = paths_cfg.get("config_dir", "config") or "config"
        matrix_path = os.path.join(config_dir, "experiment_matrix.yaml")

    only_ids: Optional[List[str]] = None
    if args.only:
        only_ids = [s.strip() for s in args.only.split(",") if s.strip()]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    runner = ExperimentMatrixRunner(cfg, matrix_path)
    runner.run(
        only_ids=only_ids,
        tag=args.tag,
        dry_run=args.dry_run,
        max_experiments=args.max_experiments,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
