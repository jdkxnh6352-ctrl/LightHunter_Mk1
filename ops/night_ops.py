# -*- coding: utf-8 -*-
"""
ops/night_ops.py

LightHunter Mk3 - NightOps 盘后科研流水线
========================================

目标：
------
在一个统一入口内完成（按能力尽可能调用实际模块，如果模块缺失则优雅降级）：

1. 环境自检（Config / DuckDB / ZeroMQ / Broker 等）
2. 数据体检（DataGuardian）
3. 数据修复（TSDataRepairer）
4. 因子 & 标签流水线（FactorEngine + Labeler）
5. 训练 & 回测（TrainingPipelines）
6. 生成简明运行报告（打印 + 写入 monitor/night_ops_report_*.json）

说明：
------
- 尽量调用已有模块：tools.regression_runner、data_guardian、ts_data_repair、
  factor_engine、ts_labeler、alpha.training_pipelines 等。
- 如果某些模块尚未完全实现/命名不同，NightOps 会捕获异常并记录为“跳过/失败”，
  不会让整个流程崩掉，方便你逐步填充后端逻辑。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config_center import get_system_config, resolve_path
from core.logging_utils import get_logger
from lab.experiment_lab import get_experiment_lab
from alpha.training_pipelines import run_training_and_backtest

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class StepReport:
    name: str
    ok: bool
    duration_sec: float
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NightOpsReport:
    date: str
    steps: List[StepReport]
    success: bool
    train_jobs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "success": self.success,
            "steps": [asdict(s) for s in self.steps],
            "train_jobs": self.train_jobs,
        }


# ----------------------------------------------------------------------
# 辅助函数：各阶段执行
# ----------------------------------------------------------------------


def _run_env_selfcheck(cfg: Dict[str, Any]) -> StepReport:
    """
    调用 regression_runner 做一轮 quick 自检。
    """
    t0 = time.time()
    try:
        from tools.regression_runner import run_tests  # type: ignore

        report = run_tests(mode="quick", with_pytest=False)
        summ = report.summary
        ok = bool(summ.get("success", False))
        msg = f"quick tests: {summ.get('passed')}/{summ.get('total')} passed, failed={summ.get('failed')}"
        extra = {"summary": summ}
        return StepReport(
            name="env_selfcheck",
            ok=ok,
            duration_sec=time.time() - t0,
            message=msg,
            extra=extra,
        )
    except Exception as e:
        log.exception("NightOps: 环境自检失败。")
        return StepReport(
            name="env_selfcheck",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"自检异常: {e!r}",
        )


def _run_data_guardian(cfg: Dict[str, Any]) -> StepReport:
    """
    调用 DataGuardian 做数据体检，如果模块不存在则友好跳过。
    """
    t0 = time.time()
    try:
        import data_guardian  # type: ignore

        # 尽量兼容多种实现方式
        if hasattr(data_guardian, "run_full_check"):
            result = data_guardian.run_full_check(cfg)  # type: ignore
        elif hasattr(data_guardian, "run_data_guardian"):
            result = data_guardian.run_data_guardian(cfg)  # type: ignore
        elif hasattr(data_guardian, "DataGuardian"):
            g = data_guardian.DataGuardian(cfg)  # type: ignore
            if hasattr(g, "run_full_check"):
                result = g.run_full_check()
            elif hasattr(g, "run"):
                result = g.run()
            else:
                result = None
        else:
            raise RuntimeError("data_guardian 模块存在，但未找到已知入口函数。")

        msg = "数据体检完成"
        extra: Dict[str, Any] = {}
        if isinstance(result, dict):
            extra.update(result)
            msg += f"，score={result.get('score')}"
        return StepReport(
            name="data_guardian",
            ok=True,
            duration_sec=time.time() - t0,
            message=msg,
            extra=extra,
        )
    except ImportError:
        log.warning("NightOps: 未找到 data_guardian 模块，跳过数据体检。")
        return StepReport(
            name="data_guardian",
            ok=False,
            duration_sec=time.time() - t0,
            message="模块缺失，已跳过。",
        )
    except Exception as e:
        log.exception("NightOps: 数据体检过程异常。")
        return StepReport(
            name="data_guardian",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"数据体检异常: {e!r}",
        )


def _run_data_repair(cfg: Dict[str, Any]) -> StepReport:
    """
    调用 TSDataRepairer 做数据修复。
    """
    t0 = time.time()
    try:
        import ts_data_repair as tdr  # type: ignore

        if hasattr(tdr, "run_repair_job"):
            result = tdr.run_repair_job(cfg, mode="night_ops")  # type: ignore
        elif hasattr(tdr, "TSDataRepairer"):
            r = tdr.TSDataRepairer(cfg)  # type: ignore
            if hasattr(r, "run_all"):
                result = r.run_all()
            elif hasattr(r, "run"):
                result = r.run()
            else:
                result = None
        else:
            raise RuntimeError("ts_data_repair 模块存在，但未找到已知入口函数。")

        msg = "数据修复完成"
        extra: Dict[str, Any] = {}
        if isinstance(result, dict):
            extra.update(result)
        return StepReport(
            name="data_repair",
            ok=True,
            duration_sec=time.time() - t0,
            message=msg,
            extra=extra,
        )
    except ImportError:
        log.warning("NightOps: 未找到 ts_data_repair 模块，跳过数据修复。")
        return StepReport(
            name="data_repair",
            ok=False,
            duration_sec=time.time() - t0,
            message="模块缺失，已跳过。",
        )
    except Exception as e:
        log.exception("NightOps: 数据修复过程异常。")
        return StepReport(
            name="data_repair",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"数据修复异常: {e!r}",
        )


def _run_factor_and_label_pipeline(cfg: Dict[str, Any]) -> StepReport:
    """
    调用 FactorEngine / Labeler 完成因子 & 标签流水线。
    """
    t0 = time.time()
    ok = True
    msgs: List[str] = []
    extra: Dict[str, Any] = {}

    # 因子流水线
    try:
        import factor_engine  # type: ignore

        if hasattr(factor_engine, "run_factor_pipeline"):
            r = factor_engine.run_factor_pipeline(cfg)  # type: ignore
        elif hasattr(factor_engine, "build_factor_panel"):
            r = factor_engine.build_factor_panel(cfg)  # type: ignore
        elif hasattr(factor_engine, "main"):
            r = factor_engine.main()  # type: ignore
        else:
            r = None
        msgs.append("因子流水线完成")
        if isinstance(r, dict):
            extra["factor"] = r
    except ImportError:
        ok = False
        msgs.append("因子模块缺失，已跳过")
        log.warning("NightOps: 未找到 factor_engine 模块，跳过因子流水线。")
    except Exception as e:
        ok = False
        msgs.append(f"因子流水线异常: {e!r}")
        log.exception("NightOps: 因子流水线异常。")

    # 标签流水线
    try:
        import ts_labeler  # type: ignore

        if hasattr(ts_labeler, "run_label_pipeline"):
            r = ts_labeler.run_label_pipeline(cfg)  # type: ignore
        elif hasattr(ts_labeler, "build_labels"):
            r = ts_labeler.build_labels(cfg)  # type: ignore
        elif hasattr(ts_labeler, "main"):
            r = ts_labeler.main()  # type: ignore
        else:
            r = None
        msgs.append("标签流水线完成")
        if isinstance(r, dict):
            extra["label"] = r
    except ImportError:
        ok = False
        msgs.append("标签模块缺失，已跳过")
        log.warning("NightOps: 未找到 ts_labeler 模块，跳过标签流水线。")
    except Exception as e:
        ok = False
        msgs.append(f"标签流水线异常: {e!r}")
        log.exception("NightOps: 标签流水线异常。")

    return StepReport(
        name="factor_and_label",
        ok=ok,
        duration_sec=time.time() - t0,
        message="; ".join(msgs),
        extra=extra,
    )


def _get_night_train_jobs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 system_config 中获取需要在 NightOps 中执行的训练任务列表。
    格式建议：
        "jobs": {
          "night_ops": {
            "train_jobs": [
              {"job_id": "ultrashort_main", "model_id": "ultrashort_xgb", "top_k": 20}
            ]
          }
        }
    若未配置，则使用 alpha.default_job + 第一个 alpha.models 作为兜底。
    """
    jobs_cfg = (cfg.get("jobs") or {}).get("night_ops") or {}
    train_jobs = jobs_cfg.get("train_jobs") or []
    if train_jobs:
        return train_jobs

    alpha_cfg = cfg.get("alpha") or {}
    default_job = alpha_cfg.get("default_job", "ultrashort_main")
    models_cfg = alpha_cfg.get("models") or {}
    if models_cfg:
        first_model_id = list(models_cfg.keys())[0]
    else:
        first_model_id = "ultrashort_xgb"

    log.warning(
        "NightOps: 未在 jobs.night_ops.train_jobs 中找到配置，使用兜底 job=%s, model=%s。",
        default_job,
        first_model_id,
    )
    return [{"job_id": default_job, "model_id": first_model_id, "top_k": 20}]


def _run_training_and_backtest(cfg: Dict[str, Any]) -> StepReport:
    """
    对配置中的每个 (job_id, model_id) 执行一次训练 + 回测。
    """
    t0 = time.time()
    jobs = _get_night_train_jobs(cfg)
    lab = get_experiment_lab(cfg)
    results: List[Dict[str, Any]] = []

    for job in jobs:
        job_id = job["job_id"]
        model_id = job["model_id"]
        top_k = int(job.get("top_k", 20))

        log.info("NightOps: 开始训练+回测 job=%s model=%s top_k=%d", job_id, model_id, top_k)
        try:
            # 使用 Mk3-Step-04 中的统一管线
            from alpha.training_pipelines import run_training_and_backtest  # type: ignore

            res = run_training_and_backtest(job_id=job_id, model_id=model_id, cfg=cfg)
            train_run_id = res["train_run_id"]
            backtest_run_id = res["backtest_run_id"]
            model_path = res["model_path"]

            # 尝试读取回测绩效（performance.json）
            bt_perf: Dict[str, Any] = {}
            try:
                run_dir = lab.get_run_dir(backtest_run_id)
                perf_path = run_dir / "performance.json"
                if perf_path.exists():
                    with perf_path.open("r", encoding="utf-8") as f:
                        bt_perf = json.load(f)
            except Exception:
                log.exception("NightOps: 读取回测绩效失败（忽略）。")

            results.append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "train_run_id": train_run_id,
                    "backtest_run_id": backtest_run_id,
                    "model_path": model_path,
                    "backtest_performance": bt_perf,
                }
            )
        except Exception as e:
            log.exception("NightOps: 训练+回测 job=%s model=%s 失败。", job_id, model_id)
            results.append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "error": repr(e),
                }
            )

    ok = all("error" not in r for r in results)
    msg = f"{len(results)} 个训练任务完成，其中成功 {sum('error' not in r for r in results)} 个。"

    return StepReport(
        name="train_and_backtest",
        ok=ok,
        duration_sec=time.time() - t0,
        message=msg,
        extra={"jobs": results},
    )


# ----------------------------------------------------------------------
# NightOps 主入口
# ----------------------------------------------------------------------


def run_night_ops(cfg: Optional[Dict[str, Any]] = None) -> NightOpsReport:
    """
    NightOps 主流程：依次执行所有阶段，并生成报告。
    """
    cfg = cfg or get_system_config()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    step_reports: List[StepReport] = []

    # 1. 环境自检
    step_reports.append(_run_env_selfcheck(cfg))

    # 2. 数据体检
    step_reports.append(_run_data_guardian(cfg))

    # 3. 数据修复
    step_reports.append(_run_data_repair(cfg))

    # 4. 因子 & 标签流水线
    step_reports.append(_run_factor_and_label_pipeline(cfg))

    # 5. 训练 & 回测
    train_step = _run_training_and_backtest(cfg)
    step_reports.append(train_step)

    success = all(s.ok for s in step_reports if s.name != "data_guardian" and s.name != "data_repair")
    train_jobs = train_step.extra.get("jobs", []) if train_step.extra else []

    report = NightOpsReport(
        date=date_str,
        steps=step_reports,
        success=success,
        train_jobs=train_jobs,
    )

    _print_night_report(report)
    _save_night_report(report, cfg)

    return report


def _print_night_report(report: NightOpsReport) -> None:
    """
    将 NightOps 结果打印到终端（便于你直接查看）。
    """
    print("=" * 70)
    print(f"NightOps Report - {report.date}")
    print(f"Overall Success: {report.success}")
    print("-" * 70)
    for s in report.steps:
        status = "OK" if s.ok else "FAIL"
        print(f"[{status}] {s.name} ({s.duration_sec:.1f}s) - {s.message}")
    print("-" * 70)
    if report.train_jobs:
        print("Train/Backtest Jobs Summary:")
        for job in report.train_jobs:
            if "error" in job:
                print(
                    f"  - job={job['job_id']} model={job['model_id']} ERROR={job['error']}"
                )
            else:
                perf = job.get("backtest_performance") or {}
                eq_ret = perf.get("equity_total_return")
                sharpe = perf.get("equity_sharpe")
                max_dd = perf.get("equity_max_drawdown")
                print(
                    "  - job={job} model={model} train_run={train} bt_run={bt} "
                    "ret={ret} sharpe={sharpe} maxDD={dd}".format(
                        job=job["job_id"],
                        model=job["model_id"],
                        train=job["train_run_id"],
                        bt=job["backtest_run_id"],
                        ret=f"{eq_ret:.2%}" if isinstance(eq_ret, (int, float)) else eq_ret,
                        sharpe=f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe,
                        dd=f"{max_dd:.2%}" if isinstance(max_dd, (int, float)) else max_dd,
                    )
                )
    print("=" * 70)


def _save_night_report(report: NightOpsReport, cfg: Dict[str, Any]) -> None:
    """
    将报告写入 monitor/night_ops_report_*.json，方便 DayOps / Dashboard 读取。
    """
    monitor_dir = resolve_path("monitor_dir", default="monitor", ensure_dir=True)
    date_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = monitor_dir / f"night_ops_report_{date_tag}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    log.info("NightOps 报告已写入：%s", path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    cfg = get_system_config()
    run_night_ops(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
