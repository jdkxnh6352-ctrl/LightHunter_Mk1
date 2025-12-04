# -*- coding: utf-8 -*-
"""
ops/day_ops.py

LightHunter Mk3 - DayOps 盘前体检
================================

目标：
------
- 在开盘前快速做一轮系统体检，确认：
    1) 配置 / DuckDB / ZeroMQ / Broker 等基本功能完好（调用 regression_runner quick）
    2) 最近一次回测结果是否合理（从 ExperimentLab 中读取最近的 backtest run 的 performance.json）
    3) 最近一次 NightOps 报告是否成功（monitor/night_ops_report_*.json）

- 输出一份简明报告（打印 + 写入 monitor/day_ops_report_*.json）供人工决策使用。

注意：
------
- DayOps 不做重型计算（不重新训练、不重跑回测），只做检查和汇总。
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

log = get_logger(__name__)


@dataclass
class StepReport:
    name: str
    ok: bool
    duration_sec: float
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DayOpsReport:
    date: str
    success: bool
    steps: List[StepReport]
    latest_backtest: Optional[Dict[str, Any]] = None
    latest_night_ops: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "success": self.success,
            "steps": [asdict(s) for s in self.steps],
            "latest_backtest": self.latest_backtest,
            "latest_night_ops": self.latest_night_ops,
        }


# ----------------------------------------------------------------------
# Step 1: 环境 quick 自检
# ----------------------------------------------------------------------


def _run_env_selfcheck(cfg: Dict[str, Any]) -> StepReport:
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
        log.exception("DayOps: 环境自检失败。")
        return StepReport(
            name="env_selfcheck",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"自检异常: {e!r}",
        )


# ----------------------------------------------------------------------
# Step 2: 最近一次 NightOps 报告
# ----------------------------------------------------------------------


def _load_latest_night_ops_report(cfg: Dict[str, Any]) -> StepReport:
    t0 = time.time()
    monitor_dir = resolve_path("monitor_dir", default="monitor", ensure_dir=True)

    try:
        reports = sorted(
            monitor_dir.glob("night_ops_report_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            msg = "未找到 NightOps 报告文件。"
            return StepReport(
                name="night_ops_report",
                ok=False,
                duration_sec=time.time() - t0,
                message=msg,
            )

        latest = reports[0]
        with latest.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ok = bool(data.get("success", False))
        msg = f"最新 NightOps 报告: file={latest.name}, success={ok}"
        return StepReport(
            name="night_ops_report",
            ok=ok,
            duration_sec=time.time() - t0,
            message=msg,
            extra={"report_path": str(latest), "report": data},
        )
    except Exception as e:
        log.exception("DayOps: 读取 NightOps 报告失败。")
        return StepReport(
            name="night_ops_report",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"读取报告异常: {e!r}",
        )


# ----------------------------------------------------------------------
# Step 3: 最近一次 backtest 绩效
# ----------------------------------------------------------------------


def _find_latest_backtest_run(cfg: Dict[str, Any]) -> Optional[str]:
    """
    从 ExperimentLab 的 run_index.json / backtest 目录中找到最近一个 backtest run_id。
    """
    from pathlib import Path

    lab = get_experiment_lab(cfg)
    experiments_dir = resolve_path("experiments_dir", ensure_dir=True)
    index_path = experiments_dir / "run_index.json"
    if not index_path.exists():
        return None

    try:
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception:
        log.exception("DayOps: 解析 run_index.json 失败。")
        return None

    # index_data: {run_id: {"run_type": "...", "path": "..."}}
    candidates: List[tuple[str, float]] = []
    for run_id, info in index_data.items():
        if info.get("run_type") != "backtest":
            continue
        path_str = info.get("path")
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            continue
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue
        candidates.append((run_id, mtime))

    if not candidates:
        return None

    latest_run_id = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
    return latest_run_id


def _load_backtest_performance(cfg: Dict[str, Any]) -> StepReport:
    t0 = time.time()
    lab = get_experiment_lab(cfg)

    try:
        run_id = _find_latest_backtest_run(cfg)
        if not run_id:
            msg = "未找到任何 backtest run。"
            return StepReport(
                name="latest_backtest",
                ok=False,
                duration_sec=time.time() - t0,
                message=msg,
            )

        run_dir = lab.get_run_dir(run_id)
        meta_path = run_dir / "meta.json"
        perf_path = run_dir / "performance.json"

        meta: Dict[str, Any] = {}
        perf: Dict[str, Any] = {}

        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        if perf_path.exists():
            with perf_path.open("r", encoding="utf-8") as f:
                perf = json.load(f)

        job_id = (meta.get("config") or {}).get("job_id")
        model_id = (meta.get("config") or {}).get("model_id")

        # 简单给出几项关键指标
        eq_ret = perf.get("equity_total_return")
        sharpe = perf.get("equity_sharpe")
        max_dd = perf.get("equity_max_drawdown")

        msg = (
            f"最新 backtest run_id={run_id}, job={job_id}, model={model_id}, "
            f"ret={eq_ret}, sharpe={sharpe}, maxDD={max_dd}"
        )

        ok = True
        if isinstance(eq_ret, (int, float)) and isinstance(max_dd, (int, float)):
            # 很粗糙的体检标准：回测收益为负且回撤过大则提示为“不健康”，但不强制失败
            if eq_ret < 0 and max_dd < -0.3:
                msg += " [WARNING: 回测收益偏差且回撤较大]"
        else:
            msg += " [INFO: 未检测到标准绩效字段]"

        return StepReport(
            name="latest_backtest",
            ok=ok,
            duration_sec=time.time() - t0,
            message=msg,
            extra={
                "run_id": run_id,
                "meta": meta,
                "performance": perf,
            },
        )
    except Exception as e:
        log.exception("DayOps: 读取 backtest 绩效失败。")
        return StepReport(
            name="latest_backtest",
            ok=False,
            duration_sec=time.time() - t0,
            message=f"读取 backtest 异常: {e!r}",
        )


# ----------------------------------------------------------------------
# DayOps 主入口
# ----------------------------------------------------------------------


def run_day_ops(cfg: Optional[Dict[str, Any]] = None) -> DayOpsReport:
    cfg = cfg or get_system_config()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    steps: List[StepReport] = []

    # 1. 环境 quick 自检
    steps.append(_run_env_selfcheck(cfg))

    # 2. 最近一次 NightOps 报告
    night_step = _load_latest_night_ops_report(cfg)
    steps.append(night_step)

    # 3. 最近一次 backtest 结果
    bt_step = _load_backtest_performance(cfg)
    steps.append(bt_step)

    # 你可以按需再加风险模块 / 网络检测等，这里先留扩展位
    # 例如：
    # steps.append(_run_riskbrain_preopen(cfg))

    # 简单总体判定：至少环境 OK，NightOps 报告成功 & backtest 可读取
    success = all(s.ok for s in steps if s.name in ("env_selfcheck", "night_ops_report", "latest_backtest"))

    latest_backtest = bt_step.extra if bt_step.extra else None
    latest_night_ops = night_step.extra.get("report") if night_step.extra else None

    report = DayOpsReport(
        date=date_str,
        success=success,
        steps=steps,
        latest_backtest=latest_backtest,
        latest_night_ops=latest_night_ops,
    )

    _print_day_report(report)
    _save_day_report(report, cfg)

    return report


def _print_day_report(report: DayOpsReport) -> None:
    print("=" * 70)
    print(f"DayOps Report - {report.date}")
    print(f"Overall Ready: {report.success}")
    print("-" * 70)
    for s in report.steps:
        status = "OK" if s.ok else "FAIL"
        print(f"[{status}] {s.name} ({s.duration_sec:.1f}s) - {s.message}")
    print("-" * 70)
    if report.latest_backtest:
        perf = report.latest_backtest.get("performance") or {}
        meta = report.latest_backtest.get("meta") or {}
        job_id = (meta.get("config") or {}).get("job_id")
        model_id = (meta.get("config") or {}).get("model_id")
        eq_ret = perf.get("equity_total_return")
        sharpe = perf.get("equity_sharpe")
        max_dd = perf.get("equity_max_drawdown")
        print("Latest Backtest Snapshot:")
        print(
            f"  job={job_id}, model={model_id}, "
            f"ret={eq_ret}, sharpe={sharpe}, maxDD={max_dd}"
        )
    if report.latest_night_ops:
        print("Latest NightOps Summary:")
        print(
            f"  success={report.latest_night_ops.get('success')}, "
            f"  train_jobs={len(report.latest_night_ops.get('train_jobs') or [])}"
        )
    print("=" * 70)


def _save_day_report(report: DayOpsReport, cfg: Dict[str, Any]) -> None:
    monitor_dir = resolve_path("monitor_dir", default="monitor", ensure_dir=True)
    date_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = monitor_dir / f"day_ops_report_{date_tag}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    log.info("DayOps 报告已写入：%s", path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    cfg = get_system_config()
    run_day_ops(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
