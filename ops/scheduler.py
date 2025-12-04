# -*- coding: utf-8 -*-
"""
ops/scheduler.py

LightHunter Mk4 - 统一任务调度中心
=================================

职责：
- 读取：
  1）config/system_config.json 中的 jobs 块（老配置）
  2）config/jobs.yaml 中的 jobs 段（采集任务矩阵等新配置）
- 将两边的 job 合并（YAML 同名 job 覆盖 system_config.jobs）
- 按 schedule 字段定时启动子进程执行对应命令

支持的 schedule.mode：
- "daily"   : 每天固定时间，支持 days_of_week（1=周一…7=周日）
- "interval": 固定间隔触发（秒），不看时间点
- "manual"  : 手动触发（只支持 --run-once，不参加轮询）

运行方式：
- 列出所有配置的 Job：
    python -m ops.scheduler --list

- 仅执行一次指定 Job：
    python -m ops.scheduler --run-once collect_daily_ohlcv_multi

- 以守护模式轮询调度：
    python -m ops.scheduler --loop

说明：
- 所有子进程的 cwd 会设置为 system_config.paths.project_root
- 建议用 systemd / supervisor 把本模块常驻运行
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: PyYAML 未安装，请先在环境中安装 pyyaml / PyYAML。", file=sys.stderr)
    raise

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
        # 兜底：找不到配置中心就用空配置
        return {}


log = get_logger(__name__)


class JobScheduler:
    """
    简易但够用的调度器：
    - 支持 daily / interval / manual
    - 从 system_config.jobs 与 config/jobs.yaml 合并任务
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)

        self.system_config = system_config or {}
        paths_cfg = self.system_config.get("paths", {}) or {}

        # 作为子进程运行时的工作目录
        self.project_root = paths_cfg.get("project_root", ".") or "."

        # jobs.yaml 路径
        self.jobs_yaml_path = paths_cfg.get("jobs_yaml", "config/jobs.yaml")

        # 载入 job 定义
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._load_jobs()

        # 记录每个 job 上次执行时间
        self._last_run_at: Dict[str, dt.datetime] = {}

        # 默认轮询间隔
        self.default_tick_sec: int = 30

    # ------------------------------------------------------------------
    # 加载 / 列出 / 单次执行
    # ------------------------------------------------------------------

    def _load_jobs(self) -> None:
        """
        从 system_config.jobs + jobs.yaml 合并任务：
        - system_config.jobs 先载入
        - jobs.yaml 中同名任务覆盖
        """
        merged: Dict[str, Dict[str, Any]] = {}

        # 来源 1：system_config.jobs
        cfg_jobs = self.system_config.get("jobs", {}) or {}
        for job_id, job_def in cfg_jobs.items():
            merged[job_id] = self._normalize_job(job_id, job_def, source="config")

        # 来源 2：jobs.yaml
        yaml_jobs = self._load_jobs_from_yaml()
        for job_id, job_def in yaml_jobs.items():
            merged[job_id] = self._normalize_job(job_id, job_def, source="yaml")

        self.jobs = merged
        log.info("Scheduler 加载了 %d 个 job（config=%d, yaml=%d）。",
                 len(self.jobs), len(cfg_jobs), len(yaml_jobs))

    def _load_jobs_from_yaml(self) -> Dict[str, Dict[str, Any]]:
        """
        从 config/jobs.yaml 中读取 jobs 定义。
        """
        if not self.jobs_yaml_path:
            return {}

        path = self.jobs_yaml_path
        if not os.path.exists(path):
            log.warning("jobs.yaml 未找到：%s，跳过 YAML jobs。", path)
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            log.error("读取 jobs.yaml 失败：%s", e)
            return {}

        jobs = data.get("jobs", {}) or {}
        if not isinstance(jobs, dict):
            log.warning("jobs.yaml 中的 jobs 字段格式异常，期待 dict。")
            return {}

        return jobs

    @staticmethod
    def _normalize_job(job_id: str, raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        标准化 job 定义，补默认值，确保后续调度逻辑简单。
        """
        job = dict(raw) if raw is not None else {}
        job["id"] = job_id
        job.setdefault("description", "")
        job.setdefault("enabled", True)

        cmd = job.get("command")
        if not isinstance(cmd, list) or not cmd:
            log.warning("Job '%s'（来源=%s）未配置 command 或格式错误，将标记为 disabled。", job_id, source)
            job["enabled"] = False

        schedule = job.get("schedule") or {}
        mode = (schedule.get("mode") or "daily").lower()
        schedule["mode"] = mode

        # daily: 默认工作日
        if mode == "daily":
            schedule.setdefault("time", "03:00")
            schedule.setdefault("days_of_week", [1, 2, 3, 4, 5])

        # interval: 默认 60 秒
        if mode == "interval":
            schedule.setdefault("interval_sec", 60)

        job["schedule"] = schedule
        return job

    def list_jobs(self) -> None:
        """
        打印所有 job 信息。
        """
        if not self.jobs:
            print("当前未配置任何 job。")
            return

        print("当前已注册的 Jobs：")
        for job_id, job in sorted(self.jobs.items()):
            sched = job.get("schedule", {})
            enabled = job.get("enabled", True)
            desc = job.get("description", "")
            mode = sched.get("mode", "daily")
            print(f"- {job_id} [{'ENABLED' if enabled else 'DISABLED'}] ({mode})  {desc}")

    def run_job_once(self, job_id: str) -> int:
        """
        只执行一次指定 job。
        """
        job = self.jobs.get(job_id)
        if not job:
            log.error("未找到 job_id=%s", job_id)
            return 1

        if not job.get("enabled", True):
            log.warning("job_id=%s 当前被标记为 disabled，仍按请求执行一次。", job_id)

        return self._spawn_job(job_id, job, wait=True)

    # ------------------------------------------------------------------
    # 守护模式
    # ------------------------------------------------------------------

    def run_forever(self, tick_interval_sec: Optional[int] = None) -> None:
        """
        守护模式运行：根据 schedule 周期性检查并启动 job。
        """
        if tick_interval_sec is None or tick_interval_sec <= 0:
            tick_interval_sec = self.default_tick_sec

        log.info("Scheduler 启动，轮询间隔=%d 秒。", tick_interval_sec)
        try:
            while True:
                now = dt.datetime.now()
                for job_id, job in self.jobs.items():
                    try:
                        if self._should_run(job_id, job, now):
                            self._spawn_job(job_id, job, wait=False)
                    except Exception as e:  # pragma: no cover
                        log.error("检查或执行 job=%s 时发生异常：%s", job_id, e)
                time.sleep(tick_interval_sec)
        except KeyboardInterrupt:
            log.info("Scheduler 收到中断信号，准备退出。")

    # ------------------------------------------------------------------
    # 运行逻辑
    # ------------------------------------------------------------------

    def _should_run(self, job_id: str, job: Dict[str, Any], now: dt.datetime) -> bool:
        """
        判断某个 job 在当前时刻是否应该触发。
        """
        if not job.get("enabled", True):
            return False

        sched = job.get("schedule", {}) or {}
        mode = (sched.get("mode") or "daily").lower()

        last_run = self._last_run_at.get(job_id)

        # 手动任务只通过 --run-once 启动
        if mode == "manual":
            return False

        # 每日定时任务
        if mode == "daily":
            # 按 isoweekday 判定（1=周一…7=周日）
            days = sched.get("days_of_week", [1, 2, 3, 4, 5])
            if days and now.isoweekday() not in days:
                return False

            time_str = sched.get("time", "03:00")
            try:
                hour, minute = [int(x) for x in time_str.split(":", 1)]
            except Exception:
                log.warning("job=%s 的 time 字段解析失败：%s", job_id, time_str)
                return False

            scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < scheduled:
                # 还没到触发时间
                return False

            if last_run is None:
                # 当天第一次跑
                return True

            # 确保一天只跑一次
            if last_run.date() < now.date():
                return True
            return False

        # 间隔任务：简单按间隔秒数触发
        if mode == "interval":
            interval_sec = int(sched.get("interval_sec", 60))
            if interval_sec <= 0:
                return False

            if last_run is None:
                return True
            delta = (now - last_run).total_seconds()
            return delta >= interval_sec

        # 未知模式：默认不跑
        log.warning("job=%s 使用未知调度模式：%s，将跳过。", job_id, mode)
        return False

    def _spawn_job(self, job_id: str, job: Dict[str, Any], wait: bool) -> int:
        """
        启动一个 job 对应的子进程。
        """
        cmd = job.get("command") or []
        if not cmd:
            log.error("job=%s 未配置 command，无法执行。", job_id)
            return 1

        desc = job.get("description", "")
        log.info("准备执行 job=%s，描述=%s，命令=%s", job_id, desc, " ".join(cmd))

        # 更新 last_run 时间
        self._last_run_at[job_id] = dt.datetime.now()

        try:
            if wait:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    check=False,
                )
                if result.returncode == 0:
                    log.info("job=%s 执行完成，返回码=0。", job_id)
                else:
                    log.warning("job=%s 执行结束，返回码=%d。", job_id, result.returncode)
                return result.returncode
            else:
                subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=None,
                    stderr=None,
                )
                log.info("job=%s 已异步启动。", job_id)
                return 0
        except FileNotFoundError as e:
            log.error("job=%s 启动失败，可执行文件未找到：%s", job_id, e)
            return 1
        except Exception as e:
            log.error("job=%s 启动异常：%s", job_id, e)
            return 1


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LightHunter 统一任务调度器")
    parser.add_argument("--list", action="store_true", help="列出所有 job 并退出")
    parser.add_argument("--run-once", dest="run_once", help="只执行一次指定 job_id 然后退出")
    parser.add_argument("--loop", action="store_true", help="以守护模式轮询调度")
    parser.add_argument("--tick-sec", type=int, default=None, help="调度轮询间隔秒数（默认 30 秒）")

    args = parser.parse_args(argv)

    scheduler = JobScheduler()

    if args.list:
        scheduler.list_jobs()
        return

    if args.run_once:
        exit_code = scheduler.run_job_once(args.run_once)
        sys.exit(exit_code)

    # 默认行为：进入守护模式
    scheduler.run_forever(tick_interval_sec=args.tick_sec)


if __name__ == "__main__":
    main()
