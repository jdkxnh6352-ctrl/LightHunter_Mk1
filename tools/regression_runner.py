# -*- coding: utf-8 -*-
"""
tools/regression_runner.py

LightHunter Mk3 - 扩展回归测试入口
===================================

用途
----
在代码或配置发生较大变动后，快速检查核心链路是否仍然可用：

1. 配置与路径检查（ConfigCenter + paths）
2. DuckDB / Parquet 存储检查
3. ZeroMQ EventBus 基础连通性（如可用）
4. ExperimentLab / PerformanceLab 基础可用性
5. 因子存储结构一致性（最近几日）
6. DatasetBuilder 构建一致性（同配置多次构建）
7. 简单策略回测稳定性（如 backtest_core 提供冒烟入口）
8. （可选）调用 pytest 跑 tests/ 下的单元 / 集成测试

建议使用方式
------------
- 上线前：手工运行，确认 PASS 后再上线
- 每日夜间：由 Scheduler 定时运行，作为日常战备自检的一部分

命令示例
--------
# 全量回归（包含 pytest）
python -m tools.regression_runner

# 仅运行快速内建检查（不跑 pytest）
python -m tools.regression_runner --quick

# 全量检查但跳过 pytest（比如 CI 中已经单独跑了 pytest）
python -m tools.regression_runner --skip-pytest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from core.logging_utils import get_logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str):
        return logging.getLogger(name)


log = get_logger(__name__)

try:  # pragma: no cover
    from config.config_center import get_system_config
except Exception:  # pragma: no cover
    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        return {}


@dataclass
class CheckResult:
    name: str
    ok: bool
    critical: bool = True
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "critical": self.critical,
            "message": self.message,
            "error": self.error,
        }


class RegressionRunner:
    """回归测试总控。"""

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        self.system_config = system_config or get_system_config(refresh=True)
        if not isinstance(self.system_config, dict):
            log.warning("get_system_config 返回值不是 dict，后续检查可能异常：%r", type(self.system_config))

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def run(self, quick: bool = False, run_pytest: bool = True) -> int:
        """
        运行回归测试。

        返回：
            0 表示所有 critical 检查通过；
            1 表示有 critical 检查失败。
        """
        results: List[CheckResult] = []

        results.append(self._check_config_paths())
        results.append(self._check_duckdb())
        results.append(self._check_event_bus())
        results.append(self._check_experiment_lab())

        if not quick:
            results.append(self._check_factor_storage_consistency())
            results.append(self._check_dataset_builder_consistency())
            results.append(self._check_backtest_stability())

        if run_pytest:
            results.append(self._run_pytest_suite())

        self._print_summary(results)

        # 只要有 critical 的检查失败，整体视为失败
        has_critical_fail = any((not r.ok) and r.critical for r in results)
        return 0 if not has_critical_fail else 1

    # ------------------------------------------------------------------
    # 单项检查
    # ------------------------------------------------------------------

    def _check_config_paths(self) -> CheckResult:
        name = "config_paths"
        try:
            cfg = self.system_config or {}
            paths = cfg.get("paths", {}) or {}

            required_keys = [
                "data_root",
                "duckdb_path",
                "factors_dir",
                "datasets_dir",
                "experiments_dir",
            ]
            missing = [k for k in required_keys if k not in paths]
            if missing:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=True,
                    message=f"paths 中缺少关键字段: {missing}",
                )

            # 检查目录存在性（不存在则尝试创建非关键目录）
            for key in ["factors_dir", "datasets_dir", "experiments_dir"]:
                d = paths.get(key)
                if not d:
                    continue
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception as e:
                    return CheckResult(
                        name=name,
                        ok=False,
                        critical=True,
                        message=f"创建/访问路径失败: {key}={d}, err={e}",
                        error=traceback.format_exc(),
                    )

            return CheckResult(
                name=name,
                ok=True,
                message="config_center & paths 检查通过",
            )
        except Exception as e:  # pragma: no cover
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message=f"读取配置失败: {e}",
                error=traceback.format_exc(),
            )

    def _check_duckdb(self) -> CheckResult:
        name = "duckdb_storage"
        try:
            import duckdb  # type: ignore
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message="duckdb 模块未安装，请在 requirements 中加入 duckdb 并安装。",
                error=str(e),
            )

        try:
            cfg = self.system_config or {}
            paths = cfg.get("paths", {}) or {}
            db_path = paths.get("duckdb_path") or "data/lighthunter.duckdb"
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            con = duckdb.connect(db_path, read_only=False)
            # 简单自检表
            con.execute("CREATE TABLE IF NOT EXISTS _regression_heartbeat (ts TIMESTAMP)")
            con.execute("INSERT INTO _regression_heartbeat VALUES (now())")
            con.execute("SELECT count(*) FROM _regression_heartbeat").fetchall()
            con.close()

            return CheckResult(
                name=name,
                ok=True,
                message=f"DuckDB 可写入可查询，路径: {db_path}",
            )
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message=f"DuckDB 检查失败: {e}",
                error=traceback.format_exc(),
            )

    def _check_event_bus(self) -> CheckResult:
        name = "event_bus"
        try:
            from bus.zmq_bus import ZmqBus  # type: ignore
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"未找到 ZmqBus（bus.zmq_bus），跳过事件总线检查: {e}",
                error=str(e),
            )

        try:
            bus = ZmqBus.from_system_config(self.system_config)  # type: ignore[attr-defined]
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message=f"ZmqBus 初始化失败，请检查 event_bus 配置: {e}",
                error=traceback.format_exc(),
            )

        # 如果 ZmqBus 有 health/ping 方法，就调用一下
        health_ok = True
        try:
            if hasattr(bus, "health_check"):
                h = bus.health_check()  # type: ignore[call-arg]
                health_ok = bool(h.get("ok", True))
            elif hasattr(bus, "ping"):
                bus.ping()  # type: ignore[call-arg]
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message=f"ZmqBus 健康检查失败: {e}",
                error=traceback.format_exc(),
            )

        if not health_ok:
            return CheckResult(
                name=name,
                ok=False,
                critical=True,
                message="ZmqBus health_check 返回非 OK",
            )

        return CheckResult(
            name=name,
            ok=True,
            message="事件总线 (ZmqBus) 初始化和基础健康检查通过",
        )

    def _check_experiment_lab(self) -> CheckResult:
        name = "experiment_lab"
        try:
            from lab.experiment_lab import ExperimentLab  # type: ignore
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"lab.experiment_lab 不存在，跳过实验记录检查: {e}",
                error=str(e),
            )

        try:
            if hasattr(ExperimentLab, "from_system_config"):
                lab = ExperimentLab.from_system_config(self.system_config)  # type: ignore[attr-defined]
            else:
                lab = ExperimentLab()  # type: ignore[call-arg]

            # 写入一个心跳实验
            exp_id = f"regression_heartbeat_{int(time.time())}"
            lab.log_experiment(
                experiment_id=exp_id,  # type: ignore[attr-defined]
                config={"type": "regression_heartbeat"},
                status="done",
                metrics={"ok": 1},
            )
            # 读取最近实验列表（若支持）
            if hasattr(lab, "list_recent_experiments"):
                _ = lab.list_recent_experiments(limit=5)

            return CheckResult(
                name=name,
                ok=True,
                message="ExperimentLab 心跳写入与基础查询通过",
            )
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"ExperimentLab 检查失败：建议检查实验目录配置及权限: {e}",
                error=traceback.format_exc(),
            )

    def _check_factor_storage_consistency(self) -> CheckResult:
        """
        因子产出一致性（存储层面）基础检查：
        - 从 factors_dir 寻找最近若干 Parquet 文件；
        - 检查最近 2~3 个文件的列集合是否一致；
        - 检查最新文件中是否存在“全为 NaN” 的因子列。
        """
        name = "factor_storage_consistency"
        try:
            if pd is None:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message="pandas 未安装，跳过因子存储一致性检查",
                )

            cfg = self.system_config or {}
            paths = cfg.get("paths", {}) or {}
            factors_dir = (
                paths.get("factors_dir")
                or paths.get("factor_dir")
                or os.path.join("data", "factors")
            )

            if not os.path.isdir(factors_dir):
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"因子目录不存在: {factors_dir}",
                )

            files = [
                os.path.join(factors_dir, f)
                for f in os.listdir(factors_dir)
                if f.lower().endswith(".parquet")
            ]
            if len(files) < 2:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"因子文件不足 2 个，当前共 {len(files)} 个，无法做一致性对比",
                )

            files.sort()
            recent_files = files[-3:]  # 最多取最近 3 个
            cols_set = None
            for fp in recent_files:
                df = pd.read_parquet(fp)
                cols = set(df.columns)
                if cols_set is None:
                    cols_set = cols
                else:
                    if cols != cols_set:
                        return CheckResult(
                            name=name,
                            ok=False,
                            critical=True,
                            message=f"因子列集合不一致：{os.path.basename(fp)} vs 其他文件",
                        )

            # 检查最新文件是否存在全 NaN 列
            latest_fp = recent_files[-1]
            df_latest = pd.read_parquet(latest_fp)
            if df_latest.shape[1] == 0:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=True,
                    message=f"最新因子文件没有任何列: {latest_fp}",
                )
            all_nan_cols = df_latest.columns[df_latest.isna().all()].tolist()
            if all_nan_cols:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"最新因子文件存在全 NaN 列: {all_nan_cols[:10]} ...",
                )

            return CheckResult(
                name=name,
                ok=True,
                message=f"因子文件列集合在最近 {len(recent_files)} 日内保持一致，无全空列",
            )

        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"因子存储一致性检查异常: {e}",
                error=traceback.format_exc(),
            )

    def _check_dataset_builder_consistency(self) -> CheckResult:
        """
        DatasetBuilder 构建一致性检查：
        - 从 alpha.dataset_builder 导入 DatasetBuilder；
        - 优先使用 list_datasets() 获取一个数据集 ID；
        - 连续构建两次，检查 shape 与列集合是否一致。
        """
        name = "dataset_builder_consistency"
        try:
            import importlib

            try:
                ds_mod = importlib.import_module("alpha.dataset_builder")
            except ImportError as e:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"未找到 alpha.dataset_builder，跳过数据集构建一致性检查: {e}",
                    error=str(e),
                )

            DatasetBuilder = getattr(ds_mod, "DatasetBuilder", None)
            if DatasetBuilder is None:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message="alpha.dataset_builder 中未找到 DatasetBuilder 类，跳过检查",
                )

            # 实例化
            if hasattr(DatasetBuilder, "from_system_config"):
                builder = DatasetBuilder.from_system_config(self.system_config)
            else:
                builder = DatasetBuilder(self.system_config)

            # 选择一个 dataset_id
            dataset_id = None
            if hasattr(builder, "list_datasets"):
                try:
                    ds_ids = builder.list_datasets()
                    if ds_ids:
                        dataset_id = ds_ids[0]
                except Exception:
                    dataset_id = None
            if not dataset_id:
                dataset_id = "default"

            # 构建两次
            ds1 = builder.build(dataset_id=dataset_id, split="train")
            ds2 = builder.build(dataset_id=dataset_id, split="train")

            # 尝试转换为 DataFrame（如果不是）
            if pd is None:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message="pandas 未安装，无法进行 dataset 一致性检查",
                )

            def to_frame(ds: Any) -> pd.DataFrame:
                if isinstance(ds, pd.DataFrame):
                    return ds
                if isinstance(ds, dict):
                    # 常见结构: {"X": DataFrame/ndarray, "y": Series/ndarray}
                    x = ds.get("X")
                    y = ds.get("y")
                    if isinstance(x, pd.DataFrame):
                        df = x.copy()
                        if y is not None:
                            if isinstance(y, pd.Series):
                                df["__label__"] = y
                            else:
                                # y 可能是 ndarray
                                df["__label__"] = pd.Series(list(y), index=df.index)
                        return df
                # 其他结构就尝试构造 DataFrame
                return pd.DataFrame(ds)

            df1 = to_frame(ds1)
            df2 = to_frame(ds2)

            if df1.shape != df2.shape:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=True,
                    message=f"DatasetBuilder 同一配置两次构建 shape 不一致: {df1.shape} vs {df2.shape}",
                )

            if set(df1.columns) != set(df2.columns):
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=True,
                    message="DatasetBuilder 同一配置两次构建列集合不一致",
                )

            # 这里不强制值完全相同（某些 builder 带随机采样），仅提示
            return CheckResult(
                name=name,
                ok=True,
                message=f"DatasetBuilder({dataset_id}) 构建一致性检查通过，shape={df1.shape}",
            )

        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"DatasetBuilder 一致性检查异常: {e}",
                error=traceback.format_exc(),
            )

    def _check_backtest_stability(self) -> CheckResult:
        """
        简易回测稳定性检查（如 backtest_core 提供冒烟函数）：

        - 尝试调用 backtest_core.run_smoke_backtest(system_config)
          或 backtest_core.smoke_test(system_config)
        - 若找到对应函数，则运行两次，对比核心指标是否稳定：
          - ret_ann / sharpe / max_dd 等差异是否在容忍范围内
        """
        name = "backtest_stability"
        try:
            import importlib

            try:
                bt_mod = importlib.import_module("backtest_core")
            except ImportError as e:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"未找到 backtest_core 模块，跳过回测稳定性检查: {e}",
                    error=str(e),
                )

            func = None
            for cand in ["run_smoke_backtest", "smoke_test"]:
                if hasattr(bt_mod, cand):
                    func = getattr(bt_mod, cand)
                    break

            if func is None:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message="backtest_core 中未找到 run_smoke_backtest/smoke_test，跳过检查",
                )

            r1 = func(self.system_config)
            r2 = func(self.system_config)

            if not isinstance(r1, dict) or not isinstance(r2, dict):
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message="backtest_core 冒烟函数未返回 dict，无法进行稳定性对比",
                )

            # 对几个常见指标进行比较
            metrics = ["ret_ann", "sharpe", "max_dd"]
            diffs = {}
            for m in metrics:
                if m in r1 and m in r2:
                    v1, v2 = float(r1[m]), float(r2[m])
                    if v1 == 0 and v2 == 0:
                        diff = 0.0
                    else:
                        diff = abs(v1 - v2) / (abs(v1) + 1e-9)
                    diffs[m] = diff

            # 如果没有任何公共指标，就视为通过但提示
            if not diffs:
                return CheckResult(
                    name=name,
                    ok=True,
                    message="backtest_core 冒烟函数返回中无常见指标（ret_ann/sharpe/max_dd），略过稳定性对比",
                )

            # 容忍 5% 的相对差异
            too_large = {m: d for m, d in diffs.items() if d > 0.05}
            if too_large:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"回测冒烟结果在重复运行间差异较大（>5%%）: {too_large}",
                )

            return CheckResult(
                name=name,
                ok=True,
                message=f"回测冒烟结果在重复运行间稳定，指标相对差异<=5%%: {diffs}",
            )

        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"回测稳定性检查异常: {e}",
                error=traceback.format_exc(),
            )

    def _run_pytest_suite(self) -> CheckResult:
        """
        调用 pytest 运行 tests/ 目录下的用例（包括我们新增的因子/数据集/回测测试）。
        """
        name = "pytest_tests"
        try:
            import pytest  # type: ignore
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"pytest 未安装，无法运行 tests/ 下的用例: {e}",
                error=str(e),
            )

        try:
            # -q 安静模式，仅输出结果
            exit_code = pytest.main(["-q", "tests"])
            if exit_code == 0:
                return CheckResult(
                    name=name,
                    ok=True,
                    message="pytest tests/ 用例全部通过",
                )
            else:
                return CheckResult(
                    name=name,
                    ok=False,
                    critical=False,
                    message=f"pytest 返回非零 exit_code={exit_code}，请查看测试输出",
                )
        except SystemExit as e:  # pytest 可能抛 SystemExit
            code = int(getattr(e, "code", 1) or 1)
            if code == 0:
                return CheckResult(
                    name=name,
                    ok=True,
                    message="pytest tests/ 用例全部通过（SystemExit=0）",
                )
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"pytest SystemExit code={code}，请查看测试输出",
            )
        except Exception as e:
            return CheckResult(
                name=name,
                ok=False,
                critical=False,
                message=f"pytest 运行异常: {e}",
                error=traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    # 总结输出
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(results: List[CheckResult]) -> None:
        log.info("=" * 72)
        log.info("LightHunter Mk3 回归测试结果汇总")
        log.info("-" * 72)

        rows = []
        for r in results:
            status = "OK" if r.ok else "FAIL"
            crit_flag = "C" if r.critical else "N"
            rows.append((r.name, status, crit_flag, r.message))

        # 简单对齐输出
        name_width = max(len(r[0]) for r in rows) if rows else 10
        log.info("%-*s | %-6s | %-3s | %s", name_width, "CHECK", "STATUS", "CRT", "MESSAGE")
        log.info("-" * (name_width + 40))
        for name, status, crit_flag, msg in rows:
            log.info("%-*s | %-6s | %-3s | %s", name_width, name, status, crit_flag, msg)

        critical_fail = [r for r in results if (not r.ok) and r.critical]
        if critical_fail:
            log.error("共有 %d 项 critical 检查失败，建议立即排查后再上线。", len(critical_fail))
        else:
            log.info("所有 critical 检查均已通过。")

        log.info("=" * 72)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LightHunter Mk3 回归测试入口")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="仅运行快速内建检查（跳过因子/dataset/回测稳定性）",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="跳过 pytest tests/ 用例执行，仅跑内建检查",
    )
    args = parser.parse_args(argv)

    cfg = get_system_config(refresh=True)
    runner = RegressionRunner(system_config=cfg)
    exit_code = runner.run(quick=args.quick, run_pytest=not args.skip_pytest)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
