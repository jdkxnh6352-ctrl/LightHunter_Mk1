# -*- coding: utf-8 -*-
"""
tests/test_backtest_stability.py

简单策略回测稳定性测试
======================

设计思路：
---------
- 假设 backtest_core 提供了一个冒烟函数：
  - run_smoke_backtest(system_config) 或
  - smoke_test(system_config)
- 函数返回 dict，包含 ret_ann / sharpe / max_dd 等指标。
- 重复运行两次，检查核心指标的相对误差在 5% 以内。
"""

from typing import Dict

import pytest

from config.config_center import get_system_config

bt_mod = pytest.importorskip("backtest_core")

_smoke_func = None
for _cand in ("run_smoke_backtest", "smoke_test"):
    if hasattr(bt_mod, _cand):
        _smoke_func = getattr(bt_mod, _cand)
        break

if _smoke_func is None:
    pytest.skip(
        "backtest_core 中未找到 run_smoke_backtest/smoke_test，"
        "请在 backtest_core 中实现一个简单冒烟回测函数用于稳定性测试。",
        allow_module_level=True,
    )


def _rel_diff(v1: float, v2: float) -> float:
    if v1 == 0 and v2 == 0:
        return 0.0
    denom = abs(v1) + 1e-9
    return abs(v1 - v2) / denom


def test_smoke_backtest_metrics_stable():
    cfg = get_system_config()
    r1: Dict = _smoke_func(cfg)
    r2: Dict = _smoke_func(cfg)

    assert isinstance(r1, dict) and isinstance(r2, dict), (
        "回测冒烟函数返回值应为 dict，包含核心绩效指标。"
    )

    metrics = ["ret_ann", "sharpe", "max_dd"]
    diffs = {}

    for m in metrics:
        if m in r1 and m in r2:
            v1, v2 = float(r1[m]), float(r2[m])
            diffs[m] = _rel_diff(v1, v2)

    if not diffs:
        pytest.skip("冒烟回测结果中未包含 ret_ann/sharpe/max_dd 等指标，无法做稳定性检查。")

    too_large = {m: d for m, d in diffs.items() if d > 0.05}  # 5% 容忍度
    assert not too_large, f"回测结果在重复运行间差异过大（>5%）：{too_large}"
