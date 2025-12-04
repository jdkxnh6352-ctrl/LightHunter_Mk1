# -*- coding: utf-8 -*-
"""
tests/test_factor_consistency.py

因子产出一致性测试
==================

检查点：
--------
1. 因子存储目录（factors_dir/factor_dir）下最近 N 个 Parquet 文件列集合一致
2. 最新因子文件中不存在“全为 NaN”的列
"""

import os
from typing import List

import pytest

import pandas as pd

from config.config_center import get_system_config


def _get_factors_dir() -> str:
    cfg = get_system_config()
    paths = cfg.get("paths", {}) or {}
    d = paths.get("factors_dir") or paths.get("factor_dir") or os.path.join("data", "factors")
    return d


def _list_factor_files(limit: int = 5) -> List[str]:
    d = _get_factors_dir()
    if not os.path.isdir(d):
        pytest.skip(f"因子目录不存在: {d}")
    files = [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith(".parquet")
    ]
    if not files:
        pytest.skip(f"因子目录中没有 parquet 文件: {d}")
    files.sort()
    return files[-limit:]


def test_factor_files_have_consistent_columns():
    files = _list_factor_files(limit=5)
    assert len(files) >= 2, "因子文件不足 2 个，无法做一致性对比"

    base_cols = None
    for fp in files:
        df = pd.read_parquet(fp)
        cols = set(df.columns)
        if base_cols is None:
            base_cols = cols
        else:
            assert cols == base_cols, (
                f"因子列集合不一致: {os.path.basename(fp)} 与其他文件列不一致，"
                f"当前列数={len(cols)}, 参考列数={len(base_cols)}"
            )


def test_latest_factor_file_has_no_all_nan_columns():
    files = _list_factor_files(limit=3)
    latest = files[-1]
    df = pd.read_parquet(latest)
    assert df.shape[1] > 0, f"最新因子文件没有任何列: {latest}"

    all_nan_cols = df.columns[df.isna().all()].tolist()
    assert not all_nan_cols, (
        f"最新因子文件存在全 NaN 因子列: {all_nan_cols[:10]} (共 {len(all_nan_cols)} 列)"
    )
