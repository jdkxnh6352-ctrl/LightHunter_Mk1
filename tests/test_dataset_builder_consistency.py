# -*- coding: utf-8 -*-
"""
tests/test_dataset_builder_consistency.py

DatasetBuilder 构建一致性测试
==============================

检查点：
--------
1. 同一 dataset_id + split，多次构建结果 shape 一致
2. 列集合一致（不要求数值逐元素完全一致——考虑到随机采样）
"""

from typing import Any

import pytest
import pandas as pd

from config.config_center import get_system_config

# 若没有 alpha.dataset_builder，就跳过整个测试文件
ds_mod = pytest.importorskip("alpha.dataset_builder")
DatasetBuilder = getattr(ds_mod, "DatasetBuilder", None)
if DatasetBuilder is None:
    pytest.skip("alpha.dataset_builder 中未找到 DatasetBuilder 类", allow_module_level=True)


def _to_frame(ds: Any) -> pd.DataFrame:
    if isinstance(ds, pd.DataFrame):
        return ds
    if isinstance(ds, dict):
        x = ds.get("X")
        y = ds.get("y")
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            if y is not None:
                if isinstance(y, pd.Series):
                    df["__label__"] = y
                else:
                    df["__label__"] = pd.Series(list(y), index=df.index)
            return df
    # 兜底：尝试直接转 DataFrame
    return pd.DataFrame(ds)


def _get_builder():
    cfg = get_system_config()
    if hasattr(DatasetBuilder, "from_system_config"):
        return DatasetBuilder.from_system_config(cfg)
    return DatasetBuilder(cfg)


def _pick_dataset_id(builder) -> str:
    if hasattr(builder, "list_datasets"):
        try:
            ids = builder.list_datasets()
            if ids:
                return ids[0]
        except Exception:
            pass
    # 没有 list_datasets 的话，用较中性的默认 ID
    return "default"


def test_dataset_builder_train_split_consistency():
    builder = _get_builder()
    dataset_id = _pick_dataset_id(builder)

    ds1 = builder.build(dataset_id=dataset_id, split="train")
    ds2 = builder.build(dataset_id=dataset_id, split="train")

    df1 = _to_frame(ds1)
    df2 = _to_frame(ds2)

    assert df1.shape == df2.shape, (
        f"DatasetBuilder({dataset_id}, 'train') 两次构建 shape 不一致: "
        f"{df1.shape} vs {df2.shape}"
    )
    assert set(df1.columns) == set(df2.columns), (
        f"DatasetBuilder({dataset_id}, 'train') 两次构建列集合不一致"
    )
