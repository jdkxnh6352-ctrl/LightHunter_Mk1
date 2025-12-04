# -*- coding: utf-8 -*-
"""
U3 预测引擎（离线版）

- 只依赖历史日线因子表 ultrashort_main.parquet
- 使用与 U2 相同的特征列（来自 u2_live_config.json.dataset.features）
- 训练一个随机森林分类器，用于替换 U2 的 Logistic 回归
- 暂时只给日度打分脚本 u2_daily_scoring.py 调用，不改实盘流程
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .u2_live_batch_backtest import U2Config


@dataclass
class U3TrainConfig:
    """
    U3 训练相关可调参数（先留简单几个，将来可以再扩）
    """
    n_estimators: int = 300
    max_depth: Optional[int] = 5
    min_samples_leaf: int = 5
    random_state: int = 42


def _get_feature_cols(cfg: U2Config) -> List[str]:
    """
    特征列先直接用 u2_live_config.json.dataset.features
    """
    feats = getattr(cfg, "features", None)
    if not feats:
        # 兜底：老版本没 features 字段时，用默认 8 因子
        feats = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vol_20",
            "amt_mean_20",
        ]
    return list(feats)


# ----------------------------------------------------------------------
# 训练 & 预测
# ----------------------------------------------------------------------


def train_u3_model(
    train_df: pd.DataFrame,
    cfg: U2Config,
    u3_cfg: Optional[U3TrainConfig] = None,
) -> Optional[RandomForestClassifier]:
    """
    在给定训练集上训练 U3 模型。

    label 规则：
        - 使用 cfg.label_col（默认 label_u2）
        - label > cfg.label_threshold 视为 1，其余为 0
    """
    if u3_cfg is None:
        u3_cfg = U3TrainConfig()

    feature_cols = _get_feature_cols(cfg)
    label_col = cfg.label_col
    label_th = getattr(cfg, "label_threshold", 0.0)

    # 去掉缺 label / 特征的样本
    cols_need = feature_cols + [label_col]
    train_df = train_df.dropna(subset=cols_need)
    if train_df.empty:
        return None

    y_raw = train_df[label_col].values.astype(float)
    y = (y_raw > float(label_th)).astype(int)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    # 样本太少或者过度不平衡就不训了
    if pos < 20 or neg < 20:
        return None

    X = train_df[feature_cols].values

    model = RandomForestClassifier(
        n_estimators=u3_cfg.n_estimators,
        max_depth=u3_cfg.max_depth,
        min_samples_leaf=u3_cfg.min_samples_leaf,
        random_state=u3_cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def score_u3_model(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    cfg: U2Config,
) -> np.ndarray:
    """
    用训练好的 U3 模型对给定样本打分，返回“上涨概率”数组。
    """
    if df.empty:
        return np.zeros(0, dtype=float)

    feature_cols = _get_feature_cols(cfg)
    df = df.dropna(subset=feature_cols)
    if df.empty:
        return np.zeros(0, dtype=float)

    X = df[feature_cols].values
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # 理论上不会触发，这里做个兜底
        preds = model.predict(X).astype(float)
        # 简单 Sigmoid 映射到 0~1
        proba = 1.0 / (1.0 + np.exp(-preds))
    return proba
