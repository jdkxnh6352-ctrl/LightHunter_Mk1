# -*- coding: utf-8 -*-
"""
features/concept_graph_builder.py

LightHunter Mk3 - Concept Graph Builder
=======================================

目标
----
针对 A 股超短线场景，基于「概念/题材 + 连板结构」构建股票之间的关系图，
为图特征（concept_graph_features）和 GNN 训练（gnn_training_pipeline）提供输入。

输入数据约定
------------
1. 概念归属表 df_concept_membership（行粒度：symbol × concept × trade_date）：
    必要列：
        - symbol        : 股票代码
        - concept_id    : 概念ID（或概念名称统一后的标识）
    可选列：
        - trade_date    : 交易日（若不提供，则认为是静态图）

2. （可选）节点特征表 df_node_features：
    - 至少包含 symbol 列
    - 其它列为数值特征（可选），例如：
        - 连板天数：board_count
        - 是否涨停：is_limit_up
        - 其它你在 factor_panel 中已经算过的因子

输出
----
ConceptGraph 数据结构：
    - nodes         : List[str]       节点对应的股票代码，顺序固定
    - edge_index    : np.ndarray(2, E)  边索引 (src, dst)
    - edge_weight   : np.ndarray(E,)    边权重（共现次数或归一后的权重）
    - node_features : Optional[np.ndarray(N, F)]  节点特征（如提供）
    - node_index    : Dict[str, int]   symbol → 节点索引
    - meta          : dict             额外元信息（如使用的特征列、日期等）

图结构
------
本实现构建的是「股票-股票」同质图：
    - 若两只股票在同一概念中出现，则在它们之间加边
    - 每个概念内构建完全图（clique），边权重 ~= 1/(概念内股票数-1)
    - 若两只股票共同属于多个概念，则权重为多次累加

可以通过参数滤掉：
    - 概念内股票数过少的小众概念（如仅1只）
    - 概念内股票数过多的「指数型」大篮子（如沪深300）以降低噪音
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ConceptGraph:
    """概念图的基础数据结构。"""

    nodes: List[str]
    edge_index: np.ndarray  # shape (2, E), dtype=int64
    edge_weight: np.ndarray  # shape (E,), dtype=float32
    node_features: Optional[np.ndarray] = None  # shape (N, F)
    node_index: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptGraphBuilderConfig:
    """控制构图行为的配置。"""

    symbol_col: str = "symbol"
    concept_col: str = "concept_id"
    trade_date_col: str = "trade_date"

    # 控制概念大小过滤，避免噪音与超大“垃圾桶”概念
    min_concept_size: int = 2
    max_concept_size: int = 80

    # 是否对边权重做简单归一，以避免大概念权重过度放大
    normalize_by_group_size: bool = True


class ConceptGraphBuilder:
    """概念图构建器。"""

    def __init__(self, cfg: Optional[ConceptGraphBuilderConfig] = None) -> None:
        self.cfg = cfg or ConceptGraphBuilderConfig()

    # ------------------------------------------------------------------
    # 外部 API
    # ------------------------------------------------------------------

    def build_graph_from_membership(
        self,
        df_membership: pd.DataFrame,
        df_node_features: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ConceptGraph:
        """
        根据完整的概念归属表构建一张概念图（不区分日期）。

        参数
        ----
        df_membership : DataFrame
            必须包含 [symbol_col, concept_col]，可选 trade_date_col。
        df_node_features : DataFrame, optional
            节点特征表，至少包含 symbol_col 列，其余数值列可作为特征。
        feature_cols : List[str], optional
            明确指定从 df_node_features 中使用哪些列作为节点特征。
            不指定时，将自动选择数值型列。
        meta : dict, optional
            附加元信息，在返回的 ConceptGraph.meta 中可用。

        返回
        ----
        ConceptGraph 实例。
        """
        cfg = self.cfg

        required_cols = {cfg.symbol_col, cfg.concept_col}
        missing = required_cols - set(df_membership.columns)
        if missing:
            raise KeyError(f"ConceptGraphBuilder: df_membership 缺少必要列: {missing}")

        # 去重防止重复记录
        df = df_membership[[cfg.symbol_col, cfg.concept_col]].drop_duplicates()

        # 节点列表：所有出现过的 symbol
        nodes = sorted(df[cfg.symbol_col].unique().tolist())
        node_index = {s: i for i, s in enumerate(nodes)}
        log.info("ConceptGraphBuilder: 节点数 N=%d", len(nodes))

        # 构建边
        edge_index, edge_weight = self._build_edges(df, node_index)

        # 节点特征（可选）
        node_features = None
        used_feature_cols: List[str] = []
        if df_node_features is not None:
            node_features, used_feature_cols = self._build_node_features(
                nodes, df_node_features, feature_cols=feature_cols
            )
            log.info(
                "ConceptGraphBuilder: 节点特征矩阵 shape=%s, 使用列=%s",
                node_features.shape,
                used_feature_cols,
            )

        meta = dict(meta or {})
        meta.update(
            {
                "feature_cols": used_feature_cols,
                "num_nodes": len(nodes),
                "num_edges": edge_index.shape[1],
            }
        )

        return ConceptGraph(
            nodes=nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_features=node_features,
            node_index=node_index,
            meta=meta,
        )

    def build_graph_for_date(
        self,
        df_membership: pd.DataFrame,
        trade_date: Any,
        df_node_features: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> ConceptGraph:
        """
        为指定交易日构建一张概念图（动态图的一帧）。

        参数
        ----
        df_membership : DataFrame
            必须包含 [symbol_col, concept_col, trade_date_col]。
        trade_date : 任意可比较对象
            将会与 trade_date_col 精确匹配（建议传 datetime.date 或 pandas.Timestamp）。
        df_node_features : DataFrame, optional
            当日节点特征表（至少包含 symbol_col，建议也包含 trade_date_col）。
        feature_cols : List[str], optional
            节点特征列名。

        返回
        ----
        ConceptGraph
        """
        cfg = self.cfg
        if cfg.trade_date_col not in df_membership.columns:
            raise KeyError(
                f"build_graph_for_date 需要 df_membership 包含列 {cfg.trade_date_col!r}"
            )

        df_day = df_membership[
            df_membership[cfg.trade_date_col] == trade_date
        ].copy()
        if df_day.empty:
            log.warning(
                "ConceptGraphBuilder: trade_date=%s 没有任何概念归属记录，返回空图。",
                trade_date,
            )
            return ConceptGraph(
                nodes=[],
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_weight=np.zeros((0,), dtype=np.float32),
            )

        df_feat_day = None
        if df_node_features is not None:
            if cfg.trade_date_col in df_node_features.columns:
                df_feat_day = df_node_features[
                    df_node_features[cfg.trade_date_col] == trade_date
                ].copy()
            else:
                df_feat_day = df_node_features.copy()
        meta = {"trade_date": pd.to_datetime(trade_date)}

        return self.build_graph_from_membership(
            df_day, df_node_features=df_feat_day, feature_cols=feature_cols, meta=meta
        )

    # ------------------------------------------------------------------
    # 内部函数：构边 & 特征
    # ------------------------------------------------------------------

    def _build_edges(
        self,
        df: pd.DataFrame,
        node_index: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """依据 concept 分组构造股票之间的边。"""
        cfg = self.cfg

        edges: Dict[Tuple[int, int], float] = {}

        for concept_id, g in df.groupby(cfg.concept_col):
            symbols = g[cfg.symbol_col].unique().tolist()
            n = len(symbols)
            if n < cfg.min_concept_size or n > cfg.max_concept_size:
                continue

            idxs = [node_index[s] for s in symbols if s in node_index]
            if len(idxs) < 2:
                continue

            if cfg.normalize_by_group_size:
                w = 1.0 / float(len(idxs) - 1)
            else:
                w = 1.0

            # 在同一概念内构建完全图（有向），后续 GCN 会加自环并做对称归一
            for i in idxs:
                for j in idxs:
                    if i == j:
                        continue
                    key = (i, j)
                    edges[key] = edges.get(key, 0.0) + w

        if not edges:
            log.warning("ConceptGraphBuilder: 未构造出任何边，返回空边集合。")
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros((0,), dtype=np.float32)
            return edge_index, edge_weight

        rows: List[int] = []
        cols: List[int] = []
        ws: List[float] = []
        for (i, j), w in edges.items():
            rows.append(i)
            cols.append(j)
            ws.append(float(w))

        edge_index = np.vstack([rows, cols]).astype(np.int64)
        edge_weight = np.asarray(ws, dtype=np.float32)

        log.info("ConceptGraphBuilder: 构造边数 E=%d", edge_index.shape[1])
        return edge_index, edge_weight

    def _build_node_features(
        self,
        nodes: List[str],
        df_node_features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """根据节点特征表构造节点特征矩阵。"""
        cfg = self.cfg

        df_feat = df_node_features.copy()
        if cfg.symbol_col in df_feat.columns:
            df_feat = df_feat.set_index(cfg.symbol_col)

        # 只保留图中出现过的节点
        df_feat = df_feat.reindex(nodes)

        if feature_cols is None:
            # 自动选取数值型列作为特征
            numeric_cols = [
                c
                for c in df_feat.columns
                if np.issubdtype(df_feat[c].dtype, np.number)
            ]
            feature_cols = numeric_cols

        if not feature_cols:
            log.warning("ConceptGraphBuilder: 未找到可用的节点特征列，将返回 None。")
            return None, []

        x = df_feat[feature_cols].to_numpy(dtype=np.float32)
        # NaN/inf 清洗
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x, feature_cols


__all__ = [
    "ConceptGraph",
    "ConceptGraphBuilderConfig",
    "ConceptGraphBuilder",
]
