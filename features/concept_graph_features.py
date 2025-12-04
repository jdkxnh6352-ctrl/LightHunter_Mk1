# -*- coding: utf-8 -*-
"""
features/concept_graph_features.py

LightHunter Mk3 - Concept Graph Feature Extractor
=================================================

目标
----
针对 ConceptGraph 计算图结构特征，用于：
    - 因子层：作为图结构相关因子（cg_*）加入 factor_panel
    - GNN 辅助：也可以作为额外输入特征或对比基线

特征列表（节点级）
-----------------
    - cg_degree                : 无权度（邻居数量）
    - cg_weighted_degree       : 加权度（按边权求和）
    - cg_pagerank              : PageRank 中心度（考虑边权）

依赖
----
    - 必选：numpy, pandas
    - 推荐：networkx（如不可用，则仅计算 degree 与 weighted_degree）

输入
----
    - ConceptGraph（来自 concept_graph_builder）

输出
----
    - DataFrame，index 为 symbol，列为上述特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from features.concept_graph_builder import ConceptGraph

log = get_logger(__name__)

try:
    import networkx as nx  # type: ignore

    HAS_NETWORKX = True
except Exception:  # pragma: no cover - optional dependency
    HAS_NETWORKX = False
    nx = None


@dataclass
class ConceptGraphFeatureConfig:
    use_degree: bool = True
    use_weighted_degree: bool = True
    use_pagerank: bool = True


def compute_concept_graph_features(
    graph: ConceptGraph,
    cfg: Optional[ConceptGraphFeatureConfig] = None,
) -> pd.DataFrame:
    """
    计算概念图上的节点级图特征。

    参数
    ----
    graph : ConceptGraph
        概念图（由 ConceptGraphBuilder 构建）。
    cfg : ConceptGraphFeatureConfig, optional
        控制计算哪些特征。

    返回
    ----
    df_feat : DataFrame
        index 为 symbol，列为 cg_* 特征。
    """
    if cfg is None:
        cfg = ConceptGraphFeatureConfig()

    nodes = graph.nodes
    N = len(nodes)
    if N == 0:
        log.warning("compute_concept_graph_features: 空图，返回空 DataFrame。")
        return pd.DataFrame(columns=["cg_degree", "cg_weighted_degree", "cg_pagerank"])

    edge_index = graph.edge_index
    edge_weight = graph.edge_weight
    if edge_index.size == 0:
        log.warning("compute_concept_graph_features: 图中无边，特征退化为 0/均值。")

    # ------------------- 基础度数计算（不依赖 networkx） -------------------
    # degree（不带权）
    if cfg.use_degree:
        deg = np.zeros(N, dtype=np.float32)
        if edge_index.size > 0:
            src = edge_index[0]
            dst = edge_index[1]
            # 无向视角：入度 + 出度
            for i in range(src.shape[0]):
                deg[src[i]] += 1.0
                deg[dst[i]] += 1.0
    else:
        deg = np.zeros(N, dtype=np.float32)

    # weighted_degree（带权）
    if cfg.use_weighted_degree:
        wdeg = np.zeros(N, dtype=np.float32)
        if edge_index.size > 0:
            src = edge_index[0]
            dst = edge_index[1]
            w = edge_weight if edge_weight is not None else np.ones_like(src, dtype=np.float32)
            for i in range(src.shape[0]):
                wdeg[src[i]] += w[i]
                wdeg[dst[i]] += w[i]
    else:
        wdeg = np.zeros(N, dtype=np.float32)

    # ------------------- PageRank 等高级特征（依赖 networkx） -------------
    if cfg.use_pagerank and HAS_NETWORKX and edge_index.size > 0:
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        src = edge_index[0]
        dst = edge_index[1]
        w = edge_weight if edge_weight is not None else np.ones_like(src, dtype=np.float32)
        for i in range(src.shape[0]):
            u = int(src[i])
            v = int(dst[i])
            weight = float(w[i])
            if G.has_edge(u, v):
                G[u][v]["weight"] += weight
            else:
                G.add_edge(u, v, weight=weight)

        # PageRank（考虑边权）
        try:
            pr_dict = nx.pagerank(G, weight="weight")
            pr = np.zeros(N, dtype=np.float32)
            for i in range(N):
                pr[i] = float(pr_dict.get(i, 0.0))
        except Exception as e:  # pragma: no cover - numeric issues
            log.warning("compute_concept_graph_features: PageRank 计算失败：%s", e)
            pr = np.zeros(N, dtype=np.float32)
    else:
        pr = np.zeros(N, dtype=np.float32)
        if cfg.use_pagerank and not HAS_NETWORKX:
            log.warning("compute_concept_graph_features: 未安装 networkx，PageRank 用 0 代替。")

    df_feat = pd.DataFrame(
        {
            "symbol": nodes,
            "cg_degree": deg,
            "cg_weighted_degree": wdeg,
            "cg_pagerank": pr,
        }
    ).set_index("symbol")

    return df_feat


__all__ = [
    "ConceptGraphFeatureConfig",
    "compute_concept_graph_features",
]
