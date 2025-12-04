# -*- coding: utf-8 -*-
"""
alpha/gnn_models.py

LightHunter Mk3 - GNN Models (GCN)
==================================

目标
----
提供一套轻量级、无 PyG 依赖的图神经网络实现，用于在概念图上做节点级任务：
    - 超短线 T+1 方向预测（节点分类）
    - 超短线收益回归（节点回归）

核心组件
--------
- build_normalized_adj : 构建 GCN 用的归一化稀疏邻接矩阵
- GraphConvLayer       : 单层 GCN 图卷积
- GCNNodeModel         : 多层 GCN 节点模型
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GCNConfig:
    in_dim: int
    hidden_dim: int = 64
    out_dim: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    use_bias: bool = True


def build_normalized_adj(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    构建适用于 GCN 的归一化邻接矩阵（稀疏张量）。

    公式：
        \hat{A} = A + I
        D_ii = sum_j \hat{A}_{ij}
        A_norm = D^{-1/2} \hat{A} D^{-1/2}

    参数
    ----
    num_nodes : int
        节点数
    edge_index : np.ndarray(2, E)
        边索引（src, dst）
    edge_weight : np.ndarray(E,), optional
        边权重；不提供则默认全 1
    device : torch.device, optional
        张量所在设备
    add_self_loops : bool
        是否添加自环。

    返回
    ----
    adj_norm : torch.sparse.FloatTensor
        归一化后的稀疏邻接矩阵。
    """
    if device is None:
        device = torch.device("cpu")

    if edge_index.size == 0:
        # 空图：只保留单位矩阵（如果 add_self_loops）
        if add_self_loops:
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(num_nodes, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes)
            ).coalesce()
        else:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
                (num_nodes, num_nodes),
            ).coalesce()

    row_np = edge_index[0].astype(np.int64)
    col_np = edge_index[1].astype(np.int64)
    if edge_weight is None:
        w_np = np.ones_like(row_np, dtype=np.float32)
    else:
        w_np = edge_weight.astype(np.float32)

    row = torch.from_numpy(row_np).to(device=device)
    col = torch.from_numpy(col_np).to(device=device)
    weight = torch.from_numpy(w_np).to(device=device)

    if add_self_loops:
        self_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
        row = torch.cat([row, self_idx], dim=0)
        col = torch.cat([col, self_idx], dim=0)
        weight = torch.cat(
            [weight, torch.ones(num_nodes, dtype=torch.float32, device=device)], dim=0
        )

    # 度数
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.scatter_add_(0, row, weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]

    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(
        indices, norm_weight, (num_nodes, num_nodes)
    ).coalesce()
    return adj


class GraphConvLayer(nn.Module):
    """单层 GCN 图卷积：H' = A_norm @ H @ W"""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # (N, F_in) -> (N, F_out)
        x = self.linear(x)
        # 稀疏矩阵乘法：A_norm @ X
        x = torch.sparse.mm(adj_norm, x)
        return x


class GCNNodeModel(nn.Module):
    """
    多层 GCN 节点模型，用于节点级预测（分类/回归）。

    使用方式：
        cfg = GCNConfig(in_dim=feat_dim, hidden_dim=64, out_dim=1, num_layers=2)
        model = GCNNodeModel(cfg)
        out = model(x, adj_norm)  # (N, out_dim)
    """

    def __init__(self, cfg: GCNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.in_dim
        for i in range(cfg.num_layers):
            out_dim = cfg.hidden_dim if i < cfg.num_layers - 1 else cfg.out_dim
            layers.append(GraphConvLayer(in_dim, out_dim, bias=cfg.use_bias))
            in_dim = out_dim

        self.layers = nn.ModuleList(layers)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_norm)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


__all__ = [
    "GCNConfig",
    "GCNNodeModel",
    "GraphConvLayer",
    "build_normalized_adj",
]
