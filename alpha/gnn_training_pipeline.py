# -*- coding: utf-8 -*-
"""
alpha/gnn_training_pipeline.py

LightHunter Mk3 - GNN Training Pipeline
=======================================

目标
----
基于 A 股「概念/连板」图结构，对股票节点做 GNN 训练：
    - 节点标签：如 T+1 方向 (ULTRA_T1_DIR_3C)、T+1 收益 (ULTRA_T1_RET_C2C) 等
    - 节点特征：来自 factor_panel（包括价量因子、订单流因子、概念/情绪因子等）
    - 图结构：来自 ConceptGraph（概念共现关系）

核心接口
--------
- build_gnn_dataset_from_graph
    将 ConceptGraph + 节点特征/标签打包为张量 (x, y, adj_norm, masks)
- train_gnn_node_model
    用 GCN 在该数据集上训练，输出模型与指标

注意
----
本模块只关注「单一时间截面」的静态图训练：
    - 例如选定某个日期的图，用过去的因子/标签做训练；
    - 时序 GNN / 时变图可以在此基础上扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from core.logging_utils import get_logger
from features.concept_graph_builder import ConceptGraph
from alpha.gnn_models import GCNConfig, GCNNodeModel, build_normalized_adj

log = get_logger(__name__)


@dataclass
class GNNTrainingConfig:
    task_type: str = "regression"  # "regression" or "classification"
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    val_ratio: float = 0.15
    test_ratio: float = 0.15

    device: Optional[str] = None  # "cuda" / "cpu" / None(auto)


@dataclass
class GNNDataset:
    x: torch.Tensor
    y: torch.Tensor
    adj_norm: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    node_symbols: List[str]


# ----------------------------------------------------------------------
# 构建数据集
# ----------------------------------------------------------------------


def build_gnn_dataset_from_graph(
    graph: ConceptGraph,
    node_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cfg: Optional[GNNTrainingConfig] = None,
    train_mask_col: Optional[str] = None,
    val_mask_col: Optional[str] = None,
    test_mask_col: Optional[str] = None,
) -> GNNDataset:
    """
    根据 ConceptGraph + 节点数据表构建 GNN 训练数据集。

    参数
    ----
    graph : ConceptGraph
        概念图结构（节点列表必须为 symbol 列表）。
    node_df : DataFrame
        节点特征与标签表，至少包含：
            - symbol
            - feature_cols 中指定的列
            - label_col 指定的标签列
        若存在 train/val/test mask 列，也可用 train_mask_col 等指定。
    feature_cols : List[str]
        用作 GNN 输入的特征列。
    label_col : str
        用作监督信号的标签列。
    cfg : GNNTrainingConfig, optional
        训练配置，用于 device 设置与数据划分默认比例。
    train_mask_col / val_mask_col / test_mask_col : str, optional
        若 node_df 中已有划分列（0/1），可以直接使用；否则将随机划分。

    返回
    ----
    GNNDataset 对象。
    """
    if cfg is None:
        cfg = GNNTrainingConfig()

    symbols = graph.nodes
    N = len(symbols)
    if N == 0:
        raise ValueError("build_gnn_dataset_from_graph: 图中无节点，无法构建数据集。")

    # 对 node_df 按 symbol 对齐
    if "symbol" not in node_df.columns:
        raise KeyError("node_df 必须包含列 'symbol'。")

    df = node_df.set_index("symbol").reindex(symbols)
    # 特征矩阵
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise KeyError(f"node_df 中缺少以下特征列: {missing_feats}")
    x_np = df[feature_cols].to_numpy(dtype=np.float32)
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 标签
    if label_col not in df.columns:
        raise KeyError(f"node_df 中缺少标签列 {label_col!r}")
    y_np = df[label_col].to_numpy()

    # 转为张量
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(y_np).to(device=device)

    # 构建归一化邻接矩阵
    adj_norm = build_normalized_adj(
        num_nodes=N,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        device=device,
        add_self_loops=True,
    )

    # 构建 mask
    train_mask, val_mask, test_mask = _build_split_masks(
        df, cfg, device, train_mask_col, val_mask_col, test_mask_col
    )

    return GNNDataset(
        x=x,
        y=y,
        adj_norm=adj_norm,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_symbols=symbols,
    )


def _build_split_masks(
    df: pd.DataFrame,
    cfg: GNNTrainingConfig,
    device: torch.device,
    train_mask_col: Optional[str],
    val_mask_col: Optional[str],
    test_mask_col: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = df.shape[0]

    if (
        train_mask_col
        and val_mask_col
        and test_mask_col
        and train_mask_col in df.columns
        and val_mask_col in df.columns
        and test_mask_col in df.columns
    ):
        train_mask = torch.from_numpy(df[train_mask_col].to_numpy().astype(bool)).to(
            device=device
        )
        val_mask = torch.from_numpy(df[val_mask_col].to_numpy().astype(bool)).to(
            device=device
        )
        test_mask = torch.from_numpy(df[test_mask_col].to_numpy().astype(bool)).to(
            device=device
        )
        return train_mask, val_mask, test_mask

    # 否则随机划分
    idx = np.arange(N)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(idx)

    n_val = int(N * cfg.val_ratio)
    n_test = int(N * cfg.test_ratio)
    n_train = N - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ----------------------------------------------------------------------
# 训练主入口
# ----------------------------------------------------------------------


def train_gnn_node_model(
    graph: ConceptGraph,
    node_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cfg: Optional[GNNTrainingConfig] = None,
    train_mask_col: Optional[str] = None,
    val_mask_col: Optional[str] = None,
    test_mask_col: Optional[str] = None,
) -> Dict[str, any]:
    """
    在概念图上训练 GCN 节点模型。

    参数
    ----
    graph, node_df, feature_cols, label_col :
        同 build_gnn_dataset_from_graph。
    cfg : GNNTrainingConfig, optional
        训练参数。
    *_mask_col : str, optional
        若 node_df 中已有划分列，则可以指定使用。

    返回
    ----
    result : dict
        - "model"  : 训练好的 GCNNodeModel
        - "metrics": 训练/验证/测试集上的损失与指标
        - "dataset": GNNDataset（方便后续推理或可视化）
    """
    if cfg is None:
        cfg = GNNTrainingConfig()

    dataset = build_gnn_dataset_from_graph(
        graph,
        node_df,
        feature_cols=feature_cols,
        label_col=label_col,
        cfg=cfg,
        train_mask_col=train_mask_col,
        val_mask_col=val_mask_col,
        test_mask_col=test_mask_col,
    )

    x = dataset.x
    y = dataset.y
    adj_norm = dataset.adj_norm
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    in_dim = x.size(1)
    if cfg.task_type == "classification":
        # 假定标签为 0 ~ C-1
        num_classes = int(y.max().item()) + 1
        out_dim = num_classes
    else:
        out_dim = 1  # 回归输出单标量

    gcn_cfg = GCNConfig(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model = GCNNodeModel(gcn_cfg).to(device=x.device)

    if cfg.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(x, adj_norm)  # (N, out_dim)

        if cfg.task_type == "classification":
            loss_train = criterion(out[train_mask], y[train_mask].long())
        else:
            # 回归：y 任意浮点标签
            pred = out[train_mask].squeeze(-1)
            loss_train = criterion(pred, y[train_mask].float())

        loss_train.backward()
        optimizer.step()

        # 验证集
        model.eval()
        with torch.no_grad():
            out = model(x, adj_norm)
            if cfg.task_type == "classification":
                loss_val = criterion(out[val_mask], y[val_mask].long())
            else:
                pred_val = out[val_mask].squeeze(-1)
                loss_val = criterion(pred_val, y[val_mask].float())

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "GNN Epoch %d/%d - train_loss=%.6f, val_loss=%.6f",
                epoch,
                cfg.num_epochs,
                loss_train.item(),
                loss_val.item(),
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评价
    model.eval()
    with torch.no_grad():
        out = model(x, adj_norm)
        if cfg.task_type == "classification":
            # 损失
            loss_train = criterion(out[train_mask], y[train_mask].long()).item()
            loss_val = criterion(out[val_mask], y[val_mask].long()).item()
            loss_test = criterion(out[test_mask], y[test_mask].long()).item()

            # 准确率
            pred_label = out.argmax(dim=-1)
            def _acc(mask):
                if mask.sum() == 0:
                    return 0.0
                correct = (pred_label[mask] == y[mask].long()).sum().item()
                return correct / mask.sum().item()

            acc_train = _acc(train_mask)
            acc_val = _acc(val_mask)
            acc_test = _acc(test_mask)

            metrics = {
                "train_loss": loss_train,
                "val_loss": loss_val,
                "test_loss": loss_test,
                "train_acc": acc_train,
                "val_acc": acc_val,
                "test_acc": acc_test,
            }
        else:
            pred_all = out.squeeze(-1)
            def _mse(mask):
                if mask.sum() == 0:
                    return float("nan")
                return nn.functional.mse_loss(
                    pred_all[mask], y[mask].float()
                ).item()

            metrics = {
                "train_mse": _mse(train_mask),
                "val_mse": _mse(val_mask),
                "test_mse": _mse(test_mask),
            }

    log.info("GNN 训练完成，指标：%s", metrics)

    return {
        "model": model,
        "metrics": metrics,
        "dataset": dataset,
    }


__all__ = [
    "GNNTrainingConfig",
    "GNNDataset",
    "build_gnn_dataset_from_graph",
    "train_gnn_node_model",
]
