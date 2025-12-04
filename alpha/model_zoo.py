# -*- coding: utf-8 -*-
"""
alpha/model_zoo.py

LightHunter Mk3 - 模型动物园（Model Zoo）
========================================

统一管理所有训练/推理用模型，包括：

1. 超短主战模型（family="ultrashort"）
   - ultra_mlp_v1       : 512-256-128 三层 MLP，分类任务主力模型
   - ultra_mlp_reg_v1   : 同结构回归版，用于收益/回撤等连续标签

2. 基线模型（family="baseline"）
   - baseline_lr        : 逻辑回归（scikit-learn）
   - baseline_mlp_small : 2x64 小 MLP，用作 sanity check

3. GNN 模型（family="gnn"）
   - gnn_gcn_v1         : 概念图上的 GCN 分类模型
   - gnn_gcn_reg_v1     : 概念图上的 GCN 回归模型

4. 多任务模型（family="multitask"）
   - mtl_mlp_alpha_risk_v1 :
        共享 MLP 主干 + 两个输出头：
        - alpha 头：1 维回归（预测超短收益/edge）
        - risk  头：3 类分类（低/中/高风险）

5. 微观结构模型（family="microstructure"）
   - micro_mlp_v1       : 256-128-64 MLP，用于订单流/盘口异常检测等任务

推荐使用方式：
--------------
>>> from alpha.model_zoo import ModelZoo, get_model
>>> zoo = ModelZoo()
>>> spec = zoo.create("ultra_mlp_v1", input_dim=128, output_dim=3)
>>> model = spec.model
>>> print(spec.config)   # 查看建议超参数

或：
>>> model = get_model("baseline_mlp_small", input_dim=64, output_dim=2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np  # 预留给后续可能的模型封装
import pandas as pd  # 同上

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    HAS_SKLEARN = False

try:
    # 之前在 gnn_models.py 里实现的 GCN 模型
    from alpha.gnn_models import GCNNodeModel, GCNConfig  # type: ignore
    HAS_GNN = True
except Exception:  # pragma: no cover
    GCNNodeModel = None  # type: ignore
    GCNConfig = None  # type: ignore
    HAS_GNN = False


# ----------------------------------------------------------------------
# 统一模型规格封装
# ----------------------------------------------------------------------


@dataclass
class ModelSpec:
    """
    模型规格：包含模型本体 + 元信息 + 配置。

    属性
    ----
    model_id   : 模型标识符（如 "ultra_mlp_v1"）
    model      : 具体的模型实例（torch.nn.Module / sklearn 模型等）
    family     : 模型家族（ultrashort / baseline / gnn / multitask / microstructure）
    task_type  : 任务类型（classification / regression / multitask）
    config     : 最终使用的超参数配置（默认配置 + 调用时覆盖）
    description: 中文/英文说明，方便在 Dashboard / HUD 中展示
    """

    model_id: str
    model: Any
    family: str
    task_type: str
    config: Dict[str, Any]
    description: str = ""


# ----------------------------------------------------------------------
# 通用 MLP / MultiTaskMLP 实现
# ----------------------------------------------------------------------


class MLP(nn.Module if HAS_TORCH else object):
    """
    通用 MLP，用于：
        - ultra_mlp_* 系列
        - baseline_mlp_small
        - micro_mlp_v1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for MLP models.")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_cfg = hidden_layers
        self.activation_name = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.output: Optional[nn.Linear]
        if output_dim is not None:
            self.output = nn.Linear(in_dim, output_dim)
        else:
            self.output = None

    def forward_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """返回最后一层隐含表示，用于多任务头等场景。"""
        return self.backbone(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.forward_features(x)
        if self.output is not None:
            return self.output(h)
        return h


class MultiTaskMLP(nn.Module if HAS_TORCH else object):
    """
    多任务 MLP：共享主干 + 多头输出。

    heads_cfg 结构示例：
    -------------------
    {
      "alpha": { "out_dim": 1, "task_type": "regression" },
      "risk" : { "out_dim": 3, "task_type": "classification" }
    }
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        heads_cfg: Dict[str, Dict[str, Any]],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for MultiTaskMLP.")
        super().__init__()
        self.heads_cfg = heads_cfg
        self.trunk = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=None,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        last_dim = hidden_layers[-1] if hidden_layers else input_dim
        self.heads = nn.ModuleDict()
        for head_name, cfg in heads_cfg.items():
            out_dim = int(cfg.get("out_dim", 1))
            self.heads[head_name] = nn.Linear(last_dim, out_dim)

    def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        h = self.trunk.forward_features(x)
        out: Dict[str, "torch.Tensor"] = {}
        for name, head in self.heads.items():
            out[name] = head(h)
        return out


# ----------------------------------------------------------------------
# 模型动物园主体
# ----------------------------------------------------------------------


class ModelZoo:
    """
    模型动物园入口。

    主要方法：
    --------
    - list_models(family=None)      : 列出全部或某一 family 的模型 ID
    - get_default_config(model_id)  : 查看某个模型的推荐配置
    - create(model_id, input_dim, output_dim=None, override_cfg=None)
        -> ModelSpec

    用法示例：
    --------
        zoo = ModelZoo()
        spec = zoo.create("ultra_mlp_v1", input_dim=128, output_dim=3)
        model = spec.model
    """

    def __init__(self) -> None:
        self.registry: Dict[str, Dict[str, Any]] = self._build_registry()

    # 注册表：每个模型一个条目
    def _build_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        每个条目格式：
        {
          "family": "ultrashort" / "baseline" / "gnn" / "multitask" / "microstructure",
          "framework": "torch" / "sklearn",
          "task_type": "classification" / "regression" / "multitask",
          "description": "...",
          "default_cfg": {...}
        }
        """
        return {
            # --------------------
            # 基线模型 Baselines
            # --------------------
            "baseline_lr": {
                "family": "baseline",
                "framework": "sklearn",
                "task_type": "classification",
                "description": "Logistic Regression baseline model for tabular classification.",
                "default_cfg": {
                    "C": 1.0,          # 正则强度
                    "penalty": "l2",
                    "max_iter": 200,
                    "class_weight": None,
                },
            },
            "baseline_mlp_small": {
                "family": "baseline",
                "framework": "torch",
                "task_type": "classification",
                "description": "Small MLP baseline: 2×64 hidden units for quick sanity checks.",
                "default_cfg": {
                    "hidden_layers": [64, 64],
                    "activation": "relu",
                    "dropout": 0.1,
                    "batch_norm": True,
                },
            },

            # --------------------
            # 超短主战模型 Ultra-Short Core
            # --------------------
            "ultra_mlp_v1": {
                "family": "ultrashort",
                "framework": "torch",
                "task_type": "classification",
                "description": "Main ultra-shortline MLP: 512-256-128, dropout 0.2, for direction / 3-class labels.",
                "default_cfg": {
                    "hidden_layers": [512, 256, 128],
                    "activation": "relu",
                    "dropout": 0.2,
                    "batch_norm": True,
                    # 可选：num_classes 用于自动推断 output_dim
                    # "num_classes": 3,
                },
            },
            "ultra_mlp_reg_v1": {
                "family": "ultrashort",
                "framework": "torch",
                "task_type": "regression",
                "description": "Regression variant of ultra_mlp_v1 for continuous return/drawdown targets.",
                "default_cfg": {
                    "hidden_layers": [512, 256, 128],
                    "activation": "relu",
                    "dropout": 0.2,
                    "batch_norm": True,
                    # "out_dim": 1,
                },
            },

            # --------------------
            # 微观结构模型 Microstructure
            # --------------------
            "micro_mlp_v1": {
                "family": "microstructure",
                "framework": "torch",
                "task_type": "classification",
                "description": "Microstructure MLP: 256-128-64 for order-flow / anomaly detection tasks.",
                "default_cfg": {
                    "hidden_layers": [256, 128, 64],
                    "activation": "relu",
                    "dropout": 0.15,
                    "batch_norm": True,
                },
            },

            # --------------------
            # GNN 概念图模型 GNNs
            # --------------------
            "gnn_gcn_v1": {
                "family": "gnn",
                "framework": "torch",
                "task_type": "classification",
                "description": "GCN node model on concept graph; hidden_dim=64, 2 layers, dropout=0.1.",
                "default_cfg": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    # "num_classes": 3,
                },
            },
            "gnn_gcn_reg_v1": {
                "family": "gnn",
                "framework": "torch",
                "task_type": "regression",
                "description": "GCN node model (regression) for node-level returns / edges.",
                "default_cfg": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    # "out_dim": 1,
                },
            },

            # --------------------
            # 多任务模型 Multi-Task
            # --------------------
            "mtl_mlp_alpha_risk_v1": {
                "family": "multitask",
                "framework": "torch",
                "task_type": "multitask",
                "description": (
                    "Shared MLP trunk with two heads: "
                    "alpha (regression, 1-d) + risk (3-class classification)."
                ),
                "default_cfg": {
                    "hidden_layers": [512, 256, 128],
                    "activation": "relu",
                    "dropout": 0.2,
                    "batch_norm": True,
                    "heads": {
                        "alpha": {
                            "out_dim": 1,
                            "task_type": "regression",
                        },
                        "risk": {
                            "out_dim": 3,
                            "task_type": "classification",
                        },
                    },
                },
            },
        }

    # ---------------- 公共 API ----------------

    def list_models(self, family: Optional[str] = None) -> List[str]:
        """列出全部模型 ID，或指定 family 下的模型列表。"""
        if family is None:
            return sorted(self.registry.keys())
        return sorted([m for m, spec in self.registry.items() if spec["family"] == family])

    def get_default_config(self, model_id: str) -> Dict[str, Any]:
        """返回某个模型在 registry 中登记的默认配置（一个 dict，可再修改）。"""
        if model_id not in self.registry:
            raise KeyError(f"Unknown model_id: {model_id}")
        return dict(self.registry[model_id]["default_cfg"])

    def get_task_type(self, model_id: str) -> str:
        """返回模型的任务类型：classification / regression / multitask。"""
        if model_id not in self.registry:
            raise KeyError(f"Unknown model_id: {model_id}")
        return str(self.registry[model_id]["task_type"])

    def create(
        self,
        model_id: str,
        input_dim: int,
        output_dim: Optional[int] = None,
        override_cfg: Optional[Dict[str, Any]] = None,
    ) -> ModelSpec:
        """
        根据 model_id 创建模型实例，并返回 ModelSpec。

        参数
        ----
        model_id   : 注册表中的模型标识符
        input_dim  : 特征维度（tabular 为因子数；GNN 为 node feature 维度）
        output_dim : 输出维度（分类为类别数；回归为输出维度）。如不指定，将尝试从配置推断。
        override_cfg : 用于覆盖 default_cfg 中的部分字段。

        返回
        ----
        ModelSpec(model_id, model, family, task_type, config, description)
        """
        if model_id not in self.registry:
            raise KeyError(f"Unknown model_id: {model_id}")

        spec = self.registry[model_id]
        family = spec["family"]
        framework = spec["framework"]
        task_type = spec["task_type"]
        description = spec.get("description", "")

        cfg = dict(spec["default_cfg"])
        if override_cfg:
            cfg.update(override_cfg)

        if family == "baseline" and model_id == "baseline_lr":
            model = self._create_baseline_lr(cfg)
        elif framework == "torch":
            model = self._create_torch_model(
                model_id=model_id,
                family=family,
                task_type=task_type,
                input_dim=input_dim,
                output_dim=output_dim,
                cfg=cfg,
            )
        else:
            raise ValueError(
                f"ModelZoo: 不支持的模型类型: model_id={model_id}, family={family}, framework={framework}"
            )

        return ModelSpec(
            model_id=model_id,
            model=model,
            family=family,
            task_type=task_type,
            config=cfg,
            description=description,
        )

    # ---------------- 具体构造函数 ----------------

    def _create_baseline_lr(self, cfg: Dict[str, Any]) -> Any:
        """构造 scikit-learn LogisticRegression 基线模型。"""
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is not installed, cannot create LogisticRegression baseline."
            )
        model = LogisticRegression(
            C=cfg.get("C", 1.0),
            penalty=cfg.get("penalty", "l2"),
            max_iter=cfg.get("max_iter", 200),
            class_weight=cfg.get("class_weight", None),
        )
        return model

    def _create_torch_model(
        self,
        model_id: str,
        family: str,
        task_type: str,
        input_dim: int,
        output_dim: Optional[int],
        cfg: Dict[str, Any],
    ) -> Any:
        """构造基于 PyTorch 的模型（MLP / GNN / MultiTask）。"""
        if not HAS_TORCH:
            raise RuntimeError(f"PyTorch is required to build model {model_id}.")

        # --- Tabular MLP 系列：baseline / ultrashort / microstructure ---
        if family in ("baseline", "ultrashort", "microstructure"):
            hidden_layers = cfg.get("hidden_layers", [128, 64])
            activation = cfg.get("activation", "relu")
            dropout = float(cfg.get("dropout", 0.0))
            batch_norm = bool(cfg.get("batch_norm", False))
            if output_dim is None:
                if task_type == "classification":
                    output_dim = int(cfg.get("num_classes", 2))
                elif task_type == "regression":
                    output_dim = int(cfg.get("out_dim", 1))
                else:
                    output_dim = 1
            model = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
            )
            return model

        # --- GNN 系列：gnn_* ---
        if family == "gnn":
            if not HAS_GNN:
                raise RuntimeError(
                    "GNN models requested but alpha.gnn_models is not available."
                )
            hidden_dim = int(cfg.get("hidden_dim", 64))
            num_layers = int(cfg.get("num_layers", 2))
            dropout = float(cfg.get("dropout", 0.1))
            if output_dim is None:
                if task_type == "classification":
                    output_dim = int(cfg.get("num_classes", 2))
                else:
                    output_dim = int(cfg.get("out_dim", 1))
            gcfg = GCNConfig(
                in_dim=input_dim,
                hidden_dim=hidden_dim,
                out_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            model = GCNNodeModel(gcfg)
            return model

        # --- 多任务系列：multitask ---
        if family == "multitask":
            hidden_layers = cfg.get("hidden_layers", [512, 256, 128])
            activation = cfg.get("activation", "relu")
            dropout = float(cfg.get("dropout", 0.2))
            batch_norm = bool(cfg.get("batch_norm", True))
            heads_cfg = cfg.get("heads", {})
            if not heads_cfg:
                raise ValueError(f"MultiTask model {model_id} requires 'heads' config.")
            model = MultiTaskMLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                heads_cfg=heads_cfg,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
            )
            return model

        raise ValueError(f"ModelZoo: unknown family {family} for torch model.")


# ----------------------------------------------------------------------
# 模块级便捷方法（方便旧代码调用）
# ----------------------------------------------------------------------

_zoo_singleton: Optional[ModelZoo] = None


def get_zoo() -> ModelZoo:
    """返回一个单例 ModelZoo，避免重复构建。"""
    global _zoo_singleton
    if _zoo_singleton is None:
        _zoo_singleton = ModelZoo()
    return _zoo_singleton


def get_model(
    model_id: str,
    input_dim: int,
    output_dim: Optional[int] = None,
    override_cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    便捷函数：直接拿到模型本体（不关心 ModelSpec 其它信息时用）。

    示例：
        model = get_model("ultra_mlp_v1", input_dim=128, output_dim=3)
    """
    spec = get_zoo().create(
        model_id=model_id,
        input_dim=input_dim,
        output_dim=output_dim,
        override_cfg=override_cfg,
    )
    return spec.model


__all__ = [
    "ModelSpec",
    "ModelZoo",
    "get_zoo",
    "get_model",
]
