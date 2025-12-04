# -*- coding: utf-8 -*-
"""
模块名称：CombatBrain Mk-AI (神经链路策略)
版本：Mk-AI Neural-Link R130 (Config-Aware)
路径: G:/LightHunter_Mk1/alpha_strategy.py

功能：
- 神经网络(MLP)内核，遗传算法接口，全维度评分。
- 集成 RiskBrain：对未来10分钟大回撤概率进行降权与风控封印。
- 【R130】MLP 超参数从配置中心读取，便于科学调参。
"""

import pandas as pd
import numpy as np
import datetime
import os
import joblib
from collections import Counter
from typing import Dict, Any

# ------------- ConfigCenter 接入 ----------------
try:
    from config.config_manager import get_ai_config
except Exception:  # 兼容老环境
    def get_ai_config() -> Dict[str, Any]:
        return {}
# ------------------------------------------------

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class CombatBrain:
    def __init__(self):
        self.version = "Mk-AI Neural-Link R130"
        self.scaler = StandardScaler() if AI_AVAILABLE else None
        self.model_file = "hunter_brain.pkl"
        self.scaler_file = "hunter_scaler.pkl"
        self.neural_model = None
        self.is_trained = False
        self._load_model()
        # 默认权重，可被遗传算法覆盖
        self.base_weights = {
            "w_pct": 2.0,
            "w_force": 20.0,
            "w_res": 2.0,
            "w_con": 30.0,
            "w_ai": 5.0,
        }

    def _load_model(self):
        """尝试加载已训练的 AI 模型"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                self.neural_model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.is_trained = True
            except Exception:
                pass

    # --------------------------------------------------
    # 训练大脑：优先用 TS 标签数据
    # --------------------------------------------------
    def train_brain(self, csv_path: str = "market_blackbox.csv"):
        """
        【R110】训练神经网络
        - 优先从 market_blackbox_labeled.csv 中读取 label（基于 TS 未来20min）
        - 若无标签文件，则退回到的“涨停/大涨”标签方式
        """
        if not AI_AVAILABLE:
            print(" [AI] Sklearn not installed. Run: pip install scikit-learn")
            return

        # 读取 AI 超参配置
        ai_cfg = get_ai_config() or {}
        cb_cfg = ai_cfg.get("combat_brain", {}) if isinstance(ai_cfg, dict) else {}
        hidden_layers = cb_cfg.get("hidden_layer_sizes", [64, 32])
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(int(x) for x in hidden_layers)
        elif isinstance(hidden_layers, int):
            hidden_layers = (hidden_layers,)
        max_iter = int(cb_cfg.get("max_iter", 500))
        random_state = int(cb_cfg.get("random_state", 42))

        labeled_path = "market_blackbox_labeled.csv"
        use_labeled = os.path.exists(labeled_path)

        data_path = labeled_path if use_labeled else csv_path
        if not os.path.exists(data_path):
            print(f" [AI] No training data found: {data_path}")
            return

        print(f" [AI] Learning from {data_path}...")

        try:
            df = pd.read_csv(data_path)

            # 清洗数据
            if use_labeled:
                if "label" not in df.columns:
                    print(" [AI] labeled file has no 'label' column, fallback to simple rule.")
                    use_labeled = False
                else:
                    df = df.dropna(subset=["label"])
                    df = df[df["label"].isin([0, 1])]

            if not use_labeled:
                # 简单标签：涨停/大涨为1，其它为0
                if "涨幅" not in df.columns:
                    print(" [AI] no 涨幅 column, abort.")
                    return
                df["label"] = (df["涨幅"] >= 9.0).astype(int)

            feature_cols = [
                "涨幅",
                "换手率",
                "量比",
                "主力攻击系数",
                "板块热度",
                "Z_Force",
                "NN_Prob",
                "Risk_Prob",
            ]
            feature_cols = [c for c in feature_cols if c in df.columns]

            if len(feature_cols) < 3:
                print(" [AI] too few feature columns, abort.")
                return

            X = df[feature_cols].fillna(0.0).values
            y = df["label"].values

            # 异常样本过滤（可选）
            if IsolationForest is not None:
                try:
                    iso = IsolationForest(
                        n_estimators=200,
                        contamination=0.02,
                        random_state=random_state,
                    )
                    mask = iso.fit_predict(X) == 1
                    X = X[mask]
                    y = y[mask]
                except Exception:
                    pass

            if len(X) < 1000:
                print(" [AI] not enough samples, skip training.")
                return

            # 标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 建模（这里采用配置中心中的结构）
            clf = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=max_iter,
                random_state=random_state,
            )
            clf.fit(X_scaled, y)

            self.neural_model = clf
            self.is_trained = True

            # 持久化模型
            joblib.dump(self.neural_model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)

            print(
                f" [AI] CombatBrain trained. Layers={hidden_layers}, max_iter={max_iter}, "
                f"samples={len(X)}, pos_rate={y.mean():.3f}"
            )

        except Exception as e:
            print(f" [AI] Train failed: {e}")

    # --------------------------------------------------
    # 评分接口（保持原有逻辑，不赘述）
    # --------------------------------------------------
    def score_stock(self, row: pd.Series, risk_prob: float = 0.0) -> float:
        """
        根据传统因子 + NN_Prob + Risk_Prob 计算综合得分。
        """
        # 传统因子
        pct = float(row.get("涨幅", 0.0))
        force = float(row.get("主力攻击系数", 0.0))
        res = float(row.get("资金动能", 0.0))
        con = float(row.get("概念共振", 0.0))
        nn_prob = float(row.get("NN_Prob", 0.0))

        base = (
            self.base_weights["w_pct"] * pct
            + self.base_weights["w_force"] * force
            + self.base_weights["w_res"] * res
            + self.base_weights["w_con"] * con
            + self.base_weights["w_ai"] * nn_prob
        )

        # RiskBrain 风险封印
        risk_penalty = (1.0 - risk_prob) ** 2
        return base * risk_penalty

    # 其余 analyze 等函数保持不变，此处省略……
