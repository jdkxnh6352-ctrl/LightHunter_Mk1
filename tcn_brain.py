# -*- coding: utf-8 -*-
"""
模块名称：TCNBrain Mk-TS
版本：Mk-TCN R10 (10m Forward Return Regressor)
路径: G:/LightHunter_Mk1/tcn_brain.py

核心目标（对应 Mk-V 白皮书 5.1 “核心预测模型”）：
- 基于 FactorEngine 生成的 ts_dataset_YYYY-MM-DD.csv（分钟级特征 + 未来10分钟收益）； 
- 构建一个轻量级 TCN（Temporal Convolutional Network）时序模型；
- 输入：单票最近 seq_len 分钟的特征序列；
- 输出：未来 10 分钟的预期收益 fwd_ret_10m（回归），同时给出“攻击概率”。

特点：
- 纯离线训练，盘后跑，不影响盘中 Commander；
- torch 不存在时优雅降级（仅提示，不报错），不破坏现有流水线；
- 不依赖 sklearn，自带简易特征归一化器。
"""

import os
import glob
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - 环境无 torch 时仅做降级提示
    torch = None  # type: ignore
    nn = object  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    TORCH_AVAILABLE = False


# ----------------------------
# 简易特征归一化器（替代 sklearn）
# ----------------------------
class FeatureScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        """
        X: (N, F) or (N * T, F)
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler is not fitted.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        fs = cls()
        if d.get("mean") is not None and d.get("std") is not None:
            fs.mean_ = np.array(d["mean"], dtype=np.float32)
            fs.std_ = np.array(d["std"], dtype=np.float32)
        return fs


# ----------------------------
# TCN 模块
# ----------------------------
if TORCH_AVAILABLE:

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.chomp_size == 0:
                return x
            return x[:, :, :-self.chomp_size]

    class TemporalBlock(nn.Module):
        def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.conv1 = nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.chomp2,
                self.relu2,
                self.dropout2,
            )
            self.downsample = (
                nn.Conv1d(n_inputs, n_outputs, 1)
                if n_inputs != n_outputs
                else None
            )
            self.out_relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.out_relu(out + res)

    class TemporalConvNet(nn.Module):
        def __init__(
            self,
            num_inputs: int,
            num_channels: List[int],
            kernel_size: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            layers: List[nn.Module] = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                in_ch = num_inputs if i == 0 else num_channels[i - 1]
                out_ch = num_channels[i]
                dilation = 2 ** i
                padding = (kernel_size - 1) * dilation
                layers.append(
                    TemporalBlock(
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=padding,
                        dropout=dropout,
                    )
                )
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T)
            return self.network(x)

    class TCNRegressor(nn.Module):
        def __init__(
            self,
            num_features: int,
            num_channels: Optional[List[int]] = None,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            if num_channels is None:
                num_channels = [32, 32, 64]
            self.tcn = TemporalConvNet(
                num_inputs=num_features,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.head = nn.Linear(num_channels[-1], 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (B, T, F) → (B, F, T) → (B, C, T)
            输出：未来10分钟收益的点预测（单位：百分比，例如 2.0 = +2%）
            """
            x = x.permute(0, 2, 1)  # (B, F, T)
            y = self.tcn(x)  # (B, C, T)
            last = y[:, :, -1]  # 取最后一个时间步
            out = self.head(last).squeeze(-1)  # (B,)
            return out


# ----------------------------
# Dataset & 配置
# ----------------------------
if TORCH_AVAILABLE:

    class SequenceDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            """
            X: (N, T, F)  y: (N,)
            """
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]


@dataclass
class TCNBrainConfig:
    seq_len: int = 60
    feature_names: Tuple[str, ...] = (
        "pct",
        "turnover_rate",
        "force",
        "amount_log",
        "ba_ratio",
    )
    label_col: str = "fwd_ret_10m"
    # 训练目标单位：百分比，例如 2.0 = +2%
    target_scale: float = 100.0
    attack_threshold_pct: float = 2.0  # 预测收益 >= 2% 视为“强攻击”


# ----------------------------
# 主类：TCNBrain
# ----------------------------
class TCNBrain:
    """
    盘后训练的 TCN 时序大脑：
    - 依赖 ts_datasets/ts_dataset_*.csv （由 FactorEngine 生成）；
    - 只读，不修改任何现有表；
    - 训练结束会生成：
        * tcn_brain.pt          : torch 模型权重
        * tcn_brain_scaler.json : 特征归一化参数
        * tcn_brain_config.json : 配置（窗口长度、特征名等）
    """

    def __init__(
        self,
        data_dir: str = "ts_datasets",
        model_path: str = "tcn_brain.pt",
        scaler_path: str = "tcn_brain_scaler.json",
        config_path: str = "tcn_brain_config.json",
    ):
        self.data_dir = data_dir
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path

        self.config = TCNBrainConfig()
        self.scaler: Optional[FeatureScaler] = None
        self.model: Optional["TCNRegressor"] = None  # type: ignore

        if TORCH_AVAILABLE:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = None

    # ------------------------
    # 数据加载 & 样本构建
    # ------------------------
    def _load_all_datasets(
        self, max_files: Optional[int] = None
    ) -> pd.DataFrame:
        """
        从 data_dir 里加载 ts_dataset_*.csv 并拼接。
        默认按文件名排序，优先使用最近的 max_files 个。
        """
        if not os.path.exists(self.data_dir):
            print(
                f"[TCN] data_dir '{self.data_dir}' 不存在，请先运行 FactorEngine 生成 ts_dataset_*.csv。"
            )
            return pd.DataFrame()

        files = sorted(
            glob.glob(os.path.join(self.data_dir, "ts_dataset_*.csv"))
        )
        if not files:
            print(
                f"[TCN] 未在 {self.data_dir} 下找到 ts_dataset_*.csv，先跑一遍 factor_engine.py。"
            )
            return pd.DataFrame()

        if max_files is not None and max_files > 0:
            files = files[-max_files:]

        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"[TCN] 读取 {fp} 失败：{e}")

        if not dfs:
            print("[TCN] 所有 ts_dataset_* 文件均为空或读取失败。")
            return pd.DataFrame()

        all_df = pd.concat(dfs, ignore_index=True)
        # 确保列存在
        needed = list(self.config.feature_names) + [self.config.label_col, "code", "ts"]
        for col in needed:
            if col not in all_df.columns:
                print(f"[TCN] 缺失列 {col}，请检查 FactorEngine 版本。")
                return pd.DataFrame()

        # 时间列转 datetime
        all_df["ts"] = pd.to_datetime(all_df["ts"], errors="coerce")
        all_df = all_df.dropna(subset=["ts"])
        return all_df

    def _build_sequences(
        self,
        df: pd.DataFrame,
        seq_len: Optional[int] = None,
        max_samples: int = 200_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将按 (code, ts) 排序的分钟数据切成序列样本：
        - X: (N, T, F)  T=seq_len
        - y: (N,)       目标 = fwd_ret_10m * target_scale
        """
        if seq_len is None:
            seq_len = self.config.seq_len

        feature_cols = list(self.config.feature_names)
        label_col = self.config.label_col
        target_scale = self.config.target_scale

        X_list: List[np.ndarray] = []
        y_list: List[float] = []

        # 先按 code 分组
        grouped = df.sort_values(["code", "ts"]).groupby("code", sort=False)

        for code, g in grouped:
            g = g.reset_index(drop=True)
            if len(g) <= seq_len + 1:
                continue

            # 构造特征矩阵
            feat = g[feature_cols].copy()

            # 对 amount_log 做处理：原始 amount 可能规模巨大，这里先 log1p
            if "amount_log" in feature_cols and "amount" in g.columns:
                # 不在这里处理，因为 FactorEngine 目前输出的是 amount，TCNBrain 内部做 log1p
                pass

            # 若当前数据里还未有 amount_log 列，则自动从 amount 构造一列
            if "amount_log" in feature_cols and "amount_log" not in g.columns:
                if "amount" in g.columns:
                    feat["amount_log"] = np.log1p(
                        g["amount"].astype(float).clip(lower=0.0)
                    )
                else:
                    feat["amount_log"] = 0.0

            # 部分列缺失则补 0
            for c in feature_cols:
                if c not in feat.columns:
                    feat[c] = 0.0

            feat_mat = (
                feat[feature_cols].astype(float).values
            )  # (len_g, F)
            target = g[label_col].astype(float).values  # (len_g,)

            # 滑动窗口
            for i in range(seq_len, len(g)):
                y_val = target[i]
                if not np.isfinite(y_val):
                    continue

                x_seq = feat_mat[i - seq_len : i, :]  # (T, F)
                X_list.append(x_seq)
                y_list.append(y_val * target_scale)

        if not X_list:
            print("[TCN] 构造出来的样本数量为 0，请检查 ts_dataset_* 数据质量。")
            return np.empty((0, seq_len, len(self.config.feature_names))), np.empty(
                (0,)
            )

        X = np.stack(X_list, axis=0)  # (N, T, F)
        y = np.array(y_list, dtype=np.float32)

        # 随机下采样，避免样本过多撑爆内存
        n = X.shape[0]
        if n > max_samples:
            idx = np.random.choice(n, size=max_samples, replace=False)
            X = X[idx]
            y = y[idx]

        return X, y

    # ------------------------
    # 训练 & 保存 / 加载
    # ------------------------
    def train(
        self,
        max_files: Optional[int] = 20,
        seq_len: Optional[int] = None,
        max_samples: int = 200_000,
        batch_size: int = 256,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        valid_ratio: float = 0.2,
    ):
        if not TORCH_AVAILABLE:
            print(
                "[TCN] 当前环境未安装 torch，无法训练 TCN 模型。"
                " 可在 WSL/conda 环境中安装后再运行：pip install torch"
            )
            return

        # 1) 载入数据
        df = self._load_all_datasets(max_files=max_files)
        if df.empty:
            return

        # 如果 ts_dataset 中没有 amount_log 列，这里统一构造一次，后续 _build_sequences 会复用
        if "amount_log" not in df.columns and "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"].astype(float).clip(lower=0.0))

        # 2) 构造序列样本
        X, y = self._build_sequences(
            df, seq_len=seq_len, max_samples=max_samples
        )
        if X.size == 0:
            return

        n_samples, T, F = X.shape
        print(
            f"[TCN] 样本量: {n_samples}, 序列长度: {T}, 特征数: {F} "
            f"(使用最近 {max_files} 个 ts_dataset_*.csv)"
        )

        # 3) 特征归一化（按特征维度）
        flat = X.reshape(-1, F)  # (N*T, F)
        self.scaler = FeatureScaler()
        flat_norm = self.scaler.fit_transform(flat)
        X_norm = flat_norm.reshape(n_samples, T, F)

        # 4) 训练/验证集拆分
        idx = np.random.permutation(n_samples)
        n_valid = int(n_samples * valid_ratio)
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]

        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_valid, y_valid = X_norm[valid_idx], y[valid_idx]

        train_ds = SequenceDataset(X_train, y_train)
        valid_ds = SequenceDataset(X_valid, y_valid)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # 5) 搭建模型
        self.model = TCNRegressor(num_features=F)
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_valid_loss = math.inf
        best_state = None

        # 6) 训练循环
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_ds)

            # 验证
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    valid_loss += loss.item() * xb.size(0)
            valid_loss /= len(valid_ds)

            print(
                f"[TCN][Epoch {epoch:02d}] "
                f"Train MSE={train_loss:.4f} | Valid MSE={valid_loss:.4f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "valid_loss": valid_loss,
                }

        if best_state is None:
            print("[TCN] 训练未产生有效模型。")
            return

        # 7) 保存模型 & scaler & config
        self._save_artifacts(best_state, F)
        print(
            f"[TCN] 训练完成，最优验证 MSE={best_valid_loss:.4f}，"
            f"模型已保存到 {self.model_path}"
        )

    def _save_artifacts(self, best_state: dict, num_features: int):
        # 模型
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(
                {
                    "model_state": best_state["model_state"],
                    "num_features": num_features,
                    "config": self.config.__dict__,
                },
                self.model_path,
            )

        # scaler
        if self.scaler is not None:
            with open(self.scaler_path, "w", encoding="utf-8") as f:
                json.dump(self.scaler.to_dict(), f, ensure_ascii=False, indent=4)

        # config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=4)

    def load(self) -> bool:
        """
        从磁盘加载 tcn_brain.pt + scaler + config，供盘中/盘后推理使用。
        """
        if not TORCH_AVAILABLE:
            print("[TCN] 未安装 torch，无法加载 TCN 模型。")
            return False

        if not os.path.exists(self.model_path):
            print(f"[TCN] 模型文件不存在：{self.model_path}")
            return False

        try:
            ckpt = torch.load(self.model_path, map_location="cpu")
        except Exception as e:
            print(f"[TCN] 加载模型失败：{e}")
            return False

        num_features = ckpt.get("num_features", len(self.config.feature_names))
        cfg_dict = ckpt.get("config")
        if cfg_dict:
            self.config = TCNBrainConfig(**cfg_dict)

        # scaler
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "r", encoding="utf-8") as f:
                    s_dict = json.load(f)
                self.scaler = FeatureScaler.from_dict(s_dict)
            except Exception as e:
                print(f"[TCN] 加载 scaler 失败：{e}")
                self.scaler = None
        else:
            self.scaler = None

        # model
        self.model = TCNRegressor(num_features=num_features)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        return True

    # ------------------------
    # 单票推理接口（实验用）
    # ------------------------
    def predict_sequence(
        self, seq_df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        对单只股票最近一段分钟数据进行预测。

        :param seq_df: 必须包含 feature_names 列，按时间升序。
                       至少包含 seq_len 条记录，多的只取最后 seq_len 条。
        :return: (pred_ret_pct, attack_prob)
                 pred_ret_pct: 预测未来10min收益（百分比，例如 2.3 表示 +2.3%）
                 attack_prob : 0~1，粗略将回归值映射为“攻击成功概率”
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("TCN 模型尚未加载或当前环境无 torch。")

        if self.scaler is None:
            raise RuntimeError("TCN 特征归一化参数尚未加载。")

        seq_len = self.config.seq_len
        feature_cols = list(self.config.feature_names)
        thr = self.config.attack_threshold_pct

        df = seq_df.sort_values("ts").copy()
        if len(df) < seq_len:
            # 不足则复制最后一条补齐
            last = df.iloc[-1:]
            need = seq_len - len(df)
            df = pd.concat([df, last.loc[last.index.repeat(need)]], ignore_index=True)

        df = df.iloc[-seq_len:]

        # 构造 amount_log
        if "amount_log" in feature_cols and "amount_log" not in df.columns:
            if "amount" in df.columns:
                df["amount_log"] = np.log1p(
                    df["amount"].astype(float).clip(lower=0.0)
                )
            else:
                df["amount_log"] = 0.0

        feat = df[feature_cols].astype(float).values  # (T, F)
        feat = feat.reshape(-1, feat.shape[1])  # (T, F)

        # 归一化（使用训练时的 mean/std）
        flat_norm = self.scaler.transform(feat)  # (T, F)
        X = flat_norm.reshape(1, seq_len, len(feature_cols))  # (1, T, F)

        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            pred_scaled = self.model(xb).cpu().numpy()[0]

        # 回到原始百分比单位
        pred_ret_pct = float(pred_scaled)

        # 粗略概率映射：sigmoid( pred / thr )
        attack_prob = 1.0 / (1.0 + math.exp(-pred_ret_pct / max(thr, 1e-3)))

        return pred_ret_pct, attack_prob


# ----------------------------
# 命令行入口
# ----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LightHunter TCNBrain Trainer (Mk-TCN R10)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ts_datasets",
        help="ts_dataset_*.csv 所在目录（默认: ts_datasets）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="训练时最多使用最近多少个 ts_dataset_*.csv（默认: 20）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="时间窗口长度（分钟，默认: 60）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数（默认: 10）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size（默认: 256）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="最大样本数，上限内随机采样（默认: 200000）",
    )

    args = parser.parse_args()

    brain = TCNBrain(data_dir=args.data_dir)
    brain.config.seq_len = int(args.seq_len)

    brain.train(
        max_files=args.max_files,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
