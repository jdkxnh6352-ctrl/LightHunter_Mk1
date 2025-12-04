# -*- coding: utf-8 -*-
"""
tools/factor_leakage_checker.py

LightHunter - 因子泄露检查工具 (FactorLeakageChecker)

用途：
- 检查特征面 / 因子面中是否存在「未来函数」或「标签泄露」风险；
- 在科研阶段作为 DataGuardian 的子模块使用，给出可解释的因子级风险报告。

支持检测：
1. 名字模式扫描：
   - 因子名中是否带 label / target / future / y_ / ret+ 等明显“像标签”的前缀。
2. 高相关度扫描：
   - 因子与目标列在不同时间滞后 lag ∈ [-max_lag, max_lag] 下的相关系数；
   - 若在某个 lag 上相关度极高（如 > 0.995），就非常可疑。
3. 近似复制检测：
   - 因子是否几乎等于标签（或标签平移若干天），即“把 y 当作特征”。
4. 滞后模式分析：
   - 哪个 lag 的相关性最大？如果在明显“未来”的方向上达到峰值，需要重点人工复核。

注意：
- 这是一个「静态体检」工具，只能给出“可疑”因子名单，真正是否泄露，需要结合因子构造逻辑人工判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class LeakageCheckConfig:
    """因子泄露检查配置。"""

    target_col: str                 # 标签列名，例如 'label_1d'
    symbol_col: str = "symbol"
    date_col: str = "trade_date"

    max_lag: int = 5                # 检查的最大滞后（天）
    min_samples: int = 50           # 计算相关系数所需的最少样本数

    high_corr_threshold: float = 0.995      # 绝对相关度超过此值视为“极高”
    suspicious_corr_threshold: float = 0.9  # 绝对相关度超过此值视为“可疑”

    # 名字模式中“看起来像标签”的关键词
    suspicious_name_tokens: Tuple[str, ...] = (
        "label",
        "target",
        "future",
        "fut_",
        "y_",
        "yhat",
        "ret+",
        "pnl",
        "profit",
        "nextret",
        "next_ret",
        "fwdret",
        "fwd_ret",
    )

    # 数值近似判断的容忍度
    duplicate_abs_tol: float = 1e-6
    duplicate_rms_ratio_tol: float = 1e-3   # RMS 差 / 标签 std 小于该值视为近似复制


class FactorLeakageChecker:
    """因子泄露检查工具。"""

    def __init__(self, cfg: LeakageCheckConfig) -> None:
        self.cfg = cfg
        self.log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def run(
        self,
        panel_df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """执行因子泄露体检。

        Args:
            panel_df: DataFrame，至少包含 [symbol_col, date_col, target_col] 和若干因子列。
            feature_cols: 要检查的因子列名列表；若为 None，则自动从数值列中筛选。

        Returns:
            DataFrame：每行对应一个因子，包含：
                factor                  : 因子名
                corr_0                  : 与标签 (lag=0) 的相关系数
                max_abs_corr            : 所有滞后中的最大绝对相关系数
                best_lag                : 对应的 lag
                corr_at_best_lag        : best_lag 下的相关系数
                suspicious_name         : 名字是否可疑
                near_duplicate          : 是否近似复制标签
                duplicate_lag           : 如复制，复制了几天的标签
                risk_level              : ok / warning / critical
                reasons                 : 文本理由
        """
        cfg = self.cfg
        df = panel_df.copy()

        for col in [cfg.symbol_col, cfg.date_col, cfg.target_col]:
            if col not in df.columns:
                raise ValueError(f"panel_df 缺少必要列: {col}")

        # 规范排序 & 日期格式
        df[cfg.symbol_col] = df[cfg.symbol_col].astype(str)
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col]).dt.date
        df = df.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)

        # 选择要检查的因子列
        if feature_cols is None:
            exclude = {cfg.symbol_col, cfg.date_col, cfg.target_col}
            feature_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            feature_cols = [c for c in feature_cols if c != cfg.target_col]

        if not feature_cols:
            raise ValueError("找不到任何可用于泄露检查的因子列。")

        self.log.info(
            "FactorLeakageChecker: 将对 %d 个因子执行泄露检查，label=%s",
            len(feature_cols),
            cfg.target_col,
        )

        # 预先按 lag 计算好标签的平移序列（按 symbol 分组）
        target_shifts: Dict[int, pd.Series] = {}
        g = df.groupby(cfg.symbol_col, sort=False)[cfg.target_col]
        for lag in range(-cfg.max_lag, cfg.max_lag + 1):
            target_shifts[lag] = g.shift(-lag)

        # 为每个因子计算相关系数 & 复制检测
        records: List[Dict[str, object]] = []
        for factor in feature_cols:
            rec = self._inspect_single_factor(df, factor, target_shifts)
            records.append(rec)

        result = pd.DataFrame(records)
        # 按风险等级排序：critical > warning > ok
        rank = {"critical": 2, "warning": 1, "ok": 0}
        result["_rank"] = result["risk_level"].map(rank).fillna(0)
        result = result.sort_values(["_rank", "max_abs_corr"], ascending=[False, False])
        result = result.drop(columns=["_rank"])
        return result

    # ------------------------------------------------------------------
    # 单因子分析
    # ------------------------------------------------------------------
    def _inspect_single_factor(
        self,
        df: pd.DataFrame,
        factor: str,
        target_shifts: Dict[int, pd.Series],
    ) -> Dict[str, object]:
        cfg = self.cfg
        x = df[factor].astype(float)

        # 计算不同 lag 下的相关系数
        corrs: Dict[int, float] = {}
        for lag, y in target_shifts.items():
            corr = self._safe_corr(x, y)
            corrs[lag] = corr

        corr_0 = corrs.get(0, np.nan)
        # 找到相关度绝对值最大的 lag
        best_lag, best_corr = self._best_lag(corrs)

        # 名字模式可疑？
        suspicious_name = self._is_suspicious_name(factor)

        # 如果相关度极高，进一步检查是否近似复制标签（或平移后的标签）
        near_duplicate = False
        duplicate_lag: Optional[int] = None

        if not np.isnan(best_corr) and abs(best_corr) >= cfg.high_corr_threshold:
            for lag, corr in corrs.items():
                if np.isnan(corr) or abs(corr) < cfg.high_corr_threshold:
                    continue
                y = target_shifts[lag]
                if self._is_near_duplicate(x, y):
                    near_duplicate = True
                    duplicate_lag = lag
                    break

        # 生成风险等级 & 理由
        risk_level, reasons = self._assess_risk(
            factor=factor,
            corr_0=corr_0,
            best_lag=best_lag,
            best_corr=best_corr,
            suspicious_name=suspicious_name,
            near_duplicate=near_duplicate,
            duplicate_lag=duplicate_lag,
        )

        rec: Dict[str, object] = {
            "factor": factor,
            "corr_0": float(corr_0) if not np.isnan(corr_0) else np.nan,
            "max_abs_corr": float(abs(best_corr)) if not np.isnan(best_corr) else np.nan,
            "best_lag": int(best_lag) if best_lag is not None else None,
            "corr_at_best_lag": float(best_corr) if not np.isnan(best_corr) else np.nan,
            "suspicious_name": suspicious_name,
            "near_duplicate": near_duplicate,
            "duplicate_lag": duplicate_lag,
            "risk_level": risk_level,
            "reasons": "; ".join(reasons),
        }
        return rec

    # ------------------------------------------------------------------
    # 相关性 / 复制判定 / 风险评级
    # ------------------------------------------------------------------
    def _safe_corr(self, x: pd.Series, y: pd.Series) -> float:
        """在样本数过少或 std=0 时返回 NaN。"""
        cfg = self.cfg
        mask = x.notna() & y.notna()
        if mask.sum() < cfg.min_samples:
            return float("nan")
        xv = x[mask].to_numpy()
        yv = y[mask].to_numpy()
        sx = xv.std()
        sy = yv.std()
        if sx == 0 or sy == 0:
            return float("nan")
        return float(np.corrcoef(xv, yv)[0, 1])

    @staticmethod
    def _best_lag(corrs: Dict[int, float]) -> Tuple[Optional[int], float]:
        """返回相关系数绝对值最大的 (lag, corr)。"""
        if not corrs:
            return None, float("nan")
        best_lag = None
        best_corr = float("nan")
        best_abs = -1.0
        for lag, c in corrs.items():
            if np.isnan(c):
                continue
            ac = abs(c)
            if ac > best_abs:
                best_abs = ac
                best_corr = c
                best_lag = lag
        return best_lag, best_corr

    def _is_suspicious_name(self, factor: str) -> bool:
        f = factor.lower()
        return any(tok in f for tok in self.cfg.suspicious_name_tokens)

    def _is_near_duplicate(self, x: pd.Series, y: pd.Series) -> bool:
        """判断 x 是否近似复制 y（在允许的误差范围内几乎相等）。"""
        cfg = self.cfg
        mask = x.notna() & y.notna()
        if mask.sum() < cfg.min_samples:
            return False
        xv = x[mask].to_numpy()
        yv = y[mask].to_numpy()
        diff = xv - yv
        max_abs = float(np.max(np.abs(diff)))
        if max_abs <= cfg.duplicate_abs_tol:
            return True
        rms = float(np.sqrt(np.mean(diff ** 2)))
        std_y = float(yv.std() or 1.0)
        if rms / std_y <= cfg.duplicate_rms_ratio_tol:
            return True
        return False

    def _assess_risk(
        self,
        *,
        factor: str,
        corr_0: float,
        best_lag: Optional[int],
        best_corr: float,
        suspicious_name: bool,
        near_duplicate: bool,
        duplicate_lag: Optional[int],
    ) -> Tuple[str, List[str]]:
        cfg = self.cfg
        reasons: List[str] = []

        # 默认等级
        level = "ok"

        if near_duplicate:
            level = "critical"
            if duplicate_lag is None or duplicate_lag == 0:
                reasons.append("因子几乎等于标签本身（疑似直接泄露）。")
            else:
                reasons.append(f"因子几乎等于平移 {duplicate_lag} 天后的标签（强烈怀疑未来数据泄露）。")
            return level, reasons

        if not np.isnan(corr_0) and abs(corr_0) >= cfg.high_corr_threshold:
            level = "critical"
            reasons.append(
                f"因子与标签在 lag=0 下绝对相关度 {abs(corr_0):.4f}，极高，建议人工复核是否包含未来信息或直接用到了 y。"
            )

        if best_lag is not None and not np.isnan(best_corr):
            if abs(best_corr) >= cfg.high_corr_threshold and best_lag != 0:
                # 在非 0 滞后上出现极高相关度，非常可疑
                level = "critical"
                reasons.append(
                    f"在 lag={best_lag} 处出现绝对相关度 {abs(best_corr):.4f} 的峰值，"
                    f"若该 lag 对应未来标签，极大概率存在未来函数。"
                )
            elif abs(best_corr) >= cfg.suspicious_corr_threshold and best_lag != 0:
                # 中度可疑
                if level != "critical":
                    level = "warning"
                reasons.append(
                    f"在 lag={best_lag} 处相关度较高 (|corr|={abs(best_corr):.4f})，建议检查该因子的构造窗口是否只用到了历史数据。"
                )

        if suspicious_name:
            if level == "ok":
                level = "warning"
            reasons.append("因子名中包含 label/target/future 等疑似标签字段，请确认该列不是误把标签当作特征。")

        if not reasons:
            reasons.append("未发现明显泄露风险。")

        return level, reasons


if __name__ == "__main__":  # 简单自测
    # 构造一个 toy 面板：y_t = ret1d; x1 = y_t（直接泄露）；x2 = y_{t+1}（未来泄露）；x3 = 正常噪声因子
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date
    df = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * n,
            "trade_date": dates,
        }
    )
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, size=n)
    df["label_1d"] = ret
    df["x_leak_direct"] = ret  # 直接等于标签
    df["x_leak_future"] = np.roll(ret, -1)  # “未来收益”
    df["x_noise"] = rng.normal(0, 1, size=n)

    cfg = LeakageCheckConfig(target_col="label_1d")
    checker = FactorLeakageChecker(cfg)
    report = checker.run(df)
    print(report)
