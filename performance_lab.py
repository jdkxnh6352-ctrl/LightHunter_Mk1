# -*- coding: utf-8 -*-
"""
performance_lab.py

PerformanceLab Mk2 - 统一绩效指标与评估中心（接入 ExperimentLab）

设计目标
========
1. 把 LightHunter 体系里**所有“效果好不好”**的问题，统一到一套指标体系上：
   - 模型预测类：IC / RankIC / MSE / MAE / 命中率 等。
   - 交易回测类：年化收益、波动率、夏普、最大回撤、胜率、Calmar 等。
2. 提供一个简单的编程接口，方便：
   - 训练脚本在每个 epoch/阶段调用；
   - 回测脚本在每次策略评估后调用；
   - NightOps 在 nightly backtest 时统一记账。
3. 与 ExperimentLab 打通：
   - PerformanceLab 负责“算指标”；
   - ExperimentLab 负责“指标落盘 + 事件轨迹”。

使用例子
========
    from lab.experiment_lab import get_experiment_lab
    from performance_lab import PerformanceLab

    lab = get_experiment_lab()
    perf = PerformanceLab()

    run_id = lab.start_run(
        name="train_ultrashort_v1",
        run_type="training",
        config={"model": "xgboost_v1"},
    )

    # 模型预测评估
    metrics_pred = perf.eval_predictions(y_true, y_pred, run_id=run_id, tag="val")

    # 回测收益评估
    metrics_bt = perf.eval_equity(daily_return, run_id=run_id, tag="backtest")

    lab.end_run(run_id, status="completed", summary_metrics={**metrics_pred, **metrics_bt})
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import math
import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config
from lab.experiment_lab import get_experiment_lab, ExperimentLab

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 数据类
# ----------------------------------------------------------------------


@dataclass
class EquityMetrics:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    cum_return: float
    final_equity: float
    win_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionMetrics:
    ic: float
    rank_ic: float
    ic_abs: float
    mse: float
    mae: float
    hit_rate: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------


def _safe_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.dropna()
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x).dropna()
    return pd.Series(x).dropna()


def compute_equity_metrics(
    daily_return: pd.Series,
    initial_equity: float = 1.0,
    annual_trading_days: int = 252,
) -> EquityMetrics:
    """
    给定日度（或 bar）收益序列，计算一组标准化绩效指标。

    Args:
        daily_return: 每期收益率，Series(index 为时间)；
        initial_equity: 初始资金（用于计算 final_equity）；
        annual_trading_days: 年化换算使用的交易期数。

    Returns:
        EquityMetrics 对象。
    """
    r = _safe_series(daily_return)
    if r.empty:
        return EquityMetrics(
            ann_return=0.0,
            ann_vol=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            cum_return=0.0,
            final_equity=initial_equity,
            win_rate=0.0,
        )

    mean = float(r.mean())
    vol = float(r.std(ddof=0))
    ann_return = (1.0 + mean) ** annual_trading_days - 1.0
    ann_vol = vol * math.sqrt(annual_trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # equity curve & drawdown
    cum = (1.0 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    cum_return = float(cum.iloc[-1] - 1.0)
    final_equity = float(initial_equity * cum.iloc[-1])

    win_rate = float((r > 0).mean()) if len(r) > 0 else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    return EquityMetrics(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        cum_return=cum_return,
        final_equity=final_equity,
        win_rate=win_rate,
    )


def compute_prediction_metrics(
    y_true: Any,
    y_pred: Any,
) -> PredictionMetrics:
    """
    针对“因子/模型预测 vs 标签”的场景，计算一组统一指标。

    - ic: 皮尔逊相关系数（可看作线性 IC）
    - rank_ic: Spearman 秩相关（排序一致性）
    - mse / mae: 回归误差
    - hit_rate: 方向预测命中率（sign 一致）
    """
    s_true = _safe_series(y_true)
    s_pred = _safe_series(y_pred)
    df = pd.concat([s_true, s_pred], axis=1, join="inner").dropna()
    df.columns = ["y_true", "y_pred"]

    n = len(df)
    if n == 0:
        return PredictionMetrics(
            ic=0.0,
            rank_ic=0.0,
            ic_abs=0.0,
            mse=0.0,
            mae=0.0,
            hit_rate=0.0,
            num_samples=0,
        )

    # 相关性
    ic = float(df["y_true"].corr(df["y_pred"])) if df["y_true"].std(ddof=0) > 0 and df["y_pred"].std(ddof=0) > 0 else 0.0
    rank_ic = float(df["y_true"].rank().corr(df["y_pred"].rank())) if n > 1 else 0.0

    # 回归误差
    diff = df["y_true"] - df["y_pred"]
    mse = float((diff ** 2).mean())
    mae = float(diff.abs().mean())

    # 方向命中率
    sign_true = np.sign(df["y_true"])
    sign_pred = np.sign(df["y_pred"])
    hit_rate = float((sign_true == sign_pred).mean())

    return PredictionMetrics(
        ic=ic,
        rank_ic=rank_ic,
        ic_abs=abs(ic),
        mse=mse,
        mae=mae,
        hit_rate=hit_rate,
        num_samples=int(n),
    )


# ----------------------------------------------------------------------
# PerformanceLab 主体
# ----------------------------------------------------------------------


class PerformanceLab:
    """
    PerformanceLab - 统一绩效评估与指标计算中心。

    职责：
        1. 负责算指标；
        2. 可选：将指标写入 ExperimentLab；
        3. 提供统一的 key 命名，避免“每个脚本自己起名字”导致混乱。
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        experiment_lab: Optional[ExperimentLab] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.lab = experiment_lab or get_experiment_lab(self.sys_cfg)
        self.log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 对外接口：预测类
    # ------------------------------------------------------------------
    def eval_predictions(
        self,
        y_true: Any,
        y_pred: Any,
        run_id: Optional[str] = None,
        tag: str = "val",
    ) -> Dict[str, Any]:
        """
        评估模型预测效果，并（可选）写入 ExperimentLab。

        返回的 key 命名统一为：
            f"{tag}_ic"
            f"{tag}_rank_ic"
            f"{tag}_ic_abs"
            f"{tag}_mse"
            f"{tag}_mae"
            f"{tag}_hit_rate"
            f"{tag}_num_samples"
        """
        pm = compute_prediction_metrics(y_true, y_pred)
        metrics = {
            f"{tag}_ic": pm.ic,
            f"{tag}_rank_ic": pm.rank_ic,
            f"{tag}_ic_abs": pm.ic_abs,
            f"{tag}_mse": pm.mse,
            f"{tag}_mae": pm.mae,
            f"{tag}_hit_rate": pm.hit_rate,
            f"{tag}_num_samples": pm.num_samples,
        }

        if run_id is not None:
            self.lab.log_metric(run_id, metrics)

        self.log.info("PerformanceLab.eval_predictions(%s): %s", tag, metrics)
        return metrics

    # ------------------------------------------------------------------
    # 对外接口：回测/收益类
    # ------------------------------------------------------------------
    def eval_equity(
        self,
        daily_return: Any,
        initial_equity: float = 1.0,
        run_id: Optional[str] = None,
        tag: str = "backtest",
    ) -> Dict[str, Any]:
        """
        评估一条收益序列（回测或实盘）的表现。

        返回的 key 命名统一为：
            f"{tag}_ann_return"
            f"{tag}_ann_vol"
            f"{tag}_sharpe"
            f"{tag}_max_drawdown"
            f"{tag}_calmar"
            f"{tag}_cum_return"
            f"{tag}_final_equity"
            f"{tag}_win_rate"
        """
        s = _safe_series(daily_return)
        em = compute_equity_metrics(s, initial_equity=initial_equity)

        metrics = {
            f"{tag}_ann_return": em.ann_return,
            f"{tag}_ann_vol": em.ann_vol,
            f"{tag}_sharpe": em.sharpe,
            f"{tag}_max_drawdown": em.max_drawdown,
            f"{tag}_calmar": em.calmar,
            f"{tag}_cum_return": em.cum_return,
            f"{tag}_final_equity": em.final_equity,
            f"{tag}_win_rate": em.win_rate,
        }

        if run_id is not None:
            self.lab.log_metric(run_id, metrics)

        self.log.info("PerformanceLab.eval_equity(%s): %s", tag, metrics)
        return metrics

    # ------------------------------------------------------------------
    # 组合评估：预测 + 回测
    # ------------------------------------------------------------------
    def eval_combo(
        self,
        y_true: Any,
        y_pred: Any,
        daily_return: Any,
        initial_equity: float = 1.0,
        run_id: Optional[str] = None,
        pred_tag: str = "val",
        equity_tag: str = "backtest",
    ) -> Dict[str, Any]:
        """
        一次性评估“模型预测 + 回测表现”，并打包指标。
        """
        m_pred = self.eval_predictions(y_true, y_pred, run_id=run_id, tag=pred_tag)
        m_eq = self.eval_equity(daily_return, initial_equity=initial_equity, run_id=run_id, tag=equity_tag)
        metrics = {**m_pred, **m_eq}
        return metrics
