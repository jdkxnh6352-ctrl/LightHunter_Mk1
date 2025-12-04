# -*- coding: utf-8 -*-
"""
lab/performance_lab.py

LightHunter Mk3 - PerformanceLab
================================

统一管理：
- 回测 / 实盘 / 仿真 的绩效指标计算
- 封装对 ExperimentLab 的指标写入接口

设计目标
--------
1. 输入：
   - equity_curve: DataFrame, 至少包含列：
       - date   : 回测时间索引（datetime/date）
       - equity : 组合权益
       - ret    : 每期收益（可选，如果没有则根据 equity 计算）
   - trades: DataFrame（可选），至少包含列：
       - pnl   : 每笔交易盈亏（货币）
       - ret   : 每笔交易收益率（可选）
       - 其它如 symbol/entry_date/exit_date 等

2. 输出：
   - metrics: 统一字段名的绩效指标字典，例如：
       - bt_ret_total     : 总收益率
       - bt_ret_ann       : 年化收益率
       - bt_vol_ann       : 年化波动率
       - bt_sharpe        : 夏普比
       - bt_max_dd        : 最大回撤
       - bt_max_dd_days   : 最大回撤持续天数
       - bt_trades        : 交易次数
       - bt_win_rate      : 胜率
       - bt_avg_win       : 平均单笔盈利（货币）
       - bt_avg_loss      : 平均单笔亏损（货币）
       - bt_profit_factor : 盈亏比（总盈利/总亏损）

3. 与 ExperimentLab 联动：
   - 提供 log_to_experiment(run, metrics, prefix="bt") 帮助方法
   - 默认把所有指标打上 bt. 前缀写入 run.log_metrics(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config

log = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """绩效计算配置。"""

    trading_days: int = 244  # A 股每年交易日约 240-250
    risk_free_rate: float = 0.0  # 年化无风险利率（例如 0.02）


class PerformanceLab:
    """
    统一绩效分析模块。

    用法示例
    --------
    >>> perf_lab = PerformanceLab()
    >>> metrics = perf_lab.evaluate_backtest(equity_curve, trades_df)
    >>> perf_lab.log_to_experiment(exp_run, metrics, prefix="bt")
    """

    def __init__(self, system_cfg: Optional[Dict[str, Any]] = None) -> None:
        if system_cfg is None:
            try:
                system_cfg = get_system_config()
            except Exception:
                system_cfg = {}

        perf_cfg = (system_cfg.get("performance") or {}) if isinstance(system_cfg, dict) else {}
        self.default_trading_days = int(perf_cfg.get("trading_days", 244))
        self.default_risk_free_rate = float(perf_cfg.get("risk_free_rate", 0.0))

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    def evaluate_backtest(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        评估一次回测的绩效。

        参数
        ----
        equity_curve : DataFrame
            至少包含 'date' 和 'equity' 列，可选 'ret'。
        trades : DataFrame, optional
            每笔交易记录，包含 'pnl' / 'ret' 等。
        config : dict, optional
            可覆盖 trading_days / risk_free_rate 等配置。

        返回
        ----
        metrics : dict
            统一绩效指标字典。
        """
        cfg = self._build_config(config)
        eq_metrics = self._calc_equity_metrics(equity_curve, cfg)
        tr_metrics: Dict[str, float] = {}

        if trades is not None and len(trades) > 0:
            tr_metrics = self._calc_trade_metrics(trades)

        metrics = {**eq_metrics, **tr_metrics}
        return metrics

    # ------------------------------------------------------------------
    # 与 ExperimentLab 的联动
    # ------------------------------------------------------------------

    def log_to_experiment(
        self,
        exp_run: Any,
        metrics: Dict[str, float],
        prefix: str = "bt",
        step: int = 0,
    ) -> None:
        """
        将指标写入 ExperimentLab 的当前 run。

        期望 exp_run 提供接口：
            exp_run.log_metrics(step: int, metrics: Dict[str, float])

        所有指标会加上前缀，如：
            bt.bt_ret_ann, bt.bt_sharpe, ...
        """
        if exp_run is None:
            return

        try:
            prefixed = {f"{prefix}.{k}": float(v) for k, v in metrics.items() if v is not None}
            exp_run.log_metrics(step=step, metrics=prefixed)
        except Exception as e:  # pragma: no cover
            log.warning("写入 ExperimentLab 指标失败: %s", e)

    # ------------------------------------------------------------------
    # 内部：构造配置 / 计算 equity 级指标 / 交易级指标
    # ------------------------------------------------------------------

    def _build_config(self, config: Optional[Dict[str, Any]]) -> PerformanceConfig:
        if config is None:
            return PerformanceConfig(
                trading_days=self.default_trading_days,
                risk_free_rate=self.default_risk_free_rate,
            )
        return PerformanceConfig(
            trading_days=int(config.get("trading_days", self.default_trading_days)),
            risk_free_rate=float(config.get("risk_free_rate", self.default_risk_free_rate)),
        )

    def _calc_equity_metrics(
        self,
        equity_curve: pd.DataFrame,
        cfg: PerformanceConfig,
    ) -> Dict[str, float]:
        if equity_curve is None or len(equity_curve) == 0:
            return {
                "bt_ret_total": np.nan,
                "bt_ret_ann": np.nan,
                "bt_vol_ann": np.nan,
                "bt_sharpe": np.nan,
                "bt_max_dd": np.nan,
                "bt_max_dd_days": np.nan,
                "bt_daily_mean": np.nan,
                "bt_daily_vol": np.nan,
                "bt_n": 0.0,
            }

        df = equity_curve.copy()
        if "equity" not in df.columns:
            raise KeyError("equity_curve 必须包含 'equity' 列。")

        df = df.sort_values("date")
        eq = df["equity"].astype(float).values

        # 收益序列
        if "ret" in df.columns:
            rets = df["ret"].astype(float).values
        else:
            # 通过权益计算
            if len(eq) < 2:
                rets = np.array([], dtype=float)
            else:
                rets = eq[1:] / eq[:-1] - 1.0

        rets = np.asarray(rets, dtype=float)
        rets = rets[~np.isnan(rets)]

        if rets.size == 0:
            return {
                "bt_ret_total": np.nan,
                "bt_ret_ann": np.nan,
                "bt_vol_ann": np.nan,
                "bt_sharpe": np.nan,
                "bt_max_dd": np.nan,
                "bt_max_dd_days": np.nan,
                "bt_daily_mean": np.nan,
                "bt_daily_vol": np.nan,
                "bt_n": 0.0,
            }

        n = rets.size
        total_ret = float(np.prod(1.0 + rets) - 1.0)

        # 年化收益 / 波动
        daily_mean = float(np.mean(rets))
        daily_vol = float(np.std(rets, ddof=1)) if n > 1 else 0.0

        if n > 0:
            ret_ann = float((1.0 + total_ret) ** (cfg.trading_days / n) - 1.0)
        else:
            ret_ann = np.nan

        vol_ann = float(daily_vol * np.sqrt(cfg.trading_days))

        # 夏普比
        if vol_ann > 0:
            sharpe = float(
                (daily_mean * cfg.trading_days - cfg.risk_free_rate) / vol_ann
            )
        else:
            sharpe = np.nan

        # 最大回撤
        cummax = np.maximum.accumulate(eq)
        dd = eq / cummax - 1.0
        max_dd = float(dd.min())

        # 最大回撤持续时间
        durations = []
        cur = 0
        for v in dd:
            if v < 0:
                cur += 1
            elif cur > 0:
                durations.append(cur)
                cur = 0
        if cur > 0:
            durations.append(cur)
        max_dd_days = float(max(durations)) if durations else 0.0

        return {
            "bt_ret_total": total_ret,
            "bt_ret_ann": ret_ann,
            "bt_vol_ann": vol_ann,
            "bt_sharpe": sharpe,
            "bt_max_dd": max_dd,
            "bt_max_dd_days": max_dd_days,
            "bt_daily_mean": daily_mean,
            "bt_daily_vol": daily_vol,
            "bt_n": float(n),
        }

    def _calc_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades is None or len(trades) == 0:
            return {
                "bt_trades": 0.0,
                "bt_win_rate": np.nan,
                "bt_avg_win": np.nan,
                "bt_avg_loss": np.nan,
                "bt_profit_factor": np.nan,
                "bt_trade_ret_mean": np.nan,
            }

        df = trades.copy()
        if "pnl" not in df.columns:
            raise KeyError("trades DataFrame 必须包含 'pnl' 列。")

        pnl = df["pnl"].astype(float).values
        n_trades = int(len(pnl))

        win_mask = pnl > 0
        loss_mask = pnl < 0

        n_win = int(win_mask.sum())
        n_loss = int(loss_mask.sum())

        win_rate = float(n_win) / n_trades if n_trades > 0 else np.nan
        avg_win = float(pnl[win_mask].mean()) if n_win > 0 else np.nan
        avg_loss = float(pnl[loss_mask].mean()) if n_loss > 0 else np.nan

        gross_profit = float(pnl[win_mask].sum())
        gross_loss = float(-pnl[loss_mask].sum()) if n_loss > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        if "ret" in df.columns:
            trade_ret_mean = float(df["ret"].astype(float).mean())
        else:
            trade_ret_mean = np.nan

        return {
            "bt_trades": float(n_trades),
            "bt_win_rate": win_rate,
            "bt_avg_win": avg_win,
            "bt_avg_loss": avg_loss,
            "bt_profit_factor": profit_factor,
            "bt_trade_ret_mean": trade_ret_mean,
        }


__all__ = ["PerformanceLab", "PerformanceConfig"]
