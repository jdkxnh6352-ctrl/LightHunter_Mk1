# -*- coding: utf-8 -*-
"""
factor_engine.py

LightHunter Mk3 - FactorEngine（含订单流 + 情绪因子）
====================================================

本版 FactorEngine 负责统一产出「超短线主战因子面板」，包括：

1. 价格/动量因子（F_*）
2. 成交量/流动性因子（F_*）
3. 风险/过热因子（F_*）
4. 订单流因子（OF_*，由 OrderFlowEngine 注入）
5. 情绪/舆情因子（SENT_*，由 SentimentEngine 注入）
6. 概念/图谱因子（CONCEPT_* / cg_*，由 ConceptGraphFeatures / GNN 注入）

使用方式
--------
1) 先构建 df_daily（日线面板）
2) 用各个引擎分别产出因子块 DataFrame，然后通过 extra_blocks 注入：

    of_block   = order_flow_engine.compute_for_daily_panel(df_daily, df_of_bar)
    sent_block = sentiment_engine.compute_for_daily_panel(df_daily, df_posts_scored)
    cg_block   = concept_graph_features(...)  # index 为 symbol/trade_date 对齐的因子块

    fe = FactorEngine()
    factor_panel = fe.compute_factor_panel(
        df_daily,
        factor_set="ultrashort_core",
        extra_blocks={
            "order_flow": of_block,
            "sentiment": sent_block,
            "concept_graph": cg_block,
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 主战因子集合定义
# ----------------------------------------------------------------------

ULTRASHORT_CORE_FACTORS: List[str] = [
    # 价格 & 动量
    "F_gap_open",
    "F_ret_intraday",
    "F_ret_5d",
    "F_high_break_5d",
    "F_close_pos_5d",
    # 成交量 & 流动性
    "F_vol_ratio_5d",
    "F_amt_ratio_5d",
    "F_turn_z_5d",
    "F_intraday_range",
    # 风险 & 过热
    "F_limit_up_freq_5d",
    "F_big_down_freq_5d",
    "F_drawdown_5d",
    "F_gap_down_risk",
    # 订单流主战指标（由 OrderFlowEngine 提供）
    "OF_buy_pressure",
    "OF_order_imbalance",
    "OF_buy_value_ratio",
    "OF_buy_tick_ratio",
    "OF_large_buy_ratio",
    "OF_bid_ask_imbalance",
    "OF_spread_mean",
    "OF_spread_norm_mean",
    # 情绪/舆情因子（由 SentimentEngine 提供）
    "SENT_stock_score",
    "SENT_stock_score_std",
    "SENT_stock_post_count",
    "SENT_stock_bull_ratio",
    "SENT_stock_bear_ratio",
    "SENT_market_score",
    "SENT_market_bull_ratio",
    "SENT_market_bear_ratio",
    "SENT_market_phase",
    # 概念/图谱因子（由 ConceptGraphFeatures / GNN 提供）
    "CONCEPT_hot_score",
    "CONCEPT_centrality",
    "cg_degree",
    "cg_weighted_degree",
    "cg_pagerank",
]

FACTOR_SETS: Dict[str, List[str]] = {
    # 完整主战因子：基础价量 + 风险 + 订单流 + 情绪 + 概念/图谱
    "ultrashort_core": ULTRASHORT_CORE_FACTORS,
    # 仅基础价量因子的轻量版
    "ultrashort_basic": [
        "F_gap_open",
        "F_ret_intraday",
        "F_ret_5d",
        "F_high_break_5d",
        "F_close_pos_5d",
        "F_vol_ratio_5d",
        "F_intraday_range",
        "F_drawdown_5d",
    ],
    # 仅风险/过热相关因子
    "risk_only": [
        "F_limit_up_freq_5d",
        "F_big_down_freq_5d",
        "F_drawdown_5d",
        "F_gap_down_risk",
    ],
}


@dataclass
class FactorEngineConfig:
    """FactorEngine 配置。"""

    lookback_short: int = 5       # 短期窗口（5 日）
    lookback_mid: int = 10        # 中期窗口（10 日）
    limit_up_threshold: float = 0.095   # 近似涨停阈值（+9.5%）
    big_down_threshold: float = -0.05   # 大阴线阈值（-5%）


class FactorEngine:
    """LightHunter Mk3 - 超短线主战因子引擎。"""

    def __init__(self, cfg: Optional[FactorEngineConfig] = None) -> None:
        self.cfg = cfg or FactorEngineConfig()

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    def compute_factor_panel(
        self,
        df_daily: pd.DataFrame,
        factor_set: str = "ultrashort_core",
        extra_blocks: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        计算主战因子面板。

        参数
        ----
        df_daily   : 日线行情数据（至少包含 symbol / open / high / low / close）
        factor_set : 预设因子集合名称（ultrashort_core / ultrashort_basic / risk_only）
        extra_blocks :
            可选字典，key 是块名（如 "order_flow" / "sentiment" / "concept_graph"），
            value 是包含若干因子列的 DataFrame，index 必须与 df_daily 可以对齐。

        返回
        ----
        factor_panel : DataFrame
            index 与 df_daily 完全一致，列为需要的因子名。
        """
        if "symbol" not in df_daily.columns:
            raise ValueError("FactorEngine: df_daily 必须包含列 'symbol'。")

        df_sorted = self._sort_by_symbol_date(df_daily)

        if factor_set not in FACTOR_SETS:
            raise KeyError(f"未知因子集合: {factor_set}，可选: {list(FACTOR_SETS.keys())}")
        target_factors = FACTOR_SETS[factor_set]

        # 1) 内置基础因子
        core_block = self._compute_core_price_volume_factors(df_sorted)

        # 2) 合并外部特征模块（订单流 / 情绪 / 概念 / GNN 等）
        merged = core_block
        if extra_blocks:
            for name, block in extra_blocks.items():
                if block is None or block.empty:
                    log.info("FactorEngine: 外部特征块 '%s' 为空，跳过。", name)
                    continue
                block_aligned = block.reindex(df_sorted.index)
                merged = pd.concat([merged, block_aligned], axis=1)
                log.info(
                    "FactorEngine: 已合并外部特征块 '%s'，列数=%d",
                    name,
                    block_aligned.shape[1],
                )

        # 3) 最终按 target_factors 过滤（不存在的列自动忽略）
        existing_cols = set(merged.columns)
        final_cols: List[str] = []
        missing: List[str] = []
        for f in target_factors:
            if f in existing_cols:
                final_cols.append(f)
            else:
                missing.append(f)

        if missing:
            log.info(
                "FactorEngine: 以下主战因子当前未能产出（可能依赖外部模块）: %s",
                ", ".join(missing),
            )

        if not final_cols:
            log.warning("FactorEngine: 目标因子列为空，请检查输入或外部模块。")

        factor_panel_sorted = merged[final_cols].copy()
        # 恢复到 df_daily 原 index 顺序
        factor_panel_sorted.index = df_sorted.index
        factor_panel = factor_panel_sorted.reindex(df_daily.index)
        return factor_panel

    # ------------------------------------------------------------------
    # 内部工具：排序 & 分组
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_symbol_date(df: pd.DataFrame) -> pd.DataFrame:
        if "trade_date" in df.columns:
            return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return df.sort_values(["symbol"]).reset_index(drop=True)

    @staticmethod
    def _groupby_symbol(df: pd.DataFrame):
        return df.groupby("symbol", group_keys=False)

    # ------------------------------------------------------------------
    # 内置基础价量因子计算（和之前版本相同）
    # ------------------------------------------------------------------

    def _compute_core_price_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"FactorEngine: 计算基础因子需要列 {required_cols}，当前缺失: {missing}"
            )

        g = self._groupby_symbol(df)
        cfg = self.cfg

        # 前收盘（用于计算缺口等），如果没有 pre_close 列，就用前一日 close 代替
        if "pre_close" in df.columns:
            pre_close = df["pre_close"].astype(float)
        else:
            pre_close = g["close"].shift(1).astype(float)

        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        # --- 价格 & 动量类 ---
        F_gap_open = (open_ / pre_close - 1.0).replace([np.inf, -np.inf], np.nan)
        F_ret_intraday = (close / open_ - 1.0).replace([np.inf, -np.inf], np.nan)

        close_shift_5 = g["close"].shift(cfg.lookback_short).astype(float)
        F_ret_5d = (close / close_shift_5 - 1.0).replace([np.inf, -np.inf], np.nan)

        rolling_high_5d = g["high"].rolling(cfg.lookback_short, min_periods=1).max()
        rolling_low_5d = g["low"].rolling(cfg.lookback_short, min_periods=1).min()

        F_high_break_5d = (close / rolling_high_5d - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )

        span = (rolling_high_5d - rolling_low_5d).replace(0, np.nan)
        F_close_pos_5d = (close - rolling_low_5d) / span

        # --- 成交量 & 流动性 ---
        if "volume" in df.columns:
            volume = df["volume"].astype(float)
            vol_mean_5d = g["volume"].rolling(cfg.lookback_short, min_periods=1).mean()
            F_vol_ratio_5d = (volume / vol_mean_5d).replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            F_vol_ratio_5d = pd.Series(np.nan, index=df.index)

        if "amount" in df.columns:
            amount = df["amount"].astype(float)
            amt_mean_5d = g["amount"].rolling(cfg.lookback_short, min_periods=1).mean()
            F_amt_ratio_5d = (amount / amt_mean_5d).replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            F_amt_ratio_5d = pd.Series(np.nan, index=df.index)

        if "turnover" in df.columns:
            turnover = df["turnover"].astype(float)
            rolling_turn_mean = g["turnover"].rolling(cfg.lookback_short, min_periods=1).mean()
            rolling_turn_std = g["turnover"].rolling(cfg.lookback_short, min_periods=1).std()
            F_turn_z_5d = (turnover - rolling_turn_mean) / rolling_turn_std.replace(
                0, np.nan
            )
        else:
            F_turn_z_5d = pd.Series(np.nan, index=df.index)

        intraday_span = (high - low).replace(0, np.nan)
        F_intraday_range = intraday_span / close.replace(0, np.nan)

        # --- 风险 & 过热 ---
        daily_ret = (close / pre_close - 1.0).replace([np.inf, -np.inf], np.nan)
        limit_up_flag = (daily_ret >= cfg.limit_up_threshold).astype("int8")
        F_limit_up_freq_5d = (
            g[limit_up_flag.name]
            .apply(lambda s: s.rolling(cfg.lookback_short, min_periods=1).sum())
        )
        F_limit_up_freq_5d = F_limit_up_freq_5d.reindex(df.index)

        big_down_flag = (daily_ret <= cfg.big_down_threshold).astype("int8")
        F_big_down_freq_5d = (
            g[big_down_flag.name]
            .apply(lambda s: s.rolling(cfg.lookback_short, min_periods=1).sum())
        )
        F_big_down_freq_5d = F_big_down_freq_5d.reindex(df.index)

        rolling_close_high_5d = g["close"].rolling(cfg.lookback_short, min_periods=1).max()
        F_drawdown_5d = (close / rolling_close_high_5d - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )

        gap_down = (open_ / pre_close - 1.0).replace([np.inf, -np.inf], np.nan)
        F_gap_down_risk = gap_down.where(gap_down < 0.0, 0.0)

        core_factors = pd.DataFrame(
            {
                "F_gap_open": F_gap_open,
                "F_ret_intraday": F_ret_intraday,
                "F_ret_5d": F_ret_5d,
                "F_high_break_5d": F_high_break_5d,
                "F_close_pos_5d": F_close_pos_5d,
                "F_vol_ratio_5d": F_vol_ratio_5d,
                "F_amt_ratio_5d": F_amt_ratio_5d,
                "F_turn_z_5d": F_turn_z_5d,
                "F_intraday_range": F_intraday_range,
                "F_limit_up_freq_5d": F_limit_up_freq_5d,
                "F_big_down_freq_5d": F_big_down_freq_5d,
                "F_drawdown_5d": F_drawdown_5d,
                "F_gap_down_risk": F_gap_down_risk,
            },
            index=df.index,
        )

        return core_factors


__all__ = [
    "FactorEngine",
    "FactorEngineConfig",
    "ULTRASHORT_CORE_FACTORS",
    "FACTOR_SETS",
]
