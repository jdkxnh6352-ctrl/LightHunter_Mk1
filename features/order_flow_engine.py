# -*- coding: utf-8 -*-
"""
features/order_flow_engine.py

LightHunter Mk3 - OrderFlowEngine 增强版
=======================================

职责：
------
在 OrderFlowReconstructor 的基础上，对重构得到的 of_* 底层数据进行
「日度聚合 + 指标强化」，产出一组与超短线强相关的订单流主战因子：

输出字段（按 symbol × trade_date 粒度）：
------------------------------------------
    - OF_buy_pressure        : (买量 - 卖量) / (买量 + 卖量)，[-1, 1]
    - OF_order_imbalance     : 与 buy_pressure 类似，可用于兼容旧代码
    - OF_buy_value_ratio     : 买入成交额 / 总成交额
    - OF_buy_tick_ratio      : 买入 tick 数 / 总 tick 数
    - OF_large_buy_ratio     : 大单买入量 / (大单买入量 + 大单卖出量)
    - OF_bid_ask_imbalance   : 盘口不平衡度日均值
    - OF_spread_mean         : 盘口价差日均值（原始价差）
    - OF_spread_norm_mean    : 盘口价差相对价格（价差/估算价格）的日均值（如可计算）

这些指标主要刻画：
    - 主动买/卖的强弱（方向 + 规模）
    - 大单主导程度（主力 vs. 零散资金）
    - 盘口压力（bid/ask 不平衡、spread 紧不紧）

输入数据预期：
--------------
df_of_bar: 由 OrderFlowReconstructor.reconstruct() 返回的 DataFrame，包含列：
    - symbol
    - ts
    - of_buy_vol, of_sell_vol, of_buy_amt, of_sell_amt
    - of_buy_ticks, of_sell_ticks, of_large_buy_vol, of_large_sell_vol
    - of_spread_mean, of_book_imbalance_mean
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class OrderFlowEngineConfig:
    """订单流因子引擎配置。"""

    eps: float = 1e-9


class OrderFlowEngine:
    """订单流因子引擎（从重构后的 of_bar 聚合到日线因子）。"""

    FEATURE_COLS: List[str] = [
        "OF_buy_pressure",
        "OF_order_imbalance",
        "OF_buy_value_ratio",
        "OF_buy_tick_ratio",
        "OF_large_buy_ratio",
        "OF_bid_ask_imbalance",
        "OF_spread_mean",
        "OF_spread_norm_mean",
    ]

    def __init__(self, cfg: Optional[OrderFlowEngineConfig] = None) -> None:
        self.cfg = cfg or OrderFlowEngineConfig()

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    def aggregate_daily(self, df_of_bar: pd.DataFrame) -> pd.DataFrame:
        """
        将重构后的订单流 bar 数据聚合为 symbol × trade_date 维度。

        返回的 DataFrame 使用 MultiIndex (symbol, trade_date)，列为 FEATURE_COLS。
        """
        if df_of_bar.empty:
            log.warning("OrderFlowEngine.aggregate_daily: 输入为空。")
            idx = pd.MultiIndex.from_arrays([[], []], names=["symbol", "trade_date"])
            return pd.DataFrame(columns=self.FEATURE_COLS, index=idx)

        required = {
            "symbol",
            "ts",
            "of_buy_vol",
            "of_sell_vol",
            "of_buy_amt",
            "of_sell_amt",
            "of_buy_ticks",
            "of_sell_ticks",
            "of_large_buy_vol",
            "of_large_sell_vol",
            "of_spread_mean",
            "of_book_imbalance_mean",
        }
        missing = required - set(df_of_bar.columns)
        if missing:
            raise ValueError(f"OrderFlowEngine.aggregate_daily: 输入缺少必要列: {missing}")

        df = df_of_bar.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df["trade_date"] = df["ts"].dt.normalize()

        group = df.groupby(["symbol", "trade_date"], sort=False)

        eps = self.cfg.eps

        agg = group.agg(
            {
                "of_buy_vol": "sum",
                "of_sell_vol": "sum",
                "of_buy_amt": "sum",
                "of_sell_amt": "sum",
                "of_buy_ticks": "sum",
                "of_sell_ticks": "sum",
                "of_large_buy_vol": "sum",
                "of_large_sell_vol": "sum",
                "of_spread_mean": "mean",
                "of_book_imbalance_mean": "mean",
            }
        )

        # 衍生
        buy_vol = agg["of_buy_vol"]
        sell_vol = agg["of_sell_vol"]
        vol_sum = buy_vol + sell_vol + eps

        buy_amt = agg["of_buy_amt"]
        sell_amt = agg["of_sell_amt"]
        amt_sum = buy_amt + sell_amt + eps

        buy_ticks = agg["of_buy_ticks"]
        sell_ticks = agg["of_sell_ticks"]
        tick_sum = buy_ticks + sell_ticks + eps

        large_buy_vol = agg["of_large_buy_vol"]
        large_sell_vol = agg["of_large_sell_vol"]
        large_sum = large_buy_vol + large_sell_vol + eps

        # 主动买卖压力
        OF_buy_pressure = (buy_vol - sell_vol) / vol_sum
        OF_order_imbalance = OF_buy_pressure.copy()  # 可视作同义，保留别名便于兼容旧代码

        # 价值权重视角的买方占比
        OF_buy_value_ratio = buy_amt / amt_sum

        # tick 计数视角的买方占比
        OF_buy_tick_ratio = buy_ticks / tick_sum

        # 大单主导程度
        OF_large_buy_ratio = large_buy_vol / large_sum

        # 盘口不平衡与价差
        OF_bid_ask_imbalance = agg["of_book_imbalance_mean"]
        OF_spread_mean = agg["of_spread_mean"]

        # 若能估得到平均价格，可以考虑归一化 spread；这里简化为：
        # spread_norm ~ spread / (买卖均价的一个近似)
        approx_price = (buy_amt + sell_amt) / (buy_vol + sell_vol + eps)
        OF_spread_norm_mean = OF_spread_mean / (approx_price.replace(0, np.nan))

        out = pd.DataFrame(
            {
                "OF_buy_pressure": OF_buy_pressure,
                "OF_order_imbalance": OF_order_imbalance,
                "OF_buy_value_ratio": OF_buy_value_ratio,
                "OF_buy_tick_ratio": OF_buy_tick_ratio,
                "OF_large_buy_ratio": OF_large_buy_ratio,
                "OF_bid_ask_imbalance": OF_bid_ask_imbalance,
                "OF_spread_mean": OF_spread_mean,
                "OF_spread_norm_mean": OF_spread_norm_mean,
            },
            index=agg.index,
        )

        return out

    def compute_for_daily_panel(
        self,
        df_daily: pd.DataFrame,
        df_of_bar: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        根据 df_daily 的 (symbol, trade_date) 行，生成同 index 的订单流因子块。

        参数
        ----
        df_daily : DataFrame
            至少需要列 ['symbol', 'trade_date']。
        df_of_bar : DataFrame
            OrderFlowReconstructor.reconstruct() 的输出。

        返回
        ----
        df_factors : DataFrame
            index 与 df_daily 对齐，列为 FEATURE_COLS。
        """
        if df_daily.empty:
            return pd.DataFrame(columns=self.FEATURE_COLS, index=df_daily.index)

        if "symbol" not in df_daily.columns or "trade_date" not in df_daily.columns:
            raise ValueError("OrderFlowEngine.compute_for_daily_panel: df_daily 需要包含 'symbol' 和 'trade_date' 列。")

        daily_of = self.aggregate_daily(df_of_bar)  # MultiIndex (symbol, trade_date)

        # 将 (symbol, trade_date) 合并到 df_daily 的行上
        key_cols = ["symbol", "trade_date"]
        mapper = (
            daily_of.reset_index()
            .set_index(key_cols)[self.FEATURE_COLS]
        )

        # 根据 df_daily 的 key_cols 去 mapper 里取值
        # 为保持顺序，逐行映射
        keys = df_daily[key_cols].astype({"symbol": str}).to_records(index=False)
        res_array = []
        for sym, d in keys:
            try:
                row = mapper.loc[(sym, d)]
            except KeyError:
                row = pd.Series({col: np.nan for col in self.FEATURE_COLS})
            res_array.append(row.values)

        df_res = pd.DataFrame(
            res_array,
            columns=self.FEATURE_COLS,
            index=df_daily.index,
        )
        return df_res


__all__ = [
    "OrderFlowEngine",
    "OrderFlowEngineConfig",
]
