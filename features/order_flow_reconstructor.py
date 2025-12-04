# -*- coding: utf-8 -*-
"""
features/order_flow_reconstructor.py

LightHunter Mk3 - OrderFlowReconstructor 增强版
===============================================

职责：
------
基于 L1 快照数据（Snapshot），通过差分和简单的微观结构推断，
重构出「订单流」相关的基础量：

    - 主动买入/卖出成交量与成交额
    - 净成交量/净成交额
    - 买卖方向 tick 计数
    - 大单买卖成交量
    - 盘口价差与盘口不平衡度

再按给定频率（默认 1 分钟）聚合，为上层 OrderFlowEngine / Risk 模块提供输入。

输入数据预期：
--------------
DataFrame df_snapshots 至少包含列：
    - symbol        : 股票代码
    - ts            : 时间戳（datetime 或可转成 datetime 的字符串）
    - volume        : 成交量（*累积*量，从开盘开始累积）
可选列：
    - amount        : 成交额（*累积*）
    - last_price    : 最新成交价（若缺失，将回退使用 close/price）
    - bid1, ask1    : 一档买卖价
    - bid1_volume, ask1_volume : 一档买卖挂单量（若有，则可更准确估计盘口不平衡）

核心输出字段（按聚合后的 bar 粒度）：
--------------------------------------
    - of_buy_vol        : 主动买入成交量
    - of_sell_vol       : 主动卖出成交量
    - of_net_vol        : 净成交量（买 - 卖）
    - of_buy_amt        : 主动买入成交额（如有 amount）
    - of_sell_amt       : 主动卖出成交额
    - of_net_amt        : 净成交额
    - of_buy_ticks      : 归为“主动买”方向的 tick 数
    - of_sell_ticks     : 归为“主动卖”方向的 tick 数
    - of_total_ticks    : 总 tick 数（买+卖）
    - of_large_buy_vol  : 大单买入成交量
    - of_large_sell_vol : 大单卖出成交量
    - of_spread_mean    : 平均盘口价差（bid1/ask1）
    - of_book_imbalance_mean : 平均盘口不平衡度

注：
----
这里的“主动买/卖”方向判断采用简化规则：
    - 默认：价格上行 => 主动买，价格下行 => 主动卖，持平 => 中性
    - 若存在 bid1 / ask1，则可近似使用 mid-price 辅助判断
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class OrderFlowReconstructorConfig:
    """订单流重构配置。"""

    resample_freq: str = "1min"          # 聚合频率
    large_trade_quantile: float = 0.9    # 将 d_vol 超过该分位数视为“大单”


class OrderFlowReconstructor:
    """基于 L1 快照重构订单流（增强版）。"""

    def __init__(self, cfg: Optional[OrderFlowReconstructorConfig] = None) -> None:
        self.cfg = cfg or OrderFlowReconstructorConfig()

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def reconstruct(self, df_snapshots: pd.DataFrame) -> pd.DataFrame:
        """
        对 L1 快照数据进行订单流重构，并按 cfg.resample_freq 聚合。

        参数
        ----
        df_snapshots : DataFrame
            至少包含列 ['symbol', 'ts', 'volume']，推荐同时包含：
                - amount
                - last_price
                - bid1, ask1
                - bid1_volume, ask1_volume

        返回
        ----
        of_bar : DataFrame
            每行是一个 (symbol, ts_bar) 的时间段，列为上述 of_* 与盘口统计量。
        """
        if df_snapshots.empty:
            log.warning("OrderFlowReconstructor: 输入为空，返回空 DataFrame。")
            return pd.DataFrame()

        required = {"symbol", "ts", "volume"}
        missing = required - set(df_snapshots.columns)
        if missing:
            raise ValueError(f"OrderFlowReconstructor: 输入缺少必要列: {missing}")

        df = df_snapshots.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

        results = []
        for symbol, g in df.groupby("symbol", group_keys=False):
            g_res = self._reconstruct_one_symbol(symbol, g)
            results.append(g_res)

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)
        # 衍生字段
        out["of_net_vol"] = out["of_buy_vol"] - out["of_sell_vol"]
        out["of_net_amt"] = out["of_buy_amt"] - out["of_sell_amt"]
        out["of_total_ticks"] = out["of_buy_ticks"] + out["of_sell_ticks"]

        return out

    # ------------------------------------------------------------------
    # 单股票重构逻辑
    # ------------------------------------------------------------------

    def _reconstruct_one_symbol(self, symbol: str, g: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        g = g.copy().sort_values("ts")

        # --- 成交量/成交额差分 ---
        g["volume"] = g["volume"].astype(float)
        g["d_vol"] = g["volume"].diff().fillna(0.0)
        # 只考虑正向成交量（防止异常数据）
        g["d_vol"] = g["d_vol"].clip(lower=0.0)

        if "amount" in g.columns:
            g["amount"] = g["amount"].astype(float)
            g["d_amt"] = g["amount"].diff().fillna(0.0).clip(lower=0.0)
        else:
            g["d_amt"] = 0.0

        # 如果 d_vol 全是 0，仍然继续走后面的逻辑（会得到全 0 的 bar）
        if g["d_vol"].sum() == 0:
            log.info("OrderFlowReconstructor: symbol=%s 无增量成交量，结果全部为 0。", symbol)

        # --- 价格与方向判断 ---
        # 价格优先使用 last_price，否则尝试使用 close / price
        price_col = None
        for cand in ("last_price", "price", "close"):
            if cand in g.columns:
                price_col = cand
                break

        if price_col is None:
            raise ValueError(
                f"OrderFlowReconstructor: symbol={symbol} 缺少价格列（last_price/price/close 至少一个）。"
            )

        g[price_col] = g[price_col].astype(float)
        g["price"] = g[price_col]
        g["price_prev"] = g["price"].shift(1)

        # 盘口信息（可选）
        if "bid1" in g.columns and "ask1" in g.columns:
            g["bid1"] = g["bid1"].astype(float)
            g["ask1"] = g["ask1"].astype(float)
            g["ba_spread"] = (g["ask1"] - g["bid1"]).clip(lower=0.0)
            if "bid1_volume" in g.columns and "ask1_volume" in g.columns:
                bvol = g["bid1_volume"].astype(float)
                svol = g["ask1_volume"].astype(float)
                g["ba_imbalance"] = (bvol - svol) / (bvol + svol + 1e-9)
            else:
                g["ba_imbalance"] = np.nan
        else:
            g["ba_spread"] = np.nan
            g["ba_imbalance"] = np.nan

        # 方向：默认基于价格变动
        cond_buy = g["price"] > g["price_prev"]
        cond_sell = g["price"] < g["price_prev"]
        side = np.where(cond_buy, 1, np.where(cond_sell, -1, 0))
        g["side"] = side

        # “大单”的阈值（按 d_vol 的分位数）
        d_vol_pos = g.loc[g["d_vol"] > 0, "d_vol"]
        if len(d_vol_pos) > 0:
            large_th = float(d_vol_pos.quantile(cfg.large_trade_quantile))
        else:
            large_th = np.inf

        is_buy = g["side"] > 0
        is_sell = g["side"] < 0
        is_large = g["d_vol"] >= large_th

        g["of_buy_vol"] = np.where(is_buy, g["d_vol"], 0.0)
        g["of_sell_vol"] = np.where(is_sell, g["d_vol"], 0.0)
        g["of_buy_amt"] = np.where(is_buy, g["d_amt"], 0.0)
        g["of_sell_amt"] = np.where(is_sell, g["d_amt"], 0.0)
        g["of_buy_ticks"] = (is_buy & (g["d_vol"] > 0)).astype("int32")
        g["of_sell_ticks"] = (is_sell & (g["d_vol"] > 0)).astype("int32")
        g["of_large_buy_vol"] = np.where(is_buy & is_large, g["d_vol"], 0.0)
        g["of_large_sell_vol"] = np.where(is_sell & is_large, g["d_vol"], 0.0)

        g = g.set_index("ts")

        # --- 频率聚合 ---
        agg = g.resample(cfg.resample_freq).agg(
            {
                "of_buy_vol": "sum",
                "of_sell_vol": "sum",
                "of_buy_amt": "sum",
                "of_sell_amt": "sum",
                "of_buy_ticks": "sum",
                "of_sell_ticks": "sum",
                "of_large_buy_vol": "sum",
                "of_large_sell_vol": "sum",
                "ba_spread": "mean",
                "ba_imbalance": "mean",
            }
        )

        agg = agg.rename_axis("ts").reset_index()
        agg["symbol"] = symbol

        # 统一列名
        agg = agg[
            [
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
                "ba_spread",
                "ba_imbalance",
            ]
        ].copy()

        # 为后续方便，直接把盘口字段改成 of_ 前缀
        agg = agg.rename(
            columns={
                "ba_spread": "of_spread_mean",
                "ba_imbalance": "of_book_imbalance_mean",
            }
        )

        return agg


__all__ = [
    "OrderFlowReconstructor",
    "OrderFlowReconstructorConfig",
]
