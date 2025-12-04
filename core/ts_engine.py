# -*- coding: utf-8 -*-
"""
core/ts_engine.py

TSEngine：统一时间序列访问引擎

职责：
- 通过 DuckDBClient 访问 Parquet 快照数据（视图：snapshots_all）；
- 统一规范时间戳字段（优先 ts，没有则用 created_at）；
- 提供原始快照和分钟级 Bar 的获取接口：
    - get_snapshots(...)
    - get_bars(...)
    - get_intraday_bars_for_date(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from core.logging_utils import get_logger
from storage.duckdb_client import get_duckdb_client, DuckDBClient

logger = get_logger(__name__)


def _to_timestamp(value: Optional[object]) -> Optional[pd.Timestamp]:
    """
    将字符串 / datetime / pandas.Timestamp 统一转成 pandas.Timestamp。
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


@dataclass
class TSQueryParams:
    codes: Optional[List[str]] = None
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None


class TSEngine:
    """
    TSEngine：统一时间序列访问引擎。

    默认数据源：
    - DuckDBClient 中的视图 snapshots_all，实际来自 data/ts_snapshots 目录下的所有 Parquet 文件；
    - 视图结构 = 所有快照记录的并集（历史 + 实时）。

    用法示例：

        from core.ts_engine import TSEngine

        ts = TSEngine()
        df_raw = ts.get_snapshots(codes=["000001", "600000"], start_ts="2025-01-01")
        df_bar = ts.get_bars(codes=["000001"], start_ts="2025-01-01 09:30:00",
                             end_ts="2025-01-01 10:30:00", freq="1min")
    """

    def __init__(self, duckdb_client: Optional[DuckDBClient] = None) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._db: DuckDBClient = duckdb_client or get_duckdb_client()

    # ---------------- 原始快照 ---------------- #

    def get_snapshots(
        self,
        codes: Optional[Sequence[str]] = None,
        start_ts: Optional[object] = None,
        end_ts: Optional[object] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        获取原始快照数据（从 snapshots_all 视图中读）。

        Args:
            codes:    代码列表；None 表示全市场（慎用，数据量可能很大）
            start_ts: 起始时间（含），字符串或 datetime；None 不限制
            end_ts:   结束时间（不含），字符串或 datetime；None 不限制
            limit:    限制返回总行数（调试用），None 不限制

        Returns:
            DataFrame，至少包含：
                - code: 股票代码
                - ts:   时间戳（统一转换/填充后字段）
            以及原始 snapshots_all 中的其他字段（price / volume 等）。
        """
        start = _to_timestamp(start_ts)
        end = _to_timestamp(end_ts)

        # 先从 DuckDB 按 code 过滤一层（避免直接把全库都拉进来）
        sql = "SELECT * FROM snapshots_all"
        params: List[object] = []
        conds: List[str] = []

        if codes:
            codes = list({str(c) for c in codes})  # 去重
            placeholders = ", ".join(["?"] * len(codes))
            conds.append(f"code IN ({placeholders})")
            params.extend(codes)

        if conds:
            sql += " WHERE " + " AND ".join(conds)

        if limit is not None and limit > 0:
            sql += f" LIMIT {int(limit)}"

        df = self._db.query_df(sql, params)
        if df is None or df.empty:
            return pd.DataFrame()

        # 统一时间戳字段：优先 ts，没有则用 created_at
        if "ts" in df.columns:
            try:
                df["ts"] = pd.to_datetime(df["ts"])
            except Exception:
                self._log.exception("无法将 snapshots_all.ts 解析为 datetime")
        elif "created_at" in df.columns:
            try:
                df["ts"] = pd.to_datetime(df["created_at"])
            except Exception:
                self._log.exception("无法将 snapshots_all.created_at 解析为 datetime")
        else:
            raise ValueError(
                "snapshots_all 视图中既没有 ts 字段也没有 created_at 字段，无法构建时间轴。"
            )

        # 时间过滤在 pandas 层执行，保证兼容性
        if start is not None:
            df = df[df["ts"] >= start]
        if end is not None:
            df = df[df["ts"] < end]

        if df.empty:
            return df

        # 排序以便后续聚合
        df = df.sort_values(["ts", "code"]).reset_index(drop=True)
        return df

    # ---------------- 分钟级 Bar 聚合 ---------------- #

    def get_bars(
        self,
        codes: Optional[Sequence[str]] = None,
        start_ts: Optional[object] = None,
        end_ts: Optional[object] = None,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        将原始快照聚合为分钟级 Bar（OHLCV）。

        Args:
            codes:    代码列表；None 表示对所有代码聚合（慎用）
            start_ts: 起始时间（含）
            end_ts:   结束时间（不含）
            freq:     频率，如 "1min" / "5min" / "15min"

        Returns:
            DataFrame，列包括：
                - code
                - ts （Bar 起始时间）
                - open, high, low, close
                - volume（若原始数据有 volume 字段）
                - amount（若原始数据有 amount 字段）
        """
        df = self.get_snapshots(codes=codes, start_ts=start_ts, end_ts=end_ts)
        if df.empty:
            return df

        if "price" not in df.columns:
            raise ValueError("snapshots 数据中缺少 price 字段，无法构建 OHLC。")

        df = df.copy()
        df = df.set_index("ts")

        # 构建 named aggregation
        group = df.groupby("code").resample(freq)

        agg_spec = {
            "open": ("price", "first"),
            "high": ("price", "max"),
            "low": ("price", "min"),
            "close": ("price", "last"),
        }
        if "volume" in df.columns:
            agg_spec["volume"] = ("volume", "sum")
        if "amount" in df.columns:
            agg_spec["amount"] = ("amount", "sum")

        bars = group.agg(**agg_spec).reset_index()

        # 过滤掉没有 open/close 的空 Bar
        bars = bars.dropna(subset=["open", "close"])

        # 统一列名：ts 为时间
        bars.rename(columns={"ts": "ts"}, inplace=True)

        # 排序
        bars = bars.sort_values(["ts", "code"]).reset_index(drop=True)
        return bars

    def get_intraday_bars_for_date(
        self,
        trade_date: str,
        codes: Optional[Sequence[str]] = None,
        freq: str = "1min",
        start_time: str = "09:00:00",
        end_time: str = "16:00:00",
    ) -> pd.DataFrame:
        """
        获取某个交易日的日内分钟线（对 snapshots 做聚合）。

        Args:
            trade_date: 交易日期字符串，如 "2025-01-01"
            codes:      代码列表；None 表示全市场（慎用）
            freq:       频率，如 "1min" / "5min"
            start_time: 当天起始时间，默认 09:00:00
            end_time:   当天结束时间（不含），默认 16:00:00

        Returns:
            分钟级 Bar DataFrame，包含该日内指定时间区间的数据。
        """
        start_ts = f"{trade_date} {start_time}"
        end_ts = f"{trade_date} {end_time}"
        return self.get_bars(codes=codes, start_ts=start_ts, end_ts=end_ts, freq=freq)


# 模块级单例（可选）
_default_ts_engine: Optional[TSEngine] = None


def get_ts_engine() -> TSEngine:
    """
    获取全局 TSEngine 单例。
    """
    global _default_ts_engine
    if _default_ts_engine is None:
        _default_ts_engine = TSEngine()
    return _default_ts_engine
