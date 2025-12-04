# -*- coding: utf-8 -*-
"""
ts_minute_engine.py

TSMinuteEngine：分钟线查询引擎（业务层友好封装）

职责：
- 为 Commander / Backtest / Lab 提供方便的分钟线获取接口；
- 内部完全复用 core.ts_engine.TSEngine，不直接操作 DuckDB / SQLite。

用法示例：

    from ts_minute_engine import TSMinuteEngine

    eng = TSMinuteEngine()
    df = eng.get_bars("000001", "2025-01-01 09:30:00", "2025-01-01 10:30:00")
    print(df.head())
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from core.logging_utils import get_logger
from core.ts_engine import TSEngine, get_ts_engine

logger = get_logger(__name__)


class TSMinuteEngine:
    """
    分钟线查询引擎。
    """

    def __init__(self, ts_engine: Optional[TSEngine] = None) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._ts_engine: TSEngine = ts_engine or get_ts_engine()

    def get_bars(
        self,
        code: str,
        start_ts: str,
        end_ts: str,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        获取单只股票在指定时间区间的分钟线。
        """
        return self.get_bars_multi(
            codes=[code], start_ts=start_ts, end_ts=end_ts, freq=freq
        )

    def get_bars_multi(
        self,
        codes: Sequence[str],
        start_ts: str,
        end_ts: str,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        获取多只股票在指定时间区间的分钟线（返回一个合并 DataFrame）。
        """
        if not codes:
            self._log.warning("get_bars_multi: codes 为空，返回空 DataFrame")
            return pd.DataFrame()

        df = self._ts_engine.get_bars(
            codes=list(codes), start_ts=start_ts, end_ts=end_ts, freq=freq
        )
        return df

    def get_day_bars(
        self,
        trade_date: str,
        codes: Optional[Sequence[str]] = None,
        freq: str = "1min",
        start_time: str = "09:00:00",
        end_time: str = "16:00:00",
    ) -> pd.DataFrame:
        """
        获取某个交易日的日内分钟线。
        """
        df = self._ts_engine.get_intraday_bars_for_date(
            trade_date=trade_date,
            codes=list(codes) if codes is not None else None,
            freq=freq,
            start_time=start_time,
            end_time=end_time,
        )
        return df
