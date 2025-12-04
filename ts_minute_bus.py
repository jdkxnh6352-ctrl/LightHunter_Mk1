# -*- coding: utf-8 -*-
"""
ts_minute_bus.py

TSMinuteBus Mk2 - 基于 TSStorage 的分钟线数据总线

职责：
    - 从 TSStorage 读取分钟级行情（minute_bars）；
    - 提供两种常用访问方式：
        1) load_intraday_frame(...)  -> 一天/多天的 DataFrame
        2) iter_time_slices(...)     -> 逐分钟切片的生成器（回测/仿真很方便）

依赖：
    - storage.ts_storage.TSStorage (DuckDB + Parquet 双写后的统一读取层)

典型用法：
    from datetime import date
    from ts_minute_bus import get_ts_minute_bus

    bus = get_ts_minute_bus()
    df = bus.load_intraday_frame(date(2024, 6, 1), symbols=["000001.SZ", "600000.SH"])

    for ts, slice_df in bus.iter_time_slices(date(2024, 6, 1), symbols=["000001.SZ"]):
        # 每分钟的切片
        do_something(slice_df)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger
from storage.ts_storage import TSStorage, get_ts_storage

log = get_logger(__name__)

DateLike = Union[date, datetime, str]


@dataclass
class TSMinuteBusConfig:
    """
    TSMinuteBus 行为配置。

    currently:
        - default_symbol_universe_key: 若未传 symbols，则可从 system_config["universe"][key] 读取；
        - ts_column: 时间列名（默认 "ts"）；
        - symbol_column: 标的列名（默认 "symbol"）。
    """

    default_symbol_universe_key: str = "a_share_symbols_file"
    ts_column: str = "ts"
    symbol_column: str = "symbol"

    @classmethod
    def from_system_config(cls, sys_cfg: Optional[Dict[str, Any]] = None) -> "TSMinuteBusConfig":
        sys_cfg = sys_cfg or get_system_config()
        raw = sys_cfg.get("ts_minute_bus") or {}
        return cls(
            default_symbol_universe_key=str(
                raw.get("default_symbol_universe_key", "a_share_symbols_file")
            ),
            ts_column=str(raw.get("ts_column", "ts")),
            symbol_column=str(raw.get("symbol_column", "symbol")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TSMinuteBus:
    """
    TSMinuteBus - 基于 TSStorage 的分钟级数据访问层。

    重点：所有“真·数据”都从 TSStorage 来（DuckDB / Parquet），
          不再直接摸 SQLite / CSV 等。
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        storage: Optional[TSStorage] = None,
        config: Optional[TSMinuteBusConfig] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg = config or TSMinuteBusConfig.from_system_config(self.sys_cfg)
        self.storage: TSStorage = storage or get_ts_storage(self.sys_cfg)
        self.log = get_logger(self.__class__.__name__)

        self.log.info(
            "TSMinuteBus 初始化: cfg=%s", self.cfg.to_dict()
        )

    # ------------------------------------------------------------------
    # 对外接口：加载单日/多日分钟线 DataFrame
    # ------------------------------------------------------------------
    def load_intraday_frame(
        self,
        start_date: DateLike,
        end_date: Optional[DateLike] = None,
        *,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        加载指定日期区间的分钟线 DataFrame。

        Args:
            start_date: 起始日期（含）
            end_date  : 结束日期（含），若为 None 则等于 start_date
            symbols   : 标的列表，若为 None 则不做 symbol 过滤（由 TSStorage 决定是否全市场）

        Returns:
            包含列至少有：
                - symbol
                - trading_date
                - ts (分钟时间戳)
            的 DataFrame。
        """
        start_d = self._to_date(start_date)
        end_d = self._to_date(end_date) if end_date is not None else start_d

        all_frames: List[pd.DataFrame] = []
        cur = start_d
        while cur <= end_d:
            df_day = self.storage.load_from_duckdb(
                kind="minute",
                trading_date=cur,
                symbols=list(symbols) if symbols is not None else None,
            )
            if df_day is not None and not df_day.empty:
                all_frames.append(df_day)
            else:
                self.log.debug(
                    "TSMinuteBus: date=%s 没有分钟线记录（可能是非交易日或尚未采集）。",
                    cur,
                )
            cur = cur + pd.Timedelta(days=1)  # 用 pandas 加日期以避免闰年问题

        if not all_frames:
            self.log.warning(
                "TSMinuteBus.load_intraday_frame: [%s, %s] 没有任何分钟线数据。",
                start_d,
                end_d,
            )
            return pd.DataFrame()

        df_all = pd.concat(all_frames, ignore_index=True)

        # 统一时间列类型 & 排序
        ts_col = self.cfg.ts_column
        sym_col = self.cfg.symbol_column

        if ts_col in df_all.columns:
            try:
                df_all[ts_col] = pd.to_datetime(df_all[ts_col])
            except Exception as e:  # pragma: no cover
                self.log.warning("TSMinuteBus: 解析 %s 列为 datetime 失败 err=%s", ts_col, e)

        sort_cols = []
        if "trading_date" in df_all.columns:
            sort_cols.append("trading_date")
        if ts_col in df_all.columns:
            sort_cols.append(ts_col)
        if sym_col in df_all.columns:
            sort_cols.append(sym_col)

        if sort_cols:
            df_all = df_all.sort_values(sort_cols).reset_index(drop=True)

        return df_all

    # ------------------------------------------------------------------
    # 对外接口：按分钟切片迭代
    # ------------------------------------------------------------------
    def iter_time_slices(
        self,
        trading_date: DateLike,
        *,
        symbols: Optional[Sequence[str]] = None,
    ) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
        """
        以“每分钟一个切片”的方式，遍历某个交易日的分钟线。

        Args:
            trading_date: 交易日期
            symbols     : 标的列表，若为 None 则不做 symbol 过滤

        Yields:
            (ts, df_slice)
            其中 ts 是该分钟的时间戳（pd.Timestamp），df_slice 是该分钟所有标的的行。
        """
        d = self._to_date(trading_date)
        df = self.load_intraday_frame(d, d, symbols=symbols)
        if df is None or df.empty:
            self.log.warning(
                "TSMinuteBus.iter_time_slices: date=%s 没有分钟线数据，直接返回空迭代器。",
                d,
            )
            return iter(())

        ts_col = self.cfg.ts_column
        if ts_col not in df.columns:
            self.log.warning(
                "TSMinuteBus.iter_time_slices: DataFrame 中缺少时间列 '%s'，无法按时间切片。",
                ts_col,
            )
            return iter(())

        # 确保 ts 列为 datetime，按 ts 聚合
        df = df.copy()
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception as e:  # pragma: no cover
            self.log.warning("TSMinuteBus: 解析 %s 列为 datetime 失败 err=%s", ts_col, e)

        for ts_val, g in df.groupby(ts_col):
            # g 是该分钟所有 symbol 的 snapshot/bars
            yield ts_val, g

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _to_date(d: DateLike) -> date:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        # str
        return datetime.strptime(str(d), "%Y-%m-%d").date()


# ----------------------------------------------------------------------
# 模块级单例 & 简易封装
# ----------------------------------------------------------------------

_default_bus: Optional[TSMinuteBus] = None


def get_ts_minute_bus(system_config: Optional[Dict[str, Any]] = None) -> TSMinuteBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = TSMinuteBus(system_config=system_config)
    return _default_bus


def load_intraday_frame(
    start_date: DateLike,
    end_date: Optional[DateLike] = None,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    模块级便捷函数：从 TSStorage 读取分钟线 DataFrame。
    """
    bus = get_ts_minute_bus()
    return bus.load_intraday_frame(start_date, end_date, symbols=symbols)


def iter_time_slices(
    trading_date: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    模块级便捷函数：按分钟切片迭代某日行情。
    """
    bus = get_ts_minute_bus()
    return bus.iter_time_slices(trading_date, symbols=symbols)
