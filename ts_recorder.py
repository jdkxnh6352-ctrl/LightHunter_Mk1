# -*- coding: utf-8 -*-
"""
ts_recorder.py

TSRecorder Mk2 - 统一接入 TSStorage 的时序数据记录器

职责
====
1. 为上游采集器（ts_collector / market_ts_collector 等）提供一个稳定的“写盘入口”；
2. 把分钟线 / 日线等数据以统一格式交给 TSStorage，由 TSStorage 负责：
   - DuckDB / SQLite 双写；
   - 后续可能扩展 Parquet 冷数据等；
3. 对上游屏蔽具体存储实现细节，只暴露清晰的 record_* 接口。

设计要点
========
- 默认按 trading_date 维度写入；
- 若传入的 DataFrame 中缺少 trading_date 列，将自动填充；
- 不对列名做过度限制，但推荐列集合：
    ["symbol", "trading_date", "ts", "open", "high", "low", "close", "volume", "amount", ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger
from storage.ts_storage import TSStorage, get_ts_storage

log = get_logger(__name__)


@dataclass
class TSRecorderConfig:
    """
    TSRecorder 行为配置。

    目前保持简单，只预留一些钩子，未来可扩展：
        - default_source: 当上游未提供 source 时的默认数据源标记；
        - enforce_trading_date: 若为 True，则强制将所有记录的 trading_date 列
          替换为传入的 trading_date 参数。
    """

    default_source: str = "unknown"
    enforce_trading_date: bool = True

    @classmethod
    def from_system_config(cls, sys_cfg: Optional[Dict[str, Any]] = None) -> "TSRecorderConfig":
        sys_cfg = sys_cfg or get_system_config()
        raw = sys_cfg.get("ts_recorder") or {}
        return cls(
            default_source=str(raw.get("default_source", "unknown")),
            enforce_trading_date=bool(raw.get("enforce_trading_date", True)),
        )


class TSRecorder:
    """
    TSRecorder Mk2 - 统一接入 TSStorage 的记录器。

    使用示例
    --------
        from datetime import date
        from ts_recorder import TSRecorder

        rec = TSRecorder.get_instance()
        rec.record_minute_bars(date(2024, 6, 1), df_minute, source="eastmoney")
    """

    _instance: Optional["TSRecorder"] = None

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        storage: Optional[TSStorage] = None,
        config: Optional[TSRecorderConfig] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg = config or TSRecorderConfig.from_system_config(self.sys_cfg)
        self.storage: TSStorage = storage or get_ts_storage()
        self.log = get_logger(self.__class__.__name__)

        self.log.info(
            "TSRecorder 初始化: default_source=%s enforce_trading_date=%s",
            self.cfg.default_source,
            self.cfg.enforce_trading_date,
        )

    # ------------------------------------------------------------------
    # 单例
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "TSRecorder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 公共接口：记录分钟线 / 日线
    # ------------------------------------------------------------------
    def record_minute_bars(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        source: Optional[str] = None,
    ) -> int:
        """
        记录某个交易日的分钟线数据。

        Args:
            trading_date: 交易日期；
            df          : 包含该交易日数据的 DataFrame；
            source      : 数据源名称（如 "eastmoney"/"ths"），若为空则使用配置里的 default_source。

        Returns:
            实际写入的行数。
        """
        if df is None or df.empty:
            self.log.warning(
                "TSRecorder.record_minute_bars: trading_date=%s 收到空 DataFrame，跳过写入。",
                trading_date,
            )
            return 0

        df_prepared = self._prepare_df_with_trading_date(df, trading_date)
        src = source or self.cfg.default_source

        n = self.storage.write_minute_bars(df_prepared, source=src)
        self.log.info(
            "TSRecorder.record_minute_bars: trading_date=%s source=%s rows=%d",
            trading_date,
            src,
            n,
        )
        return n

    def record_daily_bars(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        source: Optional[str] = None,
    ) -> int:
        """
        记录某个交易日的日线数据（一般是一整批复盘后的日线）。

        参数含义与 record_minute_bars 相同。
        """
        if df is None or df.empty:
            self.log.warning(
                "TSRecorder.record_daily_bars: trading_date=%s 收到空 DataFrame，跳过写入。",
                trading_date,
            )
            return 0

        df_prepared = self._prepare_df_with_trading_date(df, trading_date)
        src = source or self.cfg.default_source

        n = self.storage.write_daily_bars(df_prepared, source=src)
        self.log.info(
            "TSRecorder.record_daily_bars: trading_date=%s source=%s rows=%d",
            trading_date,
            src,
            n,
        )
        return n

    # ------------------------------------------------------------------
    # 通用入口：如果你有别的 TS 表，也可以用这个接口
    # ------------------------------------------------------------------
    def record_dataframe(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        table_name: str,
        source: Optional[str] = None,
    ) -> int:
        """
        通用记录接口：将 df 写入指定表名（双写）。

        仍然会对 df 补充/统一 trading_date 列。
        """
        if df is None or df.empty:
            self.log.warning(
                "TSRecorder.record_dataframe: table=%s trading_date=%s 收到空 DataFrame，跳过写入。",
                table_name,
                trading_date,
            )
            return 0

        df_prepared = self._prepare_df_with_trading_date(df, trading_date)
        src = source or self.cfg.default_source

        n = self.storage.write_dataframe(table_name, df_prepared, source=src)
        self.log.info(
            "TSRecorder.record_dataframe: table=%s trading_date=%s source=%s rows=%d",
            table_name,
            trading_date,
            src,
            n,
        )
        return n

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _prepare_df_with_trading_date(self, df: pd.DataFrame, trading_date: date) -> pd.DataFrame:
        """
        确保 df 中存在 'trading_date' 列。

        若 enforce_trading_date=True，则无论原本有没有 trading_date 列，
        都会将该列统一设置为传入的 trading_date（isoformat 字符串）。
        """
        if df is None or df.empty:
            return df

        df2 = df.copy()
        td_str = self._to_date_str(trading_date)

        if self.cfg.enforce_trading_date:
            df2["trading_date"] = td_str
        else:
            # 若已有 trading_date 列则尊重原值，否则填入当前交易日
            if "trading_date" not in df2.columns:
                df2["trading_date"] = td_str

        return df2

    @staticmethod
    def _to_date_str(d: date) -> str:
        if isinstance(d, datetime):
            return d.date().isoformat()
        return d.isoformat()


# ----------------------------------------------------------------------
# 模块级便捷函数（兼容旧代码）
# ----------------------------------------------------------------------


def get_recorder() -> TSRecorder:
    """获取全局 TSRecorder 单例（方便旧代码调用）。"""
    return TSRecorder.get_instance()
