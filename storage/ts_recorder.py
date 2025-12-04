# -*- coding: utf-8 -*-
"""
ts_recorder.py

TSRecorder：分时数据记录器

职责：
- 接收由 TSCollector / 其他模块传入的行情快照 DataFrame；
- 调用 storage.ts_storage.TSStorage，将数据双写到 SQLite + Parquet；
- 尽量保持对旧代码的兼容（保留部分传统参数名 / 方法名）。

说明：
- 目前仅负责写入“tick / snapshot”级别数据，写入表名默认 "snapshots"；
- 表的具体 schema 由传入的 DataFrame 决定（pandas.to_sql 会自动建表）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.logging_utils import get_logger
from storage.ts_storage import get_ts_storage, TSStorage

logger = get_logger(__name__)


class TSRecorder:
    """
    分时数据记录器。

    推荐用法：
        from ts_recorder import TSRecorder

        recorder = TSRecorder()
        recorder.record_snapshot_df(df)  # df 是一批行情快照记录

    兼容旧用法：
        TSRecorder("ts_data.db")  # db_path 参数会被忽略，仅用于兼容旧代码签名
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        storage: Optional[TSStorage] = None,
        table: str = "snapshots",
    ) -> None:
        """
        Args:
            db_path:  旧版本可能会传入 ts_data.db 的路径，这里为了兼容保留该参数，但会忽略；
            storage:  指定一个 TSStorage 实例，一般不需要，默认使用全局单例；
            table:    写入的表名 / 前缀名，默认 "snapshots"。
        """
        if db_path:
            logger.info(
                "TSRecorder 初始化时传入 db_path=%s（当前版本将忽略该参数，由 TSStorage 管理实际路径）。",
                db_path,
            )

        self._log = get_logger(self.__class__.__name__)
        self._storage: TSStorage = storage or get_ts_storage()
        self._table: str = table

    # ----------------- 对外写入接口 ----------------- #

    def record_snapshot_df(self, df: pd.DataFrame) -> None:
        """
        记录一批行情快照（DataFrame）。

        要求：
        - df 每一行是一只股票在某一时刻的行情记录；
        - 列结构尽量保持稳定（后续写入同一表时要保持兼容）。
        """
        if df is None or df.empty:
            self._log.warning("record_snapshot_df 收到空 DataFrame，忽略")
            return

        try:
            self._storage.write_snapshots(df, table=self._table)
            self._log.info("已写入 %d 条快照记录到表 %s", len(df), self._table)
        except Exception:
            self._log.exception("写入快照数据失败")

    # 若旧代码使用 record_snapshots / record 等名称，可以加别名方法：

    def record_snapshots(self, df: pd.DataFrame) -> None:
        """
        兼容别名：record_snapshots -> record_snapshot_df
        """
        self.record_snapshot_df(df)

    def record(self, df: pd.DataFrame) -> None:
        """
        兼容别名：record -> record_snapshot_df
        """
        self.record_snapshot_df(df)
