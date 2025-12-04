# -*- coding: utf-8 -*-
"""
storage/duckdb_client.py

DuckDBClient Mk2 - LightHunter 统一 DuckDB 访问层

职责：
    - 根据 system_config 中的 storage.duckdb 配置创建/管理 DuckDB 连接；
    - 提供统一的 query_df / execute / write_df 等高层封装；
    - 暴露模块级 get_duckdb()/query_df()/execute() 便捷函数。

依赖：
    - duckdb
    - pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from config.config_center import get_system_config, get_path
from core.logging_utils import get_logger

log = get_logger(__name__)

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

PathLike = Union[str, Path]


@dataclass
class DuckDBConfig:
    """从 system_config 抽象出的 DuckDB 配置。"""

    path: Path
    read_only: bool = False
    threads: int = 0  # 0 表示使用 DuckDB 默认线程数
    pragmas: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_system_config(cls, sys_cfg: Optional[Dict[str, Any]] = None) -> "DuckDBConfig":
        sys_cfg = sys_cfg or get_system_config()
        storage_cfg = sys_cfg.get("storage", {}) if isinstance(sys_cfg, dict) else {}
        duck_cfg = storage_cfg.get("duckdb", {}) or {}

        # 优先使用 storage.duckdb.path，如果没有则走 paths.duckdb_path
        raw_path = duck_cfg.get("path", "data/storage/lighthunter.duckdb")
        resolved_path = get_path("duckdb_path", raw_path)

        read_only = bool(duck_cfg.get("read_only", False))
        threads = int(duck_cfg.get("threads", 0))
        pragmas = duck_cfg.get("pragmas") or {}

        return cls(
            path=Path(resolved_path),
            read_only=read_only,
            threads=threads,
            pragmas=pragmas,
        )


class DuckDBClient:
    """
    DuckDBClient：全局统一的 DuckDB 访问入口。

    典型用法：

        from storage.duckdb_client import get_duckdb, query_df

        db = get_duckdb()
        df = db.query_df("SELECT 1 AS x")

        # 或者：

        df = query_df("SELECT * FROM minute_bars WHERE trading_date = ?", [date_str])
    """

    _instance: Optional["DuckDBClient"] = None

    def __init__(
        self,
        config: Optional[DuckDBConfig] = None,
        system_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if duckdb is None:
            raise RuntimeError(
                "duckdb 未安装，请先执行：pip install duckdb\n"
                "DuckDBClient 无法初始化。"
            )

        self.sys_cfg = system_config or get_system_config()
        self.config = config or DuckDBConfig.from_system_config(self.sys_cfg)
        self.log = get_logger(self.__class__.__name__)

        self._conn = self._open_connection()
        self.log.info(
            "DuckDBClient 初始化完成 path=%s read_only=%s threads=%s pragmas=%s",
            self.config.path,
            self.config.read_only,
            self.config.threads,
            self.config.pragmas,
        )

    # ------------------------------------------------------------------ #
    # 单例访问
    # ------------------------------------------------------------------ #
    @classmethod
    def get_instance(cls) -> "DuckDBClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    def _open_connection(self):
        # 确保目录存在
        try:
            self.config.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover
            pass

        conn = duckdb.connect(database=str(self.config.path), read_only=self.config.read_only)

        # 线程数
        if self.config.threads and self.config.threads > 0:
            try:
                conn.execute("SET threads = ?", [self.config.threads])
            except Exception as e:  # pragma: no cover
                self.log.warning("DuckDBClient: 设置 threads 失败 err=%s", e)

        # 其它 PRAGMA
        for key, val in (self.config.pragmas or {}).items():
            try:
                if isinstance(val, str):
                    v_expr = "'" + val.replace("'", "''") + "'"
                else:
                    v_expr = str(val)
                conn.execute(f"SET {key}={v_expr}")
            except Exception as e:  # pragma: no cover
                self.log.warning("DuckDBClient: SET %s 失败，val=%r err=%s", key, val, e)

        return conn

    @property
    def conn(self):
        return self._conn

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
                self.log.info("DuckDBClient: 连接已关闭")
        except Exception as e:  # pragma: no cover
            self.log.warning("DuckDBClient: 关闭连接失败 err=%s", e)

    # ------------------------------------------------------------------ #
    # 基础操作
    # ------------------------------------------------------------------ #
    def query_df(self, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """
        执行查询并返回 DataFrame。

        sql    : 带 ? 占位符的 SQL
        params : 可选参数列表
        """
        try:
            cur = self._conn.execute(sql, params or [])
            df = cur.df()
            return df
        except Exception as e:
            self.log.exception("DuckDBClient.query_df 失败 sql=%s params=%s err=%s", sql, params, e)
            raise

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> None:
        """
        执行写入/DDL 等不需要返回 DataFrame 的语句。
        """
        try:
            self._conn.execute(sql, params or [])
        except Exception as e:
            self.log.exception("DuckDBClient.execute 失败 sql=%s params=%s err=%s", sql, params, e)
            raise

    # ------------------------------------------------------------------ #
    # 表/Schema 工具
    # ------------------------------------------------------------------ #
    def table_exists(self, table_name: str) -> bool:
        """
        判断某个表是否存在。
        """
        sql = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """
        df = self.query_df(sql, [table_name])
        return not df.empty

    def list_tables(self) -> pd.DataFrame:
        """
        列出所有用户表。
        """
        sql = """
        SELECT table_catalog, table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
        """
        return self.query_df(sql)

    # ------------------------------------------------------------------ #
    # DataFrame 写入
    # ------------------------------------------------------------------ #
    def write_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "append",
    ) -> int:
        """
        将 DataFrame 写入指定表。

        Args:
            df        : 待写入数据
            table_name: 目标表名
            mode      :
                - "create"  : CREATE TABLE ... AS SELECT * FROM df
                - "replace" : CREATE OR REPLACE TABLE ... AS SELECT * FROM df
                - "append"  : INSERT INTO ... SELECT * FROM df

        Returns:
            实际写入行数（df 的行数）
        """
        if df is None or df.empty:
            return 0

        mode = mode.lower()
        valid_modes = {"create", "replace", "append"}
        if mode not in valid_modes:
            raise ValueError(f"不支持的写入模式: {mode}, 只能是 {valid_modes}")

        tmp_view = "__tmp_df_to_write__"
        self._conn.register(tmp_view, df)

        try:
            if mode == "create":
                self._conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM {tmp_view}"
                )
            elif mode == "replace":
                self._conn.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {tmp_view}"
                )
            elif mode == "append":
                self._conn.execute(
                    f"INSERT INTO {table_name} SELECT * FROM {tmp_view}"
                )
        except Exception as e:
            self.log.exception(
                "DuckDBClient.write_df 失败 table=%s mode=%s err=%s",
                table_name,
                mode,
                e,
            )
            raise
        finally:
            try:
                self._conn.unregister(tmp_view)
            except Exception:  # pragma: no cover
                pass

        return int(len(df))


# ---------------------------------------------------------------------- #
# 模块级便捷函数
# ---------------------------------------------------------------------- #

_default_client: Optional[DuckDBClient] = None


def get_duckdb(system_config: Optional[Dict[str, Any]] = None) -> DuckDBClient:
    """
    获取 DuckDBClient 单例实例。

    如果需要用自定义 system_config 初始化，可以在第一次调用时传入。
    """
    global _default_client
    if _default_client is None:
        _default_client = DuckDBClient(system_config=system_config)
    return _default_client


def query_df(sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    """
    模块级查询便捷函数。
    """
    client = get_duckdb()
    return client.query_df(sql, params=params)


def execute(sql: str, params: Optional[Sequence[Any]] = None) -> None:
    """
    模块级执行便捷函数。
    """
    client = get_duckdb()
    client.execute(sql, params=params)
