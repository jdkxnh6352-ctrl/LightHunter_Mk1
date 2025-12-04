# -*- coding: utf-8 -*-
"""
storage/ts_storage.py

TSStorage Mk2 - DuckDB + Parquet 双写时序存储层

设计目标
========
1. 为 LightHunter 提供统一的 TS 写入入口；
2. 支持 "双写"：同时写入 DuckDB 表 + Parquet 分区文件；
3. 不强绑具体 schema，只假定 DataFrame 至少有:
    - symbol
    - trading_date（没有则由函数参数自动补上）
4. 默认按 trading_date 做 "分区替换"：
    - 写入同一 trading_date 的数据时，会先删除该日旧数据，再插入新数据；
    - Parquet 端则直接重建该日的分区目录。

DuckDB 配置（system_config.json）默认：
    "storage": {
      "duckdb": {
        "path": "data/storage/lighthunter.duckdb",
        "read_only": false,
        "threads": 4,
        "memory_limit": "4GB",
        "temp_directory": "data/storage/duckdb_tmp",
        "ts_tables": {
          "minute": "minute_bars",
          "daily": "daily_bars",
          "factor_panel": "factor_panel",
          "label_panel": "label_panel"
        }
      },
      "parquet": {
        "root": "data/parquet"
      }
    }

如果没有配置 "ts_tables"，则会使用上述默认表名。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config.config_center import get_system_config, get_path
from core.logging_utils import get_logger
from storage.duckdb_client import DuckDBClient, get_duckdb

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 配置结构
# ----------------------------------------------------------------------


@dataclass
class TSStorageConfig:
    """
    TSStorage 行为配置。

    use_duckdb / use_parquet:
        是否启用对应后端；若某后端出现错误时，你可以在配置里临时关闭；
    parquet_root:
        Parquet 根目录，所有数据按 kind + trading_date 分区；
    duckdb_tables:
        不同数据类型对应的 DuckDB 表名，支持：
            - "minute"
            - "daily"
            - "factor_panel"
            - "label_panel"
    partition_col:
        分区列名，默认 'trading_date'。
    """

    use_duckdb: bool = True
    use_parquet: bool = True
    parquet_root: Path = Path("data/parquet")
    duckdb_tables: Dict[str, str] = field(
        default_factory=lambda: {
            "minute": "minute_bars",
            "daily": "daily_bars",
            "factor_panel": "factor_panel",
            "label_panel": "label_panel",
        }
    )
    partition_col: str = "trading_date"

    @classmethod
    def from_system_config(cls, sys_cfg: Optional[Dict[str, Any]] = None) -> "TSStorageConfig":
        sys_cfg = sys_cfg or get_system_config()
        storage_cfg = sys_cfg.get("storage") or {}
        duck_cfg = storage_cfg.get("duckdb") or {}
        pq_cfg = storage_cfg.get("parquet") or {}

        # 解析 parquet_root（相对于 paths.root）
        raw_pq_root = pq_cfg.get("root", "data/parquet")
        try:
            root = get_path("root")
        except Exception:  # pragma: no cover
            from config.config_center import PROJECT_ROOT
            root = PROJECT_ROOT
        pq_root = Path(raw_pq_root)
        if not pq_root.is_absolute():
            pq_root = (root / raw_pq_root).resolve()

        # ts_tables（可选）
        ts_tables_raw = duck_cfg.get("ts_tables") or {}
        duckdb_tables = {
            "minute": ts_tables_raw.get("minute", "minute_bars"),
            "daily": ts_tables_raw.get("daily", "daily_bars"),
            "factor_panel": ts_tables_raw.get("factor_panel", "factor_panel"),
            "label_panel": ts_tables_raw.get("label_panel", "label_panel"),
        }

        use_duckdb = True
        if isinstance(duck_cfg.get("enabled"), bool):
            use_duckdb = duck_cfg["enabled"]

        # parquet 可以通过 storage.parquet.enabled 控制
        use_parquet = True
        if isinstance(pq_cfg.get("enabled"), bool):
            use_parquet = pq_cfg["enabled"]

        return cls(
            use_duckdb=use_duckdb,
            use_parquet=use_parquet,
            parquet_root=pq_root,
            duckdb_tables=duckdb_tables,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["parquet_root"] = str(self.parquet_root)
        return d


# ----------------------------------------------------------------------
# 主体类：TSStorage
# ----------------------------------------------------------------------


class TSStorage:
    """
    TSStorage - LightHunter 时序数据存储中枢（双写）。

    使用示例：

        from datetime import date
        from storage.ts_storage import TSStorage

        storage = TSStorage()
        df_minute = ...  # 包含 symbol / ts / price / volume 等

        storage.write_minute_bars(date(2024, 6, 1), df_minute)

    默认行为：
        - DuckDB:
            * 表名：minute_bars（可在 storage.duckdb.ts_tables 中覆盖）；
            * 同一 trading_date 会先 DELETE 再 INSERT；
        - Parquet:
            * 路径：data/parquet/minute/trading_date=2024-06-01/data.parquet
            * 当天分区目录会被重建。
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        config: Optional[TSStorageConfig] = None,
        duckdb_client: Optional[DuckDBClient] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg = config or TSStorageConfig.from_system_config(self.sys_cfg)
        self.duckdb: Optional[DuckDBClient] = (
            duckdb_client if duckdb_client is not None else (get_duckdb() if self.cfg.use_duckdb else None)
        )
        self.log = get_logger(self.__class__.__name__)
        self.log.info("TSStorage 初始化: %s", self.cfg.to_dict())

    # ------------------------------------------------------------------
    # 公共写入接口（按数据类型）
    # ------------------------------------------------------------------
    def write_minute_bars(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str = "replace_partition",
    ) -> None:
        """
        写入分钟级行情（多源合并后）。

        Args:
            trading_date: 交易日期；
            df          : DataFrame；
            mode        : "replace_partition" 或 "append"。
        """
        self._write_dataset("minute", trading_date, df, mode=mode)

    def write_daily_bars(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str = "replace_partition",
    ) -> None:
        self._write_dataset("daily", trading_date, df, mode=mode)

    def write_factor_panel(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str = "replace_partition",
    ) -> None:
        self._write_dataset("factor_panel", trading_date, df, mode=mode)

    def write_label_panel(
        self,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str = "replace_partition",
    ) -> None:
        self._write_dataset("label_panel", trading_date, df, mode=mode)

    # ------------------------------------------------------------------
    # 核心写入逻辑
    # ------------------------------------------------------------------
    def _write_dataset(
        self,
        kind: str,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str = "replace_partition",
    ) -> None:
        """
        双写核心：
            - 统一补充 partition_col（trading_date）列；
            - 调用 _write_duckdb / _write_parquet。
        """
        if df is None or df.empty:
            self.log.warning("TSStorage._write_dataset: kind=%s date=%s df 为空，跳过写入", kind, trading_date)
            return

        if mode not in ("replace_partition", "append"):
            raise ValueError("mode 仅支持 'replace_partition' 或 'append'")

        df = self._ensure_partition_col(df, trading_date)

        if self.cfg.use_duckdb and self.duckdb is not None:
            self._write_duckdb(kind, trading_date, df, mode=mode)
        else:
            self.log.debug("TSStorage: DuckDB 已禁用，跳过 kind=%s date=%s 的 DuckDB 写入", kind, trading_date)

        if self.cfg.use_parquet:
            self._write_parquet(kind, trading_date, df, mode=mode)
        else:
            self.log.debug("TSStorage: Parquet 已禁用，跳过 kind=%s date=%s 的 Parquet 写入", kind, trading_date)

    # ------------------------------------------------------------------
    # DuckDB 写入
    # ------------------------------------------------------------------
    def _write_duckdb(
        self,
        kind: str,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str,
    ) -> None:
        assert self.duckdb is not None, "DuckDB 未初始化"
        table = self.cfg.duckdb_tables.get(kind, kind)
        tds = self._date_to_str(trading_date)
        part_col = self.cfg.partition_col

        self.log.info(
            "TSStorage: DuckDB 写入 kind=%s table=%s date=%s rows=%d mode=%s",
            kind,
            table,
            tds,
            len(df),
            mode,
        )

        # 如果表不存在，则直接创建并写入
        if not self.duckdb.table_exists(table):
            self.duckdb.to_table(table, df, if_exists="replace")
            return

        # 表已存在，根据 mode 决定是否删除旧分区
        if mode == "replace_partition":
            try:
                sql = f"DELETE FROM {table} WHERE {part_col} = ?"
                self.duckdb.execute(sql, [tds])
            except Exception as e:
                self.log.exception(
                    "TSStorage._write_duckdb: 删除旧分区失败 table=%s %s=%s err=%s",
                    table,
                    part_col,
                    tds,
                    e,
                )
                raise

        # 追加新数据
        self.duckdb.to_table(table, df, if_exists="append")

    # ------------------------------------------------------------------
    # Parquet 写入
    # ------------------------------------------------------------------
    def _write_parquet(
        self,
        kind: str,
        trading_date: date,
        df: pd.DataFrame,
        *,
        mode: str,
    ) -> None:
        tds = self._date_to_str(trading_date)
        part_dir = self.cfg.parquet_root / kind / f"{self.cfg.partition_col}={tds}"

        self.log.info(
            "TSStorage: Parquet 写入 kind=%s dir=%s rows=%d mode=%s",
            kind,
            part_dir,
            len(df),
            mode,
        )

        # 重建分区目录
        if mode == "replace_partition" and part_dir.exists():
            shutil.rmtree(part_dir, ignore_errors=True)
        part_dir.mkdir(parents=True, exist_ok=True)

        # 简单地写一个 data.parquet
        file_path = part_dir / "data.parquet"
        try:
            df.to_parquet(file_path, index=False)
        except Exception as e:
            self.log.exception("TSStorage._write_parquet: 写 Parquet 失败 path=%s err=%s", file_path, e)
            raise

    # ------------------------------------------------------------------
    # DuckDB 读取（简单封装）
    # ------------------------------------------------------------------
    def load_from_duckdb(
        self,
        kind: str,
        trading_date: date,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        从 DuckDB 中读取某个 kind + trading_date 的数据（可选 symbols 过滤）。

        注意：只是一个简化封装，具体 where 条件可以在上层更精细控制。
        """
        if self.duckdb is None:
            raise RuntimeError("DuckDB 未启用，无法 load_from_duckdb")

        table = self.cfg.duckdb_tables.get(kind, kind)
        tds = self._date_to_str(trading_date)
        part_col = self.cfg.partition_col

        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            sql = f"""
            SELECT *
            FROM {table}
            WHERE {part_col} = ?
              AND symbol IN ({placeholders})
            """
            params: List[Any] = [tds] + list(symbols)
        else:
            sql = f"SELECT * FROM {table} WHERE {part_col} = ?"
            params = [tds]

        return self.duckdb.query_df(sql, params=params)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _date_to_str(d: date) -> str:
        if isinstance(d, datetime):
            return d.date().isoformat()
        return d.isoformat()

    def _ensure_partition_col(self, df: pd.DataFrame, trading_date: date) -> pd.DataFrame:
        """
        确保 df 中包含 partition_col 列（默认 trading_date）。
        如果不存在，则添加同名列并填入当前 trading_date。
        """
        col = self.cfg.partition_col
        if col in df.columns:
            return df

        df2 = df.copy()
        df2[col] = self._date_to_str(trading_date)
        return df2


# ----------------------------------------------------------------------
# 模块级便捷构造
# ----------------------------------------------------------------------


_default_storage: Optional[TSStorage] = None


def get_ts_storage(system_config: Optional[Dict[str, Any]] = None) -> TSStorage:
    """
    获取全局 TSStorage 单例（方便在各模块中直接调用）。
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = TSStorage(system_config=system_config)
    return _default_storage
