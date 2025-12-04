# -*- coding: utf-8 -*-
"""
工具：migrate_sqlite_to_parquet.py
路径：tools/migrate_sqlite_to_parquet.py

功能：
- 从 SQLite ts_data.db.snapshots 读取历史分时快照；
- 按日期分区写入 Parquet 数据集（布局与 TSStorage 完全一致）；
- 便于后续通过 DuckDB 直接对 Parquet 做高性能查询 / 回测。

使用方式（在项目根目录）：
    python tools/migrate_sqlite_to_parquet.py
    python tools/migrate_sqlite_to_parquet.py --max-days 5
    python tools/migrate_sqlite_to_parquet.py --db data/db/ts_data.db --parquet-dir data/ts_parquet
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from core.logging_utils import get_logger
from storage.ts_storage import TSStorage, _load_system_config_safe

logger = get_logger(__name__)


def _resolve_paths_from_config(
    db_override: Optional[str],
    parquet_override: Optional[str],
) -> tuple[Path, Path]:
    """
    结合 system_config 和命令行参数，得出最终 SQLite / Parquet 路径。
    """
    cfg = _load_system_config_safe()
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    default_ts_db = paths_cfg.get("ts_db", "ts_data.db")
    default_data_root = paths_cfg.get("data_root", "data")

    db_path = Path(db_override or default_ts_db).resolve()
    parquet_dir = Path(
        parquet_override
        or cfg.get("ts_storage", {}).get("parquet_dir", "")
        or (Path(default_data_root) / "ts_parquet")
    ).resolve()

    return db_path, parquet_dir


def _list_trade_dates(conn: sqlite3.Connection) -> List[str]:
    """
    列出 snapshots 表中所有交易日（基于 ts 的日期部分）。
    """
    sql = "SELECT DISTINCT substr(ts,1,10) AS d FROM snapshots ORDER BY d;"
    cur = conn.cursor()
    rows = cur.execute(sql).fetchall()
    dates = [r[0] for r in rows if r and r[0]]
    return dates


def _filter_dates(
    dates: List[str],
    max_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    """
    根据 max_days / start_date / end_date 筛选需要迁移的交易日。
    """
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if max_days is not None and max_days > 0 and len(dates) > max_days:
        dates = dates[-max_days:]

    return dates


def _migrate_for_date(
    conn: sqlite3.Connection,
    storage: TSStorage,
    date_str: str,
    chunksize: int = 50000,
) -> int:
    """
    迁移指定交易日的数据到 Parquet，按 chunksize 分批处理。

    Returns:
        写入的总记录数
    """
    start = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    end = start + datetime.timedelta(days=1)
    start_ts = start.strftime("%Y-%m-%d 00:00:00")
    end_ts = end.strftime("%Y-%m-%d 00:00:00")

    sql = """
        SELECT ts, code, price, pct, amount, turnover_rate
        FROM snapshots
        WHERE ts >= ? AND ts < ?
        ORDER BY ts, code;
    """

    total_written = 0
    logger.info("Start migrate date %s ...", date_str)

    for chunk in pd.read_sql_query(
        sql, conn, params=(start_ts, end_ts), chunksize=chunksize
    ):
        if chunk is None or chunk.empty:
            continue
        n = storage.append_snapshots(chunk)
        total_written += n

    logger.info(
        "Finish migrate date %s, written %d records.", date_str, total_written
    )
    return total_written


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="迁移 ts_data.db.snapshots 到 Parquet 数据集（供 DuckDB 使用）"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite ts_data.db 路径，默认从 system_config.paths.ts_db 读取。",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=None,
        help="Parquet 输出目录，默认使用 system_config.ts_storage.parquet_dir 或 data/ts_parquet。",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="仅迁移最近 N 个交易日（基于 ts 日期部分），默认迁移全部历史。",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="仅迁移该日期(含)之后的数据，格式 YYYY-MM-DD。",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="仅迁移该日期(含)之前的数据，格式 YYYY-MM-DD。",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="每批从 SQLite 读取的行数，默认 50000。",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path, parquet_dir = _resolve_paths_from_config(args.db, args.parquet_dir)

    print(f"[MIGRATE] SQLite DB : {db_path}")
    print(f"[MIGRATE] ParquetDir: {parquet_dir}")

    parquet_dir.mkdir(parents=True, exist_ok=True)

    # 只启用 Parquet 写入，避免误写回 SQLite
    storage = TSStorage(
        sqlite_path=None,
        parquet_dir=str(parquet_dir),
        enable_sqlite=False,
        enable_parquet=True,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        dates = _list_trade_dates(conn)
        if not dates:
            print("[MIGRATE] snapshots 表为空，无需迁移。")
            return

        dates = _filter_dates(
            dates,
            max_days=args.max_days,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        if not dates:
            print("[MIGRATE] 经过筛选后没有需要迁移的日期。")
            return

        print(f"[MIGRATE] 将迁移 {len(dates)} 个交易日：{dates[0]} ... {dates[-1]}")
        total = 0
        for d in dates:
            total += _migrate_for_date(
                conn,
                storage,
                d,
                chunksize=args.chunksize,
            )

        print(f"[MIGRATE] 完成迁移，合计写入 {total} 条记录。")
    finally:
        conn.close()
        storage.close()


if __name__ == "__main__":
    main()
