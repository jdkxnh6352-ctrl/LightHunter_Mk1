# -*- coding: utf-8 -*-
"""
tools/print_data_score_report.py

DataGuardian Ω 报表查看工具：

- 从 ts_data.db.data_calendar 中读取最近 N 天的体检结果；
- 在终端打印简要的日报表，方便快速评估最近数据质量。

使用示例（项目根目录）：

    # 查看最近 10 个交易日的体检结果
    python tools/print_data_score_report.py --last 10

    # 查看指定日期区间
    python tools/print_data_score_report.py --from 2024-11-01 --to 2024-11-30
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config

logger = get_logger(__name__)

# tools/ 上一层是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_ts_db_path() -> Path:
    """
    根据 system_config 获取 ts_data.db 路径。
    """
    cfg = get_system_config()
    paths_cfg = cfg.get("paths", {}) or {}

    db_root = paths_cfg.get("db_root", "data/db")
    ts_db_name = paths_cfg.get("ts_db", "ts_data.db")

    db_path = (PROJECT_ROOT / db_root / ts_db_name).resolve()
    return db_path


def _query_calendar(
    last: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    查询 data_calendar 表。
    """
    db_path = _get_ts_db_path()
    if not db_path.exists():
        logger.warning("ts_data.db 不存在：%s", db_path)
        return pd.DataFrame()

    with sqlite3.connect(str(db_path)) as conn:
        # 确认 data_calendar 是否存在
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='data_calendar';"
        )
        if cur.fetchone() is None:
            logger.warning("ts_data.db 中不存在 data_calendar 表，请先运行 data_guardian.py。")
            return pd.DataFrame()

        sql = """
        SELECT
            trade_date,
            status,
            data_score,
            coverage_ratio,
            snapshot_ratio,
            expected_codes,
            actual_codes,
            expected_snapshots,
            actual_snapshots,
            missing_codes,
            ts_first,
            ts_last,
            note
        FROM data_calendar
        """
        conds = []
        params = []

        if start_date:
            conds.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conds.append("trade_date <= ?")
            params.append(end_date)

        if conds:
            sql += " WHERE " + " AND ".join(conds)

        sql += " ORDER BY trade_date DESC"

        if last is not None and last > 0:
            sql += f" LIMIT {int(last)}"

        df = pd.read_sql_query(sql, conn, params=params)

    return df


def _print_report(df: pd.DataFrame) -> None:
    """
    把 data_calendar 的记录打印成人类友好的报表。
    """
    if df is None or df.empty:
        print("data_calendar 表中没有记录。请先运行 data_guardian.py 生成体检结果。")
        return

    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)

    print("=== DataGuardian Ω - 最近数据质量报告 ===")
    header = (
        f"{'日期':<12} {'状态':<6} {'Score':>7} "
        f"{'Cover':>8} {'Snap':>8} {'Codes':>11} {'Miss':>6} {'时间范围':<23}"
    )
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        trade_date = str(row["trade_date"])
        status = (row["status"] or "").upper()

        score = row.get("data_score", None)
        score_str = f"{score:5.1f}" if pd.notna(score) else "  -  "

        cov = row.get("coverage_ratio", None)
        cov_str = f"{cov * 100:5.1f}%" if pd.notna(cov) else "   - "

        snap = row.get("snapshot_ratio", None)
        snap_str = f"{snap * 100:5.1f}%" if pd.notna(snap) else "   - "

        exp_codes = int(row["expected_codes"]) if pd.notna(row["expected_codes"]) else 0
        act_codes = int(row["actual_codes"]) if pd.notna(row["actual_codes"]) else 0
        codes_str = f"{act_codes}/{exp_codes}" if exp_codes > 0 else f"{act_codes}"

        miss = int(row["missing_codes"]) if pd.notna(row["missing_codes"]) else 0
        miss_str = f"{miss}" if exp_codes > 0 else "-"

        ts_first = row.get("ts_first")
        ts_last = row.get("ts_last")
        if pd.notna(ts_first) and pd.notna(ts_last):
            # 只展示时间部分 HH:MM:SS
            t1 = str(ts_first)[11:19]
            t2 = str(ts_last)[11:19]
            ts_range = f"{t1}~{t2}"
        else:
            ts_range = "-"

        line = (
            f"{trade_date:<12} {status:<6} {score_str:>7} "
            f"{cov_str:>8} {snap_str:>8} {codes_str:>11} {miss_str:>6} {ts_range:<23}"
        )
        print(line)

    print("=" * len(header))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="打印 DataGuardian Ω 数据质量报告"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="显示最近 N 个交易日（默认 10）。",
    )
    parser.add_argument(
        "--from",
        dest="start_date",
        type=str,
        default=None,
        help="起始日期（含），格式 YYYY-MM-DD。",
    )
    parser.add_argument(
        "--to",
        dest="end_date",
        type=str,
        default=None,
        help="结束日期（含），格式 YYYY-MM-DD。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = _query_calendar(
        last=args.last,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    _print_report(df)


if __name__ == "__main__":
    main()
