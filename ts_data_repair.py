# -*- coding: utf-8 -*-
"""
ts_data_repair.py

TSDataRepairer：分时数据修复 / 分钟线重建器

职责：
- 读取 ts_data.db 中的 data_calendar，找到数据质量较差的交易日；
- 对这些日期在 market_ts.db 中删除旧的 minute_bars 记录；
- 通过 TSDataPipeline 再次从 snapshots 聚合分钟线，重建 minute_bars；
- 提供命令行入口，支持：
    - 自动从 data_calendar 选出需要修复的日期；
    - 手动指定日期列表或日期区间进行重建。

注意：
- 当前版本的“修复”主要是针对分钟线：重新从 snapshots 聚合，保证分钟线与最新底层数据一致；
- 对“底层 snapshots 真缺失”的情况，目前不会去外部数据源补历史，只是保证已有数据聚合合理。
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

from core.logging_utils import get_logger
from config.config_center import get_system_config
from ts_data_pipeline import TSDataPipeline
from data_guardian import _normalize_date

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


class TSDataRepairer:
    """
    TSDataRepairer：负责分钟线的“删除 + 重建”。

    重建逻辑：
    - 删除 market_ts.db.minute_bars 中 trade_date=某日 的所有记录；
    - 调用 TSDataPipeline.run_for_dates([该日]) 从 snapshots 重算分钟线并写入。

    触发方式：
    - auto_repair_from_calendar：根据 data_calendar 中的低分日期自动修复；
    - repair_dates / repair_range：手动指定日期进行修复。
    """

    def __init__(self, freq: str = "1min") -> None:
        self._log = get_logger(self.__class__.__name__)
        self._freq = freq

        cfg = get_system_config()
        paths_cfg = cfg.get("paths", {}) or {}

        db_root = paths_cfg.get("db_root", "data/db")
        ts_db_name = paths_cfg.get("ts_db", "ts_data.db")
        market_ts_name = paths_cfg.get("market_ts_db", "market_ts.db")

        self._db_root = (PROJECT_ROOT / db_root).resolve()
        self._db_root.mkdir(parents=True, exist_ok=True)

        self._ts_db_path = (self._db_root / ts_db_name).resolve()
        self._market_ts_path = (self._db_root / market_ts_name).resolve()

        self._log.info(
            "TSDataRepairer init. ts_db=%s market_ts_db=%s",
            self._ts_db_path,
            self._market_ts_path,
        )

        # 分钟线生成管线（内部会使用 TSEngine）
        self._pipeline = TSDataPipeline(freq=self._freq)

    # ----------------- 内部辅助 ----------------- #

    def _delete_minute_bars_for_date(self, trade_date: str) -> None:
        """
        删除 market_ts.db.minute_bars 中某个 trade_date 的所有记录。

        若 minute_bars 表不存在，则静默跳过。
        """
        if not self._market_ts_path.exists():
            self._log.warning(
                "market_ts.db 不存在，跳过删除 minute_bars（trade_date=%s）。",
                trade_date,
            )
            return

        with sqlite3.connect(str(self._market_ts_path)) as conn:
            try:
                conn.execute(
                    "DELETE FROM minute_bars WHERE trade_date = ?;", (trade_date,)
                )
                conn.commit()
                self._log.info(
                    "已删除 minute_bars 中 trade_date=%s 的旧记录（如存在）。", trade_date
                )
            except sqlite3.OperationalError as e:
                # 例如：no such table: minute_bars
                self._log.warning(
                    "删除 minute_bars 时出错（可能表尚不存在）：%r", e
                )

    def _get_low_score_dates_from_calendar(
        self,
        min_score: float = 80.0,
        max_days: Optional[int] = None,
    ) -> List[str]:
        """
        从 data_calendar 中选出需要修复的交易日（score < min_score 且 score > 0）。

        Args:
            min_score: 分数阈值，低于此值会被标记为“需要修复”；
            max_days:  限制最多返回多少个最近日期（按 trade_date 倒序），None 表示不限制。

        Returns:
            按日期升序的 trade_date 列表（YYYY-MM-DD）。
        """
        if not self._ts_db_path.exists():
            self._log.warning(
                "ts_data.db 不存在（%s），无法从 data_calendar 读取信息。",
                self._ts_db_path,
            )
            return []

        with sqlite3.connect(str(self._ts_db_path)) as conn:
            # 检查 data_calendar 是否存在
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='data_calendar';"
            )
            if cur.fetchone() is None:
                self._log.warning(
                    "ts_data.db 中不存在 data_calendar 表，请先运行 data_guardian 生成。"
                )
                return []

            sql = """
                SELECT trade_date, score, status
                FROM data_calendar
                WHERE score < ? AND score > 0
                ORDER BY trade_date DESC;
            """
            rows = conn.execute(sql, (float(min_score),)).fetchall()

        if not rows:
            self._log.info(
                "data_calendar 中没有 score < %.1f 且 score > 0 的记录，无需修复。",
                min_score,
            )
            return []

        # 按日期倒序 → 只取最新的 max_days 个 → 再升序返回
        dates_desc = [str(r[0]) for r in rows if r and r[0]]
        if max_days is not None and max_days > 0 and len(dates_desc) > max_days:
            dates_desc = dates_desc[:max_days]

        dates = sorted({_normalize_date(d) for d in dates_desc})
        self._log.info(
            "从 data_calendar 中筛选到 %d 个需修复日期（score < %.1f）：%s ... %s",
            len(dates),
            min_score,
            dates[0],
            dates[-1],
        )
        return dates

    # ----------------- 核心修复逻辑 ----------------- #

    def rebuild_minute_bars_for_date(self, trade_date: str) -> None:
        """
        对单个交易日执行“分钟线重建”。

        步骤：
        1. 删除 market_ts.db.minute_bars 中该日记录；
        2. 调用 TSDataPipeline，从 snapshots 重算分钟线并写回。
        """
        d = _normalize_date(trade_date)
        self._log.info("开始重建分钟线：trade_date=%s, freq=%s", d, self._freq)

        # 1. 删除旧记录
        self._delete_minute_bars_for_date(d)

        # 2. 重算分钟线并写入
        self._pipeline.run_for_dates(
            dates=[d],
            codes=None,
        )

        self._log.info("分钟线重建完成：trade_date=%s", d)

    def repair_dates(
        self,
        dates: Iterable[str],
    ) -> None:
        """
        对给定多个日期执行分钟线重建。
        """
        for d in dates:
            self.rebuild_minute_bars_for_date(d)

    def repair_range(self, start_date: str, end_date: str) -> None:
        """
        对 [start_date, end_date] 闭区间的日期逐日执行分钟线重建。
        """
        d0 = datetime.strptime(_normalize_date(start_date), "%Y-%m-%d").date()
        d1 = datetime.strptime(_normalize_date(end_date), "%Y-%m-%d").date()
        if d1 < d0:
            d0, d1 = d1, d0

        cur = d0
        while cur <= d1:
            td = cur.strftime("%Y-%m-%d")
            self.rebuild_minute_bars_for_date(td)
            cur += timedelta(days=1)

    def auto_repair_from_calendar(
        self,
        min_score: float = 80.0,
        max_days: Optional[int] = None,
    ) -> None:
        """
        根据 data_calendar 自动选择“低分日”，执行分钟线重建。
        """
        dates = self._get_low_score_dates_from_calendar(
            min_score=min_score,
            max_days=max_days,
        )
        if not dates:
            return
        self.repair_dates(dates)


# ----------------- 命令行入口 ----------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSDataRepairer - 分钟线重建 / 数据修复工具"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-calendar",
        action="store_true",
        help="根据 data_calendar 自动选择低分日期进行修复。",
    )
    group.add_argument(
        "--dates",
        nargs="+",
        metavar="DATE",
        help="手动指定一个或多个日期进行修复（YYYY-MM-DD 或 YYYYMMDD）。",
    )
    group.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help="指定起止日期（含），对区间内全部日期进行修复。",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=80.0,
        help="自动修复模式下的评分阈值，score < min_score 会被选为修复对象（默认 80.0）。",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="自动修复模式下：最多修复最近 N 天（按 trade_date 倒序），默认不限制。",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1min",
        help="分钟线频率，默认 1min。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repairer = TSDataRepairer(freq=args.freq)

    if args.from_calendar:
        repairer.auto_repair_from_calendar(
            min_score=args.min_score,
            max_days=args.max_days,
        )
    elif args.dates:
        repairer.repair_dates(args.dates)
    else:
        start, end = args.range
        repairer.repair_range(start, end)


if __name__ == "__main__":
    main()
