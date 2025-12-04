# -*- coding: utf-8 -*-
"""
ts_collector.py

TSCollector：分时行情采集器

职责：
- 定期从行情适配层（MarketAdapter）获取一篮子股票的实时简要行情；
- 将采集到的 DataFrame 通过 TSRecorder 写入 TSStorage（SQLite + Parquet）；
- 提供一次性采集 / 循环采集两种模式。

说明：
- 股票池来源：优先从 system_config.paths.universe_file 指定的文本文件加载；
- 若未配置 universe_file 或文件不存在，则使用初始化传入的 codes 参数；
- 默认使用 source_id="tx_quote"（腾讯行情），与 data_sources 配置对应。
"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config
from adapters.market_adapter import MarketAdapter
from data_core import load_stock_universe
from ts_recorder import TSRecorder

logger = get_logger(__name__)


class TSCollector:
    """
    分时行情采集器。

    用法示例（脚本中）：

        from ts_collector import TSCollector

        collector = TSCollector()
        collector.run_loop(iterations=10)

    """

    def __init__(
        self,
        codes: Optional[List[str]] = None,
        source_id: str = "tx_quote",
        enable_record: bool = True,
    ) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._system_cfg = get_system_config()

        commander_cfg = self._system_cfg.get("commander", {}) or {}
        self._scan_interval_sec: int = int(
            commander_cfg.get("scan_interval_sec", 5)
        )

        self._source_id = source_id
        self._adapter = MarketAdapter()
        self._recorder: Optional[TSRecorder] = TSRecorder() if enable_record else None

        # 加载股票池
        if codes is not None:
            self._codes = [c for c in codes if c]
        else:
            paths_cfg = self._system_cfg.get("paths", {}) or {}
            universe_file = paths_cfg.get("universe_file")
            self._codes = load_stock_universe(universe_file)

        if not self._codes:
            self._log.warning(
                "TSCollector 初始化时股票池为空。你可以："
                "1) 在 system_config.paths.universe_file 指定股票池文件；"
                "2) 初始化 TSCollector(codes=[...]) 传入代码列表。"
            )

    # ----------------- 核心采集逻辑 ----------------- #

    def collect_once(self) -> pd.DataFrame:
        """
        采集当前时刻的一次全股票池快照，返回 DataFrame。
        并在 enable_record=True 时自动写入存储。
        """
        if not self._codes:
            self._log.warning("collect_once: 股票池为空，返回空 DataFrame")
            return pd.DataFrame()

        df = self._adapter.get_realtime_quotes(
            self._codes,
            source_id=self._source_id,
        )

        self._log.info(
            "采集完成：source=%s, 股票数=%d, 有效行数=%d",
            self._source_id,
            len(self._codes),
            len(df),
        )

        if self._recorder is not None and df is not None and not df.empty:
            try:
                self._recorder.record_snapshot_df(df)
            except Exception:
                self._log.exception("写入 TSStorage 时发生异常")

        return df

    def run_loop(self, iterations: Optional[int] = None) -> None:
        """
        循环采集模式。

        Args:
            iterations: 采集轮数，如果为 None 则无限循环（Ctrl+C 结束）。
        """
        i = 0
        while True:
            i += 1
            try:
                df = self.collect_once()
                self._log.info(
                    "第 %d 轮采集完成，记录数=%d", i, len(df) if df is not None else 0
                )
            except KeyboardInterrupt:
                self._log.info("收到中断信号，停止采集循环")
                break
            except Exception:
                self._log.exception("采集过程中出现异常")

            if iterations is not None and i >= iterations:
                self._log.info("达到预设采集轮数 iterations=%d，停止", iterations)
                break

            time.sleep(self._scan_interval_sec)


# ----------------- 命令行入口（可选） ----------------- #

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="LightHunter TSCollector - 分时行情采集器"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只采集一轮并退出",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="采集轮数，默认为无限循环。如果 --once 为真则该参数无效。",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="tx_quote",
        help="数据源 ID（默认 tx_quote，对应腾讯行情）",
    )
    parser.add_argument(
        "--codes",
        type=str,
        nargs="*",
        default=None,
        help="覆盖默认股票池的代码列表，例如：--codes 000001 600000 300750",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="仅采集不写入存储（调试用）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collector = TSCollector(
        codes=args.codes,
        source_id=args.source,
        enable_record=not args.no_record,
    )

    if args.once:
        df = collector.collect_once()
        print(df.head())
        return

    collector.run_loop(iterations=args.iterations)


if __name__ == "__main__":
    main()
