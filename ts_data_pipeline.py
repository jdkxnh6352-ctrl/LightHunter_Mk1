# -*- coding: utf-8 -*-
"""
ts_data_pipeline.py

TS 数据管线骨架（Mk2 配置收敛版）

职责：
    - 从 ConfigCenter 读取 TS 管线相关配置（ts_pipeline.json + system_config.paths/storage）；
    - 提供一个 TSDataPipeline 类，对外暴露 run_for_date / run_range 等接口；
    - 内部再去调用你原来的各类 builder/aggregator/factor_engine/labeler。

这里不强行重写你已有的细节实现，只是把“参数来源”统一抽到配置层。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional

from config.config_center import get_system_config, get_config_center
from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TSPipelineConfig:
    """TS 管线的一些核心配置。"""

    lookback_days: int = 60
    min_minutes_per_day: int = 200
    universe_key: str = "a_share_symbols_file"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookback_days": self.lookback_days,
            "min_minutes_per_day": self.min_minutes_per_day,
            "universe_key": self.universe_key,
        }


class TSDataPipeline:
    """
    TS 数据管线统一入口。

    典型使用：
        pipeline = TSDataPipeline()
        pipeline.run_for_date(date(2024, 6, 1))
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[TSPipelineConfig] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg_center = get_config_center()

        ts_cfg_raw = self.cfg_center.ts_pipeline or {}
        lookback_days = int(ts_cfg_raw.get("lookback_days", 60))
        min_minutes_per_day = int(ts_cfg_raw.get("min_minutes_per_day", 200))
        universe_key = str(ts_cfg_raw.get("universe_key", "a_share_symbols_file"))

        if pipeline_config is None:
            pipeline_config = TSPipelineConfig(
                lookback_days=lookback_days,
                min_minutes_per_day=min_minutes_per_day,
                universe_key=universe_key,
            )

        self.cfg = pipeline_config
        self.log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def run_for_date(self, trading_date: date) -> None:
        """
        生成指定交易日的特征 & 标签数据集。
        内部可以调用：
            - ts_aggregator / ts_minute_engine
            - factor_engine
            - ts_labeler
        这里只做骨架 & 配置统一。
        """
        self.log.info(
            "TSDataPipeline.run_for_date: date=%s cfg=%s",
            trading_date.isoformat(),
            self.cfg.to_dict(),
        )

        # 这里你可以接入自己的实现：
        try:
            # from factor_engine import FactorEngine
            # from ts_labeler import TSLabeler
            # fe = FactorEngine(system_config=self.sys_cfg)
            # labeler = TSLabeler(system_config=self.sys_cfg)
            # fe.build_factors_for_date(trading_date)
            # labeler.build_labels_for_date(trading_date)
            pass
        except Exception as e:  # pragma: no cover
            self.log.exception("TSDataPipeline.run_for_date: 执行管线失败 err=%s", e)
            raise

    def run_range(self, start_date: date, end_date: date) -> None:
        """
        批量跑一段日期区间。
        """
        cur = start_date
        while cur <= end_date:
            self.run_for_date(cur)
            cur = cur.fromordinal(cur.toordinal() + 1)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_date(s: str) -> date:
        return datetime.strptime(s, "%Y-%m-%d").date()
