# -*- coding: utf-8 -*-
"""
ts_health_lab.py

TSHealthLab - 时序数据健康度分析实验室

职责：
    - 检查单一数据源的 TS 数据内部质量：
        * 每个 symbol 的 bars 数量是否足够；
        * 0 成交量/0 价格比例；
        * 时间间隔是否存在异常大 gap；
    - 输出：
        * 每个 symbol 的健康指标；
        * 整体 Dataset 层面的聚合指标。

输入：
    DataFrame，至少包含：
        symbol, ts, close, volume

注意：
    - ts 字段会使用 pd.to_datetime() 处理；
    - 若实际列名不同，可以在上游重命名后再传入。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class SymbolHealthMetrics:
    symbol: str
    n_bars: int
    zero_volume_ratio: float
    zero_price_ratio: float
    large_gap_ratio: float
    time_monotonic_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetHealthMetrics:
    num_symbols: int
    mean_bars_per_symbol: float
    median_bars_per_symbol: float
    zero_volume_ratio_mean: float
    zero_price_ratio_mean: float
    large_gap_ratio_mean: float
    time_monotonic_ok_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TSHealthLab:
    """
    TSHealthLab - 时序健康检查。

    expected_bars_per_symbol:
        单日单标的的期望 bars 数量（暂时只用于 log 提示，真正评分在 DataGuardian 那边）。
    expected_step_seconds:
        理论时间间隔（秒），用来判断“时间 gap 是否异常大”；
        对分钟线可以设为 60，5 分钟线设为 300 等。
    gap_factor:
        如果时间间隔 > expected_step_seconds * gap_factor，则认为出现了“大 gap”。
    """

    def __init__(
        self,
        expected_bars_per_symbol: int = 240,
        expected_step_seconds: int = 60,
        gap_factor: float = 3.0,
    ) -> None:
        self.expected_bars_per_symbol = expected_bars_per_symbol
        self.expected_step_seconds = expected_step_seconds
        self.gap_factor = gap_factor
        self.log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        *,
        symbol_col: str = "symbol",
        time_col: str = "ts",
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> Tuple[DatasetHealthMetrics, pd.DataFrame]:
        """
        对整份 DataFrame 进行健康度分析。

        Returns:
            (DatasetHealthMetrics, per_symbol_metrics_df)
        """
        if df is None or df.empty:
            ds = DatasetHealthMetrics(
                num_symbols=0,
                mean_bars_per_symbol=0.0,
                median_bars_per_symbol=0.0,
                zero_volume_ratio_mean=0.0,
                zero_price_ratio_mean=0.0,
                large_gap_ratio_mean=0.0,
                time_monotonic_ok_ratio=1.0,
            )
            per_symbol_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "n_bars",
                    "zero_volume_ratio",
                    "zero_price_ratio",
                    "large_gap_ratio",
                    "time_monotonic_ok",
                ]
            )
            return ds, per_symbol_df

        if symbol_col not in df.columns:
            raise KeyError(f"DataFrame 缺少 symbol 列: {symbol_col}")
        if time_col not in df.columns:
            raise KeyError(f"DataFrame 缺少时间列: {time_col}")

        per_symbol_metrics: List[SymbolHealthMetrics] = []

        # 按 symbol 分组进行检查
        for symbol, g in df.groupby(symbol_col):
            m = self._compute_symbol_metrics(
                symbol,
                g,
                time_col=time_col,
                price_col=price_col,
                volume_col=volume_col,
            )
            per_symbol_metrics.append(m)

        per_symbol_df = pd.DataFrame([m.to_dict() for m in per_symbol_metrics])
        if per_symbol_df.empty:
            ds = DatasetHealthMetrics(
                num_symbols=0,
                mean_bars_per_symbol=0.0,
                median_bars_per_symbol=0.0,
                zero_volume_ratio_mean=0.0,
                zero_price_ratio_mean=0.0,
                large_gap_ratio_mean=0.0,
                time_monotonic_ok_ratio=1.0,
            )
            return ds, per_symbol_df

        ds = DatasetHealthMetrics(
            num_symbols=int(len(per_symbol_df)),
            mean_bars_per_symbol=float(per_symbol_df["n_bars"].mean()),
            median_bars_per_symbol=float(per_symbol_df["n_bars"].median()),
            zero_volume_ratio_mean=float(per_symbol_df["zero_volume_ratio"].mean()),
            zero_price_ratio_mean=float(per_symbol_df["zero_price_ratio"].mean()),
            large_gap_ratio_mean=float(per_symbol_df["large_gap_ratio"].mean()),
            time_monotonic_ok_ratio=float(
                (per_symbol_df["time_monotonic_ok"].astype(bool)).mean()
            ),
        )
        return ds, per_symbol_df

    # ------------------------------------------------------------------
    def _compute_symbol_metrics(
        self,
        symbol: str,
        g: pd.DataFrame,
        *,
        time_col: str,
        price_col: str,
        volume_col: str,
    ) -> SymbolHealthMetrics:
        """单个 symbol 的健康度指标计算。"""
        n_bars = int(len(g))

        # 0 成交量 / 0 价格比例
        if volume_col in g.columns:
            zero_volume_ratio = float((g[volume_col].astype(float) <= 0).mean())
        else:
            zero_volume_ratio = 0.0

        if price_col in g.columns:
            zero_price_ratio = float((g[price_col].astype(float) <= 0).mean())
        else:
            zero_price_ratio = 0.0

        # 时间间隔 & 大 gap 检查
        large_gap_ratio = 0.0
        time_monotonic_ok = True

        try:
            ts_series = pd.to_datetime(g[time_col])
            ts_sorted = ts_series.sort_values()
            if len(ts_sorted) > 1:
                diffs = ts_sorted.diff().dt.total_seconds().dropna()
                if len(diffs) > 0:
                    expected = max(1.0, float(self.expected_step_seconds))
                    threshold = expected * float(self.gap_factor)
                    large_gap_ratio = float((diffs > threshold).mean())
        except Exception as e:  # pragma: no cover
            self.log.warning(
                "TSHealthLab: symbol=%s 时间列解析失败，跳过 gap 检查 err=%s",
                symbol,
                e,
            )

        return SymbolHealthMetrics(
            symbol=symbol,
            n_bars=n_bars,
            zero_volume_ratio=zero_volume_ratio,
            zero_price_ratio=zero_price_ratio,
            large_gap_ratio=large_gap_ratio,
            time_monotonic_ok=time_monotonic_ok,
        )
