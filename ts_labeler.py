# -*- coding: utf-8 -*-
"""
ts_labeler.py

TSLabeler Mk2 - 基于 DuckDB/TSStorage 的标签生成器

职责：
    - 从 TSStorage 读取 T 日 & T+1 日的日线数据；
    - 按 label_spec / LabelCenter 定义，计算 T 日对应的各类标签；
    - 将结果写入 TSStorage 的 label_panel（DuckDB + Parquet 双写）。

标签体系（详见 config/label_spec.json）：
    - 回归标签：
        * next_day_gap_open
        * next_day_oc_ret
        * next_day_high_open_ret
        * next_day_low_open_ret
    - 分类标签：
        * cls_next_day_oc_3cls
        * cls_next_day_gap_up

注意：
    - T 日的标签用到了 T+1 日的数据；
    - 若 T+1 在数据库中不存在（例如数据尚未更新或节假日），则对应 T 日的标签为 NaN。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger
from labels.label_center import get_label_center, LabelCenter, LabelDef
from storage.ts_storage import TSStorage, get_ts_storage

log = get_logger(__name__)

DateLike = Union[date, datetime, str]


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------


@dataclass
class TSLabelerConfig:
    """
    TSLabeler 行为配置。

    字段含义：
        use_ts_storage : 是否通过 TSStorage 访问数据（当前版本始终为 True）
        default_labels : 若调用方未指定 label_names，则使用该列表；
                         若为 None，则默认使用 label_spec.json 中的全部标签。
        symbol_column  : 标的代码列名（默认 "symbol"）
    """

    use_ts_storage: bool = True
    default_labels: Optional[List[str]] = None
    symbol_column: str = "symbol"

    @classmethod
    def from_system_config(cls, sys_cfg: Optional[Dict[str, Any]] = None) -> "TSLabelerConfig":
        sys_cfg = sys_cfg or get_system_config()
        raw = sys_cfg.get("ts_labeler") or {}
        default_labels = raw.get("default_labels")
        if isinstance(default_labels, str):
            default_labels = [default_labels]
        return cls(
            use_ts_storage=bool(raw.get("use_ts_storage", True)),
            default_labels=default_labels,
            symbol_column=str(raw.get("symbol_column", "symbol")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# 主体类
# ----------------------------------------------------------------------


class TSLabeler:
    """
    TSLabeler Mk2 - 基于日线数据的标签生成器。

    推荐使用方式：
        from datetime import date
        from ts_labeler import get_ts_labeler

        labeler = get_ts_labeler()
        df_label = labeler.run_for_date(date(2024, 6, 1))
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        storage: Optional[TSStorage] = None,
        label_center: Optional[LabelCenter] = None,
        config: Optional[TSLabelerConfig] = None,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg = config or TSLabelerConfig.from_system_config(self.sys_cfg)
        self.storage: TSStorage = storage or get_ts_storage(self.sys_cfg)
        self.center: LabelCenter = label_center or get_label_center(self.sys_cfg)
        self.log = get_logger(self.__class__.__name__)

        self.log.info(
            "TSLabeler 初始化: cfg=%s, labels_available=%d",
            self.cfg.to_dict(),
            len(self.center.all_label_names()),
        )

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run_for_date(
        self,
        trading_date: DateLike,
        symbols: Optional[Sequence[str]] = None,
        *,
        label_names: Optional[Sequence[str]] = None,
        write_to_storage: bool = True,
    ) -> pd.DataFrame:
        """
        为某个交易日 T 生成标签（基于 T 和 T+1 的日线数据）。

        Args:
            trading_date   : 交易日 T
            symbols        : 限定的股票列表（可选），为 None 时不做 symbol 过滤
            label_names    : 需要生成的标签名列表；若为 None，则使用配置/LabelCenter 默认
            write_to_storage: True 则写入 TSStorage 的 label_panel，False 仅返回 df

        Returns:
            label_panel DataFrame：
                - 列：["trading_date", "symbol", <各标签列>]
        """
        d0 = self._to_date(trading_date)
        d1 = d0 + timedelta(days=1)

        active_labels = self._resolve_label_names(label_names)

        if not active_labels:
            self.log.warning("TSLabeler.run_for_date: 未选择任何标签，直接返回空 df。")
            return pd.DataFrame(columns=["trading_date", self.cfg.symbol_column])

        self.log.info(
            "TSLabeler.run_for_date: date=%s, next_date=%s, labels=%s",
            d0,
            d1,
            active_labels,
        )

        # 1. 读取 T / T+1 日的日线数据（来自 TSStorage -> DuckDB）
        df_t = self.storage.load_from_duckdb("daily", d0, symbols=list(symbols) if symbols is not None else None)
        df_t1 = self.storage.load_from_duckdb("daily", d1, symbols=list(symbols) if symbols is not None else None)

        if df_t is None or df_t.empty:
            self.log.warning("TSLabeler: date=%s 没有日线数据，返回空标签。", d0)
            return pd.DataFrame(columns=["trading_date", self.cfg.symbol_column] + list(active_labels))

        if df_t1 is None or df_t1.empty:
            self.log.warning("TSLabeler: date=%s 的 T+1 日(%s) 没有日线数据，返回空标签。", d0, d1)
            return pd.DataFrame(columns=["trading_date", self.cfg.symbol_column] + list(active_labels))

        sym_col = self.cfg.symbol_column
        need_cols_t = [sym_col, "close"]
        need_cols_t1 = [sym_col, "open", "high", "low", "close"]

        for c in need_cols_t:
            if c not in df_t.columns:
                raise KeyError(f"TSLabeler: T 日日线数据缺少列 {c}")
        for c in need_cols_t1:
            if c not in df_t1.columns:
                raise KeyError(f"TSLabeler: T+1 日日线数据缺少列 {c}")

        df_t = df_t[need_cols_t].rename(columns={"close": "close_T"})
        df_t1 = df_t1[need_cols_t1].rename(
            columns={"open": "open_T1", "high": "high_T1", "low": "low_T1", "close": "close_T1"}
        )

        merged = pd.merge(df_t, df_t1, on=sym_col, how="inner")
        if merged.empty:
            self.log.warning(
                "TSLabeler: date=%s 与 %s 在 symbol 上没有交集，返回空标签。",
                d0,
                d1,
            )
            return pd.DataFrame(columns=["trading_date", sym_col] + list(active_labels))

        # 2. 计算基础回归标签（next_day_*）
        base_labels = self._compute_base_labels(merged)

        # 3. 利用 label_spec 中的定义，按需生成回归/分类标签
        result = pd.DataFrame({
            "trading_date": [d0 for _ in range(len(merged))],
            sym_col: merged[sym_col].values,
        })

        for name in active_labels:
            spec = self.center.get_label(name)

            if spec.task_type == "regression":
                series = base_labels.get(name)
                if series is None:
                    self.log.warning(
                        "TSLabeler: 未实现的回归标签 %s（label_spec 中存在，但 ts_labeler 未提供计算），用 NaN 填充。",
                        name,
                    )
                    result[name] = np.nan
                else:
                    result[name] = series.values

            elif spec.task_type == "classification":
                base_name = spec.base_label
                if not base_name:
                    self.log.warning("TSLabeler: 分类标签 %s 缺少 base_label 定义，全部 NaN。", name)
                    result[name] = np.nan
                    continue

                base_series = base_labels.get(base_name)
                if base_series is None:
                    self.log.warning(
                        "TSLabeler: 分类标签 %s 依赖的基础标签 %s 未实现，全部 NaN。",
                        name,
                        base_name,
                    )
                    result[name] = np.nan
                    continue

                thresholds = spec.thresholds or []
                cls_codes = self._apply_thresholds(base_series, thresholds)
                result[name] = cls_codes

            else:
                self.log.warning("TSLabeler: 未知 task_type=%s 对于标签 %s，全部 NaN。", spec.task_type, name)
                result[name] = np.nan

        # 列顺序整理
        ordered_cols = ["trading_date", sym_col] + list(active_labels)
        result = result[ordered_cols]

        # 4. 写入 TSStorage 的 label_panel
        if write_to_storage and not result.empty:
            try:
                self.storage.write_label_panel(d0, result)
                self.log.info(
                    "TSLabeler: date=%s 标签写入 TSStorage 完成 rows=%d labels=%s",
                    d0,
                    len(result),
                    active_labels,
                )
            except Exception as e:
                self.log.exception("TSLabeler: 写入 label_panel 失败 err=%s", e)

        return result

    # ------------------------------------------------------------------
    # 基础标签计算（回归）
    # ------------------------------------------------------------------
    def _compute_base_labels(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        给定合并后的 DataFrame（包含 close_T/ open_T1 / high_T1 / low_T1 / close_T1），
        计算所有内置的基础回归标签：

            - next_day_gap_open
            - next_day_oc_ret
            - next_day_high_open_ret
            - next_day_low_open_ret
        """
        close_T = df["close_T"].astype(float)
        open_T1 = df["open_T1"].astype(float)
        high_T1 = df["high_T1"].astype(float)
        low_T1 = df["low_T1"].astype(float)
        close_T1 = df["close_T1"].astype(float)

        def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
            num_arr = num.to_numpy(dtype="float64")
            den_arr = den.to_numpy(dtype="float64")
            out = np.full_like(num_arr, np.nan, dtype="float64")
            mask = (den_arr != 0) & ~np.isnan(num_arr) & ~np.isnan(den_arr)
            out[mask] = num_arr[mask] / den_arr[mask]
            return pd.Series(out, index=num.index)

        # T+1 开盘相对 T 收盘的跳空：open_{T+1} / close_T - 1
        next_day_gap_open = safe_div(open_T1, close_T) - 1.0

        # T+1 开盘 -> 收盘收益：close_{T+1} / open_{T1} - 1
        next_day_oc_ret = safe_div(close_T1, open_T1) - 1.0

        # T+1 最高价相对开盘浮盈：high_{T+1} / open_{T1} - 1
        next_day_high_open_ret = safe_div(high_T1, open_T1) - 1.0

        # T+1 最低价相对开盘回撤：low_{T+1} / open_{T1} - 1
        next_day_low_open_ret = safe_div(low_T1, open_T1) - 1.0

        base = {
            "next_day_gap_open": next_day_gap_open,
            "next_day_oc_ret": next_day_oc_ret,
            "next_day_high_open_ret": next_day_high_open_ret,
            "next_day_low_open_ret": next_day_low_open_ret,
        }
        return base

    # ------------------------------------------------------------------
    # 阈值分档 -> 分类编码
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_thresholds(base_series: pd.Series, thresholds: List[float]) -> pd.Series:
        """
        基于 thresholds 将连续数值分档，返回类别编码（Int64）：

            thresholds = [t1, t2, ..., tk]

            则档位：
                0: x <= t1
                1: t1 < x <= t2
                ...
                k: x > tk

        NaN 会被映射为 <NA>。
        """
        if not thresholds:
            # 没有阈值则全部 NaN
            return pd.Series([np.nan] * len(base_series), index=base_series.index, dtype="float64")

        bins = np.array(sorted(float(t) for t in thresholds), dtype="float64")
        vals = base_series.to_numpy(dtype="float64")

        # 初始化为 -1，后面再将 NaN 替换为 pd.NA
        codes = np.full(shape=vals.shape, fill_value=-1, dtype="int64")

        mask_valid = ~np.isnan(vals)
        if mask_valid.any():
            codes[mask_valid] = np.searchsorted(bins, vals[mask_valid], side="right")

        s = pd.Series(codes, index=base_series.index, dtype="Int64")
        # 将原始 NaN 对应的位置设为 <NA>
        s[~mask_valid] = pd.NA
        return s

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _to_date(d: DateLike) -> date:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        return datetime.strptime(str(d), "%Y-%m-%d").date()

    def _resolve_label_names(self, label_names: Optional[Sequence[str]]) -> List[str]:
        if label_names is not None:
            return list(label_names)
        if self.cfg.default_labels:
            return list(self.cfg.default_labels)
        return self.center.all_label_names()


# ----------------------------------------------------------------------
# 模块级单例 & 便捷函数
# ----------------------------------------------------------------------

_default_labeler: Optional[TSLabeler] = None


def get_ts_labeler(system_config: Optional[Dict[str, Any]] = None) -> TSLabeler:
    global _default_labeler
    if _default_labeler is None:
        _default_labeler = TSLabeler(system_config=system_config)
    return _default_labeler


def run_labeler_for_date(
    trading_date: DateLike,
    symbols: Optional[Sequence[str]] = None,
    *,
    label_names: Optional[Sequence[str]] = None,
    write_to_storage: bool = True,
) -> pd.DataFrame:
    """
    模块级便捷入口：一行代码生成某天标签。
    """
    labeler = get_ts_labeler()
    return labeler.run_for_date(
        trading_date=trading_date,
        symbols=symbols,
        label_names=label_names,
        write_to_storage=write_to_storage,
    )


# ----------------------------------------------------------------------
# 简单 CLI（可选）
# ----------------------------------------------------------------------


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="LightHunter TSLabeler Mk2")
    parser.add_argument("--date", type=str, required=True, help="交易日期 YYYY-MM-DD（T 日）")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="仅计算不写入 TSStorage（默认会写入）"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    d = args.date
    write = not args.no_write

    labeler = get_ts_labeler()
    df = labeler.run_for_date(d, symbols=None, write_to_storage=write)
    print(df.head())


if __name__ == "__main__":  # pragma: no cover
    main()
