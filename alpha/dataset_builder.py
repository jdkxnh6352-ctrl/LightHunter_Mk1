# -*- coding: utf-8 -*-
"""
alpha/dataset_builder.py

简化版 DatasetBuilder，用于支持 U2 的 Walk‑Forward 回测。

设计思路：
- 假设已经有一个“宽表”形式的日频数据集，存放在 data/datasets 目录：
    data/datasets/<job_id>.parquet  或  data/datasets/<job_id>.csv

- 表结构要求（字段名可以稍有不同，Builder 会尽量自动识别）：
    - 日期列：  "trading_date" / "date" / "trade_date"  之一
    - 代码列：  "symbol" / "code" / "ts_code"  之一
    - 标签列：  "label" / "y" / "target"  之一
    - 其它数值列：统一视为特征列（feature columns）

- TrainingPipelines 会调用：
    builder = DatasetBuilder(job_id=..., system_config=...)
    train_df, valid_df, test_df = builder.build_for_walkforward(
        strategy_id=..., train_start=..., train_end=..., test_start=..., test_end=...
    )
  然后再通过 builder.feature_columns / label_column / date_column / symbol_column
  来拿到列名。

这个实现本质上只是负责：
    1）加载完整数据集；
    2）按照日期做切片；
    3）拆分 train / valid / test；
    4）记录好列名供后续使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config


log = get_logger(__name__)


# ----------------------------------------------------------------------
# 配置对象：只负责确定数据集路径 & 列名约定
# ----------------------------------------------------------------------


@dataclass
class DatasetBuilderConfig:
    dataset_root: Path
    default_dataset_name: str = "ultrashort_main"
    # 列名约定（只是默认值，后面会做自动识别）
    date_column: str = "trading_date"
    symbol_column: str = "symbol"
    label_column: str = "label"

    @classmethod
    def from_system_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "DatasetBuilderConfig":
        if cfg is None:
            cfg = get_system_config()
        paths = cfg.get("paths", {}) or {}
        dataset_dir = paths.get("dataset_dir") or "data/datasets"
        root = Path(dataset_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(dataset_root=root)


# ----------------------------------------------------------------------
# 主类：DatasetBuilder
# ----------------------------------------------------------------------


class DatasetBuilder:
    """
    简化版 DatasetBuilder。

    目前只实现了 walk‑forward 回测所需的接口：

        - __init__(system_config, job_id=..., experiment_group=..., fold_spec=...)
        - build_for_walkforward(...)
        - 属性：feature_columns / label_column / date_column / symbol_column

    若以后需要支持更多花样（例如 GNN、分钟线序列等），在这个类基础上继续扩展即可。
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        experiment_group: Optional[str] = None,
        fold_spec: Optional[Any] = None,
        **_: Any,
    ) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.cfg = DatasetBuilderConfig.from_system_config(self.sys_cfg)

        # job_id 会决定默认的数据集文件名
        self.job_id = job_id or self.cfg.default_dataset_name
        self.experiment_group = experiment_group
        self.fold_spec = fold_spec

        # 这些属性会在 build_for_walkforward() 里被填充
        self.feature_columns: List[str] = []
        self.label_column: str = self.cfg.label_column
        self.date_column: str = self.cfg.date_column
        self.symbol_column: str = self.cfg.symbol_column

        self._dataset_cache: Optional[pd.DataFrame] = None

        log.info(
            "DatasetBuilder 初始化完成: job_id=%s, dataset_root=%s",
            self.job_id,
            self.cfg.dataset_root,
        )

    # ------------------------------------------------------------------
    # 对外主接口：为一次 Walk‑Forward 折构建 train/valid/test
    # ------------------------------------------------------------------

    def build_for_walkforward(
        self,
        strategy_id: str,
        *,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        **_: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        为某个策略的一折 Walk‑Forward 构建数据集切片。

        参数说明（TrainingPipelines 会用关键字参数调用）：
            strategy_id : 策略 ID（目前只用于日志）
            train_start : 训练集开始日期（含），格式 YYYY-MM-DD
            train_end   : 训练集结束日期（含）
            test_start  : 测试集开始日期（含）
            test_end    : 测试集结束日期（含）

        返回：
            (train_df, valid_df, test_df)
        """
        log.info(
            "DatasetBuilder.build_for_walkforward: job_id=%s, strategy_id=%s, "
            "train=[%s, %s], test=[%s, %s]",
            self.job_id,
            strategy_id,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        df_all = self._load_full_dataset()
        if df_all is None or df_all.empty:
            log.warning(
                "DatasetBuilder: 数据集为空，返回空的 train/valid/test。"
                " 请先在 data/datasets 下准备 %s.(parquet|csv)。",
                self.job_id,
            )
            empty = pd.DataFrame()
            return empty.copy(), empty.copy(), empty.copy()

        # 统一日期列为 pandas.Timestamp（只取日期部分）
        date_col = self.date_column
        df = df_all.copy()
        try:
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
        except Exception as e:
            log.error("DatasetBuilder: 将 %s 转为日期失败: %s", date_col, e)
            empty = pd.DataFrame()
            return empty.copy(), empty.copy(), empty.copy()

        # 生成布尔掩码
        t_start = pd.to_datetime(train_start).date()
        t_end = pd.to_datetime(train_end).date()
        s_start = pd.to_datetime(test_start).date()
        s_end = pd.to_datetime(test_end).date()

        train_mask_all = (df[date_col] >= t_start) & (df[date_col] <= t_end)
        test_mask = (df[date_col] >= s_start) & (df[date_col] <= s_end)

        df_train_all = df.loc[train_mask_all].reset_index(drop=True)
        df_test = df.loc[test_mask].reset_index(drop=True)

        if df_train_all.empty or df_test.empty:
            log.warning(
                "DatasetBuilder: 本折训练或测试区间内没有任何样本。"
                " train_rows=%d, test_rows=%d",
                len(df_train_all),
                len(df_test),
            )
            empty = pd.DataFrame()
            return empty.copy(), empty.copy(), empty.copy()

        # 在训练区间内，再按日期切一块验证集：用最后 20% 的日期做 valid
        unique_dates = sorted(df_train_all[date_col].unique())
        if len(unique_dates) <= 5:
            # 日期太少，就不再拆 valid 了，全部算 train
            df_train = df_train_all
            df_valid = pd.DataFrame(columns=df.columns)
        else:
            n_dates = len(unique_dates)
            n_valid = max(1, int(n_dates * 0.2))
            valid_dates = set(unique_dates[-n_valid:])

            valid_mask = df_train_all[date_col].isin(valid_dates)
            train_mask = ~valid_mask

            df_train = df_train_all.loc[train_mask].reset_index(drop=True)
            df_valid = df_train_all.loc[valid_mask].reset_index(drop=True)

        # 推断列名（特征列 & 标签列）
        self._infer_columns(df)

        log.info(
            "DatasetBuilder: 构建完成 train_rows=%d, valid_rows=%d, test_rows=%d, "
            "features=%d, label=%s",
            len(df_train),
            len(df_valid),
            len(df_test),
            len(self.feature_columns),
            self.label_column,
        )

        return df_train, df_valid, df_test

    # ------------------------------------------------------------------
    # 内部工具：加载完整数据集（带简单缓存）
    # ------------------------------------------------------------------

    def _resolve_dataset_paths(self) -> List[Path]:
        """
        按优先级返回可能存在的数据集路径列表：
            <root>/<job_id>.parquet
            <root>/<job_id>.feather
            <root>/<job_id>.csv
        """
        base = self.cfg.dataset_root
        name = self.job_id or self.cfg.default_dataset_name
        candidates = [
            base / f"{name}.parquet",
            base / f"{name}.feather",
            base / f"{name}.csv",
        ]
        return candidates

    def _load_full_dataset(self) -> Optional[pd.DataFrame]:
        if self._dataset_cache is not None:
            return self._dataset_cache

        paths = self._resolve_dataset_paths()
        for p in paths:
            if not p.exists():
                continue
            try:
                if p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                elif p.suffix.lower() == ".feather":
                    df = pd.read_feather(p)
                else:
                    df = pd.read_csv(p)
            except Exception as e:
                log.error("DatasetBuilder: 读取数据集失败: %s err=%s", p, e)
                continue

            if df is None or df.empty:
                log.warning("DatasetBuilder: 文件存在但为空: %s", p)
                continue

            log.info(
                "DatasetBuilder: 已从 %s 载入数据集 rows=%d cols=%d",
                p,
                df.shape[0],
                df.shape[1],
            )
            self._dataset_cache = df
            # 尝试自动识别列名
            self._auto_detect_columns(df)
            return df

        log.error(
            "DatasetBuilder: 未找到可用的数据集文件，请在 %s 下创建 %s.(parquet|csv)。",
            self.cfg.dataset_root,
            self.job_id,
        )
        return None

    # ------------------------------------------------------------------
    # 内部工具：列名推断
    # ------------------------------------------------------------------

    def _auto_detect_columns(self, df: pd.DataFrame) -> None:
        """
        根据常见列名自动识别 日期 / 代码 / 标签列。
        """
        # 日期列
        date_candidates = [
            self.cfg.date_column,
            "trading_date",
            "date",
            "trade_date",
        ]
        for c in date_candidates:
            if c in df.columns:
                self.date_column = c
                break

        # 代码列
        symbol_candidates = [
            self.cfg.symbol_column,
            "symbol",
            "code",
            "ts_code",
            "stock_code",
        ]
        for c in symbol_candidates:
            if c in df.columns:
                self.symbol_column = c
                break

        # 标签列
        label_candidates = [
            self.cfg.label_column,
            "label",
            "y",
            "target",
            "cls_label",
            "ret_label",
        ]
        for c in label_candidates:
            if c in df.columns:
                self.label_column = c
                break

        log.info(
            "DatasetBuilder: 自动识别列名完成 date_col=%s, symbol_col=%s, label_col=%s",
            self.date_column,
            self.symbol_column,
            self.label_column,
        )

    def _infer_columns(self, df: pd.DataFrame) -> None:
        """
        在已知 date/symbol/label 列名的前提下，从数值列中推断特征列。
        """
        date_col = self.date_column
        sym_col = self.symbol_column
        label_col = self.label_column

        exclude = {date_col, sym_col, label_col}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c not in exclude]

        # 若没有数值列，退而求其次：所有非日期/代码/标签列都当作特征
        if not feat_cols:
            feat_cols = [
                c
                for c in df.columns
                if c not in exclude
            ]

        self.feature_columns = sorted(feat_cols)
