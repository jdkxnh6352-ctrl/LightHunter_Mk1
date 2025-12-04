# -*- coding: utf-8 -*-
"""
features/concept_engine.py

ConceptEngine：概念 / 情绪中枢（基础版）

设计目标：
- 将“股票-概念”的静态成分信息，和当日分钟线行情结合，
  计算出：
    1) 概念层面的强度 / 情绪；
    2) 个股维度的概念因子；
    3) 简单的全市场情绪指标（market_sentiment）。

假定输入：
- 概念成分文件：data/concepts/concept_membership.csv
    必备列：
        code         : 股票代码（6位，和 minute_bars 里的 code 对应）
        concept_id   : 概念 ID
        concept_name : 概念名称
    可选列：
        trade_date   : 若存在，则可按日期过滤成分（<= trade_date 的记录视为有效）

- 分钟线数据：通常来自 TSDataPipeline 写入前 / 读出的 DataFrame，至少包含：
    trade_date, code, ts, open, high, low, close, volume
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ConceptFeatureConfig:
    """
    概念 / 情绪因子配置。
    """
    membership_file: str = "data/concepts/concept_membership.csv"
    # 未来可以扩展：概念日度指标文件 / 舆情文件等


class ConceptEngine:
    """
    概念 / 情绪中枢。

    主要对外接口：
        - build_stock_concept_features(trade_date, minute_bars)

    输出字段（个股维度）包括：
        trade_date
        code
        cpt_count                 : 概念数量
        cpt_strength_max          : 所有归属概念中的最强概念强度
        cpt_strength_mean         : 概念强度均值
        cpt_sentiment_max         : 概念情绪（这里等价于强度）最大值
        cpt_sentiment_mean        : 概念情绪均值
        cpt_top_concept_id        : 最强概念 ID
        cpt_top_concept_name      : 最强概念名称
        market_sentiment          : 全市场情绪指标（同一天所有股票共用）
    """

    def __init__(self, config: Optional[ConceptFeatureConfig] = None) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._cfg = config or ConceptFeatureConfig()

        sys_cfg = get_system_config()
        paths_cfg = sys_cfg.get("paths", {}) or {}

        membership_file = paths_cfg.get(
            "concept_membership_file", self._cfg.membership_file
        )
        p = Path(membership_file)
        if not p.is_absolute():
            p = (PROJECT_ROOT / membership_file).resolve()
        self._membership_path = p

        self._log.info(
            "ConceptEngine init. membership_file=%s",
            self._membership_path,
        )

    # ----------------- 内部辅助：加载成分 ----------------- #

    def _load_membership(self, trade_date: str) -> pd.DataFrame:
        """
        从 CSV 加载概念成分。结构约定（至少）：
            code, concept_id, concept_name [, trade_date]
        """
        if not self._membership_path.exists():
            self._log.warning(
                "概念成分文件不存在：%s，概念因子将为空。",
                self._membership_path,
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(self._membership_path)
        except Exception:
            self._log.exception("读取概念成分文件失败：%s", self._membership_path)
            return pd.DataFrame()

        required = {"code", "concept_id", "concept_name"}
        if not required.issubset(set(df.columns)):
            self._log.warning(
                "概念成分文件缺少必要列：%s，当前列=%s；概念因子将为空。",
                ",".join(sorted(required)),
                ",".join(df.columns),
            )
            return pd.DataFrame()

        df["code"] = df["code"].astype(str).str.zfill(6)
        df["concept_id"] = df["concept_id"].astype(str)

        if "trade_date" in df.columns:
            # 若有 trade_date，则保留所有 <= 当前 trade_date 的成分
            df["trade_date"] = df["trade_date"].astype(str)
            df = df[df["trade_date"] <= trade_date]

        if df.empty:
            self._log.warning(
                "在 trade_date<=%s 的概念成分记录为空。", trade_date
            )

        return df[["code", "concept_id", "concept_name"]].drop_duplicates()

    # ----------------- 对外主接口 ----------------- #

    def build_stock_concept_features(
        self,
        trade_date: str,
        minute_bars: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        根据当日分钟线 + 概念成分，构建个股概念 / 情绪因子。

        Args:
            trade_date: 交易日期（YYYY-MM-DD）
            minute_bars: 当日分钟线 DataFrame（至少包含：code, ts, open, close, high, low, volume）

        Returns:
            DataFrame，按 (trade_date, code) 聚合后的因子表。
        """
        membership = self._load_membership(trade_date)
        if membership.empty:
            self._log.warning("概念成分为空，返回空的概念因子表。")
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "code",
                    "cpt_count",
                    "cpt_strength_max",
                    "cpt_strength_mean",
                    "cpt_sentiment_max",
                    "cpt_sentiment_mean",
                    "cpt_top_concept_id",
                    "cpt_top_concept_name",
                    "market_sentiment",
                ]
            )

        if minute_bars is None or minute_bars.empty:
            # 没有分钟线，无法算强度，直接返回“只有 count 的因子”
            codes = sorted(membership["code"].unique())
            tmp = (
                membership.groupby("code", as_index=False)["concept_id"]
                .nunique()
                .rename(columns={"concept_id": "cpt_count"})
            )
            tmp["trade_date"] = trade_date
            tmp["cpt_strength_max"] = np.nan
            tmp["cpt_strength_mean"] = np.nan
            tmp["cpt_sentiment_max"] = np.nan
            tmp["cpt_sentiment_mean"] = np.nan
            tmp["cpt_top_concept_id"] = None
            tmp["cpt_top_concept_name"] = None
            tmp["market_sentiment"] = np.nan
            return tmp

        # -------- 1) 计算个股日内涨跌 & 成交量 -------- #
        mb = minute_bars.copy()
        required_cols = {"code", "ts", "open", "close", "high", "low", "volume"}
        missing = required_cols - set(mb.columns)
        if missing:
            self._log.warning(
                "分钟线缺少计算概念强度所需列：%s，概念强度将为 NaN。",
                ",".join(sorted(missing)),
            )
            # 退化为只算概念数量
            return self.build_stock_concept_features(trade_date, minute_bars=None)

        mb["code"] = mb["code"].astype(str).str.zfill(6)
        try:
            mb["ts"] = pd.to_datetime(mb["ts"])
        except Exception:
            self._log.exception("无法将 ts 字段转换为 datetime，概念强度可能异常。")

        stock_rows = []
        for code, g in mb.groupby("code"):
            g = g.sort_values("ts")
            o = float(g["open"].iloc[0])
            c = float(g["close"].iloc[-1])
            high = float(g["high"].max())
            low = float(g["low"].min())
            vol_sum = float(g["volume"].sum())

            if o != 0:
                ret_oc = (c - o) / o
                amp = (high - low) / o
            else:
                ret_oc = 0.0
                amp = 0.0

            stock_rows.append(
                {
                    "code": code,
                    "ret_oc": ret_oc,
                    "amp": amp,
                    "vol": vol_sum,
                }
            )

        stock_df = pd.DataFrame(stock_rows)
        if stock_df.empty:
            self._log.warning(
                "当日分钟线汇总后为空（trade_date=%s），概念强度将为 NaN。", trade_date
            )
            return self.build_stock_concept_features(trade_date, minute_bars=None)

        # -------- 2) 概念层聚合：计算概念强度 / 情绪 -------- #
        mem = membership.merge(stock_df, on="code", how="left")

        def _agg_concept(group: pd.DataFrame) -> pd.Series:
            vol = group["vol"].fillna(0.0)
            ret = group["ret_oc"].fillna(0.0)

            total_vol = float(vol.sum())
            if total_vol > 0:
                strength = float((ret * vol).sum() / total_vol)
            else:
                strength = float(ret.mean())

            sentiment = float(ret.mean())

            return pd.Series(
                {
                    "concept_strength": strength,
                    "concept_sentiment": sentiment,
                    "concept_size": total_vol,
                    "concept_name": group["concept_name"].iloc[0],
                }
            )

        concept_df = (
            mem.groupby("concept_id", as_index=False)
            .apply(_agg_concept)
            .reset_index(drop=True)
        )

        if concept_df.empty:
            self._log.warning(
                "概念聚合结果为空，概念强度将为 NaN。"
            )
            return self.build_stock_concept_features(trade_date, minute_bars=None)

        # -------- 3) 全市场情绪：按概念规模加权的 strength -------- #
        total_size = float(concept_df["concept_size"].sum())
        if total_size > 0:
            market_sentiment = float(
                (concept_df["concept_strength"] * concept_df["concept_size"]).sum()
                / total_size
            )
        else:
            market_sentiment = float(concept_df["concept_strength"].mean())

        # -------- 4) 回到个股层，聚合概念因子 -------- #
        mem2 = mem.merge(
            concept_df[
                ["concept_id", "concept_strength", "concept_sentiment", "concept_name"]
            ],
            on="concept_id",
            how="left",
            suffixes=("", "_c"),
        )

        rows = []
        for code, g in mem2.groupby("code"):
            cs = g["concept_strength"]
            ss = g["concept_sentiment"]

            cpt_count = int(g["concept_id"].nunique())
            strength_max = float(cs.max()) if cs.notna().any() else np.nan
            strength_mean = float(cs.mean()) if cs.notna().any() else np.nan
            sent_max = float(ss.max()) if ss.notna().any() else np.nan
            sent_mean = float(ss.mean()) if ss.notna().any() else np.nan

            top_cid = None
            top_cname = None
            if cs.notna().any():
                try:
                    idx = cs.idxmax()
                    top_cid = g.loc[idx, "concept_id"]
                    top_cname = g.loc[idx, "concept_name"]
                except Exception:
                    pass

            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "cpt_count": cpt_count,
                    "cpt_strength_max": strength_max,
                    "cpt_strength_mean": strength_mean,
                    "cpt_sentiment_max": sent_max,
                    "cpt_sentiment_mean": sent_mean,
                    "cpt_top_concept_id": top_cid,
                    "cpt_top_concept_name": top_cname,
                    "market_sentiment": market_sentiment,
                }
            )

        stock_cpt_df = pd.DataFrame(rows)
        return stock_cpt_df
