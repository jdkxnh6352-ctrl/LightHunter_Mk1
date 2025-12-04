# -*- coding: utf-8 -*-
"""
features/sentiment_engine.py

LightHunter Mk3 - 情绪/舆情因子引擎
===================================

职责
----
1. 对舆情文本逐条打分（-1 ~ 1）
2. 聚合为「个股 × 交易日」情绪因子
3. 聚合为「市场级」情绪阶段（极冷/偏冷/中性/偏热/极热）

与 FactorEngine 的接口
----------------------
最终通过 `compute_for_daily_panel(df_daily, df_posts_scored)` 产出：

    - SENT_stock_score
    - SENT_stock_score_std
    - SENT_stock_post_count
    - SENT_stock_bull_ratio
    - SENT_stock_bear_ratio
    - SENT_market_score
    - SENT_market_bull_ratio
    - SENT_market_bear_ratio
    - SENT_market_phase

并且返回的 DataFrame index 与 df_daily 对齐，可作为 FactorEngine.extra_blocks["sentiment"] 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)

# 尝试加载 SnowNLP，若不可用则回退到简单词典
try:
    from snownlp import SnowNLP  # type: ignore

    HAS_SNOWNLP = True
except Exception:  # pragma: no cover
    HAS_SNOWNLP = False
    SnowNLP = None  # type: ignore


@dataclass
class SentimentEngineConfig:
    """情绪引擎配置。"""

    pos_threshold: float = 0.2   # > 0.2 判为看多
    neg_threshold: float = -0.2  # < -0.2 判为看空
    min_posts_stock: int = 2     # 个股至少多少帖子才认为有统计意义
    min_posts_market: int = 50   # 市场级至少多少帖子才启用阶段识别

    # 用于简单词典法的词表
    positive_words: Tuple[str, ...] = (
        "大涨",
        "涨停",
        "连板",
        "妖股",
        "起飞",
        "爆发",
        "机会",
        "利好",
        "翻倍",
        "龙头",
        "走强",
        "超预期",
        "放量上涨",
    )
    negative_words: Tuple[str, ...] = (
        "跌停",
        "暴跌",
        "核按钮",
        "杀跌",
        "崩盘",
        "利空",
        "出货",
        "套牢",
        "踩雷",
        "见顶",
        "退潮",
        "砸盘",
        "大阴线",
    )


class SentimentEngine:
    """情绪/舆情因子引擎。"""

    STOCK_FEATURE_COLS: List[str] = [
        "SENT_stock_score",
        "SENT_stock_score_std",
        "SENT_stock_post_count",
        "SENT_stock_bull_ratio",
        "SENT_stock_bear_ratio",
    ]

    MARKET_FEATURE_COLS: List[str] = [
        "SENT_market_score",
        "SENT_market_bull_ratio",
        "SENT_market_bear_ratio",
        "SENT_market_phase",
    ]

    def __init__(self, cfg: Optional[SentimentEngineConfig] = None) -> None:
        self.cfg = cfg or SentimentEngineConfig()

    # ------------------------------------------------------------------
    # 1. 文本打分
    # ------------------------------------------------------------------

    def score_posts(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        对舆情原始数据逐条打分。

        输入 df_raw 预期至少包含列：
            - symbol : 股票代码
            - ts     : 时间戳
            - content: 文本内容（如果没有 content，则用 title 替代）

        返回 df_scored，在原基础上新增列：
            - date              : 交易日（按 ts.normalize()）
            - sentiment_score   : [-1, 1]
            - is_bullish        : bool
            - is_bearish        : bool
        """
        if df_raw.empty:
            return df_raw.copy()

        df = df_raw.copy()
        if "symbol" not in df.columns or "ts" not in df.columns:
            raise KeyError("SentimentEngine.score_posts 需要 df_raw 至少包含 'symbol' 和 'ts'。")

        if "content" not in df.columns:
            if "title" in df.columns:
                df["content"] = df["title"].fillna("").astype(str)
            else:
                raise KeyError("df_raw 中缺少 'content' 字段（也没有 'title' 可替代）。")

        df["symbol"] = df["symbol"].astype(str)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["date"] = df["ts"].dt.normalize()
        df["content"] = df["content"].fillna("").astype(str)

        scores = []
        for text in df["content"]:
            s = self._score_text(text)
            scores.append(s)
        df["sentiment_score"] = np.asarray(scores, dtype="float32")

        cfg = self.cfg
        df["is_bullish"] = df["sentiment_score"] > cfg.pos_threshold
        df["is_bearish"] = df["sentiment_score"] < cfg.neg_threshold

        return df

    def _score_text(self, text: str) -> float:
        """单条文本打分：[-1, 1]。优先使用 SnowNLP，否则用简单词典法。"""
        text = (text or "").strip()
        if not text:
            return 0.0

        if HAS_SNOWNLP:
            try:
                s = SnowNLP(text).sentiments  # [0, 1]
                score = float(2.0 * s - 1.0)  # 映射到 [-1, 1]
                return float(np.clip(score, -1.0, 1.0))
            except Exception:  # pragma: no cover
                pass

        # 简单词典法：正负词出现次数
        text_l = text.lower()
        pos_cnt = sum(text_l.count(w) for w in self.cfg.positive_words)
        neg_cnt = sum(text_l.count(w) for w in self.cfg.negative_words)
        if pos_cnt == 0 and neg_cnt == 0:
            return 0.0
        score = (pos_cnt - neg_cnt) / float(pos_cnt + neg_cnt)
        return float(np.clip(score, -1.0, 1.0))

    # ------------------------------------------------------------------
    # 2. 聚合为情绪因子
    # ------------------------------------------------------------------

    def _aggregate_stock_daily(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        对 df_scored 按 (symbol, date) 聚合，得到个股级情绪因子。
        返回 MultiIndex (symbol, date) 的 DataFrame。
        """
        if df_scored.empty:
            idx = pd.MultiIndex.from_arrays([[], []], names=["symbol", "date"])
            return pd.DataFrame(columns=self.STOCK_FEATURE_COLS, index=idx)

        grouped = df_scored.groupby(["symbol", "date"])

        sent_mean = grouped["sentiment_score"].mean()
        sent_std = grouped["sentiment_score"].std().fillna(0.0)
        post_count = grouped["sentiment_score"].size()
        bull_ratio = grouped["is_bullish"].mean()
        bear_ratio = grouped["is_bearish"].mean()

        df_stock = pd.DataFrame(
            {
                "SENT_stock_score": sent_mean,
                "SENT_stock_score_std": sent_std,
                "SENT_stock_post_count": post_count.astype("int32"),
                "SENT_stock_bull_ratio": bull_ratio,
                "SENT_stock_bear_ratio": bear_ratio,
            }
        )

        return df_stock

    def _aggregate_market_daily(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        对 df_scored 按 date 聚合，得到市场级情绪因子与阶段。
        返回 index=date 的 DataFrame。
        """
        if df_scored.empty:
            return pd.DataFrame(columns=self.MARKET_FEATURE_COLS)

        grouped = df_scored.groupby("date")
        sent_mean = grouped["sentiment_score"].mean()
        bull_ratio = grouped["is_bullish"].mean()
        bear_ratio = grouped["is_bearish"].mean()
        post_count = grouped["sentiment_score"].size()

        df_m = pd.DataFrame(
            {
                "SENT_market_score": sent_mean,
                "SENT_market_bull_ratio": bull_ratio,
                "SENT_market_bear_ratio": bear_ratio,
                "SENT_market_post_count": post_count.astype("int32"),
            }
        )

        # 计算情绪阶段：-2/-1/0/1/2
        phase = []
        for dt_i, row in df_m.iterrows():
            if row["SENT_market_post_count"] < self.cfg.min_posts_market:
                phase.append(0)
                continue
            s = row["SENT_market_score"]
            br = row["SENT_market_bull_ratio"]
            nr = row["SENT_market_bear_ratio"]
            # 简化的划分规则，可以根据实际盘感调参
            if s > 0.5 and br > 0.6:
                phase.append(2)   # 极热
            elif s > 0.2 and br > 0.55:
                phase.append(1)   # 偏热
            elif s < -0.5 and nr > 0.6:
                phase.append(-2)  # 极冷
            elif s < -0.2 and nr > 0.55:
                phase.append(-1)  # 偏冷
            else:
                phase.append(0)   # 中性

        df_m["SENT_market_phase"] = np.asarray(phase, dtype="int8")
        return df_m[self.MARKET_FEATURE_COLS + ["SENT_market_post_count"]]

    # ------------------------------------------------------------------
    # 3. 与 df_daily 联合，对齐 index
    # ------------------------------------------------------------------

    def compute_for_daily_panel(
        self,
        df_daily: pd.DataFrame,
        df_scored_posts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        根据 df_daily 的 (symbol, trade_date) 构造与其 index 对齐的情绪因子块。

        参数
        ----
        df_daily       : 日线面板，至少包含 ['symbol', 'trade_date']。
        df_scored_posts: score_posts 的输出，包含 symbol/date/sentiment_score 等。

        返回
        ----
        df_factors : DataFrame
            index 与 df_daily 一致，列为 SENT_* 情绪因子。
        """
        if df_daily.empty:
            return pd.DataFrame(
                columns=self.STOCK_FEATURE_COLS + self.MARKET_FEATURE_COLS,
                index=df_daily.index,
            )

        if "symbol" not in df_daily.columns or "trade_date" not in df_daily.columns:
            raise KeyError("SentimentEngine.compute_for_daily_panel: df_daily 需要包含 'symbol' 和 'trade_date'。")

        if df_scored_posts.empty:
            log.info("SentimentEngine: df_scored_posts 为空，返回全 0/NaN 情绪因子。")
            return pd.DataFrame(
                columns=self.STOCK_FEATURE_COLS + self.MARKET_FEATURE_COLS,
                index=df_daily.index,
            )

        df_daily_local = df_daily.copy()
        df_daily_local["symbol"] = df_daily_local["symbol"].astype(str)
        df_daily_local["trade_date"] = pd.to_datetime(
            df_daily_local["trade_date"], errors="coerce"
        ).dt.normalize()

        # 聚合个股 & 市场情绪
        stock_daily = self._aggregate_stock_daily(df_scored_posts)
        market_daily = self._aggregate_market_daily(df_scored_posts)

        # 与 df_daily 做 join
        key = ["symbol", "trade_date"]
        stock_daily_reset = stock_daily.reset_index().rename(
            columns={"date": "trade_date"}
        )
        df_join = df_daily_local[key].merge(
            stock_daily_reset,
            on=key,
            how="left",
        )

        market_daily_reset = market_daily.reset_index().rename(
            columns={"date": "trade_date"}
        )
        df_join = df_join.merge(
            market_daily_reset[
                ["trade_date"]
                + self.MARKET_FEATURE_COLS
            ],
            on="trade_date",
            how="left",
        )

        # 缺失值填充：帖子数量缺失视为 0，情绪分数缺失视为 0（中性）
        for col in self.STOCK_FEATURE_COLS:
            if col not in df_join.columns:
                df_join[col] = np.nan
        for col in self.MARKET_FEATURE_COLS:
            if col not in df_join.columns:
                df_join[col] = np.nan

        df_join["SENT_stock_post_count"] = df_join["SENT_stock_post_count"].fillna(0).astype("int32")
        df_join["SENT_stock_score"] = df_join["SENT_stock_score"].fillna(0.0)
        df_join["SENT_stock_score_std"] = df_join["SENT_stock_score_std"].fillna(0.0)
        df_join["SENT_stock_bull_ratio"] = df_join["SENT_stock_bull_ratio"].fillna(0.0)
        df_join["SENT_stock_bear_ratio"] = df_join["SENT_stock_bear_ratio"].fillna(0.0)

        df_join["SENT_market_score"] = df_join["SENT_market_score"].fillna(0.0)
        df_join["SENT_market_bull_ratio"] = df_join["SENT_market_bull_ratio"].fillna(0.0)
        df_join["SENT_market_bear_ratio"] = df_join["SENT_market_bear_ratio"].fillna(0.0)
        df_join["SENT_market_phase"] = df_join["SENT_market_phase"].fillna(0).astype("int8")

        # 最终只保留因子列，按原 df_daily index 对齐
        df_res = df_join[self.STOCK_FEATURE_COLS + self.MARKET_FEATURE_COLS]
        df_res.index = df_daily.index
        return df_res


__all__ = [
    "SentimentEngine",
    "SentimentEngineConfig",
]
