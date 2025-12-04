# -*- coding: utf-8 -*-
"""
consensus_lab.py

ConsensusLab - 多源一致性分析实验室

职责：
    - 对多数据源的 TS 数据做“价格/成交量一致性”对比；
    - 产出：
        * per_source 维度的一致性评分（consensus_score）；
        * per_symbol 维度的不一致率（disagreement_ratio）；
        * per_pair   维度（源对源）的简单统计（可选）。

输入：
    data_by_source: Dict[str, pd.DataFrame]
    每个 DataFrame 至少包含列：
        symbol, ts, close, volume

输出（compute_consensus_summary 返回）：
    {
      "per_source": {
        "eastmoney": {
          "num_pair_rows": int,
          "num_disagree_rows": int,
          "consensus_score": float   # 0~1，越大越一致
        },
        ...
      },
      "per_symbol": pd.DataFrame[
          ["symbol", "num_pairs", "num_disagree", "disagreement_ratio"]
      ],
      "pair_stats": pd.DataFrame[
          ["src_a", "src_b", "num_rows", "num_disagree", "disagree_ratio"]
      ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Optional

import pandas as pd

from core.logging_utils import get_logger

log = get_logger(__name__)


class ConsensusLab:
    """多源一致性分析工具类。"""

    def __init__(self) -> None:
        self.log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def compute_consensus_summary(
        self,
        data_by_source: Dict[str, pd.DataFrame],
        *,
        price_tol: float = 1e-3,
        volume_tol: float = 0.05,
    ) -> Dict[str, Any]:
        """
        计算多源一致性摘要。

        Args:
            data_by_source: {source_name: DataFrame}
            price_tol:      价格相对差阈值（超过则视为不一致）
            volume_tol:     成交量相对差阈值

        Returns:
            见文件头部说明。
        """
        sources = sorted([s for s, df in data_by_source.items() if df is not None])
        per_source_stats: Dict[str, Dict[str, Any]] = {
            s: {"num_pair_rows": 0, "num_disagree_rows": 0} for s in sources
        }

        # per_symbol 统计
        sym_stats: Dict[str, Dict[str, int]] = {}

        pair_rows: list[Dict[str, Any]] = []

        if len(sources) < 2:
            # 只有一个源或没有源，视为全部一致
            for s in sources:
                per_source_stats[s]["consensus_score"] = 1.0
            per_symbol_df = pd.DataFrame(
                columns=["symbol", "num_pairs", "num_disagree", "disagreement_ratio"]
            )
            pair_stats_df = pd.DataFrame(
                columns=["src_a", "src_b", "num_rows", "num_disagree", "disagree_ratio"]
            )
            return {
                "per_source": per_source_stats,
                "per_symbol": per_symbol_df,
                "pair_stats": pair_stats_df,
            }

        for src_a, src_b in combinations(sources, 2):
            df_a = data_by_source.get(src_a)
            df_b = data_by_source.get(src_b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue

            sub_a = df_a[["symbol", "ts", "close", "volume"]].copy()
            sub_b = df_b[["symbol", "ts", "close", "volume"]].copy()

            sub_a = sub_a.rename(
                columns={"close": f"close_{src_a}", "volume": f"volume_{src_a}"}
            )
            sub_b = sub_b.rename(
                columns={"close": f"close_{src_b}", "volume": f"volume_{src_b}"}
            )

            merged = pd.merge(
                sub_a,
                sub_b,
                on=["symbol", "ts"],
                how="inner",
            )

            if merged.empty:
                continue

            # 价格 / 成交量相对差
            c_a = merged[f"close_{src_a}"].astype(float)
            c_b = merged[f"close_{src_b}"].astype(float)
            v_a = merged[f"volume_{src_a}"].astype(float)
            v_b = merged[f"volume_{src_b}"].astype(float)

            denom_price = c_b.abs().clip(lower=1e-6)
            denom_vol = v_b.abs().clip(lower=1.0)

            rel_p = (c_a - c_b).abs() / denom_price
            rel_v = (v_a - v_b).abs() / denom_vol

            disagree_mask = (rel_p > price_tol) | (rel_v > volume_tol)

            num_rows = len(merged)
            num_disagree = int(disagree_mask.sum())

            # 更新 per_source 聚合
            for src in (src_a, src_b):
                per_source_stats[src]["num_pair_rows"] += num_rows
                per_source_stats[src]["num_disagree_rows"] += num_disagree

            # per_symbol 统计
            total_per_sym = merged.groupby("symbol").size()
            disagree_per_sym = (
                merged.loc[disagree_mask].groupby("symbol").size()
                if num_disagree > 0
                else pd.Series(dtype="int64")
            )

            for sym, total_n in total_per_sym.items():
                sym_entry = sym_stats.setdefault(
                    sym, {"num_pairs": 0, "num_disagree": 0}
                )
                sym_entry["num_pairs"] += int(total_n)

            for sym, dis_n in disagree_per_sym.items():
                sym_entry = sym_stats.setdefault(
                    sym, {"num_pairs": 0, "num_disagree": 0}
                )
                sym_entry["num_disagree"] += int(dis_n)

            # pair 层面统计
            disagree_ratio = num_disagree / num_rows if num_rows > 0 else 0.0
            pair_rows.append(
                {
                    "src_a": src_a,
                    "src_b": src_b,
                    "num_rows": num_rows,
                    "num_disagree": num_disagree,
                    "disagree_ratio": disagree_ratio,
                }
            )

        # 汇总 per_source
        for src, stats in per_source_stats.items():
            total = stats["num_pair_rows"]
            disagree = stats["num_disagree_rows"]
            if total <= 0:
                stats["consensus_score"] = 1.0
            else:
                stats["consensus_score"] = max(
                    0.0, min(1.0, 1.0 - disagree / float(total))
                )

        # per_symbol DataFrame
        sym_rows = []
        for sym, s in sym_stats.items():
            num_pairs = s.get("num_pairs", 0)
            num_disagree = s.get("num_disagree", 0)
            if num_pairs <= 0:
                ratio = 0.0
            else:
                ratio = num_disagree / float(num_pairs)
            sym_rows.append(
                {
                    "symbol": sym,
                    "num_pairs": int(num_pairs),
                    "num_disagree": int(num_disagree),
                    "disagreement_ratio": float(max(0.0, min(1.0, ratio))),
                }
            )
        per_symbol_df = pd.DataFrame(sym_rows)

        # per_pair DataFrame
        pair_stats_df = pd.DataFrame(pair_rows)

        return {
            "per_source": per_source_stats,
            "per_symbol": per_symbol_df,
            "pair_stats": pair_stats_df,
        }
