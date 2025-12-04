# -*- coding: utf-8 -*-
"""
concept_lab.py

LightHunter ConceptLab - 概念数据中枢。

职责：
- 统一管理“概念（主题/行业/题材）”及其与标的（symbol）的映射关系；
- 为概念图谱构建（ConceptGraphBuilder）提供标准化的 membership 数据；
- 为后续 GNN / 因子引擎提供概念相关特征的基础表。

数据落地格式（Parquet）：
- 概念元数据：  concepts.parquet
    字段示例：
        concept_id      : 概念唯一 ID（如东财概念代码，或自定义）
        concept_name    : 概念名称（锂电池、机器人等）
        category        : 类别（industry/theme/region/other）
        source          : 数据来源（eastmoney、ths、自定义）
        first_seen      : 首次出现日期
        last_seen       : 最近一次更新日期

- 概念归属表： concept_membership.parquet
    字段示例：
        symbol          : 股票代码（如 600519.SH）
        concept_id      : 概念 ID
        concept_name    : 概念名称
        as_of           : 该记录对应的日期（一般为交易日）
        weight          : 权重（可选，默认为 1.0）
        source          : 数据来源

注意：
- 本模块本身不负责“抓取网页”，而是作为下游数据的统一收敛点；
- collectors 中抓到的概念数据，可以清洗为 DataFrame 后调用 ConceptLab.save_membership() 进行增量更新；
- ConceptGraphBuilder 只依赖 get_graph_membership(as_of) 这一入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.logging_utils import get_logger
from config.config_center import get_system_config


@dataclass
class ConceptLabPaths:
    root: Path
    concepts_path: Path
    membership_path: Path


class ConceptLab:
    """概念数据中枢。

    主要对外接口：
        - load_concepts() / save_concepts()
        - load_membership(as_of=None)
        - save_membership(df, mode="merge")
        - get_graph_membership(as_of=None) -> 用于 ConceptGraphBuilder
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.log = get_logger(self.__class__.__name__)
        self.sys_cfg = config or get_system_config()
        self.paths = self._init_paths(self.sys_cfg)

        self.log.info(
            "ConceptLab 初始化完成。root=%s", self.paths.root,
        )

    # ------------------------------------------------------------------
    # 路径初始化
    # ------------------------------------------------------------------
    @staticmethod
    def _init_paths(cfg: Dict[str, Any]) -> ConceptLabPaths:
        paths_cfg = cfg.get("paths", {}) or {}
        dataset_dir = paths_cfg.get("dataset_dir", "data/datasets")
        root = Path(dataset_dir).expanduser().resolve() / "concepts"
        root.mkdir(parents=True, exist_ok=True)

        concepts_path = root / "concepts.parquet"
        membership_path = root / "concept_membership.parquet"

        return ConceptLabPaths(
            root=root,
            concepts_path=concepts_path,
            membership_path=membership_path,
        )

    # ------------------------------------------------------------------
    # 概念元数据：加载 / 保存
    # ------------------------------------------------------------------
    def load_concepts(self) -> pd.DataFrame:
        """加载概念元数据表（concepts.parquet）。"""
        p = self.paths.concepts_path
        if not p.exists():
            return pd.DataFrame(
                columns=[
                    "concept_id",
                    "concept_name",
                    "category",
                    "source",
                    "first_seen",
                    "last_seen",
                ]
            )
        try:
            df = pd.read_parquet(p)
        except Exception:
            self.log.exception("加载概念元数据失败: %s", p)
            return pd.DataFrame()

        # 规范化字段
        for col in ["concept_id", "concept_name", "category", "source"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        for col in ["first_seen", "last_seen"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def save_concepts(self, df: pd.DataFrame, *, mode: str = "merge") -> None:
        """保存概念元数据。

        Args:
            df  : 至少包含 concept_id / concept_name 列
            mode:
                - "replace" : 完全覆盖旧文件；
                - "append"  : 直接追加；
                - "merge"   : 与旧数据合并后去重（默认）。
        """
        if df is None or df.empty:
            return

        df = df.copy()
        if "concept_id" not in df.columns and "concept_name" in df.columns:
            df["concept_id"] = df["concept_name"]

        for col in ["concept_id", "concept_name", "category", "source"]:
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(str)

        today = datetime.utcnow().date()
        if "first_seen" not in df.columns:
            df["first_seen"] = today
        if "last_seen" not in df.columns:
            df["last_seen"] = today

        p = self.paths.concepts_path
        if mode == "replace" or not p.exists():
            df.to_parquet(p, index=False)
            self.log.info("ConceptLab.save_concepts: 写入 %s 行到 %s (mode=replace)", len(df), p)
            return

        try:
            old = pd.read_parquet(p)
        except Exception:
            self.log.exception("读取旧 concepts 失败，将直接覆盖写入。")
            df.to_parquet(p, index=False)
            return

        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["concept_id"], keep="last")
        if mode == "append":
            # append 模式其实也会去重 concept_id（避免膨胀）
            pass

        merged.to_parquet(p, index=False)
        self.log.info(
            "ConceptLab.save_concepts: 合并写入 %s 行到 %s (mode=%s)",
            len(merged),
            p,
            mode,
        )

    # ------------------------------------------------------------------
    # 概念归属表：加载 / 保存
    # ------------------------------------------------------------------
    def load_membership(self, as_of: Optional[date] = None) -> pd.DataFrame:
        """加载概念归属表。

        数据格式参考文件头注释。

        Args:
            as_of: 若提供，则尝试筛选出在该日期有效的归属关系。
        """
        p = self.paths.membership_path
        if not p.exists():
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "concept_id",
                    "concept_name",
                    "as_of",
                    "weight",
                    "source",
                ]
            )

        try:
            df = pd.read_parquet(p)
        except Exception:
            self.log.exception("加载概念归属表失败: %s", p)
            return pd.DataFrame()

        # 列名适配
        if "trade_date" in df.columns and "as_of" not in df.columns:
            df["as_of"] = df["trade_date"]

        for col in ["symbol", "concept_id", "concept_name", "source"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        if "as_of" in df.columns:
            df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
        else:
            df["as_of"] = pd.NaT

        if "weight" not in df.columns:
            df["weight"] = 1.0

        if as_of is None:
            return df

        as_of_ts = pd.to_datetime(as_of)

        # 若存在有效期字段，则用区间过滤
        if "start_date" in df.columns or "end_date" in df.columns:
            if "start_date" in df.columns:
                df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
            else:
                df["start_date"] = pd.NaT
            if "end_date" in df.columns:
                df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
            else:
                df["end_date"] = pd.NaT

            mask = (df["start_date"].isna() | (df["start_date"] <= as_of_ts)) & (
                df["end_date"].isna() | (df["end_date"] >= as_of_ts)
            )
            return df[mask].copy()

        # 否则按 as_of 取“最近一次不超过该日期”的记录
        df = df[~df["as_of"].isna()].copy()
        if df.empty:
            return df

        df = df[df["as_of"] <= as_of_ts]
        if df.empty:
            return df

        df = df.sort_values(["symbol", "concept_id", "as_of"])
        df = df.drop_duplicates(subset=["symbol", "concept_id"], keep="last")
        return df

    def save_membership(
        self,
        df: pd.DataFrame,
        *,
        mode: str = "merge",
    ) -> None:
        """保存概念归属表。

        Args:
            df  : 至少包含 symbol / concept_name 列，推荐包含 as_of / concept_id / weight / source
            mode:
                - "replace" : 完全覆盖旧文件；
                - "append"  : 直接追加；
                - "merge"   : 追加后按 (symbol, concept_id, as_of) 去重（默认）。
        """
        if df is None or df.empty:
            return

        df = df.copy()

        # 列名适配
        if "concept_name" not in df.columns:
            for alt in ["concept", "name", "label"]:
                if alt in df.columns:
                    df["concept_name"] = df[alt]
                    break

        if "concept_id" not in df.columns:
            for alt in ["concept_code", "code", "id"]:
                if alt in df.columns:
                    df["concept_id"] = df[alt]
                    break
        if "concept_id" not in df.columns and "concept_name" in df.columns:
            df["concept_id"] = df["concept_name"]

        if "as_of" not in df.columns:
            if "trade_date" in df.columns:
                df["as_of"] = df["trade_date"]
            else:
                df["as_of"] = datetime.utcnow().date()

        if "weight" not in df.columns:
            df["weight"] = 1.0
        if "source" not in df.columns:
            df["source"] = "unknown"

        for col in ["symbol", "concept_id", "concept_name", "source"]:
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(str)

        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")

        p = self.paths.membership_path

        if mode == "replace" or not p.exists():
            df.to_parquet(p, index=False)
            self.log.info("ConceptLab.save_membership: 写入 %s 行到 %s (mode=replace)", len(df), p)
            return

        try:
            old = pd.read_parquet(p)
        except Exception:
            self.log.exception("读取旧 concept_membership 失败，将直接覆盖写入。")
            df.to_parquet(p, index=False)
            return

        merged = pd.concat([old, df], ignore_index=True)

        if mode == "merge":
            if "as_of" in merged.columns:
                merged["as_of"] = pd.to_datetime(merged["as_of"], errors="coerce")
                merged = merged.sort_values(["symbol", "concept_id", "as_of"])
                merged = merged.drop_duplicates(
                    subset=["symbol", "concept_id", "as_of"], keep="last"
                )
            else:
                merged = merged.drop_duplicates(
                    subset=["symbol", "concept_id"], keep="last"
                )
        elif mode == "append":
            pass

        merged.to_parquet(p, index=False)
        self.log.info(
            "ConceptLab.save_membership: 合并写入 %s 行到 %s (mode=%s)",
            len(merged),
            p,
            mode,
        )

    # ------------------------------------------------------------------
    # 概念图谱构建专用：图输入格式
    # ------------------------------------------------------------------
    def get_graph_membership(self, as_of: Optional[date] = None) -> pd.DataFrame:
        """为概念图谱构建提供标准化的 membership 表。

        返回字段：
            symbol
            concept_id
            concept_name
            weight
        """
        df = self.load_membership(as_of=as_of)
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["symbol", "concept_id", "concept_name", "weight"]
            )

        df = df.copy()

        # 列名适配已在 load_membership 中处理，这里只做兜底
        if "concept_name" not in df.columns:
            for alt in ["concept", "name", "label"]:
                if alt in df.columns:
                    df["concept_name"] = df[alt]
                    break
        if "concept_id" not in df.columns and "concept_name" in df.columns:
            df["concept_id"] = df["concept_name"]

        if "weight" not in df.columns:
            df["weight"] = 1.0

        for col in ["symbol", "concept_id", "concept_name"]:
            if col not in df.columns:
                raise ValueError(f"ConceptLab.get_graph_membership 缺少必要字段: {col}")

        df["symbol"] = df["symbol"].astype(str)
        df["concept_id"] = df["concept_id"].astype(str)
        df["concept_name"] = df["concept_name"].astype(str)
        df["weight"] = df["weight"].astype(float)

        df = df.drop_duplicates(subset=["symbol", "concept_id"])
        return df[["symbol", "concept_id", "concept_name", "weight"]]


# 便捷函数：获取全局 ConceptLab 实例（可选）
_global_concept_lab: Optional[ConceptLab] = None


def get_concept_lab() -> ConceptLab:
    global _global_concept_lab
    if _global_concept_lab is None:
        _global_concept_lab = ConceptLab()
    return _global_concept_lab


if __name__ == "__main__":  # 简单自测
    lab = ConceptLab()
    # 构造一小撮示例 membership
    demo = pd.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH", "300750.SZ"],
            "concept_name": ["白酒", "贵州本地股", "锂电池"],
            "as_of": [date(2024, 1, 2)] * 3,
            "source": ["demo"] * 3,
        }
    )
    lab.save_membership(demo, mode="merge")
    m = lab.get_graph_membership(as_of=date(2024, 1, 2))
    print(m)
