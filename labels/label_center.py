# -*- coding: utf-8 -*-
"""
labels/label_center.py

LightHunter Mk3 - LabelCenter 最终版
====================================

职责：
------
- 读取 config/label_spec.json，管理超短线标签规范
- 为上层组件（TSLabeler / DatasetBuilder / TrainingPipelines）提供：
    - 标签元信息（名称、类型、horizon 等）
    - 按任务获取主标签 + 辅助标签
    - 基于日线行情 DataFrame 计算一批核心价格类标签：
        * ULTRA_T1_RET_C2C      : 隔日 log 收益
        * ULTRA_T1_DIR_3C       : 隔日方向三分类
        * ULTRA_T1_HIT_TP       : 次日是否触及止盈
        * ULTRA_T1_HIT_SL       : 次日是否触及止损
        * ULTRA_T3_MAX_UP       : 3 日最大浮盈
        * ULTRA_T3_MAX_DD       : 3 日最大回撤
        * ULTRA_T3_OUTCOME_5C   : 3 日交易结果五分类（基于 max_up/max_dd）
        * ULTRA_EMO_PHASE_5C    : 情绪阶段（从列 emo_phase 直接拷贝）

约定的数据格式：
----------------
传入的日线行情 DataFrame 应至少包含以下列：
    - symbol      : 股票代码
    - trade_date  : 交易日期（datetime 或可转为 datetime 的字符串）
    - open / high / low / close

可选列（如果需要情绪标签）：
    - emo_phase   : 整数型 0~4，对应不同市场情绪阶段

输出：
------
计算函数会在原有 DataFrame 上新增若干列（列名 = label_id），并返回新的 DataFrame 副本，
不会原地修改传入的 df（除非设置 inplace=True）。
"""  # noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class LabelDef:
    """单个标签的配置定义。"""

    id: str
    name: str
    name_zh: str
    kind: str              # regression / classification / auxiliary
    family: str            # return / direction / max_runup / max_drawdown / barrier_tp / barrier_sl / emo_phase / outcome_3d
    horizon_days: int = 0
    window_days: int = 0
    base_price: str = "close"
    base_label: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    raw_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDef:
    """训练/回测任务使用的一组标签。"""

    name: str
    primary_label: str
    aux_labels: List[str]


# ----------------------------------------------------------------------
# LabelCenter
# ----------------------------------------------------------------------


class LabelCenter:
    """统一管理标签规范与计算逻辑的中枢。"""

    def __init__(
        self,
        label_defs: Dict[str, LabelDef],
        task_defs: Dict[str, TaskDef],
    ) -> None:
        self._label_defs = label_defs
        self._task_defs = task_defs

    # ---------- 加载 ----------
    @classmethod
    def from_default_config(cls) -> "LabelCenter":
        cfg = get_system_config()
        root_dir = Path(__file__).resolve().parents[1]  # 项目根目录
        paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        spec_rel = paths_cfg.get("label_spec", "config/label_spec.json")
        spec_path = root_dir / spec_rel
        if not spec_path.exists():
            # 兜底：尝试直接在 config/ 下找
            alt = root_dir / "config" / "label_spec.json"
            if alt.exists():
                spec_path = alt
        log.info("LabelCenter: 使用标签规范文件 %s", spec_path)
        return cls.from_path(spec_path)

    @classmethod
    def from_path(cls, path: Path) -> "LabelCenter":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        labels_cfg: Dict[str, Any] = cfg.get("labels", {})
        tasks_cfg: Dict[str, Any] = cfg.get("tasks", {})

        label_defs: Dict[str, LabelDef] = {}
        for lid, lc in labels_cfg.items():
            label_defs[lid] = LabelDef(
                id=lid,
                name=lc.get("name", lid),
                name_zh=lc.get("name_zh", lid),
                kind=lc.get("kind", "regression"),
                family=lc.get("family", "return"),
                horizon_days=int(lc.get("horizon_days", lc.get("horizon", 0))),
                window_days=int(lc.get("window_days", lc.get("window", 0))),
                base_price=lc.get("base_price", "close"),
                base_label=lc.get("base_label"),
                params=lc.get("params", {}),
                raw_cfg=lc,
            )

        task_defs: Dict[str, TaskDef] = {}
        for tid, tc in tasks_cfg.items():
            task_defs[tid] = TaskDef(
                name=tc.get("name", tid),
                primary_label=tc["primary_label"],
                aux_labels=list(tc.get("aux_labels", [])),
            )

        return cls(label_defs=label_defs, task_defs=task_defs)

    # ---------- 元信息 ----------
    @property
    def label_ids(self) -> List[str]:
        return list(self._label_defs.keys())

    @property
    def task_ids(self) -> List[str]:
        return list(self._task_defs.keys())

    def get_label_def(self, label_id: str) -> LabelDef:
        return self._label_defs[label_id]

    def get_task_def(self, task_id: str) -> TaskDef:
        return self._task_defs[task_id]

    def get_labels_for_task(self, task_id: str, include_aux: bool = True) -> List[str]:
        task = self.get_task_def(task_id)
        if include_aux:
            return [task.primary_label] + task.aux_labels
        return [task.primary_label]

    # ------------------------------------------------------------------
    # 计算入口
    # ------------------------------------------------------------------

    def compute_labels(
        self,
        df: pd.DataFrame,
        label_ids: Iterable[str],
        inplace: bool = False,
    ) -> pd.DataFrame:
        """在给定的日线行情 df 上，计算指定 label_ids 对应的标签列。

        参数：
            df        : 必须包含 symbol, trade_date, open/high/low/close 等列
            label_ids : 要计算的标签 ID 列表
            inplace   : 是否原地写入 df

        返回：
            含有新标签列的 DataFrame（如果 inplace=False，则返回复制后的新 df）
        """
        if not inplace:
            df = df.copy()

        # 标准化日期 + 排序
        if not np.issubdtype(df["trade_date"].dtype, np.datetime64):
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        # 预先构建 symbol 分组
        g = df.groupby("symbol", sort=False, group_keys=False)

        for lid in label_ids:
            if lid not in self._label_defs:
                raise KeyError(f"未知的 label_id: {lid}")
            ldef = self._label_defs[lid]
            log.info("LabelCenter: 计算标签 %s (%s)", lid, ldef.name_zh)

            if ldef.family == "return":
                df[lid] = self._compute_return(g, df, ldef)
            elif ldef.family == "direction":
                df[lid] = self._compute_direction(df, ldef)
            elif ldef.family == "barrier_tp":
                df[lid] = self._compute_barrier_tp(g, df, ldef)
            elif ldef.family == "barrier_sl":
                df[lid] = self._compute_barrier_sl(g, df, ldef)
            elif ldef.family == "max_runup":
                df[lid] = self._compute_max_runup(g, df, ldef)
            elif ldef.family == "max_drawdown":
                df[lid] = self._compute_max_drawdown(g, df, ldef)
            elif ldef.family == "outcome_3d":
                df[lid] = self._compute_outcome_3d(g, df, ldef)
            elif ldef.family == "emo_phase":
                df[lid] = self._copy_emo_phase(df, ldef)
            else:
                raise NotImplementedError(f"暂不支持的 family 类型: {ldef.family} (label={lid})")

        return df

    # ------------------------------------------------------------------
    # 各类 family 的具体实现
    # ------------------------------------------------------------------

    # -- return: T+N 收益 ------------------------------------------------

    def _compute_return(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        base = ldef.base_price
        if base not in df.columns:
            raise KeyError(f"计算收益标签 {ldef.id} 需要列 {base}")

        horizon = max(int(ldef.horizon_days), 1)
        price = g[base].transform(lambda s: s.astype(float))
        future = g[base].shift(-horizon).astype(float)

        # 避免除以 0
        with np.errstate(divide="ignore", invalid="ignore"):
            if ldef.raw_cfg.get("log_return", False):
                ret = np.log(future / price)
            else:
                ret = future / price - 1.0

        # 可选截断
        clip = ldef.raw_cfg.get("clip", None)
        if clip is not None:
            ret = ret.clip(lower=-float(clip), upper=float(clip))
        return ret

    # -- direction: 基于某个 return label 的多分类 ------------------------

    def _compute_direction(self, df: pd.DataFrame, ldef: LabelDef) -> pd.Series:
        base_label = ldef.base_label
        if not base_label:
            raise ValueError(f"direction family 的标签 {ldef.id} 必须指定 base_label")

        if base_label not in df.columns:
            raise KeyError(
                f"计算方向标签 {ldef.id} 需要先计算 base_label={base_label}。"
            )

        base = df[base_label].astype(float)
        th = ldef.raw_cfg.get("thresholds", {})
        neg = float(th.get("neg", -0.0))
        pos = float(th.get("pos", 0.0))

        # -1: base <= neg, 1: base >= pos, 0: 其它
        labels = np.where(
            base <= neg,
            -1,
            np.where(base >= pos, 1, 0),
        )
        return labels.astype("int8")

    # -- barrier_tp: 次日是否触及止盈 ------------------------------------

    def _compute_barrier_tp(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        base = ldef.base_price
        high_col = ldef.raw_cfg.get("price_high_col", "high")
        if base not in df.columns or high_col not in df.columns:
            raise KeyError(
                f"计算 {ldef.id} 需要列: {base}, {high_col}"
            )

        horizon = max(int(ldef.horizon_days), 1)
        tp_pct = float(ldef.raw_cfg.get("tp_pct", 0.03))

        entry = g[base].transform(lambda s: s.astype(float))
        fut_high = g[high_col].shift(-horizon).astype(float)

        tp_price = entry * (1.0 + tp_pct)
        hit = (fut_high >= tp_price).astype("int8")
        # 无未来数据行设为 0
        hit[fut_high.isna()] = 0
        return hit

    # -- barrier_sl: 次日是否触及止损 ------------------------------------

    def _compute_barrier_sl(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        base = ldef.base_price
        low_col = ldef.raw_cfg.get("price_low_col", "low")
        if base not in df.columns or low_col not in df.columns:
            raise KeyError(
                f"计算 {ldef.id} 需要列: {base}, {low_col}"
            )

        horizon = max(int(ldef.horizon_days), 1)
        sl_pct = float(ldef.raw_cfg.get("sl_pct", -0.03))

        entry = g[base].transform(lambda s: s.astype(float))
        fut_low = g[low_col].shift(-horizon).astype(float)

        sl_price = entry * (1.0 + sl_pct)
        hit = (fut_low <= sl_price).astype("int8")
        hit[fut_low.isna()] = 0
        return hit

    # -- max_runup: N 日最大浮盈 -----------------------------------------

    def _compute_max_runup(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        base = ldef.base_price
        high_col = ldef.raw_cfg.get("price_high_col", "high")
        if base not in df.columns or high_col not in df.columns:
            raise KeyError(
                f"计算 {ldef.id} 需要列: {base}, {high_col}"
            )

        window = max(int(ldef.window_days or ldef.horizon_days or 1), 1)

        def f(group: pd.DataFrame) -> pd.Series:
            entry = group[base].astype(float).to_numpy()
            high = group[high_col].astype(float).to_numpy()
            n = len(group)
            out = np.full(n, np.nan, dtype=float)
            for i in range(n):
                j_end = min(n, i + window + 1)
                if i + 1 >= j_end:
                    continue
                future_high = np.max(high[i + 1: j_end])
                out[i] = future_high / entry[i] - 1.0
            return pd.Series(out, index=group.index)

        return g.apply(f)

    # -- max_drawdown: N 日最大回撤 --------------------------------------

    def _compute_max_drawdown(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        base = ldef.base_price
        low_col = ldef.raw_cfg.get("price_low_col", "low")
        if base not in df.columns or low_col not in df.columns:
            raise KeyError(
                f"计算 {ldef.id} 需要列: {base}, {low_col}"
            )

        window = max(int(ldef.window_days or ldef.horizon_days or 1), 1)

        def f(group: pd.DataFrame) -> pd.Series:
            entry = group[base].astype(float).to_numpy()
            low = group[low_col].astype(float).to_numpy()
            n = len(group)
            out = np.full(n, np.nan, dtype=float)
            for i in range(n):
                j_end = min(n, i + window + 1)
                if i + 1 >= j_end:
                    continue
                future_low = np.min(low[i + 1: j_end])
                out[i] = future_low / entry[i] - 1.0
            return pd.Series(out, index=group.index)

        return g.apply(f)

    # -- outcome_3d: 基于 3 日 max_up / max_dd 的五分类 -------------------

    def _compute_outcome_3d(
        self,
        g: pd.core.groupby.generic.DataFrameGroupBy,
        df: pd.DataFrame,
        ldef: LabelDef,
    ) -> pd.Series:
        """大赢/小赢/震荡/小亏/大亏 五分类标签。

        规则（以 params 为例）：
            - max_up >= tp_big        -> 2 (大赢)
            - max_up >= tp_small      -> 1 (小赢)
            - max_dd <= sl_big        -> -2 (大亏)
            - max_dd <= sl_small      -> -1 (小亏)
            - 其它                    -> 0 (震荡)
        """  # noqa: E501

        # 若已有 ULTRA_T3_MAX_UP / ULTRA_T3_MAX_DD，则直接使用；
        # 否则现场计算一次，避免重复代码。
        up_label = None
        dd_label = None
        for candidate in ("ULTRA_T3_MAX_UP", "T3_MAX_UP", "max_up_3d"):
            if candidate in df.columns:
                up_label = candidate
                break
        for candidate in ("ULTRA_T3_MAX_DD", "T3_MAX_DD", "max_dd_3d"):
            if candidate in df.columns:
                dd_label = candidate
                break

        if up_label is None:
            # on-the-fly compute
            tmp_ldef = LabelDef(
                id="__tmp_max_up__",
                name="tmp",
                name_zh="tmp",
                kind="regression",
                family="max_runup",
                horizon_days=ldef.horizon_days,
                window_days=ldef.raw_cfg.get("window_days", 3),
                base_price=ldef.base_price,
                params={},
                raw_cfg={
                    "price_high_col": ldef.raw_cfg.get("price_high_col", "high")
                },
            )
            df["__tmp_max_up__"] = self._compute_max_runup(g, df, tmp_ldef)
            up = df["__tmp_max_up__"]
        else:
            up = df[up_label].astype(float)

        if dd_label is None:
            tmp_ldef = LabelDef(
                id="__tmp_max_dd__",
                name="tmp",
                name_zh="tmp",
                kind="regression",
                family="max_drawdown",
                horizon_days=ldef.horizon_days,
                window_days=ldef.raw_cfg.get("window_days", 3),
                base_price=ldef.base_price,
                params={},
                raw_cfg={
                    "price_low_col": ldef.raw_cfg.get("price_low_col", "low")
                },
            )
            df["__tmp_max_dd__"] = self._compute_max_drawdown(g, df, tmp_ldef)
            dd = df["__tmp_max_dd__"]
        else:
            dd = df[dd_label].astype(float)

        params = {
            "tp_big": 0.06,
            "tp_small": 0.03,
            "sl_big": -0.06,
            "sl_small": -0.03,
        }
        params.update(ldef.raw_cfg.get("params", {}))

        tp_big = float(params["tp_big"])
        tp_small = float(params["tp_small"])
        sl_big = float(params["sl_big"])
        sl_small = float(params["sl_small"])

        out = np.zeros(len(df), dtype="int8")

        # 顺序：先看大赢、大亏，再看小赢、小亏
        out[up >= tp_big] = 2
        out[(up >= tp_small) & (out == 0)] = 1
        out[dd <= sl_big] = -2
        out[(dd <= sl_small) & (out == 0)] = -1

        return pd.Series(out, index=df.index)

    # -- emo_phase: 直接从 emo_phase 列复制 -------------------------------

    def _copy_emo_phase(self, df: pd.DataFrame, ldef: LabelDef) -> pd.Series:
        col = ldef.raw_cfg.get("source_column", "emo_phase")
        if col not in df.columns:
            raise KeyError(
                f"计算情绪阶段标签 {ldef.id} 需要列 {col}，通常由情绪引擎预先写入。"  # noqa: E501
            )
        return df[col].astype("int8")


# ----------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------


_default_label_center: Optional[LabelCenter] = None


def get_label_center() -> LabelCenter:
    global _default_label_center
    if _default_label_center is None:
        _default_label_center = LabelCenter.from_default_config()
    return _default_label_center
