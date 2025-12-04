from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.config_center import get_system_config

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFoldSpec:
    """单个 Walk‑Forward 折的时间切片定义。"""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
        }


@dataclass
class BacktestResult:
    """单个策略在单个折上的回测结果。"""

    strategy_id: str
    fold_index: int
    test_start: datetime
    test_end: datetime

    equity_curve: pd.Series          # index=date, value=equity
    daily_returns: pd.Series         # index=date, value=ret
    trades: pd.DataFrame             # 记录每天持仓（目标权重）及信号、标签
    metrics: Dict[str, float]        # 聚合指标（年化收益/Sharpe/回撤等）


# ---------------------------------------------------------------------------
# 时间切片工具
# ---------------------------------------------------------------------------

def build_walkforward_folds(
    start_date: str,
    end_date: str,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
    mode: str = "sliding",
) -> List[WalkForwardFoldSpec]:
    """
    根据起止日期和训练/测试窗口长度，生成一系列 Walk‑Forward 折定义。

    约定（实现层面简化版）：
    - 所有日期按“自然日”处理，真正的交易日过滤交给 DatasetBuilder；
    - `mode` 目前对折的边界影响不大，主要由 DatasetBuilder 决定是否使用
      expanding（从最早开始一直到 train_end）还是 sliding（只取最近 train_days）。

    这里的策略：
    - 连续往前滚动 test_days，一次一折；
    - 对第 k 折：
        - 先确定 test_start_k / test_end_k；
        - 训练区间的结束日期 train_end_k = test_start_k - 1；
        - 训练区间的开始日期 train_start_k = train_end_k - train_days + 1。
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    if step_days is None:
        step_days = test_days

    folds: List[WalkForwardFoldSpec] = []
    idx = 0

    # 第一个测试窗口从 start+train_days 开始，确保有足够历史
    test_start = start + timedelta(days=train_days)
    while test_start <= end:
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > end:
            test_end = end

        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)

        if train_start < start:
            train_start = start

        folds.append(
            WalkForwardFoldSpec(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        idx += 1

        # 下一个测试窗口起点往前挪 step_days
        test_start = test_start + timedelta(days=step_days)

    return folds


# ---------------------------------------------------------------------------
# 底层指标计算
# ---------------------------------------------------------------------------

def _compute_max_drawdown(equity: pd.Series) -> Tuple[float, float, float]:
    """
    计算最大回撤。

    返回：
    - max_dd: 最大回撤（负数，如 -0.23）
    - peak: 峰值
    - valley: 谷值
    """
    if equity.empty:
        return 0.0, np.nan, np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    min_dd = drawdown.min()
    valley_idx = drawdown.idxmin()
    if pd.isna(valley_idx):
        return 0.0, float(equity.iloc[0]), float(equity.iloc[-1])
    peak_idx = equity.loc[:valley_idx].idxmax()
    return float(min_dd), float(equity.loc[peak_idx]), float(equity.loc[valley_idx])


def compute_basic_metrics(
    daily_returns: pd.Series,
    equity: Optional[pd.Series] = None,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """给定日度收益序列，计算一套基础绩效指标。"""
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    if n == 0:
        return {}

    if equity is None:
        equity = (1.0 + daily_returns).cumprod()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    ann_factor = trading_days_per_year / max(n, 1)
    ann_return = (1.0 + total_return) ** ann_factor - 1.0

    vol = float(daily_returns.std(ddof=1) * np.sqrt(trading_days_per_year))
    sharpe = ann_return / vol if vol > 0 else 0.0

    max_dd, peak, valley = _compute_max_drawdown(equity)
    win_ratio = float((daily_returns > 0).mean())

    return {
        "n_days": float(n),
        "total_return": total_return,
        "annual_return": float(ann_return),
        "volatility": vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_peak": peak,
        "max_drawdown_valley": valley,
        "win_ratio": win_ratio,
    }


# ---------------------------------------------------------------------------
# 简单的长多组合回测（Top‑K 等权）
# ---------------------------------------------------------------------------

def run_long_only_topk_backtest(
    strategy_id: str,
    fold_index: int,
    test_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    score_col: str = "pred",
    label_col: str = "y_t1_oc",
    top_k: int = 20,
    initial_cash: Optional[float] = None,
    trading_cost_bps: float = 1.5,
) -> BacktestResult:
    """
    使用 Top‑K 等权 + 简单费用模型的长多回测。

    约定：
    - test_df 每行是一只股票在某个 trade_date 的样本；
    - score_col 为模型预测（越大越好）；
    - label_col 为该日期对应的“未来持有一期”的实际收益（如 T+1 open->close）；
    - 每个交易日收盘，用当天 score 排序选择 Top‑K 多头持仓，
      第二天实现 label 对应的收益。
    """
    if test_df.empty:
        raise ValueError("run_long_only_topk_backtest: test_df 为空")

    cfg = get_system_config()
    if initial_cash is None:
        initial_cash = float(
            cfg.get("trade_core", {}).get("starting_cash", 1_000_000.0)
        )

    df = test_df.copy()
    # 规范化日期列
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, score_col], ascending=[True, False])

    dates = df[date_col].drop_duplicates().sort_values().tolist()

    daily_returns: List[float] = []
    equity_list: List[float] = []
    date_index: List[pd.Timestamp] = []
    trades_records: List[Dict[str, Any]] = []

    equity = float(initial_cash)
    prev_weights: Dict[str, float] = {}

    cost_rate = trading_cost_bps / 10000.0

    for d in dates:
        df_d = df[df[date_col] == d]

        # 过滤标签为空的样本
        df_d = df_d.dropna(subset=[label_col])
        if df_d.empty:
            daily_returns.append(0.0)
            equity_list.append(equity)
            date_index.append(d)
            prev_weights = {}
            continue

        # 选 Top‑K
        df_sel = df_d.head(top_k)
        symbols = df_sel[symbol_col].tolist()
        n_sel = len(symbols)

        if n_sel == 0:
            daily_returns.append(0.0)
            equity_list.append(equity)
            date_index.append(d)
            prev_weights = {}
            continue

        weight = 1.0 / n_sel
        cur_weights: Dict[str, float] = {s: weight for s in symbols}

        # 组合收益：sum(w_i * label_i)
        ret_raw = float((df_sel[label_col] * weight).sum())

        # 估算换手率 ≈ Σ|w_t - w_{t-1}|（这里忽略 1/2 因子，直接用总变化作为费用基数）
        turnover = 0.0
        all_symbols = set(prev_weights.keys()) | set(cur_weights.keys())
        for s in all_symbols:
            w_prev = prev_weights.get(s, 0.0)
            w_cur = cur_weights.get(s, 0.0)
            turnover += abs(w_cur - w_prev)

        cost = turnover * cost_rate
        ret_net = ret_raw - cost

        equity *= (1.0 + ret_net)

        daily_returns.append(ret_net)
        equity_list.append(equity)
        date_index.append(d)

        # 记录当日“目标持仓快照”
        for _, row in df_sel.iterrows():
            trades_records.append(
                {
                    "date": d,
                    "strategy_id": strategy_id,
                    "symbol": row[symbol_col],
                    "target_weight": weight,
                    "score": row[score_col],
                    "label": row[label_col],
                    "turnover": turnover,
                }
            )

        prev_weights = cur_weights

    daily_ret_ser = pd.Series(daily_returns, index=pd.to_datetime(date_index))
    equity_ser = pd.Series(equity_list, index=pd.to_datetime(date_index))
    trades_df = pd.DataFrame(trades_records)

    metrics = compute_basic_metrics(daily_ret_ser, equity_ser)

    return BacktestResult(
        strategy_id=strategy_id,
        fold_index=fold_index,
        test_start=min(date_index),
        test_end=max(date_index),
        equity_curve=equity_ser,
        daily_returns=daily_ret_ser,
        trades=trades_df,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 辅助：按年份/阶段汇总（可供外部调用）
# ---------------------------------------------------------------------------

def summarize_by_year(
    results: Sequence[BacktestResult],
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    把多个 BacktestResult 的日度收益合并后，按 年份 + 策略 维度汇总指标。
    """
    if not results:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for br in results:
        ser = br.daily_returns.dropna()
        if ser.empty:
            continue
        df = ser.to_frame(name="ret")
        df["strategy_id"] = br.strategy_id
        df["fold_index"] = br.fold_index
        df["year"] = df.index.year

        # 对每个 (strategy_id, year) 分组统计
        for (sid, year), grp in df.groupby(["strategy_id", "year"]):
            equity = (1.0 + grp["ret"]).cumprod()
            m = compute_basic_metrics(grp["ret"], equity, trading_days_per_year)
            rec = {
                "strategy_id": sid,
                "year": int(year),
                "fold_count": int(grp["fold_index"].nunique()),
            }
            rec.update(m)
            records.append(rec)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out = (
        out.groupby(["strategy_id", "year"], as_index=False)
        .agg(
            {
                "fold_count": "max",
                "n_days": "sum",
                "total_return": "mean",
                "annual_return": "mean",
                "volatility": "mean",
                "sharpe": "mean",
                "max_drawdown": "mean",
                "win_ratio": "mean",
            }
        )
        .sort_values(["strategy_id", "year"])
    )
    return out


def summarize_by_fold(
    results: Sequence[BacktestResult],
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    按 (strategy_id, fold_index) 维度汇总指标，可理解为“阶段级”统计。
    """
    if not results:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for br in results:
        ser = br.daily_returns.dropna()
        if ser.empty:
            continue
        equity = (1.0 + ser).cumprod()
        m = compute_basic_metrics(ser, equity, trading_days_per_year)
        rec = {
            "strategy_id": br.strategy_id,
            "fold_index": int(br.fold_index),
            "test_start": br.test_start.strftime("%Y-%m-%d"),
            "test_end": br.test_end.strftime("%Y-%m-%d"),
        }
        rec.update(m)
        records.append(rec)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out = out.sort_values(["strategy_id", "fold_index"])
    return out
# ---------------------------------------------------------------------------
# GeneticOptimizer 占位实现（兼容旧入口）
# ---------------------------------------------------------------------------

class GeneticOptimizer:
    """
    简化版 GeneticOptimizer。

    作用：
      - 让 main.py / ts_pipeline 里的
            from backtest_core import GeneticOptimizer
        不再报错；
      - 至少能生成/更新一个 gene_config.json 文件，后续 Commander /
        CombatBrain 可以读取这个文件里的权重。

    后续如果要上真正的遗传算法，只需要保持同样的调用方式：
        optimizer = GeneticOptimizer(...)
        best_gene = optimizer.run_evolution()
    把 run_evolution 的内部实现换掉就可以了。
    """

    def __init__(
        self,
        csv_file: str = "market_blackbox.csv",
        labeled_file: Optional[str] = None,
        output_file: str = "gene_config.json",
    ) -> None:
        self.csv_file = csv_file
        self.labeled_file = labeled_file
        self.output_file = output_file

    # 读取现有基因配置，如果没有就给一份默认值
    def _load_existing_gene(self) -> Dict[str, float]:
        import json
        import os

        # 默认一份比较稳妥的权重，等以后 GA 做好再替换
        default_gene: Dict[str, float] = {
            "w_pct": 2.0,
            "w_force": 20.0,
            "w_res": 2.0,
            "w_con": 30.0,
            "w_ai": 5.0,
        }

        if not os.path.exists(self.output_file):
            return default_gene

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                # 只保留数值类型
                return {
                    str(k): float(v)
                    for k, v in data.items()
                    if isinstance(v, (int, float))
                }
            return default_gene
        except Exception:
            logger.exception("读取现有 gene_config 失败，使用默认基因。")
            return default_gene

    def run_evolution(self) -> Dict[str, float]:
        """
        当前是占位实现：不做真正的“进化”，只是：

        1）优先读取已有 gene_config.json；
        2）如果没有，就用一份默认基因；
        3）把基因写回 gene_config.json；
        4）返回这个基因 dict。

        这样：
        - main.py 菜单 3 可以正常运行并打印一个基因；
        - ts_pipeline 阶段 4 也能顺利结束；
        - 后续我们可以在不改调用方式的前提下，把这里升级成真正的 GA。
        """
        import json

        gene = self._load_existing_gene()
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(gene, f, ensure_ascii=False, indent=2)
            logger.info(
                "GeneticOptimizer 占位版：gene_config 已写入 %s，gene=%s",
                self.output_file,
                gene,
            )
        except Exception:
            logger.exception(
                "GeneticOptimizer 占位版：写入 %s 失败（但仍返回基因 dict）",
                self.output_file,
            )

        return gene
