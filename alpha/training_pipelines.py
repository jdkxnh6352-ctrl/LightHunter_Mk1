# alpha/training_pipelines.py
"""
简化版 Alpha 训练 & Walk‑Forward 回测管线（U2 专用，支持 ultrashort_main）

说明：
- 只依赖已经存在的 backtest_core、system_config 和我们做好的 ultrashort_main 数据集；
- 不再使用 alpha.strategies / STRATEGY_REGISTRY；
- 主要给 tools.run_walkforward_backtests 使用。

当前假设：
- 数据集文件位于：
    data/datasets/ultrashort_main.parquet   或
    data/datasets/ultrashort_main.csv
- 至少包含这些列：
    code, trading_date,
    open, high, low, close, volume, amount,
    ret_1, ret_5, ret_20, vol_20, amt_mean_20,
    label_u2   （1 = 将来 20 根内最高价涨幅 >= 2%）

训练：
- 用所有特征列（除 code / trading_date / label_u2 / ret_1）去预测 label_u2。
回测：
- 以模型给出的“买入概率”作为 score；
- 用 ret_1 作为持有一期的实际收益；
- 调用 backtest_core.run_long_only_topk_backtest 做 Top‑K 等权回测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtest_core import (
    BacktestResult,
    WalkForwardFoldSpec,
    build_walkforward_folds,
    run_long_only_topk_backtest,
    summarize_by_fold,
    summarize_by_year,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Walk‑Forward 配置
# ----------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """
    Walk‑Forward 回测配置（和 tools.run_walkforward_backtests 的参数一一对应）

    Attributes
    ----------
    start_date : str   起始日期（含），格式 YYYY‑MM‑DD
    end_date   : str   结束日期（含），格式 YYYY‑MM‑DD
    train_days : int   每折训练集交易日数
    test_days  : int   每折测试集交易日数
    step_days  : int   相邻折之间向前滚动的步长（交易日数）
    mode       : str   "sliding" 或 "expanding"
    """

    start_date: str
    end_date: str
    train_days: int
    test_days: int
    step_days: int
    mode: str = "sliding"

    @classmethod
    def from_args(cls, args: Any) -> "WalkForwardConfig":
        return cls(
            start_date=args.start_date,
            end_date=args.end_date,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            mode=args.mode,
        )


# ----------------------------------------------------------------------
# 主入口：TrainingPipelines
# ----------------------------------------------------------------------


class TrainingPipelines:
    """
    简化版训练管线，只做一件事：
    - 给定 job_id（目前用 ultrashort_main）和策略列表（例如 ["U2"]），
      在指定时间段内做 Walk‑Forward 回测。
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        # system_config 由 tools.run_walkforward_backtests 传进来
        self.cfg = system_config or {}
        self.paths_cfg: Dict[str, Any] = self.cfg.get("paths", {}) or {}
        self.trade_core_cfg: Dict[str, Any] = self.cfg.get("trade_core", {}) or {}

        # 数据集目录
        dataset_dir = self.paths_cfg.get("dataset_dir", "data/datasets")
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()

        logger.info(
            "TrainingPipelines 初始化完成，dataset_dir=%s", self.dataset_dir
        )

    # ------------------------------------------------------------------
    # 对外主入口：给 tools.run_walkforward_backtests 调用
    # ------------------------------------------------------------------

    def run_walkforward_for_strategies(
        self,
        job_id: str,
        strategy_ids: Sequence[str],
        wf_conf: WalkForwardConfig,
        *,
        experiment_group: Optional[str] = None,  # 目前只是透传出来方便你之后扩展
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        对若干策略（目前主要是 U2）做 Walk‑Forward 回测。

        Parameters
        ----------
        job_id : str
            数据集 ID（现在我们假定就是 ultrashort_main）
        strategy_ids : Sequence[str]
            策略名列表，例如 ["U2"]
        wf_conf : WalkForwardConfig
            Walk‑Forward 配置
        experiment_group : Optional[str]
            仅用于以后扩展 ExperimentLab 分组；当前逻辑里暂未使用。
        top_k : int
            每天持有的标的数量（Top‑K 等权）

        Returns
        -------
        Dict[str, Any]
            {
              "fold_results": List[BacktestResult],
              "by_year": pd.DataFrame,
              "by_fold": pd.DataFrame,
            }
        """
        logger.info(
            "开始 Walk‑Forward 回测：job_id=%s, strategies=%s, cfg=%s",
            job_id,
            ",".join(strategy_ids),
            wf_conf,
        )

        # 1) 载入完整数据集
        df_all = self._load_full_dataset(job_id)

        # 2) 用 backtest_core 生成折配置（只管日期，不碰数据）
        folds: List[WalkForwardFoldSpec] = build_walkforward_folds(
            start_date=wf_conf.start_date,
            end_date=wf_conf.end_date,
            train_days=wf_conf.train_days,
            test_days=wf_conf.test_days,
            step_days=wf_conf.step_days,
            mode=wf_conf.mode,
        )

        logger.info("共生成 %d 个 Walk‑Forward 折。", len(folds))
        if not folds:
            raise RuntimeError("Walk‑Forward 折数量为 0，请检查日期区间和天数参数。")

        all_results: List[BacktestResult] = []

        for strategy_name in strategy_ids:
            logger.info("====== 开始策略 [%s] 的 Walk‑Forward 回测 ======", strategy_name)
            for fold in folds:
                logger.info(
                    "[%s] Fold #%d: train=%s~%s  test=%s~%s",
                    strategy_name,
                    fold.index,
                    fold.train_start.date(),
                    fold.train_end.date(),
                    fold.test_start.date(),
                    fold.test_end.date(),
                )
                br = self._run_single_fold_for_strategy(
                    job_id=job_id,
                    strategy_name=strategy_name,
                    df_all=df_all,
                    fold_spec=fold,
                    top_k=top_k,
                    experiment_group=experiment_group or job_id,
                )
                all_results.append(br)

        # 3) 汇总
        by_year = summarize_by_year(all_results)
        by_fold = summarize_by_fold(all_results)

        logger.info("Walk‑Forward 回测完成，共 %d 条 BacktestResult。", len(all_results))
        return {
            "fold_results": all_results,
            "by_year": by_year,
            "by_fold": by_fold,
        }

    # ------------------------------------------------------------------
    # 单折训练 + 回测
    # ------------------------------------------------------------------

    def _run_single_fold_for_strategy(
        self,
        *,
        job_id: str,
        strategy_name: str,
        df_all: pd.DataFrame,
        fold_spec: WalkForwardFoldSpec,
        top_k: int,
        experiment_group: str,  # 目前未使用，保留扩展位
    ) -> BacktestResult:
        """
        单折逻辑：
        1) 按 fold_spec 切出 train / test；
        2) 用简单逻辑回归拟合 label_u2；
        3) 用概率作为 score，调用 run_long_only_topk_backtest。
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "需要 scikit‑learn 才能跑训练，请先 `pip install scikit-learn`"
            ) from e

        train_df, _, test_df = self._build_datasets_for_fold(df_all, fold_spec)

        label_col = "label_u2"
        ret_col = "ret_1"  # 用 ret_1 作为“持有一期”的实际收益

        # 丢掉没标签的样本
        train_df = train_df.dropna(subset=[label_col])
        test_df = test_df.dropna(subset=[label_col, ret_col])

        if train_df.empty or test_df.empty:
            raise RuntimeError(
                f"[{strategy_name}] fold={fold_spec.index} 训练或测试样本为空，请检查数据集与时间区间。"
            )

        # 特征列：去掉明显不是特征的列
        drop_cols = {"code", "trading_date", label_col, ret_col}
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        if not feature_cols:
            raise RuntimeError("找不到可用的特征列，请检查数据集。")

        logger.info(
            "[%s] Fold #%d 使用特征列：%s",
            strategy_name,
            fold_spec.index,
            ", ".join(feature_cols),
        )

        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df[label_col].astype(int).to_numpy()

        model = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            n_jobs=None,
        )
        model.fit(X_train, y_train)

        X_test = test_df[feature_cols].to_numpy()
        # 预测正类概率作为 score
        proba = model.predict_proba(X_test)[:, 1]

        # 组装回测需要的 DataFrame
        bt_df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(test_df["trading_date"]),
                "symbol": test_df["code"].astype(str),
                "pred": proba.astype("float64"),
                "future_ret": test_df[ret_col].astype("float64"),
            }
        )

        br = run_long_only_topk_backtest(
            strategy_id=strategy_name,
            fold_index=fold_spec.index,
            test_df=bt_df,
            date_col="trade_date",
            symbol_col="symbol",
            score_col="pred",
            label_col="future_ret",
            top_k=top_k,
        )

        logger.info(
            "[%s] Fold #%d 回测完成：annual_return=%.4f, sharpe=%.3f, max_dd=%.4f",
            strategy_name,
            fold_spec.index,
            br.metrics.get("annual_return", 0.0),
            br.metrics.get("sharpe", 0.0),
            br.metrics.get("max_drawdown", 0.0),
        )

        return br

    # ------------------------------------------------------------------
    # 数据加载 & 按折切分
    # ------------------------------------------------------------------

    def _load_full_dataset(self, job_id: str) -> pd.DataFrame:
        """
        按 job_id 从 data/datasets 下面加载数据。
        现在我们只支持 ultrashort_main，所以直接用这个名字。
        """
        dataset_name = job_id  # 目前直接用 job_id 作为文件名前缀
        candidates = [
            self.dataset_dir / f"{dataset_name}.parquet",
            self.dataset_dir / f"{dataset_name}.csv",
        ]

        for p in candidates:
            if p.exists():
                logger.info("加载数据集：%s", p)
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                break
        else:
            raise FileNotFoundError(
                f"在 {self.dataset_dir} 下未找到 {dataset_name}.parquet / {dataset_name}.csv"
            )

        if "trading_date" not in df.columns:
            raise RuntimeError("数据集中缺少 trading_date 列，这是切分折所必须的。")

        # 规范化日期
        df = df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df = df.sort_values(["trading_date", "code"])

        return df

    def _build_datasets_for_fold(
        self,
        df_all: pd.DataFrame,
        fold_spec: WalkForwardFoldSpec,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据 fold_spec 切分出 train / valid / test。

        简化策略：
        - train: [train_start, train_end]
        - test : [test_start,  test_end]
        - valid: 取训练集最后 10% 的日期作为简单验证集（当前版本其实没怎么用到）
        """
        df = df_all

        mask_train = (df["trading_date"] >= fold_spec.train_start) & (
            df["trading_date"] <= fold_spec.train_end
        )
        mask_test = (df["trading_date"] >= fold_spec.test_start) & (
            df["trading_date"] <= fold_spec.test_end
        )

        train_df = df.loc[mask_train].copy()
        test_df = df.loc[mask_test].copy()

        # 简单切出验证集（最后 10% 日期）
        if train_df.empty:
            valid_df = train_df.copy()
        else:
            dates = sorted(train_df["trading_date"].unique())
            if len(dates) <= 1:
                valid_df = train_df.copy()
            else:
                k = max(1, int(len(dates) * 0.1))
                split_date = dates[-k]
                valid_mask = train_df["trading_date"] >= split_date
                valid_df = train_df.loc[valid_mask].copy()
                train_df = train_df.loc[~valid_mask].copy()

        logger.info(
            "Fold #%d 数据切分完成：train=%d, valid=%d, test=%d",
            fold_spec.index,
            len(train_df),
            len(valid_df),
            len(test_df),
        )

        return train_df, valid_df, test_df


__all__ = [
    "TrainingPipelines",
    "WalkForwardConfig",
]
