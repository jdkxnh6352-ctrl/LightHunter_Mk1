# -*- coding: utf-8 -*-
"""
tools/u2_daily_scoring_backtest.py

U2/U3 日常打分流水线 - 历史回测版
================================

作用：
- 在已经准备好的 ultrashort_main 数据集上，模拟“每天收盘跑 日常打分”的流程；
- 每个交易日：
    1）用之前若干个交易日的数据训练一个模型（Logistic 回归）；
    2）在当天的截面上给所有股票打分（预测 label_u2 = 1 的概率）；
    3）按打分从高到低选出 Top-K 作为“候选股名单”。

支持两种引擎：
- engine = "u2": 使用原来的 U2 特征；
- engine = "u3": 使用 U3 特征（开放价量 + 20 日均量、均额）。

输出：
- 一张汇总 CSV：每一行是一只股票在某一天被选为候选股时的记录，
  包含：日期、代码、打分、排名、真实标签 label_u2、收盘价、成交额等。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 可选：scikit-learn（若没有会友好报错）
# -----------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    _SKLEARN_OK = True
except Exception as e:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    make_pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore
    _SKLEARN_OK = False
    _SKLEARN_ERR = e

# -----------------------------------------------------------------------------
# 日志 & 配置中心：优先用工程内工具，失败时回退到最简实现
# -----------------------------------------------------------------------------
try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str):
        return logging.getLogger(name)


log = get_logger("U2DailyScoringBacktest")


try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        """
        兜底版：直接从 ./config/system_config.json 读取。
        只在 config_center 不可用时使用。
        """
        cfg_path_candidates = [
            os.path.join("config", "system_config.json"),
            "system_config.json",
        ]
        for p in cfg_path_candidates:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}


# -----------------------------------------------------------------------------
# 一些常量定义
# -----------------------------------------------------------------------------
# U2 引擎用到的特征
U2_FEATURE_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "amt_mean_20",
]

# U3 引擎用到的特征（和我们在 U3 Research 里用的一致）
U3_FEATURE_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vol_20",
    "amt_mean_20",
]

LABEL_COL = "label_u2"
CODE_COL = "code"
DATE_COL = "trading_date"


# -----------------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------------
def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U2/U3 日常打分流水线 - 历史回测版",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--job-id",
        type=str,
        default="ultrashort_main",
        help="数据集 job_id，对应 data/datasets/{job-id}.parquet",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="回测开始日期 YYYY-MM-DD（默认用数据里的最早交易日）",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="回测结束日期 YYYY-MM-DD（默认用数据里的最晚交易日）",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=120,
        help="训练窗口长度（按交易日计，使用最近 N 个交易日作为训练集）",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每天最多选出的候选股数量（<=0 则表示不过滤）",
    )
    p.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="过滤：预测为正样本(label_u2=1)的概率至少要达到多少才保留",
    )
    p.add_argument(
        "--min-price",
        type=float,
        default=0.0,
        help="过滤：收盘价 >= min-price（0 表示不限制）",
    )
    p.add_argument(
        "--max-price",
        type=float,
        default=0.0,
        help="过滤：收盘价 <= max-price（0 表示不限制）",
    )
    p.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="过滤：成交额(amount) >= min-amount（单位：元，0 表示不限制）",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="u2",
        choices=["u2", "u3"],
        help="打分引擎类型：u2（默认）或 u3",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果 CSV 输出路径（默认写到 reports/ 里，自动命名，文件名会带上 engine）",
    )
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def _resolve_paths(
    cfg: Dict[str, Any], job_id: str, engine: str, output: Optional[str]
) -> Dict[str, str]:
    """根据 system_config 推断 dataset / reports 路径。"""
    paths_cfg = cfg.get("paths", {}) or {}
    project_root = paths_cfg.get("project_root", ".") or "."
    dataset_dir = paths_cfg.get("dataset_dir", os.path.join("data", "datasets")) or os.path.join(
        "data", "datasets"
    )
    reports_dir = paths_cfg.get("reports_dir", "reports") or "reports"

    # 统一切到 project_root，跟其他工具行为一致
    try:
        os.chdir(project_root)
    except Exception as e:  # pragma: no cover
        log.warning("切换到 project_root=%s 失败：%s", project_root, e)

    dataset_path = os.path.join(dataset_dir, f"{job_id}.parquet")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"未找到数据集文件：{dataset_path}")

    os.makedirs(reports_dir, exist_ok=True)

    if output:
        out_path = output
    else:
        out_name = f"{engine}_daily_scoring_backtest_{job_id}.csv"
        out_path = os.path.join(reports_dir, out_name)

    return {
        "dataset_path": dataset_path,
        "reports_dir": reports_dir,
        "output_path": out_path,
    }


def _check_dataset_columns(df: pd.DataFrame) -> None:
    # U2/U3 所有可能用到的列都检查一下
    need_cols = sorted(
        set(U2_FEATURE_COLS + U3_FEATURE_COLS + [LABEL_COL, CODE_COL, DATE_COL])
    )
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"数据集中缺少必要列：{missing}，请确认已经按教程构建了 ultrashort_main 数据集。"
        )


# -----------------------------------------------------------------------------
# 主逻辑
# -----------------------------------------------------------------------------
def run_backtest(args: argparse.Namespace) -> None:
    if not _SKLEARN_OK:
        log.error(
            "当前环境未安装 scikit-learn，无法运行回测。\n"
            "请先在虚拟环境中执行：pip install scikit-learn\n"
            "原始错误：%r",
            _SKLEARN_ERR,
        )
        return

    engine = (args.engine or "u2").lower()
    if engine not in ("u2", "u3"):
        log.error("不支持的 engine=%s，仅支持 u2 / u3。", engine)
        return

    feature_cols = U2_FEATURE_COLS if engine == "u2" else U3_FEATURE_COLS
    prob_col = "u2_prob" if engine == "u2" else "u3_prob"

    sys_cfg = get_system_config(refresh=False) or {}
    paths = _resolve_paths(sys_cfg, args.job_id, engine, args.output)

    dataset_path = paths["dataset_path"]
    out_path = paths["output_path"]

    log.info("使用引擎: %s", engine)
    log.info("读取数据集：%s", dataset_path)
    df = pd.read_parquet(dataset_path)

    _check_dataset_columns(df)

    # 规范化类型
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values([DATE_COL, CODE_COL]).reset_index(drop=True)

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 日期范围
    all_dates = sorted(df[DATE_COL].dt.date.unique().tolist())
    if not all_dates:
        log.error("数据集里没有任何交易日，无法回测。")
        return

    start_date = _parse_date(args.start_date) or all_dates[0]
    end_date = _parse_date(args.end_date) or all_dates[-1]

    log.info(
        "回测日期区间：%s ~ %s（数据全体区间：%s ~ %s）",
        start_date,
        end_date,
        all_dates[0],
        all_dates[-1],
    )

    # 只在 [start_date, end_date] 之间做“日常打分”
    test_dates: List[dt.date] = [
        d for d in all_dates if (start_date <= d <= end_date)
    ]
    if not test_dates:
        log.error("指定的日期区间内没有任何交易日。")
        return

    # 训练窗口以“交易日个数”为单位
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    results: List[pd.DataFrame] = []

    for td in test_dates:
        idx = date_to_idx[td]
        if idx == 0:
            continue  # 没有历史数据，跳过第一天

        # 训练区间：使用该日之前的最近 train_days 个交易日
        train_start_idx = max(0, idx - args.train_days)
        train_dates = all_dates[train_start_idx:idx]
        if len(train_dates) < max(20, min(args.train_days, 60)):
            # 训练样本太少时可以选择跳过
            log.warning(
                "交易日 %s 可用训练天数仅有 %d，略过该日打分。",
                td,
                len(train_dates),
            )
            continue

        train_df = df[df[DATE_COL].dt.date.isin(train_dates)].copy()
        # 防止只有单一标签
        if train_df[LABEL_COL].nunique() < 2:
            log.warning("交易日 %s 的训练集标签只有一个取值，跳过。", td)
            continue

        X_train = train_df[feature_cols].astype(float).values
        y_train = train_df[LABEL_COL].astype(int).values

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            ),
        )

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            log.exception("训练模型时出错（交易日 %s），跳过本日。错误：%s", td, e)
            continue

        # 当天截面
        today_df = df[df[DATE_COL].dt.date == td].copy()
        if today_df.empty:
            continue

        X_test = today_df[feature_cols].astype(float).values
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            log.exception("预测时出错（交易日 %s），跳过本日。错误：%s", td, e)
            continue

        today_df[prob_col] = prob

        # -----------------------
        # 过滤条件（与日常打分脚本同一风格）
        # -----------------------
        mask = np.ones(len(today_df), dtype=bool)

        if args.min_price > 0:
            mask &= today_df["close"] >= float(args.min_price)
        if args.max_price > 0:
            mask &= today_df["close"] <= float(args.max_price)
        if args.min_amount > 0:
            mask &= today_df["amount"] >= float(args.min_amount)
        if args.min_prob > 0.0:
            mask &= today_df[prob_col] >= float(args.min_prob)

        cand = today_df.loc[mask].copy()
        if cand.empty:
            log.info("交易日 %s 过滤后没有任何候选股。", td)
            continue

        cand = cand.sort_values(prob_col, ascending=False)
        if args.top_k > 0:
            cand = cand.head(args.top_k)

        cand["rank"] = np.arange(1, len(cand) + 1)
        cand["trade_date"] = cand[DATE_COL].dt.date
        cand["engine"] = engine

        # 只保留你关心的字段（其余你可以自己再加）
        keep_cols = (
            [
                "trade_date",
                "engine",
                CODE_COL,
                prob_col,
                "rank",
                LABEL_COL,
                "close",
                "amount",
            ]
            + [c for c in feature_cols if c not in ("close", "amount")]
        )
        keep_cols = [c for c in keep_cols if c in cand.columns]

        results.append(cand[keep_cols])

    if not results:
        log.error("整个区间内没有任何候选样本，请检查过滤条件是否过严。")
        return

    res_df = pd.concat(results, ignore_index=True)

    # -----------------------
    # 一些简单汇总指标
    # -----------------------
    total_days = res_df["trade_date"].nunique()
    total_samples = len(res_df)
    hit_rate = (
        float(res_df[LABEL_COL].mean())
        if LABEL_COL in res_df.columns
        else float("nan")
    )

    log.info(
        "[%s Backtest] 总体统计：交易日数=%d, 样本数=%d, 命中率(label_u2=1 比例)=%.4f",
        engine.upper(),
        total_days,
        total_samples,
        hit_rate,
    )

    # 按年份拆分一下命中率
    res_df["year"] = pd.to_datetime(res_df["trade_date"]).dt.year
    by_year = (
        res_df.groupby("year")
        .agg(
            n_days=("trade_date", "nunique"),
            n_samples=(CODE_COL, "count"),
            hit_rate=(LABEL_COL, "mean"),
        )
        .reset_index()
    )

    if not by_year.empty:
        log.info(
            "[%s Backtest] 按年份统计：\n%s",
            engine.upper(),
            by_year.to_string(index=False, float_format=lambda x: f"{x:0.4f}"),
        )

    # -----------------------
    # 写出 CSV
    # -----------------------
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("候选股明细已写入：%s", out_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_backtest(args)


if __name__ == "__main__":  # pragma: no cover
    main()
