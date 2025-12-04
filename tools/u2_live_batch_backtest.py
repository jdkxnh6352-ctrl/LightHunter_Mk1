# -*- coding: utf-8 -*-
"""
tools/u2_live_batch_backtest.py

U2 Live 批量回测脚本（支持过滤方案 filter_tag，用于 AB 对比）。

用法示例：

    # 单次体检
    python -m tools.u2_live_batch_backtest ^
        --config config/u2_live_config.json ^
        --start-date 2019-01-01 ^
        --end-date 2025-11-28 ^
        --tag live_check ^
        --filter-tag base

    # 给不同过滤方案打 tag 做 AB 测试
    python -m tools.u2_live_batch_backtest ^
        --config config/u2_live_config.json ^
        --start-date 2019-01-01 ^
        --end-date 2025-11-28 ^
        --tag live_abtest_aggressive ^
        --filter-tag aggressive
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # 需要 scikit-learn

from config.config_manager import get_paths_config, get_u2_filter_config


# ----------------------------------------------------------------------
# 配置结构
# ----------------------------------------------------------------------


@dataclass
class U2Config:
    job_id: str
    dataset_path: Path
    label_col: str
    ret_col: str
    features: List[str]
    train_days: int
    min_train_days: int

    min_price: float
    max_price: float
    min_amount: float

    top_k: int
    min_prob: float

    max_positions: int
    position_weight: float
    stop_loss: float
    take_profit: float

    output_dir: Path
    filter_tag: Optional[str] = None


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------


def _to_date(s: str) -> pd.Timestamp:
    """把 'YYYY-MM-DD' 转成 pandas 时间戳，方便比较。"""
    return pd.to_datetime(s).normalize()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_u2_config(config_path: str, filter_tag: Optional[str] = None) -> U2Config:
    """
    读取 config/u2_live_config.json，并根据需要叠加全局过滤方案（filter_tag）。

    - u2_live_config.json 负责：
        * 数据集路径 / 特征列表 / 训练窗口长度等“结构性配置”
        * 一套默认的过滤 / 打分 / 仓位参数
    - light_hunter_config.json 里的 u2_live_filter_profiles 负责：
        * base / aggressive / defensive 等不同档位的过滤参数
        * 这里用 get_u2_filter_config(tag) 取出覆盖掉本地默认值
    """
    paths_cfg = get_paths_config() or {}
    project_root = Path(paths_cfg.get("project_root", ".")).resolve()

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    raw = _load_json(cfg_path)

    job_id = str(raw.get("job_id", "u2_live_main"))

    # ---- 数据集配置 ----
    ds_cfg = raw.get("dataset", {}) or {}
    ds_path = Path(ds_cfg.get("path", "data/datasets/ultrashort_main.parquet"))
    if not ds_path.is_absolute():
        ds_path = project_root / ds_path

    label_col = str(ds_cfg.get("label_col", "label_u2"))
    ret_col = str(ds_cfg.get("ret_col", "ret_1"))
    features = list(ds_cfg.get("features", []))

    train_days = int(ds_cfg.get("train_days", 120))
    min_train_days = int(ds_cfg.get("min_train_days", 80))

    # ---- 本地默认过滤 / 打分 / 仓位参数 ----
    filter_cfg = raw.get("filter", {}) or {}
    min_price = float(filter_cfg.get("min_price", 3.0))
    max_price = float(filter_cfg.get("max_price", 80.0))
    min_amount = float(filter_cfg.get("min_amount", 20_000_000.0))

    scoring_cfg = raw.get("scoring", {}) or {}
    top_k = int(scoring_cfg.get("top_k", 4))
    min_prob = float(scoring_cfg.get("min_prob", 0.65))

    pos_cfg = raw.get("position", {}) or {}
    max_positions = int(pos_cfg.get("max_positions", 3))
    position_weight = float(pos_cfg.get("position_weight", 0.35))
    stop_loss = float(pos_cfg.get("stop_loss", 0.05))
    take_profit = float(pos_cfg.get("take_profit", 0.08))

    out_cfg = raw.get("output", {}) or {}
    out_dir = Path(out_cfg.get("dir", "reports/u2_live_batch"))
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    # ---- 若指定 filter_tag，则叠加全局过滤方案 ----
    if filter_tag:
        global_filter = get_u2_filter_config(filter_tag) or {}
        # 只覆盖关键过滤和打分参数，其它（比如持仓权重）沿用本地配置
        min_price = float(global_filter.get("min_price", min_price))
        max_price = float(global_filter.get("max_price", max_price))
        # 这里兼容两种 key：min_amount_today / min_amount
        min_amount = float(
            global_filter.get(
                "min_amount_today", global_filter.get("min_amount", min_amount)
            )
        )
        top_k = int(global_filter.get("top_k", top_k))
        min_prob = float(global_filter.get("min_prob", min_prob))

    return U2Config(
        job_id=job_id,
        dataset_path=ds_path,
        label_col=label_col,
        ret_col=ret_col,
        features=features,
        train_days=train_days,
        min_train_days=min_train_days,
        min_price=min_price,
        max_price=max_price,
        min_amount=min_amount,
        top_k=top_k,
        min_prob=min_prob,
        max_positions=max_positions,
        position_weight=position_weight,
        stop_loss=stop_loss,
        take_profit=take_profit,
        output_dir=out_dir,
        filter_tag=filter_tag,
    )


# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------


def _calc_basic_stats(equity_df: pd.DataFrame) -> Dict[str, float]:
    """根据资金曲线计算一组常用指标。"""
    if equity_df.empty:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_ratio": 0.0,
        }

    n_days = len(equity_df)
    equity = equity_df["equity"].astype(float)
    daily_ret = equity_df["daily_return"].astype(float)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if n_days > 0:
        ann_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    else:
        ann_return = 0.0

    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    cum_max = equity.cummax()
    dd = equity / cum_max - 1.0
    max_drawdown = float(dd.min()) if not dd.empty else 0.0

    win_ratio = float((daily_ret > 0).mean()) if len(daily_ret) > 0 else 0.0

    return {
        "n_days": int(n_days),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_ratio": win_ratio,
    }


# ----------------------------------------------------------------------
# 回测主流程
# ----------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    cfg: U2Config,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    简版滚动回测（和 u2_daily_backtest_v2 同一套逻辑，参数用 U2 实盘配置）。
    """

    df = df.copy()

    # === 关键修复点：兼容 trade_date / trading_date 两种列名 ===
    if "trade_date" not in df.columns:
        if "trading_date" in df.columns:
            print(
                "[U2] 数据集中没有 trade_date 列，发现 trading_date 列，"
                "已自动重命名为 trade_date 用于批量回测……"
            )
            df = df.rename(columns={"trading_date": "trade_date"})
        else:
            raise RuntimeError(
                "数据集中缺少 trade_date 列（也未发现 trading_date 列），"
                "请检查数据集。"
            )

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    start_ts = _to_date(start_date)
    end_ts = _to_date(end_date)

    # 只保留训练 + 回测所需的日期窗口
    min_train_start = start_ts - pd.Timedelta(days=cfg.train_days + 10)
    df = df[(df["trade_date"] >= min_train_start) & (df["trade_date"] <= end_ts)].copy()

    # 防御：缺失特征列补 0
    for col in cfg.features:
        if col not in df.columns:
            df[col] = 0.0

    # 回测日期列表（只看 start~end 范围）
    all_dates = sorted(
        df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)][
            "trade_date"
        ].unique()
    )

    equity_records: List[Tuple[datetime, float, float, int]] = []
    capital = 1.0

    for cur_date in all_dates:
        cur_date = pd.Timestamp(cur_date)

        # ---- 训练窗口 ----
        train_start = cur_date - pd.Timedelta(days=cfg.train_days)
        train_df = df[(df["trade_date"] < cur_date) & (df["trade_date"] >= train_start)]
        n_train_days = train_df["trade_date"].nunique()

        if n_train_days < cfg.min_train_days:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        train_df = train_df.dropna(subset=[cfg.label_col])
        if train_df.empty:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        # 这里默认 label 已经是 0/1
        y_train = train_df[cfg.label_col].astype(int).values
        if len(np.unique(y_train)) < 2:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        X_train = train_df[cfg.features].astype(float).values

        model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            n_jobs=-1,
        )
        try:
            model.fit(X_train, y_train)
        except Exception:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        # ---- 当日选股 ----
        day_df = df[df["trade_date"] == cur_date].copy()
        if day_df.empty:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        # 过滤：价格 / 成交额
        if "close" not in day_df.columns or "amount" not in day_df.columns:
            raise RuntimeError("数据集中缺少 close / amount 列，无法做过滤。")

        day_df = day_df[
            (day_df["close"] >= cfg.min_price)
            & (day_df["close"] <= cfg.max_price)
            & (day_df["amount"] >= cfg.min_amount)
        ]
        if day_df.empty:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        X_day = day_df[cfg.features].astype(float).values
        prob = model.predict_proba(X_day)[:, 1]
        day_df["prob"] = prob

        # 按概率阈值过滤
        day_df = day_df[day_df["prob"] >= cfg.min_prob]
        if day_df.empty:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        # 概率排序 + 控制持仓数量
        day_df = day_df.sort_values("prob", ascending=False)

        # 根据单票权重限制最多持仓数量
        max_pos_by_weight = int(np.floor(1.0 / max(cfg.position_weight, 1e-6)))
        max_positions = max(1, min(cfg.max_positions, max_pos_by_weight))

        day_df = day_df.head(min(cfg.top_k, max_positions))
        n_pos = len(day_df)
        if n_pos == 0:
            equity_records.append((cur_date, capital, 0.0, 0))
            continue

        # 统一的单票权重（如果超配则整体缩放）
        weight = cfg.position_weight
        total_weight = weight * n_pos
        if total_weight > 1.0:
            weight = weight / total_weight
        day_df["weight"] = weight

        if cfg.ret_col not in day_df.columns:
            raise RuntimeError(f"数据集中缺少收益列 {cfg.ret_col}")

        day_ret = float((day_df[cfg.ret_col].astype(float) * day_df["weight"]).sum())
        capital *= 1.0 + day_ret
        equity_records.append((cur_date, capital, day_ret, n_pos))

    equity_df = pd.DataFrame(
        equity_records,
        columns=["trade_date", "equity", "daily_return", "n_positions"],
    )
    if not equity_df.empty:
        equity_df["trade_date"] = pd.to_datetime(equity_df["trade_date"])

    summary_stats = _calc_basic_stats(equity_df)

    # 按年度拆分
    if equity_df.empty:
        yearly_df = pd.DataFrame(
            columns=[
                "year",
                "n_days",
                "total_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "win_ratio",
            ]
        )
    else:
        equity_df["year"] = equity_df["trade_date"].dt.year
        rows = []
        for year, g in equity_df.groupby("year"):
            stats = _calc_basic_stats(g)
            rows.append(
                {
                    "year": int(year),
                    **stats,
                }
            )
        yearly_df = pd.DataFrame(rows)

    return equity_df, yearly_df, summary_stats


# ----------------------------------------------------------------------
# 输出 & CLI
# ----------------------------------------------------------------------


def save_reports(
    cfg: U2Config,
    tag: str,
    equity: pd.DataFrame,
    yearly: pd.DataFrame,
    summary: Dict[str, float],
) -> None:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_file = out_dir / f"u2_batch_equity_{tag}.csv"
    yearly_file = out_dir / f"u2_batch_yearly_{tag}.csv"
    trades_file = out_dir / f"u2_batch_trades_{tag}.csv"
    summary_file = out_dir / f"u2_batch_summary_{tag}.txt"

    equity.to_csv(equity_file, index=False)
    yearly.to_csv(yearly_file, index=False)

    # 目前还没细化逐笔交易，这里给一个空壳，方便后续扩展
    trades_df = pd.DataFrame(
        columns=["trade_date", "code", "prob", "ret", "weight", "pnl"]
    )
    trades_df.to_csv(trades_file, index=False)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"U2 Live 批量回测汇总（tag={tag}）\n")
        if cfg.filter_tag:
            f.write(f"过滤方案 tag : {cfg.filter_tag}\n")
        f.write(f"回测区间     : {summary.get('n_days', 0)} 个交易日\n")
        f.write(f"累计收益     : {summary.get('total_return', 0.0):.2%}\n")
        f.write(f"年化收益     : {summary.get('ann_return', 0.0):.2%}\n")
        f.write(f"年化波动     : {summary.get('ann_vol', 0.0):.2%}\n")
        f.write(f"Sharpe       : {summary.get('sharpe', 0.0):.2f}\n")
        f.write(f"最大回撤     : {summary.get('max_drawdown', 0.0):.2%}\n")
        f.write(f"日度胜率     : {summary.get('win_ratio', 0.0):.2%}\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U2 Live 批量回测（支持 filter_tag AB 测试）")
    parser.add_argument(
        "--config",
        default="config/u2_live_config.json",
        help="U2 实盘配置 json 路径（默认：config/u2_live_config.json）",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="回测开始日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="回测结束日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--tag",
        default="live_check",
        help="输出文件用的 tag（会体现在文件名里）",
    )
    parser.add_argument(
        "--filter-tag",
        default=None,
        help="过滤方案标签（例如 base / aggressive / defensive 等）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg = load_u2_config(args.config, filter_tag=args.filter_tag)

    print(
        f"[U2] 开始批量回测：config={args.config}, "
        f"start={args.start_date}, end={args.end_date}, tag={args.tag}"
    )
    if cfg.filter_tag:
        print(f"[U2] 使用过滤方案 tag = {cfg.filter_tag}")

    print(f"[U2] 读取数据集: {cfg.dataset_path}")
    df = pd.read_parquet(cfg.dataset_path)

    equity, yearly, summary = run_backtest(df, cfg, args.start_date, args.end_date)
    save_reports(cfg, args.tag, equity, yearly, summary)

    print("[U2] 批量回测完成。")


if __name__ == "__main__":
    main()
