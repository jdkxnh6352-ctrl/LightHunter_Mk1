# -*- coding: utf-8 -*-
"""
tools/build_ultrashort_dataset.py

从 ts_data.db.snapshots 自动构建 U2 超短策略训练集：
- 一行 = (trade_date, code)
- 特征列以 F_ 开头
- 标签列 = ULTRA_T1_DIR_3C  (T+1 收盘相对当日收盘的 3 分类方向)

生成文件：
    data/datasets/ultrashort_main/ultrashort_main.parquet
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger

log = get_logger("BuildU2Dataset")

# tools/.. -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------ 小工具：日期处理 ------------------ #

def _parse_date(s: str) -> str:
    """
    把用户输入的 YYYY-MM-DD 字符串标准化一下（内部都用字符串比较）。
    """
    d = datetime.strptime(s, "%Y-%m-%d").date()
    return d.strftime("%Y-%m-%d")


def _list_trade_dates(conn: sqlite3.Connection) -> List[str]:
    """
    从 ts_data.db.snapshots 里列出所有有数据的交易日（YYYY-MM-DD）。
    """
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT substr(ts, 1, 10) AS d FROM snapshots ORDER BY d")
    rows = cur.fetchall()
    return [r[0] for r in rows if r and r[0]]


def _filter_dates(
    dates: Sequence[str],
    start_date: str | None,
    end_date: str | None,
) -> List[str]:
    out = []
    for d in dates:
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        out.append(d)
    return out


# ------------------ 第一步：日线聚合 ------------------ #

def _build_daily_from_snapshots(
    conn: sqlite3.Connection,
    dates: Sequence[str],
) -> pd.DataFrame:
    """
    从 snapshots 按天聚合成简单的日线：
    - close: 当天最后一条快照价格
    - pct_close: 当天最后一条 pct
    - amount_sum: 当天成交额总和
    - turnover_end: 当天最后一条换手率
    """
    frames: list[pd.DataFrame] = []
    for d in dates:
        log.info("读取分时数据: %s", d)
        df = pd.read_sql_query(
            """
            SELECT ts, code, price, pct, amount, turnover_rate
            FROM snapshots
            WHERE substr(ts, 1, 10) = ?
            ORDER BY code, ts
            """,
            conn,
            params=(d,),
        )
        if df.empty:
            continue

        df["ts"] = pd.to_datetime(df["ts"])
        df["trade_date"] = df["ts"].dt.strftime("%Y-%m-%d")

        agg = (
            df.groupby(["trade_date", "code"])
            .agg(
                close=("price", "last"),
                pct_close=("pct", "last"),
                amount_sum=("amount", "sum"),
                turnover_end=("turnover_rate", "last"),
            )
            .reset_index()
        )
        frames.append(agg)

    if not frames:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "code",
                "close",
                "pct_close",
                "amount_sum",
                "turnover_end",
            ]
        )

    daily = pd.concat(frames, ignore_index=True)
    return daily


# ------------------ 第二步：算特征 + 标签 ------------------ #

def _add_features_and_label(daily: pd.DataFrame) -> pd.DataFrame:
    """
    对每日每只股票：
    - 根据 close / amount_sum / turnover_end 算出 F_ 开头的特征；
    - 用 T+1 收盘涨跌幅做 3 分类标签 ULTRA_T1_DIR_3C。
    """
    if daily.empty:
        return daily

    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["code", "trade_date"])

    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("trade_date").copy()

        # 简单动量特征：过去 N 日收盘涨跌幅
        g["F_RET_1D"] = g["close"].pct_change(1)
        g["F_RET_3D"] = g["close"].pct_change(3)
        g["F_RET_5D"] = g["close"].pct_change(5)

        # 过去 3 日换手率 & 成交额均值（做个 log1p 缩放）
        g["F_TURN_3D"] = g["turnover_end"].rolling(3, min_periods=1).mean()
        g["F_AMT_3D"] = np.log1p(
            g["amount_sum"].rolling(3, min_periods=1).mean()
        )

        # T+1 收盘涨跌幅
        future_close = g["close"].shift(-1)
        future_ret = (future_close - g["close"]) / g["close"]

        # 3 分类标签：上涨 >= +3% 为 1，下跌 <= -3% 为 -1，其余为 0
        th = 0.03
        label = np.where(
            future_ret >= th,
            1,
            np.where(future_ret <= -th, -1, 0),
        )
        g["ULTRA_T1_DIR_3C"] = label

        return g

    df = df.groupby("code", group_keys=False).apply(_per_code)

    # 去掉没有标签的样本（最后一天没有 T+1）
    df = df.dropna(subset=["ULTRA_T1_DIR_3C"])

    # 去掉含 NaN 的特征
    feature_cols = [c for c in df.columns if c.startswith("F_")]
    df = df.dropna(subset=feature_cols)

    # trade_date 再转回 YYYY-MM-DD 字符串，方便后面按字符串切分
    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")

    keep_cols = ["trade_date", "code"] + feature_cols + ["ULTRA_T1_DIR_3C"]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ------------------ 路径解析 ------------------ #

def _resolve_paths_from_config() -> Tuple[Path, Path]:
    """
    从 system_config 里解析 ts_data.db 和 数据集输出目录。
    """
    cfg = get_system_config(refresh=False)
    paths_cfg = cfg.get("paths", {}) or {}

    db_root = paths_cfg.get("db_root", "data/db")
    ts_db_name = paths_cfg.get("ts_db", "ts_data.db")
    dataset_root = paths_cfg.get("dataset_dir", "data/datasets")

    ts_db_path = (PROJECT_ROOT / db_root / ts_db_name).resolve()
    ds_root = (PROJECT_ROOT / dataset_root).resolve()
    out_dir = ds_root / "ultrashort_main"

    return ts_db_path, out_dir


# ------------------ 主流程 ------------------ #

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="从 ts_data.db 构建 U2 超短策略训练集 ultrashort_main"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期 YYYY-MM-DD（默认=数据库最早日期）",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期 YYYY-MM-DD（默认=数据库最晚日期）",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    ts_db_path, out_dir = _resolve_paths_from_config()

    if not ts_db_path.exists():
        log.error("未找到 ts_data.db: %s", ts_db_path)
        print("请确认 data/db 目录下已经有 ts_data.db。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ultrashort_main.parquet"

    log.info("使用 ts_data.db: %s", ts_db_path)
    log.info("输出数据集: %s", out_path)

    conn = sqlite3.connect(str(ts_db_path))
    try:
        all_dates = _list_trade_dates(conn)
        if not all_dates:
            log.error("snapshots 表为空，无法构建数据集。")
            return

        start_date = _parse_date(args.start_date) if args.start_date else all_dates[0]
        end_date = _parse_date(args.end_date) if args.end_date else all_dates[-1]

        dates = _filter_dates(all_dates, start_date, end_date)
        if not dates:
            log.error("筛选后的日期区间为空，请检查 start/end 参数。")
            return

        log.info("将使用 %d 个交易日：%s ... %s", len(dates), dates[0], dates[-1])

        daily = _build_daily_from_snapshots(conn, dates)
    finally:
        conn.close()

    if daily.empty:
        log.error("日线聚合结果为空，无法构建数据集。")
        return

    dataset = _add_features_and_label(daily)
    if dataset.empty:
        log.error("特征+标签结果为空，可能阈值过高或数据质量有问题。")
        return

    dataset.to_parquet(out_path, index=False)
    log.info("已写入数据集：%s 行", len(dataset))
    print(
        f"[U2DATA] Done. rows={len(dataset)}, "
        f"from {dataset['trade_date'].min()} to {dataset['trade_date'].max()}"
    )


if __name__ == "__main__":
    main()
