#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
u0_data_gap_check.py

功能：
- 对指定 Parquet 数据集做“缺口 & 质量”体检：
  1）行数、股票数、日期范围
  2）每个交易日的样本量（是否有异常少的日子）
  3）每只股票在全局交易日上的“缺口”统计
  4）各列缺失值比例
  5）对“收益类列”做极端值扫描

输出：
- 一份 markdown 报告：reports/u0_data_gap_<job_id>.md

用法示例（在项目根目录下）：
    python tools/u0_data_gap_check.py --job-id ultrashort_main

    # 如果你有别的 parquet：
    python tools/u0_data_gap_check.py --job-id my_ds --dataset-path data/my_ds.parquet
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def detect_date_col(df: pd.DataFrame) -> str:
    """
    自动检测日期列名。

    优先顺序：
    1) trade_date
    2) trading_date（会自动重命名为 trade_date，和其它脚本保持一致）
    3) date
    4) as_of
    5) 兜底：任意 datetime 类型列（取第一列）
    """
    # 先看是否已有规范化列
    if "trade_date" in df.columns:
        return "trade_date"

    # 兼容 trading_date：发现后自动重命名
    if "trading_date" in df.columns:
        if "trade_date" not in df.columns:
            df.rename(columns={"trading_date": "trade_date"}, inplace=True)
        return "trade_date"

    # 其它常见名字
    for col in ["date", "as_of"]:
        if col in df.columns:
            return col

    # 兜底：找 datetime 类型列
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    raise ValueError(
        "未找到日期列（期望列名之一：trade_date / trading_date / date / as_of，"
        "且未发现 datetime 类型列）"
    )


def detect_symbol_col(df: pd.DataFrame) -> str:
    """自动检测股票代码列名：symbol / ts_code / code"""
    for col in ["symbol", "ts_code", "code"]:
        if col in df.columns:
            return col
    raise ValueError("未找到股票代码列（期望列名之一：symbol / ts_code / code）")


def find_return_cols(df: pd.DataFrame):
    """简单猜一下哪些是收益/涨跌幅列"""
    candidates = []
    for col in df.columns:
        lc = col.lower()
        if any(key in lc for key in ["ret", "return", "pct_chg", "chg_pct", "收益", "涨跌"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="数据缺口体检脚本")
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="数据集标识，用于命名报告，例如 ultrashort_main",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="parquet 路径，默认 data/<job_id>.parquet",
    )
    args = parser.parse_args()

    job_id = args.job_id
    dataset_path = Path(args.dataset_path) if args.dataset_path else DATA_DIR / f"{job_id}.parquet"

    if not dataset_path.exists():
        print(f"[ERROR] 找不到数据文件：{dataset_path}")
        return

    print(f"[INFO] 读取数据集：{dataset_path}")
    df = pd.read_parquet(dataset_path)

    n_rows = len(df)
    n_cols = df.shape[1]
    print(f"[INFO] 行数：{n_rows}，列数：{n_cols}")

    # 检测列
    date_col = detect_date_col(df)
    symbol_col = detect_symbol_col(df)

    # 转换日期
    print(f"[INFO] 使用日期列：{date_col}，股票列：{symbol_col}")
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # 基本信息
    unique_dates = np.sort(df[date_col].unique())
    unique_symbols = df[symbol_col].unique()

    first_date, last_date = unique_dates[0], unique_dates[-1]

    # 每日样本量
    daily_counts = df.groupby(date_col)[symbol_col].nunique().reset_index(name="n_symbols")

    # 缺口统计：以全局日期集合为基准
    symbol_gaps = []

    print("[INFO] 统计每只股票的日期缺口（可能比较慢，大约几秒钟）.")
    for sym, g in df.groupby(symbol_col):
        dates = sorted(g[date_col].unique())
        if len(dates) == 0:
            continue
        sym_start, sym_end = dates[0], dates[-1]
        # 在全局日期范围中，本票应该出现的所有日期
        expected_days = [d for d in unique_dates if sym_start <= d <= sym_end]
        missing = [d for d in expected_days if d not in set(dates)]
        if missing:
            symbol_gaps.append((sym, len(missing), sym_start, sym_end))

    symbol_gaps_sorted = sorted(symbol_gaps, key=lambda x: x[1], reverse=True)

    # 缺失值统计
    na_ratio = df.isna().mean().sort_values(ascending=False)

    # 收益类列极端值
    ret_cols = find_return_cols(df)
    ret_extreme_stats = {}
    for col in ret_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q01, q05, q50, q95, q99 = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        max_v, min_v = s.max(), s.min()
        ret_extreme_stats[col] = {
            "q01": q01,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "q99": q99,
            "max": max_v,
            "min": min_v,
            "count": len(s),
        }

    # 生成报告
    report_path = REPORT_DIR / f"u0_data_gap_{job_id}.md"
    print(f"[INFO] 写入报告：{report_path}")

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# 数据缺口体检报告 - {job_id}\n\n")
        f.write(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 数据文件：`{dataset_path}`\n")
        f.write(f"- 行数：{n_rows}\n")
        f.write(f"- 列数：{n_cols}\n")
        f.write(f"- 股票数：{len(unique_symbols)}\n")
        f.write(f"- 交易日数：{len(unique_dates)}（{first_date} ~ {last_date}）\n\n")

        # 每日样本量
        f.write("## 按交易日的样本量（前 10 条 & 后 10 条）\n\n")
        f.write("| 日期 | 股票数 |\n|------|--------|\n")
        for _, row in daily_counts.head(10).iterrows():
            f.write(f"| {row[date_col]} | {row['n_symbols']} |\n")
        f.write("\n. .（中间省略）. .\n\n")
        for _, row in daily_counts.tail(10).iterrows():
            f.write(f"| {row[date_col]} | {row['n_symbols']} |\n")
        f.write("\n")

        # 股票缺口 Top
        f.write("## 每只股票的日期缺口统计（缺口最多的前 20 只）\n\n")
        if not symbol_gaps_sorted:
            f.write("> 未发现明显日期缺口（以全局交易日日历为准）。\n\n")
        else:
            f.write("| 股票 | 缺失天数 | 起始日期 | 结束日期 |\n|------|----------|----------|----------|\n")
            for sym, n_miss, s_date, e_date in symbol_gaps_sorted[:20]:
                f.write(f"| {sym} | {n_miss} | {s_date} | {e_date} |\n")
            f.write("\n")

        # 缺失值比例
        f.write("## 各列缺失值比例（Top 30）\n\n")
        f.write("| 列名 | 缺失比例 |\n|------|----------|\n")
        for col, ratio in na_ratio.head(30).items():
            f.write(f"| {col} | {ratio:.2%} |\n")
        f.write("\n")

        # 收益列极端值
        f.write("## 收益类列的极端值统计\n\n")
        if not ret_extreme_stats:
            f.write("> 未自动识别到收益类列（包含 ret/return/pct_chg/涨跌 的列名）。\n\n")
        else:
            for col, stats in ret_extreme_stats.items():
                f.write(f"### 列：`{col}`（样本数：{stats['count']}）\n\n")
                f.write(
                    f"- 分位数：1%={stats['q01']:.4f}，5%={stats['q05']:.4f}，"
                    f"50%={stats['q50']:.4f}，95%={stats['q95']:.4f}，99%={stats['q99']:.4f}\n"
                )
                f.write(f"- 极值：min={stats['min']:.4f}，max={stats['max']:.4f}\n\n")

        f.write("---\n\n")
        f.write("> 说明：\n")
        f.write("> - 日期缺口是以**全局交易日日历**为基准（即所有股票出现过的日期集合），\n")
        f.write(">   某只股票在其起止日期范围内缺少的交易日，可能是停牌，也可能是数据缺失。\n")
        f.write("> - 后续我们可以针对缺口大的股票/日期，写专门的“补数脚本”。\n")

    print("[OK] 数据缺口体检完成。请用 VSCode / 记事本 打开 reports 目录下的 markdown 报告查看。")


if __name__ == "__main__":
    main()
