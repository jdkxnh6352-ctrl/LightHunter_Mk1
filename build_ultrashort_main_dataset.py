#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 ultrashort_main 数据集

用法示例：
    python build_ultrashort_main_dataset.py --job-id ultrashort_main

逻辑说明：
1. 优先寻找最新的原始 CSV：
   - data/ultrashort_main_raw.csv
   - data/ultrashort_raw.csv
   - data/raw/ultrashort_main_raw.csv
   - data/raw/ultrashort_raw.csv
   找到第一个存在的就用它，否则报错退出。

2. 读取 CSV，规范日期列（trade_date / date / as_of 三选一）为字符串 'YYYYMMDD'，
   按 [日期, code] 排序，然后写成 parquet。

3. 为了兼容旧脚本，同时写两个输出：
   - data/ultrashort_main.parquet
   - data/datasets/ultrashort_main.parquet
   这样无论别的工具读哪一个，都是最新数据。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def find_raw_csv(job_id: str) -> Path | None:
    """按优先级查找原始 CSV 文件。"""
    candidates: list[Path] = [
        DATA_DIR / f"{job_id}_raw.csv",
        DATA_DIR / "ultrashort_main_raw.csv",
        DATA_DIR / "ultrashort_raw.csv",
        DATA_DIR / "raw" / f"{job_id}_raw.csv",
        DATA_DIR / "raw" / "ultrashort_main_raw.csv",
        DATA_DIR / "raw" / "ultrashort_raw.csv",
    ]

    seen = []
    for p in candidates:
        if p in seen:
            continue
        seen.append(p)
        if p.exists():
            return p

    return None


def detect_date_column(df: pd.DataFrame) -> str:
    """从 df 中自动找出日期列名称。"""
    for col in ("trade_date", "date", "as_of"):
        if col in df.columns:
            return col
    raise ValueError("在原始 CSV 中没有找到日期列（期望列名之一：trade_date / date / as_of）")


def build_dataset(job_id: str) -> None:
    print("==== 构建 ultrashort_main 数据集 ====")
    print(f"[BUILD] job_id        : {job_id}")

    raw_csv = find_raw_csv(job_id)
    if raw_csv is None:
        # 旧数据集路径（看有没有老 parquet 可以沿用）
        legacy_parquet1 = DATA_DIR / f"{job_id}.parquet"
        legacy_parquet2 = DATA_DIR / "datasets" / f"{job_id}.parquet"

        has_legacy = legacy_parquet1.exists() or legacy_parquet2.exists()
        if has_legacy:
            print(
                "[ERROR] 找不到任何原始 CSV（例如 data/ultrashort_main_raw.csv 或 data/ultrashort_raw.csv），"
                "但已经存在旧的历史数据集。"
            )
            print(
                "[ERROR] 要想更新到最新行情，请先准备 / 更新原始 CSV，"
                "然后重新运行 build_ultrashort_main_dataset.py。"
            )
        else:
            print(
                "[ERROR] 找不到任何原始 CSV（例如 data/ultrashort_main_raw.csv 或 data/ultrashort_raw.csv），"
                "且也没有现成的 parquet 数据集。"
            )
            print("[ERROR] 请先通过你的行情抓取脚本生成原始 CSV。")
        sys.exit(1)

    print(f"[BUILD] 原始 CSV 路径 : {raw_csv}")

    # 输出路径：同时写两个位置，兼容旧脚本
    out_parquet_main = DATA_DIR / f"{job_id}.parquet"
    out_parquet_ds = DATA_DIR / "datasets" / f"{job_id}.parquet"
    out_paths = [out_parquet_main, out_parquet_ds]

    print(f"[BUILD] 输出数据集路径 : {out_parquet_main}")
    print(f"[BUILD] 兼容输出路径 : {out_parquet_ds}")

    # 读取 CSV
    df = pd.read_csv(raw_csv)

    # 规范日期列
    date_col = detect_date_column(df)
    print(f"[BUILD] 使用日期列   : {date_col}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d")

    # 简单排序 & 去重，保证顺序稳定
    sort_cols = [date_col]
    if "code" in df.columns:
        sort_cols.append("code")
    df = df.sort_values(sort_cols).drop_duplicates()

    # 保证输出目录存在
    out_parquet_ds.parent.mkdir(parents=True, exist_ok=True)

    # 写 parquet 到两个位置
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        print(f"[BUILD] 已写入: {p}  （{len(df)} 条记录）")

    print("[BUILD] 数据集构建完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 ultrashort_main 数据集")
    parser.add_argument(
        "--job-id",
        required=True,
        help="数据集 job_id，例如 ultrashort_main",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.job_id)


if __name__ == "__main__":
    main()
