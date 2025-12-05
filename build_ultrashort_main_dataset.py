#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_ultrashort_main_dataset.py

作用：
- 根据 job_id = ultrashort_main，构建 / 更新主回测用的数据集 parquet 文件
- 现在支持两种情况：
  1）存在 data/ultrashort_main_raw.csv → 从 raw CSV 构建 / 更新 parquet
  2）不存在 raw CSV 但已存在 data/datasets/ultrashort_main.parquet → 打印警告，跳过更新，沿用旧数据

后面等我们把「原始数据抓取」做好，只要每天把增量日线数据 append 到
data/ultrashort_main_raw.csv，再跑这个脚本，就会自动更新数据集。
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def detect_date_col(df: pd.DataFrame) -> str:
    """
    自动检测日期列名，支持：
    - trade_date
    - date
    - as_of
    """
    for col in ["trade_date", "date", "as_of"]:
        if col in df.columns:
            return col
    raise ValueError("原始数据中找不到日期列（期望列名之一：trade_date / date / as_of）")


def build_dataset(raw_csv: Path, output: Path) -> None:
    print(f"[BUILD] 原始数据 CSV 路径: {raw_csv}")
    if not raw_csv.exists():
        raise FileNotFoundError(f"找不到原始数据文件: {raw_csv}")

    df = pd.read_csv(raw_csv)

    # 自动找日期列并统一成 trade_date
    date_col = detect_date_col(df)
    if date_col != "trade_date":
        df = df.rename(columns={date_col: "trade_date"})

    # 统一日期格式
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # 去重、按日期排序（根据你现有数据结构可以再细化）
    df = df.drop_duplicates().sort_values(["trade_date"])

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[BUILD] 写出数据集到: {output}  行数: {len(df):,}")
    df.to_parquet(output, index=False)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        required=True,
        help="数据集 ID，例如 ultrashort_main",
    )
    parser.add_argument(
        "--raw-csv",
        default=None,
        help="可选：指定原始 CSV 路径；默认使用 data/{job_id}_raw.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：指定输出 parquet 路径；默认使用 data/datasets/{job_id}.parquet",
    )

    args = parser.parse_args(argv)

    job_id = args.job_id
    data_dir = Path("data")
    datasets_dir = data_dir / "datasets"

    raw_csv = Path(args.raw_csv) if args.raw_csv else data_dir / f"{job_id}_raw.csv"
    output = Path(args.output) if args.output else datasets_dir / f"{job_id}.parquet"

    print("===== 构建 ultrashort_main 数据集 =====")
    print(f"[BUILD] job_id         : {job_id}")
    print(f"[BUILD] 原始 CSV 路径  : {raw_csv}")
    print(f"[BUILD] 输出数据集路径 : {output}")

    if not raw_csv.exists():
        # 没有原始 CSV 的情况
        if output.exists():
            print(
                f"[WARN] 找不到原始数据文件: {raw_csv}，"
                f"但已经存在历史数据集: {output}"
            )
            print("[WARN] 当前先跳过数据集构建，沿用已有数据。")
            print("[WARN] 等接好原始数据抓取后，再启用从 raw CSV 构建数据集的逻辑。")
            return
        else:
            # 连旧数据集都没有，只能报错
            raise FileNotFoundError(
                f"既找不到原始数据文件 {raw_csv}，也不存在历史数据集 {output}，"
                f"请先准备一份原始 CSV 或手动放置初始 parquet 数据集。"
            )

    # 正常从 raw CSV 构建 / 更新数据集
    build_dataset(raw_csv, output)
    print("[BUILD] 数据集构建完成。")


if __name__ == "__main__":
    main()
