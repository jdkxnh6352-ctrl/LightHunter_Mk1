#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 / 更新 ultrashort_main 数据集的工具脚本。

用法示例：
    python build_ultrashort_main_dataset.py --job-id ultrashort_main

逻辑说明：
1. 默认假设原始日线数据 CSV 为：
       data/{job_id}_raw.csv
   输出 parquet 数据集为：
       data/datasets/{job_id}.parquet

2. 如果找到 raw CSV：
   - 读入 CSV，并自动识别日期列（优先 trade_date / date / as_of）。
   - 如果已有旧 parquet，则与之合并去重（按 [code, 日期列] 去重），
     再按日期排序后写回 parquet。
   - 如果没有旧 parquet，就直接用 CSV 构建。

3. 如果找不到 raw CSV：
   - 如果已有旧 parquet：给出 [WARN]，保留旧数据集，不报错退出；
   - 如果连旧 parquet 都没有：给出 [ERROR]，并抛异常退出。
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def log_build(msg: str) -> None:
    print(f"[BUILD] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    """在常见列名里自动找日期列。"""
    candidates = ["trade_date", "date", "as_of"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_dataset(job_id: str) -> None:
    print(f"==== 构建 {job_id} 数据集 ====")

    data_dir = Path("data")
    raw_csv = data_dir / f"{job_id}_raw.csv"
    ds_dir = data_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ds_dir / f"{job_id}.parquet"

    log_build(f"job_id           : {job_id}")
    log_build(f"原始 CSV 路径    : {raw_csv}")
    log_build(f"输出数据集路径   : {ds_path}")

    # 情况 1：存在 raw CSV，执行增量 / 全量构建
    if raw_csv.exists():
        log_build("发现原始 CSV 文件，开始读取并构建数据集……")

        df_raw = pd.read_csv(raw_csv)
        date_col = detect_date_col(df_raw)
        if date_col is None:
            raise ValueError(
                "在原始 CSV 中未找到日期列（期待列名之一：trade_date / date / as_of）。"
            )

        # 规范日期类型
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])

        # 如果有 code 列，尽量保证是字符串
        if "code" in df_raw.columns:
            df_raw["code"] = df_raw["code"].astype(str)

        # 如果存在历史 parquet，则做增量合并
        if ds_path.exists():
            log_build("检测到已有历史 parquet 数据集，正在与 raw CSV 做增量合并……")
            df_old = pd.read_parquet(ds_path)

            # 尽量保证旧数据类型一致
            if date_col in df_old.columns:
                df_old[date_col] = pd.to_datetime(df_old[date_col])
            if "code" in df_old.columns:
                df_old["code"] = df_old["code"].astype(str)

            # 合并去重
            if "code" in df_raw.columns and "code" in df_old.columns:
                subset_cols = ["code", date_col]
            else:
                subset_cols = [date_col]

            df_all = pd.concat([df_old, df_raw], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=subset_cols)
            df_all = df_all.sort_values(date_col).reset_index(drop=True)
        else:
            log_build("未检测到历史 parquet 数据集，本次将直接用 raw CSV 构建。")
            df_all = df_raw.sort_values(date_col).reset_index(drop=True)

        # 写回 parquet
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(ds_path, index=False)

        last_trade = df_all[date_col].max()
        log_build(f"数据集构建完成，共 {len(df_all)} 行，"
                  f"最新交易日: {last_trade:%Y-%m-%d}")

    # 情况 2：没有 raw CSV，尝试沿用旧 parquet
    else:
        if ds_path.exists():
            log_warn(
                f"找不到原始数据文件: {raw_csv}，但已经存在历史数据集: {ds_path}"
            )
            log_warn("当前先跳过数据集构建，沿用已有数据。")
            log_warn("等每日原始数据抓取脚本就绪后，再从 raw CSV 构建 / 更新数据集。")
        else:
            log_error(f"既找不到原始数据文件: {raw_csv}，也找不到历史数据集: {ds_path}")
            log_error("无法构建数据集，请先准备好原始 CSV 再重试。")
            # 抛异常让上层流程感知失败
            raise FileNotFoundError(
                f"missing raw csv ({raw_csv}) and dataset ({ds_path})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 / 更新 ultrashort_main 数据集"
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="任务 ID，例如 ultrashort_main",
    )
    args = parser.parse_args()
    build_dataset(args.job_id)


if __name__ == "__main__":
    main()
