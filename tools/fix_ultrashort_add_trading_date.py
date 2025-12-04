# -*- coding: utf-8 -*-
"""
给 ultrashort_main 数据集补一列 trading_date，并写回 parquet/csv。

用法：
(venv) G:\LightHunter_Mk1> python tools\fix_ultrashort_add_trading_date.py
"""

import pandas as pd
from pathlib import Path

# 以当前项目根目录为工作目录运行即可
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "datasets"

parquet_path = DATASET_DIR / "ultrashort_main.parquet"
csv_path = DATASET_DIR / "ultrashort_main.csv"

print("[FIX] 读取数据集:", parquet_path)
df = pd.read_parquet(parquet_path)
print("[FIX] 原始列:", list(df.columns))

# 如果已经有 trading_date，就不重复加了
if "trading_date" not in df.columns:
    # 优先从 datetime 派生，没有就看有没有 date 列
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    else:
        raise RuntimeError("找不到 datetime / date 列，无法派生 trading_date。")

    # 只保留日期部分，比如 2020-01-02
    df["trading_date"] = dt.dt.date
    print("[FIX] 已根据 datetime/date 新增 trading_date 列。")
else:
    print("[FIX] 已存在 trading_date 列，无需新增。")

# 按日期、代码排序一下，顺手 reset_index
df = df.sort_values(["trading_date", "code"]).reset_index(drop=True)

# 写回 parquet & csv
df.to_parquet(parquet_path, index=False)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("[FIX] 已写回 parquet:", parquet_path)
print("[FIX] 已写回 csv    :", csv_path)
print("[FIX] 最终行数      :", len(df))
