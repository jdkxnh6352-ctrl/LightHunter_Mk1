# -*- coding: utf-8 -*-
"""
build_ultrashort_main_dataset.py

作用：
- 从 data/ultrashort_raw.csv 这样的日线行情文件中
  构建一个给 U2 / ultrashort_main 用的训练数据集；
- 输出到 data/datasets/ultrashort_main.parquet。

输入 CSV 要包含至少：
    - 日期: date 或 trade_date
    - 代码: code 或 symbol
    - 收盘价: close
    - 成交量: volume

标签:
    y_ret_1d_close = 下一交易日收盘相对当前收盘收益率 (float)

特征:
    F_ret_1d_prev  = 当前日相对上一交易日收益率
    F_vol_norm     = 当前日成交量 / 过去20日平均成交量
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"原始数据缺少列，至少需要其中一个: {candidates!r}。当前列: {list(df.columns)!r}")


def build_dataset(
    raw_csv: str = "data/ultrashort_raw.csv",
    output_path: str = "data/datasets/ultrashort_main.parquet",
) -> None:
    raw_path = Path(raw_csv)
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到原始数据文件: {raw_path}")

    print(f"[BUILD] 读取原始数据: {raw_path}")
    df = pd.read_csv(raw_path)

    # 1) 统一列名
    date_col = _find_col(df, ["trade_date", "date", "交易日期"])
    code_col = _find_col(df, ["symbol", "code", "证券代码", "股票代码"])
    close_col = _find_col(df, ["close", "收盘价"])
    vol_col = _find_col(df, ["volume", "vol", "成交量"])

    df = df.rename(
        columns={
            date_col: "trade_date",
            code_col: "symbol",
            close_col: "close",
            vol_col: "volume",
        }
    )

    # 转日期、排序
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # 只保留我们要用的几列（防止奇怪列干扰）
    keep_cols = ["trade_date", "symbol", "close", "volume"]
    extra_cols = [c for c in df.columns if c not in keep_cols]
    if extra_cols:
        print(f"[BUILD] 原始文件中额外列会被忽略: {extra_cols}")
    df = df[keep_cols]

    # 2) 计算特征
    # 当前日相对于前一交易日的收益率
    df["F_ret_1d_prev"] = (
        df.groupby("symbol")["close"].pct_change().astype("float32")
    )

    # 成交量 / 过去20日均量
    rolling_vol = (
        df.groupby("symbol")["volume"]
        .rolling(window=20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["F_vol_norm"] = (df["volume"] / rolling_vol).astype("float32")

    # 3) 构造标签：下一交易日收盘收益率
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["y_ret_1d_close"] = (
        (next_close - df["close"]) / df["close"]
    ).astype("float32")

    # 去掉一开始和最后一天等算不出特征/标签的行
    df = df.dropna(subset=["F_ret_1d_prev", "F_vol_norm", "y_ret_1d_close"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("构建后的数据集为空，请检查原始数据是否太少，或日期顺序是否正确。")

    # 最终输出列顺序
    out_cols = [
        "trade_date",
        "symbol",
        "F_ret_1d_prev",
        "F_vol_norm",
        "y_ret_1d_close",
    ]
    df = df[out_cols]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"[BUILD] 数据集已写入: {out_path}  行数={len(df)} 列数={len(df.columns)}")
    print("[BUILD] 列: ", out_cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 ultrashort_main/U2 用训练数据集"
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="data/ultrashort_raw.csv",
        help="原始日线行情 CSV 路径（默认: data/ultrashort_raw.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/ultrashort_main.parquet",
        help="输出数据集路径（默认: data/datasets/ultrashort_main.parquet）",
    )
    args = parser.parse_args()
    build_dataset(args.raw_csv, args.output)


if __name__ == "__main__":
    main()
