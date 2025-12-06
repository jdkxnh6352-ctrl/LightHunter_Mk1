# tools/u1_features_v2_plus.py
"""
U1 v2_plus 因子包：在原始日线数据基础上，计算一批更丰富的价格 / 成交量因子。

使用方式（示例）：
    # 就地在原数据上加列
    python -m tools.u1_features_v2_plus \
        --input  data/datasets/ultrashort_main.parquet

    # 或者生成一个新的 v2_plus 数据集
    python -m tools.u1_features_v2_plus \
        --input  data/datasets/ultrashort_main.parquet \
        --output data/datasets/ultrashort_main_v2_plus.parquet
"""

import argparse
import numpy as np
import pandas as pd


# === 这份列表可以直接用到模型里作为 feature_cols ===
V2_PLUS_FEATURES = [
    # 原来 v1 用到的基础特征
    "amount",
    "vol_20",
    "ret_1",
    "ret_5",
    "ret_20",

    # v2_plus 新特征（全部以 u2_ 开头，避免与旧列冲突）
    "u2_range_pct",          # 当日振幅 (high-low)/close
    "u2_oc_ret",             # 当日开收盘收益 (close-open)/open
    "u2_gap_ret",            # 当日开盘缺口 (open - 前一日close)/前一日close
    "u2_high_close_gap",     # (high-close)/close
    "u2_close_low_gap",      # (close-low)/close

    "u2_mom_3",              # 3 日收盘价动量
    "u2_mom_10",             # 10 日收盘价动量

    "u2_ret_std_5",          # ret_1 的 5 日波动率（未年化）
    "u2_ret_std_10",         # ret_1 的 10 日波动率
    "u2_ret_std_20",         # ret_1 的 20 日波动率

    "u2_amt_z_5",            # 成交额 5 日 Z 分数
    "u2_amt_z_20",           # 成交额 20 日 Z 分数
    "u2_vol_z_5",            # 成交量 5 日 Z 分数
    "u2_vol_z_20",           # 成交量 20 日 Z 分数

    "u2_down_vs_up_vol_20",  # 20 日“下跌波动/上涨波动”比值
]


def _check_required_columns(df: pd.DataFrame) -> None:
    required = [
        "code",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "ret_1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据集中缺少必须列：{missing}")


def add_u1_v2_plus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在传入的 DataFrame 上直接增加 v2_plus 因子列（就地修改），并返回 df 本身。

    约定：
        - 行粒度：日线
        - 每行一只股票一个交易日
        - code + trading_date 唯一
    """
    _check_required_columns(df)

    # 按股票 + 日期排序，保证 rolling / shift 结果正确
    df.sort_values(["code", "trading_date"], inplace=True)
    g = df.groupby("code", group_keys=False)

    close_safe = df["close"].replace(0, np.nan)
    open_safe = df["open"].replace(0, np.nan)

    # === 1. 当日 K 线形态相关 ===
    df["u2_range_pct"] = (df["high"] - df["low"]) / close_safe
    df["u2_oc_ret"] = (df["close"] - df["open"]) / open_safe
    df["u2_high_close_gap"] = (df["high"] - df["close"]) / close_safe
    df["u2_close_low_gap"] = (df["close"] - df["low"]) / close_safe

    # === 2. 与前一日收盘相关的“缺口因子” ===
    prev_close = g["close"].shift(1)
    df["u2_gap_ret"] = (df["open"] - prev_close) / prev_close

    # === 3. 价量动量 / 波动率 ===
    df["u2_mom_3"] = g["close"].pct_change(3)
    df["u2_mom_10"] = g["close"].pct_change(10)

    for win in (5, 10, 20):
        df[f"u2_ret_std_{win}"] = g["ret_1"].transform(
            lambda x: x.rolling(win).std()
        )

    # === 4. 成交额 / 成交量 相对强度（Z 分数） ===
    for win in (5, 20):
        mean_amt = g["amount"].transform(lambda x: x.rolling(win).mean())
        std_amt = g["amount"].transform(lambda x: x.rolling(win).std())
        df[f"u2_amt_z_{win}"] = (df["amount"] - mean_amt) / std_amt

        mean_vol = g["volume"].transform(lambda x: x.rolling(win).mean())
        std_vol = g["volume"].transform(lambda x: x.rolling(win).std())
        df[f"u2_vol_z_{win}"] = (df["volume"] - mean_vol) / std_vol

    # === 5. 下跌波动 / 上涨波动 比值（看风格更偏追涨还是抄底） ===
    def _down_vs_up_vol(x: pd.Series, win: int = 20) -> pd.Series:
        neg = x.where(x < 0.0, 0.0)
        pos = x.where(x > 0.0, 0.0)
        neg_std = neg.rolling(win).std()
        pos_std = pos.rolling(win).std()
        return neg_std / pos_std

    df["u2_down_vs_up_vol_20"] = g["ret_1"].transform(_down_vs_up_vol)

    # === NA 处理 ===
    df[V2_PLUS_FEATURES] = df[V2_PLUS_FEATURES].replace([np.inf, -np.inf], np.nan)
    df[V2_PLUS_FEATURES] = df[V2_PLUS_FEATURES].fillna(0.0)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="在 ultrashort_main 数据集中增加 U1 v2_plus 因子列"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/ultrashort_main.parquet",
        help="输入 parquet 路径（默认覆盖原来的 ultrashort_main）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 parquet 路径（缺省时会覆盖 input 文件）",
    )
    args = parser.parse_args()

    print(f"[U2Feat] 读取数据集: {args.input}")
    df = pd.read_parquet(args.input)

    add_u1_v2_plus_features(df)

    out_path = args.output or args.input
    print(f"[U2Feat] 写入带 v2_plus 因子的数据集: {out_path}")
    df.to_parquet(out_path)
    print("[U2Feat] 因子计算完成。")


if __name__ == "__main__":
    main()
