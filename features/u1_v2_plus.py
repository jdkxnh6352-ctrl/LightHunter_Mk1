# -*- coding: utf-8 -*-
"""
U1 v2 PLUS 因子包

设计目标：
- 在现有 U1 v1（amount, vol_20, ret_1, ret_5, ret_20）的基础上，
  再补一层更细的“短线选股因子”，提升模型对强势股 / 异动股的识别能力。
- 所有因子都只使用历史 + 当日收盘前的数据，不用未来信息，避免前视偏差。

使用方式（建议）：
1）在 build_ultrashort_main_dataset.py 里读完 parquet 之后：
    from features.u1_v2_plus import add_u1_v2_plus_features, U1_V2_PLUS_COLUMNS
    df = add_u1_v2_plus_features(df)

2）训练模型时，把 U1_V2_PLUS_COLUMNS 追加到特征列表里即可。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
#  因子列名列表（方便在训练脚本里直接引用）
# ============================================================

U1_V2_PLUS_COLUMNS = [
    # 多周期动量
    "f_ret_1",
    "f_ret_3",
    "f_ret_5",
    "f_ret_10",
    "f_ret_20",

    # K 线形态 & 波动
    "f_intraday_amp",        # 当日振幅
    "f_true_range",          # True Range / 收盘价
    "f_upper_shadow",        # 上影线占比
    "f_lower_shadow",        # 下影线占比

    # 成交量 / 成交额放大
    "f_vol_ratio_5",
    "f_vol_ratio_20",
    "f_amt_ratio_5",
    "f_amt_ratio_20",

    # 收益波动率（短周期）
    "f_volatility_5",
    "f_volatility_20",

    # 与最近高低点的距离
    "f_close_to_5d_high",
    "f_close_to_20d_high",
    "f_close_to_5d_low",
    "f_close_to_20d_low",

    # 极端涨跌 & 情绪因子
    "f_limit_up_like",       # 近似涨停
    "f_limit_down_like",     # 近似跌停
    "f_recent_big_up",       # 近几日是否出现过大阳
    "f_recent_big_down",     # 近几日是否出现过大阴
]


# ============================================================
#  核心入口
# ============================================================

def add_u1_v2_plus_features(
    df: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    在传入的行情 DataFrame 上直接追加 U1 v2 PLUS 因子列。

    参数
    ----
    df : DataFrame
        必须至少包含：
        [code_col, date_col, open, high, low, close, volume, amount]
    code_col : str, 默认 "code"
    date_col : str, 默认 "trade_date"

    返回
    ----
    DataFrame
        在原 df 的基础上增加若干 f_* 列，并保持原排序。
    """
    if df.empty:
        return df

    # 保障排序：按代码 + 交易日
    df = df.sort_values([code_col, date_col]).copy()
    g = df.groupby(code_col, sort=False)

    # -----------------------------
    # 1. 多周期动量（用对数收益）
    # -----------------------------
    for win in (1, 3, 5, 10, 20):
        col = f"f_ret_{win}"
        # log-return：避免高价股和低价股尺度差太大
        df[col] = np.log(g["close"].shift(0) / g["close"].shift(win))
        # 第一 win 天会是 NaN，后面模型会统一处理 / 填充

    # -----------------------------
    # 2. K 线形态 & 波动
    # -----------------------------
    # 当日振幅：高低价差 / 收盘价
    df["f_intraday_amp"] = (df["high"] - df["low"]) / df["close"].clip(lower=1e-3)

    # True Range（高 - 低，和与昨收价差三者取最大），再除以收盘价
    prev_close = g["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["f_true_range"] = true_range / df["close"].clip(lower=1e-3)

    # 上影线 / 下影线 占比
    real_body_max = df[["open", "close"]].max(axis=1)
    real_body_min = df[["open", "close"]].min(axis=1)
    df["f_upper_shadow"] = (df["high"] - real_body_max) / df["close"].clip(lower=1e-3)
    df["f_lower_shadow"] = (real_body_min - df["low"]) / df["close"].clip(lower=1e-3)

    # 简单裁一下极值，避免个别脏数据太夸张
    for c in ["f_intraday_amp", "f_true_range", "f_upper_shadow", "f_lower_shadow"]:
        df[c] = df[c].clip(lower=-1.0, upper=1.0)

    # -----------------------------
    # 3. 成交量 / 成交额放大因子
    # -----------------------------
    for win in (5, 20):
        mean_vol = g["volume"].rolling(win, min_periods=1).mean()
        mean_amt = g["amount"].rolling(win, min_periods=1).mean()
        df[f"f_vol_ratio_{win}"] = df["volume"] / mean_vol.replace(0, np.nan)
        df[f"f_amt_ratio_{win}"] = df["amount"] / mean_amt.replace(0, np.nan)

    # 合理裁剪一下放大量级
    for c in ["f_vol_ratio_5", "f_vol_ratio_20", "f_amt_ratio_5", "f_amt_ratio_20"]:
        df[c] = df[c].clip(lower=0.0, upper=10.0)

    # -----------------------------
    # 4. 短周期收益波动率
    #    用日收益的滚动 std * sqrt(年化)
    # -----------------------------
    # 如果你已有 ret_1，这里也可以直接用；否则按收盘价现算一份
    daily_ret = g["close"].pct_change()
    for win in (5, 20):
        vol = daily_ret.groupby(df[code_col]).rolling(win, min_periods=1).std().reset_index(level=0, drop=True)
        df[f"f_volatility_{win}"] = vol * np.sqrt(252.0)

    # 裁异常
    for c in ["f_volatility_5", "f_volatility_20"]:
        df[c] = df[c].clip(lower=0.0, upper=1.0)

    # -----------------------------
    # 5. 与最近高 / 低点距离
    # -----------------------------
    for win in (5, 20):
        roll_max = g["close"].rolling(win, min_periods=1).max()
        roll_min = g["close"].rolling(win, min_periods=1).min()

        df[f"f_close_to_{win}d_high"] = df["close"] / roll_max.replace(0, np.nan) - 1.0
        df[f"f_close_to_{win}d_low"] = df["close"] / roll_min.replace(0, np.nan) - 1.0

    for c in ["f_close_to_5d_high", "f_close_to_20d_high",
              "f_close_to_5d_low", "f_close_to_20d_low"]:
        df[c] = df[c].clip(lower=-0.5, upper=0.5)

    # -----------------------------
    # 6. 极端涨跌 & 情绪类因子
    #    —— 近似“涨停 / 跌停”、大阳线、大阴线
    #    主板日涨跌停一般 ±10%，这里稍微留一点余地
    # -----------------------------
    if "ret_1" in df.columns:
        one_day_ret = df["ret_1"]
    else:
        one_day_ret = daily_ret  # 上面已经算过

    df["f_limit_up_like"] = (one_day_ret > 0.095).astype("int8")
    df["f_limit_down_like"] = (one_day_ret < -0.095).astype("int8")

    # 近 N 日是否出现过“大阳 / 大阴”（>7% 或 <-7%）
    big_up = (one_day_ret > 0.07).astype("int8")
    big_down = (one_day_ret < -0.07).astype("int8")

    df["f_recent_big_up"] = (
        big_up.groupby(df[code_col])
        .rolling(5, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype("int8")
    )

    df["f_recent_big_down"] = (
        big_down.groupby(df[code_col])
        .rolling(5, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype("int8")
    )

    return df


# ============================================================
#  简单自测入口（可选）
# ============================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="在 ultrashort_main 数据集上追加 U1 v2 PLUS 因子"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 parquet 文件路径，例如 data/datasets/ultrashort_main.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 parquet 文件路径，默认在文件名后面加 _u1_v2_plus",
    )
    parser.add_argument("--code-col", type=str, default="code")
    parser.add_argument("--date-col", type=str, default="trade_date")

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_u1_v2_plus.parquet")

    print(f"[U1_V2_PLUS] 读取数据集: {inp}")
    data = pd.read_parquet(inp)
    data = add_u1_v2_plus_features(data, code_col=args.code_col, date_col=args.date_col)

    print(f"[U1_V2_PLUS] 新增因子列数: {len(U1_V2_PLUS_COLUMNS)}")
    print(f"[U1_V2_PLUS] 写出到: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out)
