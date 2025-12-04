# tools/build_ultrashort_dataset_from_raw.py
# 作用：把 data/ultrashort_raw.csv 转成
#       data/datasets/ultrashort_main.parquet / ultrashort_main.csv
#       给 U2 策略做训练 / 回测用

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "ultrashort_raw.csv"
OUT_DIR = ROOT / "data" / "datasets"
OUT_PATH_PARQUET = OUT_DIR / "ultrashort_main.parquet"
OUT_PATH_CSV = OUT_DIR / "ultrashort_main.csv"


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"[ERROR] 找不到原始文件: {RAW_PATH} ，请先运行 build_ultrashort_raw_sample。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] 读取原始文件: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # ---------- 1. 时间排序 ----------
    time_col = None
    for c in ["datetime", "time", "ts", "timestamp"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise SystemExit("[ERROR] csv 里没找到时间列，请确认有 'datetime' 这一列。")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(["code", time_col]).reset_index(drop=True)

    # ---------- 2. 构造一些基础特征 ----------
    # 1 / 5 / 20 根 K 线收益
    df["ret_1"] = df.groupby("code")["close"].pct_change(1)
    df["ret_5"] = df.groupby("code")["close"].pct_change(5)
    df["ret_20"] = df.groupby("code")["close"].pct_change(20)

    # 过去 20 根的波动率（收益标准差）
    df["vol_20"] = (
        df.groupby("code")["close"]
          .pct_change()
          .rolling(20, min_periods=5)
          .std()
    )

    # 过去 20 根的成交额均值（如果有 amount 这一列）
    if "amount" in df.columns:
        df["amt_mean_20"] = (
            df.groupby("code")["amount"]
              .rolling(20, min_periods=5)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # ---------- 3. 构造一个简单的 U2 标签 ----------
    # 定义：未来 20 根 K 线内，如果最高价相对当前收盘涨幅 >= 2%，记为 1，否则记为 0
    horizon = 20

    high_fwd_max = (
        df.groupby("code")["high"]
          .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    )
    df["label_u2_raw"] = high_fwd_max / df["close"] - 1.0
    df["label_u2"] = (df["label_u2_raw"] >= 0.02).astype("int8")

    # 去掉每只股票最后 horizon 行（未来数据不完整）
    def drop_tail(g):
        return g.iloc[:-horizon] if len(g) > horizon else g.iloc[0:0]

    df = df.groupby("code", group_keys=False).apply(drop_tail)

    # 去掉 rolling 不完整的行
    df = df.dropna(subset=["ret_5", "ret_20", "vol_20"]).reset_index(drop=True)

    feature_cols = [
        "open", "high", "low", "close", "volume", "amount",
        "ret_1", "ret_5", "ret_20", "vol_20", "amt_mean_20",
    ]
    available_feature_cols = [c for c in feature_cols if c in df.columns]

    cols = ["code", time_col] + available_feature_cols + ["label_u2"]
    df_out = df[cols].copy()

    print(f"[DATA] 最终样本数: {len(df_out)} 行")
    print(f"[DATA] 特征列: {available_feature_cols}")
    print("[DATA] 标签列: label_u2  (1 = 未来20根内最高价涨幅 >= 2%)")

    df_out.to_parquet(OUT_PATH_PARQUET, index=False)
    df_out.to_csv(OUT_PATH_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 已保存数据集到: {OUT_PATH_PARQUET}")
    print(f"[OK] 同时保存为:     {OUT_PATH_CSV}")


if __name__ == "__main__":
    main()
