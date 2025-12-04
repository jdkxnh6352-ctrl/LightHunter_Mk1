# tools/fix_ultrashort_drop_datetime.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data") / "datasets"
PARQUET_PATH = DATA_DIR / "ultrashort_main.parquet"
CSV_PATH = DATA_DIR / "ultrashort_main.csv"

def main() -> None:
    print(f"[FIX] 读取数据集: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)
    print("[FIX] 原始列:", list(df.columns))

    # 删除不应该参与建模的 datetime 列
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])
        print("[FIX] 已删除 datetime 列.")
    else:
        print("[FIX] 没有 datetime 列, 无需处理.")

    print("[FIX] 最终列:", list(df.columns))

    # 回写 parquet
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[FIX] 已写回 parquet: {PARQUET_PATH}")

    # 同步一份 CSV（方便你用 Excel 看）
    df.to_csv(CSV_PATH, index=False)
    print(f"[FIX] 同步保存 CSV: {CSV_PATH}")
    print(f"[FIX] 最终行数: {len(df)}")

if __name__ == "__main__":
    main()
