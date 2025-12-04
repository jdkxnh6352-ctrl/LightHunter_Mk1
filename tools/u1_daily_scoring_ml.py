# tools/u1_daily_scoring_ml.py
"""
U1 v1 多因子 + RF 日打分引擎

默认配置：
- 数据集: data/datasets/{job_id}.parquet   (ultrashort_main)
- 因子:   base = ['amount', 'vol_20', 'ret_1', 'ret_5', 'ret_20']
- 模型:   RandomForestRegressor
- 训练窗口: 最近 train_days 个交易日 (默认 240 天)
- 最小训练样本天数: min_train_days (默认 80 天)
- 打分目标: ret_1
- 流动性过滤: 价格 [min_price, max_price]，amount >= min_amount
- 输出: reports/u1_scores_{job_id}_{tag}_{as_of}_top{top_k}.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ----------------- 工具函数 ----------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [U1Score] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("U1 v1 daily scoring (base + RF)")

    parser.add_argument("--job-id", required=True,
                        help="数据集 ID，对应 data/datasets/{job_id}.parquet，例如 ultrashort_main")

    parser.add_argument("--as-of", default=None,
                        help="打分交易日，YYYY-MM-DD；默认用数据集中最后一个交易日")

    parser.add_argument("--train-days", type=int, default=240,
                        help="训练窗口长度（最多使用最近多少个交易日）")

    parser.add_argument("--min-train-days", type=int, default=80,
                        help="最少训练天数，不足则报错")

    parser.add_argument("--top-k", type=int, default=3,
                        help="每天选出的标的数量")

    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)

    parser.add_argument("--ret-col", default="ret_1",
                        help="作为打分目标的收益列，默认 ret_1")

    parser.add_argument(
        "--features",
        default="amount,vol_20,ret_1,ret_5,ret_20",
        help="特征列表，逗号分隔。默认是 U1 v1 base 因子"
    )

    parser.add_argument("--tag", default="u1_v1_base_rf",
                        help="策略标签，用于输出文件命名")

    parser.add_argument("--output-dir", default="reports",
                        help="结果输出目录，默认 reports/")

    return parser.parse_args()


def load_dataset(job_id: str) -> pd.DataFrame:
    dataset_path = Path("data/datasets") / f"{job_id}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")

    logging.info("读取数据集: %s", dataset_path)
    df = pd.read_parquet(dataset_path)

    # 统一日期列名
    if "trade_date" not in df.columns:
        if "trading_date" in df.columns:
            df = df.rename(columns={"trading_date": "trade_date"})
            logging.info("数据集中没有 trade_date 列，发现 trading_date 列，已重命名为 trade_date 用于分析……")
        else:
            raise KeyError("数据集中既没有 trade_date 也没有 trading_date 列。")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def infer_price_col(df: pd.DataFrame) -> str | None:
    candidates = ["close", "adj_close", "close_adj", "price", "close_1d"]
    for c in candidates:
        if c in df.columns:
            logging.info("使用价格列: %s", c)
            return c
    logging.warning("未找到价格列(%s)，将跳过价格过滤。", candidates)
    return None


def infer_code_col(df: pd.DataFrame) -> str | None:
    candidates = ["ts_code", "symbol", "sec_code", "code", "stock_code"]
    for c in candidates:
        if c in df.columns:
            logging.info("使用代码列: %s", c)
            return c
    logging.warning("未找到证券代码列(%s)，输出中将不包含代码。", candidates)
    return None


def build_train_and_today(
    df: pd.DataFrame,
    as_of_str: str | None,
    train_days: int,
    min_train_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    # 确定 as_of
    all_dates = np.sort(df["trade_date"].unique())
    last_date = pd.to_datetime(all_dates[-1])

    if as_of_str is None:
        as_of = last_date
        logging.info("未指定 as-of，默认使用数据集中最后一个交易日: %s", as_of.date())
    else:
        as_of = pd.to_datetime(as_of_str)
        if as_of not in all_dates:
            # 使用最近的一个历史交易日
            as_of = pd.to_datetime(all_dates[all_dates <= as_of][-1])
            logging.warning("指定 as-of 不在数据集中，自动调整为最近的交易日: %s", as_of.date())

    # 只考虑 as_of 之前的数据
    history = df[df["trade_date"] <= as_of].copy()
    history_dates = np.sort(history["trade_date"].unique())

    if len(history_dates) < min_train_days:
        raise ValueError(
            f"历史交易日只有 {len(history_dates)} 天，少于 min_train_days={min_train_days}，"
            f"无法训练模型。"
        )

    as_of_idx = np.where(history_dates == as_of)[0][0]
    start_idx = max(0, as_of_idx - train_days)
    train_dates = history_dates[start_idx:as_of_idx]  # 不含 as_of 当天

    if len(train_dates) < min_train_days:
        raise ValueError(
            f"训练窗口内只有 {len(train_dates)} 天，小于 min_train_days={min_train_days}。"
        )

    train_df = history[history["trade_date"].isin(train_dates)].copy()
    today_df = history[history["trade_date"] == as_of].copy()

    logging.info(
        "训练窗口: [%s, %s]，共 %d 个交易日，训练样本数: %d",
        pd.to_datetime(train_dates[0]).date(),
        pd.to_datetime(train_dates[-1]).date(),
        len(train_dates),
        len(train_df),
    )
    logging.info("打分交易日: %s，候选样本数(过滤前): %d", as_of.date(), len(today_df))

    return train_df, today_df, as_of


def apply_universe_filters(
    df: pd.DataFrame,
    price_col: str | None,
    min_price: float,
    max_price: float,
    min_amount: float,
) -> pd.DataFrame:
    out = df.copy()

    if price_col is not None and price_col in out.columns:
        out = out[(out[price_col] >= min_price) & (out[price_col] <= max_price)]

    if "amount" in out.columns and min_amount is not None:
        out = out[out["amount"] >= min_amount]
    else:
        logging.warning("未找到 amount 列，无法做成交额过滤。")

    return out


def train_rf_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    ret_col: str,
) -> RandomForestRegressor:
    cols_needed = feature_cols + [ret_col]
    train_df = train_df.dropna(subset=cols_needed).copy()

    if train_df.empty:
        raise ValueError("训练数据在去除缺失值后为空，请检查特征列或数据质量。")

    X = train_df[feature_cols].values
    y = train_df[ret_col].values

    logging.info("开始训练模型: rf ……")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    logging.info("模型训练完成。训练样本数: %d", X.shape[0])
    return model


def score_today(
    model: RandomForestRegressor,
    today_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    today_df = today_df.dropna(subset=feature_cols).copy()
    if today_df.empty:
        raise ValueError("打分当日样本在去除缺失值后为空。")

    X_today = today_df[feature_cols].values
    scores = model.predict(X_today)
    today_df["u1_score"] = scores
    today_df = today_df.sort_values("u1_score", ascending=False).reset_index(drop=True)
    return today_df


# ----------------- 主流程 ----------------- #

def main():
    setup_logging()
    args = parse_args()

    # 读取数据
    df = load_dataset(args.job_id)

    # 解析特征列
    raw_features = [c.strip() for c in args.features.split(",") if c.strip()]
    feature_cols = [c for c in raw_features if c in df.columns]
    missing = [c for c in raw_features if c not in df.columns]

    if missing:
        logging.warning("以下特征在数据集中不存在，将被忽略: %s", missing)
    if not feature_cols:
        raise ValueError(
            f"有效特征列为空，请检查 --features 或数据列名。"
        )

    logging.info("使用特征字段 %d 个: %s", len(feature_cols), feature_cols)

    # 拆分训练集 + 当日
    train_df, today_df, as_of = build_train_and_today(
        df,
        args.as_of,
        args.train_days,
        args.min_train_days,
    )

    # 推断价格列 / 代码列
    price_col = infer_price_col(df)
    code_col = infer_code_col(df)

    # 流动性过滤
    train_df = apply_universe_filters(
        train_df, price_col, args.min_price, args.max_price, args.min_amount
    )
    today_df = apply_universe_filters(
        today_df, price_col, args.min_price, args.max_price, args.min_amount
    )

    logging.info("训练样本(过滤后)数量: %d", len(train_df))
    logging.info("当日候选(过滤后)数量: %d", len(today_df))

    if len(today_df) == 0:
        raise ValueError("当日候选股票数为 0，无法打分。")

    # 训练模型
    model = train_rf_model(train_df, feature_cols, args.ret_col)

    # 当日打分
    scored_df = score_today(model, today_df, feature_cols)

    # 取 top-k
    top_k = min(args.top_k, len(scored_df))
    top_df = scored_df.head(top_k).copy()
    top_df["weight"] = 1.0 / top_k

    logging.info("选出候选股票 %d 个 (top_k=%d)。", top_k, args.top_k)

    # 打印前几名到日志
    show_cols = []
    if code_col:
        show_cols.append(code_col)
    if price_col:
        show_cols.append(price_col)
    if "amount" in scored_df.columns:
        show_cols.append("amount")
    show_cols.append("u1_score")
    preview_cols = [c for c in show_cols if c in top_df.columns]

    if preview_cols:
        logging.info("Top %d 预览:", top_k)
        for i, row in top_df.head(min(10, top_k)).iterrows():
            values = ", ".join(f"{col}={row[col]}" for col in preview_cols)
            logging.info("  #%d %s", i + 1, values)

    # 保存结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = as_of.strftime("%Y%m%d")

    top_path = out_dir / f"u1_scores_{args.job_id}_{args.tag}_{date_str}_top{top_k}.csv"
    full_path = out_dir / f"u1_scores_{args.job_id}_{args.tag}_{date_str}_full.csv"

    top_df.to_csv(top_path, index=False)
    scored_df.to_csv(full_path, index=False)

    logging.info("Top-k 结果已保存到: %s", top_path)
    logging.info("完整打分结果已保存到: %s", full_path)
    logging.info("U1 v1 日打分完成。")


if __name__ == "__main__":
    main()
