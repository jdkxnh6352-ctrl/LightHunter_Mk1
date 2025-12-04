# -*- coding: utf-8 -*-
"""
U2 / U3 日度打分脚本（离线版）

- 数据源：u2_live_config.json.dataset.path（默认 data/datasets/ultrashort_main.parquet）
- 根据配置里的 train_days / min_train_days 自动切窗口
- engine = u2 : 使用 U2 Logistic 模型
- engine = u3 : 使用 U3 随机森林模型（u3_predictor）
- 结果保存到：u2_live_config.json.output.dir（默认 reports/u2_daily）

注意：这里只在历史因子表上跑，不改任何实盘流程。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style, init as color_init

from .u2_live_batch_backtest import (
    U2Config,
    load_u2_config,
)
from .u3_predictor import train_u3_model, score_u3_model


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------


def _apply_static_filters(df: pd.DataFrame, cfg: U2Config) -> pd.DataFrame:
    """
    与 u2_live_batch_backtest 中一致的静态过滤：价格 + 成交额。
    """
    out = df.copy()
    if "close" in out.columns:
        out = out[(out["close"] >= cfg.min_price) & (out["close"] <= cfg.max_price)]
    if "amount" in out.columns:
        out = out[out["amount"] >= cfg.min_amount]
    return out


def _load_dataset(cfg: U2Config) -> pd.DataFrame:
    print(Fore.CYAN + f"[U2] 读取数据集: {cfg.dataset_path}" + Style.RESET_ALL)
    df = pd.read_parquet(cfg.dataset_path)
    # trade_date / trading_date 兼容处理
    if "trade_date" not in df.columns:
        if "trading_date" in df.columns:
            df = df.rename(columns={"trading_date": "trade_date"})
        else:
            raise RuntimeError("数据集中既没有 trade_date 列，也没有 trading_date 列。")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    return df


def _pick_dates_for_scoring(
    df: pd.DataFrame,
    cfg: U2Config,
    as_of: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    根据 as_of 选择训练窗口 & 当天样本。
    返回：train_df, today_df, train_dates
    """
    all_dates = df["trade_date"].dropna().drop_duplicates().sort_values()
    if as_of not in all_dates.values:
        raise RuntimeError(f"打分日期 {as_of.date()} 不在数据集 trade_date 范围内。")

    # 找到 as_of 的位置
    idx = int(np.where(all_dates.values == as_of.to_datetime64())[0][0])

    train_days = int(cfg.train_days)
    min_train_days = int(getattr(cfg, "min_train_days", max(40, train_days // 2)))

    start_idx = max(0, idx - train_days)
    train_dates = all_dates[start_idx:idx]
    if len(train_dates) < min_train_days:
        raise RuntimeError(
            f"训练窗口交易日数量不足：{len(train_dates)} < min_train_days={min_train_days}"
        )

    train_df = df[df["trade_date"].isin(train_dates)].copy()
    today_df = df[df["trade_date"] == as_of].copy()
    return train_df, today_df, train_dates


def _format_date(d: pd.Timestamp) -> str:
    return d.strftime("%Y-%m-%d")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------


def run_daily_scoring(
    cfg: U2Config,
    engine: str = "u2",
    as_of_str: str | None = None,
) -> Path:
    """
    对某个交易日做一次“U2 壳 + (U2/U3) 引擎”的离线打分。
    """
    df = _load_dataset(cfg)

    # as_of 默认 = 数据中最后一个交易日
    all_dates = df["trade_date"].dropna().drop_duplicates().sort_values()
    if all_dates.empty:
        raise RuntimeError("数据集中没有任何 trade_date。")

    if as_of_str:
        as_of = pd.to_datetime(as_of_str).normalize()
    else:
        as_of = all_dates.iloc[-1]

    print(
        Fore.CYAN
        + f"[U2] 打分交易日: {as_of.date()} | engine={engine}"
        + Style.RESET_ALL
    )

    train_df, today_df, train_dates = _pick_dates_for_scoring(df, cfg, as_of)

    print(
        Fore.CYAN
        + "[U2] 训练窗口: "
        + f"{_format_date(train_dates.iloc[0])} ~ {_format_date(train_dates.iloc[-1])} "
        + f"(共 {len(train_dates)} 个交易日, {len(train_df)} 条样本)"
        + Style.RESET_ALL
    )
    print(
        Fore.CYAN
        + f"[U2] 当日打分样本数(过滤前): {len(today_df)}"
        + Style.RESET_ALL
    )

    # ----------------- 训练模型 -----------------
    if engine == "u2":
        # 直接复用 u2_live_batch_backtest 里的逻辑：
        from sklearn.linear_model import LogisticRegression

        feats = getattr(cfg, "features", None)
        if not feats:
            feats = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "vol_20",
                "amt_mean_20",
            ]
        # 去掉缺 label / 特征的样本
        cols_need = feats + [cfg.label_col]
        tdf = train_df.dropna(subset=cols_need)
        if tdf.empty:
            raise RuntimeError("U2 训练集为空，无法训练模型。")

        y = (tdf[cfg.label_col] > getattr(cfg, "label_threshold", 0.0)).astype(int)
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos < 20 or neg < 20:
            raise RuntimeError(f"U2 训练样本过少：pos={pos}, neg={neg}")

        X = tdf[feats].values
        model: LogisticRegression = LogisticRegression(max_iter=1000, n_jobs=-1)
        model.fit(X, y)

        # 对当天样本打分
        df_today = today_df.dropna(subset=feats).copy()
        proba = model.predict_proba(df_today[feats].values)[:, 1]

    elif engine == "u3":
        # 使用 U3 随机森林
        from .u3_predictor import U3TrainConfig

        u3_cfg = U3TrainConfig()
        model = train_u3_model(train_df, cfg, u3_cfg)
        if model is None:
            raise RuntimeError("U3 训练失败（样本过少或异常），无法打分。")

        df_today = today_df.copy()
        proba = score_u3_model(model, df_today, cfg)
        # score_u3_model 内部已经做过缺特征的处理
        valid_mask = ~np.isnan(proba)
        df_today = df_today.loc[valid_mask].copy()
        proba = proba[valid_mask]
    else:
        raise ValueError(f"未知 engine: {engine}")

    if len(df_today) == 0:
        raise RuntimeError("当日样本在训练/特征过滤后为空，无法打分。")

    df_today = df_today.copy()
    df_today["prob"] = proba

    # ----------------- 静态过滤 + Top-K -----------------
    before_filter = len(df_today)
    df_today = _apply_static_filters(df_today, cfg)
    after_filter = len(df_today)

    print(
        Fore.CYAN
        + f"[U2] 静态过滤后样本数: {after_filter} (价格/成交额过滤前为 {before_filter})"
        + Style.RESET_ALL
    )

    df_today = df_today[df_today["prob"] >= cfg.min_prob]
    if df_today.empty:
        print(
            Fore.YELLOW
            + "[U2] 没有任何股票满足 prob 阈值，今日无候选。"
            + Style.RESET_ALL
        )

    df_today = df_today.sort_values("prob", ascending=False)

    # 组合持仓数量限制：Top-K + max_positions
    n_sel = min(cfg.top_k, cfg.max_positions, len(df_today))
    candidates = df_today.head(n_sel).copy()

    print(
        Fore.CYAN
        + f"[U2] 本次候选股票数: {len(candidates)} "
        f"(top_k={cfg.top_k}, max_positions={cfg.max_positions})"
        + Style.RESET_ALL
    )

    # ----------------- 保存结果 -----------------
    out_dir: Path = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if engine == "u2":
        fname = f"u2_candidates_{as_of.strftime('%Y%m%d')}.csv"
    else:
        # U3 结果单独存一份，不影响原来 u2_candidates_*.csv
        fname = f"u3_candidates_{as_of.strftime('%Y%m%d')}.csv"

    out_path = out_dir / fname
    candidates.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(
        Fore.GREEN
        + f"[U2] 候选列表已保存到: {out_path}"
        + Style.RESET_ALL
    )

    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def main() -> None:
    color_init(autoreset=True)

    parser = argparse.ArgumentParser(
        description="U2 / U3 日度打分（离线，只用历史因子表）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/u2_live_config.json",
        help="U2 实盘配置 json 路径（默认 config/u2_live_config.json）",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["u2", "u3"],
        default="u2",
        help="打分引擎：u2=原 Logistic, u3=随机森林（默认 u2）",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="打分日期 YYYY-MM-DD（默认=数据集中最后一个 trade_date）",
    )
    parser.add_argument(
        "--filter-tag",
        type=str,
        default=None,
        help="可选：叠加 light_hunter_config.json 中的过滤方案标签，例如 base/aggressive/defensive",
    )

    args = parser.parse_args()

    # 通过 u2_live_batch_backtest 的加载函数统一读取配置 + 叠加 filter_tag
    cfg = load_u2_config(args.config, filter_tag=args.filter_tag)

    run_daily_scoring(cfg, engine=args.engine, as_of_str=args.as_of)


if __name__ == "__main__":
    main()
