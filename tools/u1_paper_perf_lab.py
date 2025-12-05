# -*- coding: utf-8 -*-
"""
U1 纸上实盘绩效分析实验室

从 data/live/paper_trades_<job_id>.csv 中读取纸上实盘记录，
结合 data/<job_id>.parquet 里的行情/收益数据，计算日度组合收益、
权益曲线、年度统计，并输出 markdown 报告。

用法示例：

    python -m tools.u1_paper_perf_lab \
        --job-id ultrashort_main \
        --tag u1_v1_base_rf

可选参数：
    --start-date 2025-01-01
    --end-date   2025-12-31
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


# ========= 工具函数 =========

def find_dataset_path(job_id: str) -> Path:
    """尽量智能地找到 parquet 数据集路径。"""
    candidates = [
        Path("data") / f"{job_id}.parquet",
        Path("data") / f"{job_id}_main.parquet",
        Path("data/datasets") / f"{job_id}.parquet",
        Path("data/datasets") / f"{job_id}_main.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到数据文件，请确认路径：{candidates}")


def detect_date_col(df: pd.DataFrame) -> str:
    """在 DataFrame 中自动识别日期列名。"""
    for col in ["trade_date", "trading_date", "date", "as_of"]:
        if col in df.columns:
            return col
    raise ValueError("未找到日期列（期望列名之一：trade_date / trading_date / date / as_of）")


def detect_code_col(df: pd.DataFrame) -> str:
    """自动识别股票代码列名。"""
    for col in ["code", "ts_code", "stock_code", "sec_code"]:
        if col in df.columns:
            return col
    raise ValueError("未找到股票代码列（期望列名之一：code / ts_code / stock_code / sec_code）")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_float(x: float) -> str:
    return f"{x:.4f}"


def ensure_reports_dir() -> Path:
    path = Path("reports")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ========= 核心逻辑 =========

def load_market_data(job_id: str) -> pd.DataFrame:
    path = find_dataset_path(job_id)
    print(f"[U1Perf] 读取数据集: {path}")
    df = pd.read_parquet(path)

    date_col = detect_date_col(df)
    code_col = detect_code_col(df)

    if date_col != "trade_date":
        df = df.rename(columns={date_col: "trade_date"})
    if code_col != "code":
        df = df.rename(columns={code_col: "code"})

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    if "ret_1" not in df.columns:
        raise ValueError("数据集中未找到 ret_1 列，无法计算次日收益。")

    return df[["trade_date", "code", "ret_1"]].copy()


def load_paper_trades(job_id: str, tag: str) -> pd.DataFrame:
    path = Path("data/live") / f"paper_trades_{job_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到纸上实盘日志文件: {path}")

    print(f"[U1Perf] 读取纸上实盘日志: {path}")
    df = pd.read_csv(path, encoding="utf-8")

    # 自动识别日期 / 代码列
    date_col = detect_date_col(df)
    code_col = detect_code_col(df)

    if date_col != "as_of":
        df = df.rename(columns={date_col: "as_of"})
    if code_col != "code":
        df = df.rename(columns={code_col: "code"})

    df["as_of"] = pd.to_datetime(df["as_of"])

    # 过滤 job_id / tag（如果有这两列的话）
    if "job_id" in df.columns:
        df = df[df["job_id"] == job_id]
    if "tag" in df.columns:
        df = df[df["tag"] == tag]

    if df.empty:
        raise ValueError(f"纸上实盘日志中没有符合条件的记录: job_id={job_id}, tag={tag}")

    df = df.sort_values(["as_of", "code"]).reset_index(drop=True)
    print(f"[U1Perf] 过滤后记录数: {len(df)}")
    return df


def apply_date_filter(df: pd.DataFrame,
                      start_date: Optional[str],
                      end_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        dt = pd.to_datetime(start_date)
        df = df[df["as_of"] >= dt]
    if end_date:
        dt = pd.to_datetime(end_date)
        df = df[df["as_of"] <= dt]
    return df


def merge_with_market(paper: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """把纸上实盘记录与行情数据按 as_of + code 对齐，取 ret_1 作为次日收益。"""
    merged = paper.merge(
        market,
        left_on=["code", "as_of"],
        right_on=["code", "trade_date"],
        how="left",
    )

    missing = merged["ret_1"].isna().sum()
    if missing > 0:
        print(f"[U1Perf] 警告：有 {missing} 条记录在行情数据中找不到 ret_1，已按 0 处理。")
        merged["ret_1"] = merged["ret_1"].fillna(0.0)
    return merged


def compute_daily_returns(merged: pd.DataFrame) -> pd.DataFrame:
    """
    根据合并后的表计算组合日度收益：
      - 优先使用 target_weight / weight / amount 作为权重
      - 否则等权
    """
    df = merged.copy()

    # 统一权重列
    w_col = None
    for col in ["target_weight", "weight", "w", "amount"]:
        if col in df.columns:
            w_col = col
            break

    if w_col is None:
        # 没有任何权重列 -> 等权
        print("[U1Perf] 未找到权重列，默认按等权处理。")
        df["__w"] = 1.0
    else:
        if w_col == "amount":
            # amount -> 在每个交易日内按金额归一化
            df["__w_raw"] = df["amount"].astype(float)
            df["__w"] = df["__w_raw"] / df.groupby("as_of")["__w_raw"].transform("sum")
        else:
            df["__w"] = df[w_col].astype(float)

    def _agg(group: pd.DataFrame) -> float:
        w = group["__w"].values
        r = group["ret_1"].values
        denom = float(abs(w).sum())
        if denom <= 0:
            return float(r.mean())
        return float((w * r).sum() / denom)

    daily = (
        df.groupby("as_of")
        .apply(_agg)
        .to_frame("daily_ret")
        .reset_index()
        .sort_values("as_of")
        .reset_index(drop=True)
    )

    print(f"[U1Perf] 有效交易日数: {len(daily)}")
    return daily


def compute_equity_and_stats(daily: pd.DataFrame) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """从日度收益计算权益曲线、整体统计与年度统计。"""
    daily = daily.copy()
    daily["equity"] = (1.0 + daily["daily_ret"]).cumprod()

    n_days = len(daily)
    total_return = float(daily["equity"].iloc[-1] - 1.0)

    if n_days > 1:
        ann_return = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
        ann_vol = float(daily["daily_ret"].std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        ann_return = total_return
        ann_vol = 0.0

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    cummax = daily["equity"].cummax()
    drawdown = daily["equity"] / cummax - 1.0
    max_drawdown = float(drawdown.min())

    win_ratio = float((daily["daily_ret"] > 0).mean())

    summary = {
        "n_days": n_days,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_ratio": win_ratio,
    }

    # 年度拆分
    yearly = daily.copy()
    yearly["year"] = yearly["as_of"].dt.year
    rows = []
    for year, g in yearly.groupby("year"):
        nd = len(g)
        eq = g["equity"]
        tr = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        if nd > 1:
            ar = (1.0 + tr) ** (TRADING_DAYS_PER_YEAR / nd) - 1.0
            av = float(g["daily_ret"].std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            ar = tr
            av = 0.0
        sh = ar / av if av > 0 else 0.0
        cm = eq.cummax()
        dd = float((eq / cm - 1.0).min())
        wr = float((g["daily_ret"] > 0).mean())

        rows.append(
            dict(
                year=int(year),
                n_days=nd,
                total_return=tr,
                ann_return=ar,
                ann_vol=av,
                sharpe=sh,
                max_drawdown=dd,
                win_ratio=wr,
            )
        )

    yearly_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return daily, summary, yearly_df


def write_reports(job_id: str,
                  tag: str,
                  daily: pd.DataFrame,
                  summary: dict,
                  yearly: pd.DataFrame) -> None:
    reports_dir = ensure_reports_dir()

    equity_path = reports_dir / f"u1_paper_equity_{job_id}_{tag}.csv"
    yearly_path = reports_dir / f"u1_paper_yearly_{job_id}_{tag}.csv"
    md_path = reports_dir / f"u1_paper_perf_{job_id}_{tag}.md"

    daily_out = daily.copy()
    daily_out["date"] = daily_out["as_of"].dt.strftime("%Y-%m-%d")
    daily_out = daily_out[["date", "daily_ret", "equity"]]
    daily_out.to_csv(equity_path, index=False, encoding="utf-8-sig")

    yearly_out = yearly.copy()
    yearly_out.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    print(f"[U1Perf] 已保存每日权益曲线到: {equity_path}")
    print(f"[U1Perf] 已保存年度统计到: {yearly_path}")

    # 生成 markdown 报告
    lines = []
    lines.append(f"# U1 纸上实盘绩效报告 - {job_id} / {tag}")
    lines.append("")
    lines.append("## 整体统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| ---- | ---- |")
    lines.append(f"| 交易日数 | {summary['n_days']} |")
    lines.append(f"| 总收益 | {fmt_pct(summary['total_return'])} |")
    lines.append(f"| 年化收益 | {fmt_pct(summary['ann_return'])} |")
    lines.append(f"| 年化波动 | {fmt_pct(summary['ann_vol'])} |")
    lines.append(f"| Sharpe | {fmt_float(summary['sharpe'])} |")
    lines.append(f"| 最大回撤 | {fmt_pct(summary['max_drawdown'])} |")
    lines.append(f"| 胜率（按日） | {fmt_pct(summary['win_ratio'])} |")
    lines.append("")
    lines.append("## 按年份拆分统计")
    lines.append("")
    if yearly.empty:
        lines.append("暂无年度统计。")
    else:
        lines.append("| 年份 | 交易日数 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 |")
        lines.append("| ---- | -------- | ------ | -------- | -------- | ------ | -------- | ---- |")
        for _, row in yearly.iterrows():
            lines.append(
                "| {year} | {n_days} | {total} | {ann_ret} | {ann_vol} | {sharpe} | {max_dd} | {win_ratio} |".format(
                    year=int(row["year"]),
                    n_days=int(row["n_days"]),
                    total=fmt_pct(row["total_return"]),
                    ann_ret=fmt_pct(row["ann_return"]),
                    ann_vol=fmt_pct(row["ann_vol"]),
                    sharpe=fmt_float(row["sharpe"]),
                    max_dd=fmt_pct(row["max_drawdown"]),
                    win_ratio=fmt_pct(row["win_ratio"]),
                )
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[U1Perf] 绩效报告已保存到: {md_path}")


# ========= 命令行入口 =========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 纸上实盘绩效分析实验室（paper performance lab）"
    )
    parser.add_argument("--job-id", required=True, help="数据/任务 ID，例如 ultrashort_main")
    parser.add_argument("--tag", required=True, help="模型/参数版本标签，例如 u1_v1_base_rf")
    parser.add_argument("--start-date", help="起始日期，格式 YYYY-MM-DD，可选")
    parser.add_argument("--end-date", help="结束日期，格式 YYYY-MM-DD，可选")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_id = args.job_id
    tag = args.tag

    print(f"[U1Perf] 任务信息: job_id={job_id}, tag={tag}")

    market = load_market_data(job_id)
    paper = load_paper_trades(job_id, tag)
    paper = apply_date_filter(paper, args.start_date, args.end_date)

    if paper.empty:
        raise ValueError("按日期过滤后纸上实盘记录为空，请检查 start/end-date 参数。")

    merged = merge_with_market(paper, market)
    daily = compute_daily_returns(merged)
    daily, summary, yearly = compute_equity_and_stats(daily)

    print("==== U1 纸上实盘回测统计（整体） ====")
    print(f"交易日数   : {summary['n_days']}")
    print(f"累计收益   : {fmt_pct(summary['total_return'])}")
    print(f"年化收益   : {fmt_pct(summary['ann_return'])}")
    print(f"年化波动   : {fmt_pct(summary['ann_vol'])}")
    print(f"Sharpe     : {fmt_float(summary['sharpe'])}")
    print(f"最大回撤   : {fmt_pct(summary['max_drawdown'])}")
    print(f"胜率(按日) : {fmt_pct(summary['win_ratio'])}")

    write_reports(job_id, tag, daily, summary, yearly)
    print("[U1Perf] 纸上实盘绩效分析完成。")


if __name__ == "__main__":
    main()
