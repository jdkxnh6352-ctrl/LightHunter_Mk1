# tools/u1_strategy_replay.py
# -*- coding: utf-8 -*-
"""
U1 策略级历史回放脚本

功能：
- 读取基础数据：data/datasets/{job_id}.parquet
- 扫描历史打分结果：reports/u1_scores_{job_id}_{tag}_YYYYMMDD_top{top_k}.csv
- 假设规则：每天等权买入 top_k，只持有 1 天（使用 ret_col，例如 ret_1），T+1 收盘卖出
- 输出：
    * 每日策略收益 / 全市场平均收益
    * 整体区间表现（累计收益、年化、年化波动、Sharpe、最大回撤）
    * 按年份拆分表现
    * 对比“什么都不干（当天全市场等权持有）”的大致水平
- 以 markdown 报告形式保存到：
    reports/u1_strategy_replay_report_{job_id}_{tag}_{ret_col}_top{top_k}.md

使用示例（在项目根目录运行）：
    python -m tools.u1_strategy_replay ^
        --job-id ultrashort_main ^
        --tag u1_v1_base_rf ^
        --ret-col ret_1 ^
        --top-k 3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================== 通用工具函数 ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(prefix: str, msg: str) -> None:
    print(f"[U1Strat] {prefix}：{msg}")


def ensure_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保数据里有 trade_date 列；如果是 trading_date / date / as_of 就自动重命名。
    """
    if "trade_date" in df.columns:
        pass
    elif "trading_date" in df.columns:
        log("INFO", "数据集中没有 trade_date 列，发现 trading_date 列，已重命名为 trade_date 用于分析……")
        df = df.rename(columns={"trading_date": "trade_date"})
    elif "date" in df.columns:
        log("INFO", "数据集中没有 trade_date 列，发现 date 列，已重命名为 trade_date 用于分析……")
        df = df.rename(columns={"date": "trade_date"})
    elif "as_of" in df.columns:
        log("INFO", "数据集中没有 trade_date 列，发现 as_of 列，已重命名为 trade_date 用于分析……")
        df = df.rename(columns={"as_of": "trade_date"})
    else:
        raise ValueError("未在基础数据集中找到日期列（期望列名之一：trade_date / trading_date / date / as_of）")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def fmt_pct(x: float) -> str:
    if np.isnan(x):
        return "NA"
    return f"{x * 100:.2f}%"


def fmt_float(x: float) -> str:
    if np.isnan(x):
        return "NA"
    return f"{x:.4f}"


@dataclass
class PerfStats:
    n_days: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float


def compute_perf(returns: pd.Series) -> PerfStats:
    """
    根据每日收益序列计算整体绩效指标。
    returns: 每日收益，单位为小数（例如 0.01 表示 +1%）
    """
    returns = returns.dropna()
    n_days = len(returns)
    if n_days == 0:
        return PerfStats(0, np.nan, np.nan, np.nan, np.nan, np.nan)

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0

    ann_vol = returns.std(ddof=1) * np.sqrt(252.0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_dd = drawdown.min()  # 通常为负值

    return PerfStats(
        n_days=n_days,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


# ============================== 主逻辑 ==============================

def load_base_dataset(job_id: str) -> pd.DataFrame:
    ds_path = PROJECT_ROOT / "data" / "datasets" / f"{job_id}.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(f"找不到基础数据集：{ds_path}")
    log("INFO", f"读取基础数据集：{ds_path}")
    df = pd.read_parquet(ds_path)
    df = ensure_trade_date(df)
    return df


def parse_as_of_from_scores_path(path: Path) -> Optional[pd.Timestamp]:
    """
    从文件名中解析 as_of 日期。

    期望文件名形如：
        u1_scores_{job_id}_{tag}_20251031_top3.csv
    即倒数第二段为 YYYYMMDD。
    """
    stem = path.stem  # 不带扩展名
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # 一般形式：u1_scores + job_id... + tag... + YYYYMMDD + top3
    cand = parts[-2]
    if not re.fullmatch(r"\d{8}", cand):
        return None
    try:
        return pd.to_datetime(cand, format="%Y%m%d")
    except Exception:
        return None


def collect_scores_files(job_id: str, tag: str, top_k: int) -> List[Tuple[pd.Timestamp, Path]]:
    """
    在 reports/ 下搜集所有符合模式的 top_k 文件，并按日期排序。
    """
    pattern = f"u1_scores_{job_id}_{tag}_*_top{top_k}.csv"
    report_dir = PROJECT_ROOT / "reports"
    files = sorted(report_dir.glob(pattern))

    results: List[Tuple[pd.Timestamp, Path]] = []
    for p in files:
        as_of = parse_as_of_from_scores_path(p)
        if as_of is None:
            log("WARN", f"无法从文件名解析日期，已忽略：{p.name}")
            continue
        results.append((as_of, p))

    results.sort(key=lambda x: x[0])
    log("INFO", f"找到历史打分文件 {len(results)} 个（top{top_k}）。")
    return results


def simulate_strategy(
    base_df: pd.DataFrame,
    scores_files: List[Tuple[pd.Timestamp, Path]],
    ret_col: str,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据每天 top_k 选股 + ret_col 做 T+1 收益回放。
    返回：
        daily_df: 含每日策略收益和全市场收益
        trades_df: 每日交易明细（可选）
    """
    if ret_col not in base_df.columns:
        raise KeyError(f"基础数据集中找不到收益列：{ret_col}")

    records_daily = []
    records_trades = []

    # 确保基础数据按日期索引加速过滤
    base_df = base_df.copy()
    base_df["trade_date"] = pd.to_datetime(base_df["trade_date"])
    base_df.set_index("trade_date", inplace=True)

    for as_of, csv_path in scores_files:
        day_str = as_of.strftime("%Y-%m-%d")
        log("Replay", f"模拟交易日 {day_str} ……")

        scores_df = pd.read_csv(csv_path)
        if "code" not in scores_df.columns:
            raise KeyError(f"{csv_path.name} 中未找到 code 列")

        # 仅取前 top_k 行（文件本身应该已经是 top_k）
        codes = scores_df["code"].head(top_k).tolist()

        # 取当天基础数据
        if as_of not in base_df.index:
            log("WARN", f"基础数据中不存在交易日 {day_str}，已跳过。")
            continue

        day_all = base_df.loc[as_of]
        # 有可能 day_all 是 Series（当天只有一只），统一转成 DataFrame
        if isinstance(day_all, pd.Series):
            day_all = day_all.to_frame().T

        # 全市场等权收益（当做“指数”）
        mkt_ret = day_all[ret_col].dropna().mean()

        # 策略持仓（top_k）
        day_sel = day_all[day_all["code"].isin(codes)].copy()
        day_sel = day_sel.dropna(subset=[ret_col])

        n_trades = len(day_sel)
        if n_trades == 0:
            log("WARN", f"{day_str} top{top_k} 中无有效 ret 值，视为当日空仓。")
            strat_ret = 0.0
            hit_ratio = np.nan
        else:
            strat_ret = day_sel[ret_col].mean()
            hit_ratio = (day_sel[ret_col] > 0).mean()

        records_daily.append(
            {
                "trade_date": as_of,
                "strategy_ret": strat_ret,
                "market_ret": mkt_ret if not np.isnan(mkt_ret) else 0.0,
                "n_trades": n_trades,
                "hit_ratio": hit_ratio,
            }
        )

        for _, row in day_sel.iterrows():
            records_trades.append(
                {
                    "trade_date": as_of,
                    "code": row["code"],
                    "ret": row[ret_col],
                }
            )

    if not records_daily:
        raise RuntimeError("没有得到任何有效的历史交易日，无法生成策略回放结果。")

    daily_df = pd.DataFrame(records_daily).sort_values("trade_date")
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
    daily_df.set_index("trade_date", inplace=True)

    trades_df = pd.DataFrame(records_trades)
    return daily_df, trades_df


def yearly_stats_from_daily(returns: pd.Series) -> pd.DataFrame:
    """
    根据每日收益计算按年份拆分的绩效。
    """
    if returns.empty:
        return pd.DataFrame()

    df = returns.to_frame("ret")
    df["year"] = df.index.year

    rows = []
    for year, grp in df.groupby("year"):
        stats = compute_perf(grp["ret"])
        rows.append(
            {
                "year": int(year),
                "n_days": stats.n_days,
                "total_return": stats.total_return,
                "ann_return": stats.ann_return,
                "ann_vol": stats.ann_vol,
                "sharpe": stats.sharpe,
                "max_drawdown": stats.max_drawdown,
            }
        )

    return pd.DataFrame(rows).sort_values("year")


def save_markdown_report(
    job_id: str,
    tag: str,
    ret_col: str,
    top_k: int,
    daily_df: pd.DataFrame,
    strat_perf: PerfStats,
    mkt_perf: PerfStats,
    yearly_strat: pd.DataFrame,
    yearly_mkt: pd.DataFrame,
) -> Path:
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    out_path = report_dir / f"u1_strategy_replay_report_{job_id}_{tag}_{ret_col}_top{top_k}.md"
    log("INFO", f"开始写入策略回放报告：{out_path}")

    n_days = strat_perf.n_days
    total_trades = int(daily_df["n_trades"].sum())

    with out_path.open("w", encoding="utf-8") as f:
        def w(line: str = "") -> None:
            f.write(line + "\n")

        w(f"# U1 策略级历史回放报告")
        w("")
        w(f"- **job_id**：`{job_id}`")
        w(f"- **tag**：`{tag}`")
        w(f"- **ret_col**：`{ret_col}`")
        w(f"- **每天持仓**：top{top_k} 等权买入，T+1 收盘卖出")
        w(f"- **交易日数量**：{n_days}")
        w(f"- **总交易笔数**：{total_trades}")
        w("")

        # 1. 整体表现
        w("## 1. 整体表现（策略 vs 全市场等权）")
        w("")
        w("| 指标 | 策略 | 全市场等权 |")
        w("| ---- | ---- | ---------- |")
        w(f"| 交易日数量 | {strat_perf.n_days} | {mkt_perf.n_days} |")
        w(f"| 累计收益 | {fmt_pct(strat_perf.total_return)} | {fmt_pct(mkt_perf.total_return)} |")
        w(f"| 年化收益 | {fmt_pct(strat_perf.ann_return)} | {fmt_pct(mkt_perf.ann_return)} |")
        w(f"| 年化波动 | {fmt_pct(strat_perf.ann_vol)} | {fmt_pct(mkt_perf.ann_vol)} |")
        w(f"| Sharpe | {fmt_float(strat_perf.sharpe)} | {fmt_float(mkt_perf.sharpe)} |")
        w(f"| 最大回撤 | {fmt_pct(strat_perf.max_drawdown)} | {fmt_pct(mkt_perf.max_drawdown)} |")
        w("")

        # 2. 按年份拆分
        w("## 2. 按年份拆分表现")
        w("")
        if yearly_strat.empty:
            w("（无数据）")
        else:
            w("### 2.1 策略按年份表现")
            w("")
            w("| 年份 | 交易日数 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |")
            w("| ---- | -------- | -------- | -------- | -------- | ------ | -------- |")
            for _, row in yearly_strat.iterrows():
                w(
                    f"| {int(row['year'])} "
                    f"| {int(row['n_days'])} "
                    f"| {fmt_pct(row['total_return'])} "
                    f"| {fmt_pct(row['ann_return'])} "
                    f"| {fmt_pct(row['ann_vol'])} "
                    f"| {fmt_float(row['sharpe'])} "
                    f"| {fmt_pct(row['max_drawdown'])} |"
                )
            w("")

        if not yearly_mkt.empty:
            w("### 2.2 全市场等权按年份表现（对比基准）")
            w("")
            w("| 年份 | 交易日数 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |")
            w("| ---- | -------- | -------- | -------- | -------- | ------ | -------- |")
            for _, row in yearly_mkt.iterrows():
                w(
                    f"| {int(row['year'])} "
                    f"| {int(row['n_days'])} "
                    f"| {fmt_pct(row['total_return'])} "
                    f"| {fmt_pct(row['ann_return'])} "
                    f"| {fmt_pct(row['ann_vol'])} "
                    f"| {fmt_float(row['sharpe'])} "
                    f"| {fmt_pct(row['max_drawdown'])} |"
                )
            w("")

        # 3. 其它简单诊断
        w("## 3. 其它简单诊断")
        w("")
        avg_hit = daily_df["hit_ratio"].mean()
        w(f"- 平均日内胜率（top{top_k} 内个股上涨比例的平均）：{fmt_pct(avg_hit) if not np.isnan(avg_hit) else 'NA'}")
        w(f"- 单日最大回撤（策略）：{fmt_pct(((1 + daily_df['strategy_ret']).cumprod().cummax() / (1 + daily_df['strategy_ret']).cumprod() - 1).max())}")
        w("")

    log("OK", "策略回放报告写入完成。")
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="U1 策略级历史回放脚本")
    parser.add_argument(
        "--job-id",
        required=True,
        help="数据集 job_id，例如 ultrashort_main",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="U1 模型 tag，例如 u1_v1_base_rf",
    )
    parser.add_argument(
        "--ret-col",
        required=True,
        help="用于回放收益的列名，例如 ret_1",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每天持仓的 top_k 数量（默认 3）",
    )

    args = parser.parse_args(argv)

    log(
        "START",
        f"开始策略级历史回放：job_id={args.job_id}, tag={args.tag}, "
        f"ret_col={args.ret_col}, top_k={args.top_k}",
    )

    base_df = load_base_dataset(args.job_id)
    scores_files = collect_scores_files(args.job_id, args.tag, args.top_k)

    if not scores_files:
        raise RuntimeError("没有找到任何历史打分文件（u1_scores_*_top{top_k}.csv），无法进行策略回放。")

    daily_df, trades_df = simulate_strategy(
        base_df=base_df,
        scores_files=scores_files,
        ret_col=args.ret_col,
        top_k=args.top_k,
    )

    # 计算整体绩效
    strat_perf = compute_perf(daily_df["strategy_ret"])
    mkt_perf = compute_perf(daily_df["market_ret"])

    # 按年份拆分
    yearly_strat = yearly_stats_from_daily(daily_df["strategy_ret"])
    yearly_mkt = yearly_stats_from_daily(daily_df["market_ret"])

    # 保存报告
    report_path = save_markdown_report(
        job_id=args.job_id,
        tag=args.tag,
        ret_col=args.ret_col,
        top_k=args.top_k,
        daily_df=daily_df,
        strat_perf=strat_perf,
        mkt_perf=mkt_perf,
        yearly_strat=yearly_strat,
        yearly_mkt=yearly_mkt,
    )

    # 也顺手把每日收益 / 交易明细存一下，方便你后面做更细分析
    out_daily_csv = PROJECT_ROOT / "reports" / f"u1_strategy_replay_daily_{args.job_id}_{args.tag}_{args.ret_col}_top{args.top_k}.csv"
    out_trades_csv = PROJECT_ROOT / "reports" / f"u1_strategy_replay_trades_{args.job_id}_{args.tag}_{args.ret_col}_top{args.top_k}.csv"
    daily_df.to_csv(out_daily_csv)
    trades_df.to_csv(out_trades_csv, index=False)

    log("OK", f"每日收益已保存到：{out_daily_csv}")
    log("OK", f"交易明细已保存到：{out_trades_csv}")
    log("DONE", f"策略级历史回放完成，报告：{report_path}")


if __name__ == "__main__":
    main()
