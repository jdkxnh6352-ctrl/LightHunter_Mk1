# tools/u1_factor_lab.py
"""
U1 因子体检 / 单因子实验脚本

功能：
- 读取 data/datasets/{job_id}.parquet
- 过滤日期区间（以及可选的价格、成交额过滤）
- 对指定的一组特征，相对于标签列（默认 ret_1）做：
    * 缺失值 & 基本分布统计
    * 与标签的 Pearson / Spearman 相关
    * 按日期分组的日度 IC（均值、IR、正 IC 比例）
    * 分层收益：每个交易日按特征分位数分组，统计各层的平均收益
- 把所有结果写成一份 Markdown 报告：reports/u1_factor_lab_{job_id}.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s)


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NaN"
    return f"{x * 100:.2f}%"


def _fmt_float(x: float) -> str:
    if pd.isna(x):
        return "NaN"
    return f"{x:.4f}"


@dataclass
class FactorStats:
    name: str
    n_total: int
    n_valid: int
    n_missing: int
    missing_ratio: float
    desc: pd.Series

    pearson: float
    spearman: float

    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_pos_ratio: float
    ic_n_days: int

    # 分层收益：quantile -> (n, mean_ret)
    bucket_stats: List[Tuple[int, int, float]]


# ----------------------------------------------------------------------
# 加载 & 过滤数据
# ----------------------------------------------------------------------


def load_dataset(job_id: str) -> pd.DataFrame:
    data_path = Path("data") / "datasets" / f"{job_id}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据集：{data_path}")
    print(f"[U1Factor] 读取数据集: {data_path}")
    df = pd.read_parquet(data_path)

    # 兼容 trading_date -> trade_date
    if "trade_date" not in df.columns and "trading_date" in df.columns:
        print(
            "[U1Factor] 数据集中没有 trade_date 列，发现 trading_date 列，"
            "已自动重命名为 trade_date 用于分析……"
        )
        df = df.rename(columns={"trading_date": "trade_date"})

    return df


def filter_universe(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    min_price: Optional[float],
    max_price: Optional[float],
    min_amount: Optional[float],
) -> pd.DataFrame:
    out = df.copy()

    # 日期过滤
    if "trade_date" in out.columns:
        dates = pd.to_datetime(out["trade_date"], errors="coerce")
        if start_date is not None:
            out = out[dates >= start_date]
        if end_date is not None:
            out = out[dates <= end_date]

    # 价格过滤（尽量猜测价格列名称）
    price_col = None
    for cand in ["close", "close_price", "adj_close", "price"]:
        if cand in out.columns:
            price_col = cand
            break

    if price_col is not None:
        if min_price is not None:
            out = out[out[price_col] >= float(min_price)]
        if max_price is not None:
            out = out[out[price_col] <= float(max_price)]
    else:
        if min_price is not None or max_price is not None:
            print(
                "[U1Factor] 警告：未发现价格列（close / close_price / adj_close / price），"
                "将忽略 min-price / max-price 过滤。"
            )

    # 成交额过滤（amount / 成交额）
    amt_col = None
    for cand in ["amount", "成交额", "turnover"]:
        if cand in out.columns:
            amt_col = cand
            break

    if amt_col is not None and min_amount is not None:
        out = out[out[amt_col] >= float(min_amount)]
    elif min_amount is not None:
        print(
            "[U1Factor] 警告：未发现成交额列（amount / 成交额 / turnover），"
            "将忽略 min-amount 过滤。"
        )

    return out


# ----------------------------------------------------------------------
# 因子统计
# ----------------------------------------------------------------------


def analyze_factor(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    date_col: str,
    n_quantiles: int = 5,
) -> FactorStats:
    if factor_col not in df.columns:
        raise KeyError(f"数据集中没有因子列: {factor_col}")
    if label_col not in df.columns:
        raise KeyError(f"数据集中没有标签列: {label_col}")
    if date_col not in df.columns:
        raise KeyError(f"数据集中没有日期列: {date_col}")

    series_x = pd.to_numeric(df[factor_col], errors="coerce")
    series_y = pd.to_numeric(df[label_col], errors="coerce")
    mask = series_x.notna() & series_y.notna()
    x = series_x[mask]
    y = series_y[mask]

    n_total = len(df)
    n_valid = mask.sum()
    n_missing = n_total - n_valid
    missing_ratio = n_missing / n_total if n_total > 0 else np.nan

    desc = x.describe()

    pearson = float(x.corr(y)) if n_valid > 1 else np.nan
    spearman = float(x.corr(y, method="spearman")) if n_valid > 1 else np.nan

    # 日度 IC
    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "x": series_x,
            "y": series_y,
        }
    ).dropna()

    def _daily_ic(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return np.nan
        return float(g["x"].corr(g["y"]))

    ic_by_day = tmp.groupby("date").apply(_daily_ic)
    ic_by_day = ic_by_day.dropna()

    if len(ic_by_day) > 0:
        ic_mean = float(ic_by_day.mean())
        ic_std = float(ic_by_day.std(ddof=1)) if len(ic_by_day) > 1 else 0.0
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else np.nan
        ic_pos_ratio = float((ic_by_day > 0).mean())
        ic_n_days = int(len(ic_by_day))
    else:
        ic_mean = ic_std = ic_ir = ic_pos_ratio = np.nan
        ic_n_days = 0

    # 分层收益：每个交易日按 x 分位数分层，然后统计各层整体平均收益
    bucket_stats: List[Tuple[int, int, float]] = []
    if n_quantiles >= 2:
        df_q = tmp.copy()
        # 对每个交易日做分位数
        def _assign_bucket(g: pd.DataFrame) -> pd.Series:
            n = len(g)
            if n < n_quantiles:
                # 样本太少就整组放在中间 bucket
                return pd.Series([n_quantiles // 2] * n, index=g.index)
            try:
                return pd.qcut(
                    g["x"],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except Exception:
                # 全相等等情况
                return pd.Series([n_quantiles // 2] * n, index=g.index)

        df_q["bucket"] = df_q.groupby("date", group_keys=False).apply(_assign_bucket)
        df_q = df_q.dropna(subset=["bucket"])
        df_q["bucket"] = df_q["bucket"].astype(int)

        for b in range(n_quantiles):
            g = df_q[df_q["bucket"] == b]
            n_b = int(len(g))
            mean_ret = float(g["y"].mean()) if n_b > 0 else np.nan
            bucket_stats.append((b, n_b, mean_ret))

    return FactorStats(
        name=factor_col,
        n_total=n_total,
        n_valid=n_valid,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
        desc=desc,
        pearson=pearson,
        spearman=spearman,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_pos_ratio=ic_pos_ratio,
        ic_n_days=ic_n_days,
        bucket_stats=bucket_stats,
    )


# ----------------------------------------------------------------------
# 报告生成
# ----------------------------------------------------------------------


def build_markdown_report(
    job_id: str,
    df: pd.DataFrame,
    factors: List[str],
    label_col: str,
    date_col: str,
    stats_list: List[FactorStats],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    min_price: Optional[float],
    max_price: Optional[float],
    min_amount: Optional[float],
    n_quantiles: int,
) -> str:
    lines: List[str] = []

    lines.append(f"# U1 因子体检报告 - {job_id}")
    lines.append("")
    lines.append("## 1. 数据总体信息")
    lines.append("")
    lines.append(f"- 样本总数：**{len(df)}**")
    if "trade_date" in df.columns:
        dates = pd.to_datetime(df["trade_date"], errors="coerce")
        lines.append(f"- 日期范围：{dates.min().date()} ~ {dates.max().date()}")
    lines.append(f"- 标签列：`{label_col}`")
    lines.append(f"- 日期列：`{date_col}`")
    lines.append(f"- 分层数量（分位数）：{n_quantiles}")
    lines.append("")
    lines.append("过滤条件：")
    lines.append(f"- 起始日期：{start_date.date() if start_date is not None else '无'}")
    lines.append(f"- 结束日期：{end_date.date() if end_date is not None else '无'}")
    lines.append(f"- 最小价格：{min_price if min_price is not None else '无'}")
    lines.append(f"- 最大价格：{max_price if max_price is not None else '无'}")
    lines.append(f"- 最小成交额：{min_amount if min_amount is not None else '无'}")
    lines.append("")
    lines.append("## 2. 因子列表")
    lines.append("")
    for f in factors:
        lines.append(f"- `{f}`")
    lines.append("")

    for fs in stats_list:
        lines.append(f"---")
        lines.append(f"## 因子 `{fs.name}`")
        lines.append("")
        lines.append("### 2.1 缺失值与分布")
        lines.append("")
        lines.append(f"- 总样本数：{fs.n_total}")
        lines.append(f"- 有效样本数：{fs.n_valid}")
        lines.append(f"- 缺失样本数：{fs.n_missing}")
        lines.append(f"- 缺失比例：{_fmt_pct(fs.missing_ratio)}")
        lines.append("")
        lines.append("描述性统计：")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if k in fs.desc.index:
                lines.append(f"| {k} | {_fmt_float(float(fs.desc[k]))} |")
        lines.append("")

        lines.append("### 2.2 与标签相关性")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| Pearson 相关 | {_fmt_float(fs.pearson)} |")
        lines.append(f"| Spearman 相关 | {_fmt_float(fs.spearman)} |")
        lines.append("")
        lines.append("日度 IC（按 trade_date 分组）：")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| 平均 IC | {_fmt_float(fs.ic_mean)} |")
        lines.append(f"| IC 标准差 | {_fmt_float(fs.ic_std)} |")
        lines.append(f"| IR = IC_mean / IC_std | {_fmt_float(fs.ic_ir)} |")
        lines.append(f"| 正 IC 比例 | {_fmt_pct(fs.ic_pos_ratio)} |")
        lines.append(f"| 有效交易日数 | {fs.ic_n_days} |")
        lines.append("")

        lines.append("### 2.3 分层收益统计（基于日度收益）")
        lines.append("")
        lines.append(
            "分层规则：每个交易日按该因子做分位数切分（"
            f"{n_quantiles} 层，0 为最低层，{n_quantiles - 1} 为最高层），"
            "然后汇总各层的总体平均标签收益。"
        )
        lines.append("")
        lines.append("| 分层 (bucket) | 样本数 | 平均日度收益 |")
        lines.append("|---------------|--------|--------------|")
        for b, n_b, mean_ret in fs.bucket_stats:
            lines.append(f"| {b} | {n_b} | {_fmt_float(mean_ret)} |")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.u1_factor_lab",
        description="U1 因子体检 / 单因子实验脚本",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="数据集 ID，例如 ultrashort_main（会读取 data/datasets/{job_id}.parquet）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="起始日期，YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="结束日期，YYYY-MM-DD",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default="ret_1",
        help="标签列名，默认 ret_1",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="trade_date",
        help="日期列名，默认 trade_date（若不存在会尝试用 trading_date）",
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help=(
            "要分析的因子列表，逗号分隔，例如："
            "log_amount,log_volume,log_amt_mean_20,amt_to_mean_20,vol_20,ret_1,ret_5,ret_20,rev_1"
        ),
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        help="最小价格过滤，可选",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="最大价格过滤，可选",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=None,
        help="最小成交额过滤，可选，单位 = 原始数据单位",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=5,
        help="分层数量（分位数），默认 5 层",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 Markdown 报告路径，默认 reports/u1_factor_lab_{job_id}.md",
    )

    args = parser.parse_args(argv)

    job_id = args.job_id
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    label_col = args.ret_col
    date_col = args.date_col
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    n_quantiles = max(int(args.n_quantiles), 2)

    df = load_dataset(job_id)
    df = filter_universe(
        df,
        start_date=start_date,
        end_date=end_date,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
    )

    print(f"[U1Factor] 过滤后样本数: {len(df)}")

    stats_list: List[FactorStats] = []
    for f in features:
        print(f"[U1Factor] 分析因子: {f}")
        try:
            fs = analyze_factor(
                df=df,
                factor_col=f,
                label_col=label_col,
                date_col=date_col,
                n_quantiles=n_quantiles,
            )
            stats_list.append(fs)
        except Exception as e:
            print(f"[U1Factor] 因子 {f} 分析失败: {e}")

    if not stats_list:
        print("[U1Factor] 没有成功分析的因子，退出。")
        return

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output is not None
        else reports_dir / f"u1_factor_lab_{job_id}.md"
    )

    md = build_markdown_report(
        job_id=job_id,
        df=df,
        factors=features,
        label_col=label_col,
        date_col=date_col,
        stats_list=stats_list,
        start_date=start_date,
        end_date=end_date,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        n_quantiles=n_quantiles,
    )

    output_path.write_text(md, encoding="utf-8")
    print(f"[U1Factor] 因子体检报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
