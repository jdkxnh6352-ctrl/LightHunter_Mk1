import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ==============================
# 基础工具
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[UISignal] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}")


# ==============================
# 数据集加载 & 列名自动识别
# ==============================

def detect_date_col(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    在基础数据集中自动识别“日期列”。

    优先级：
    1) trade_date
    2) trading_date  -> 自动重命名为 trade_date
    3) date
    4) as_of
    """
    candidates = ["trade_date", "trading_date", "date", "as_of"]
    for col in candidates:
        if col in df.columns:
            if col == "trading_date":
                # 统一改名为 trade_date，便于后续脚本复用
                df = df.rename(columns={"trading_date": "trade_date"})
                log("数据集中没有 trade_date 列，发现 trading_date 列，"
                    "已自动重命名为 trade_date 用于分析……")
                return "trade_date", df
            else:
                return col, df

    error("未在数据集中找到日期列（期望列名之一：trade_date / trading_date / date / as_of）")
    raise SystemExit(1)


def detect_code_col(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    自动识别股票代码列：code / ts_code / stock_code
    统一改成 code。
    """
    candidates = ["code", "ts_code", "stock_code"]
    for col in candidates:
        if col in df.columns:
            if col != "code":
                df = df.rename(columns={col: "code"})
                log(f"股票代码列 {col} 已重命名为 code。")
            return "code", df

    error("未在数据集中找到股票代码列（期望列名之一：code / ts_code / stock_code）")
    raise SystemExit(1)


def load_base_dataset(job_id: str) -> Tuple[pd.DataFrame, str]:
    """
    加载基础数据集 data/datasets/<job_id>.parquet
    并自动识别日期列、代码列。
    """
    path = PROJECT_ROOT / "data" / "datasets" / f"{job_id}.parquet"
    if not path.exists():
        error(f"找不到基础数据集文件: {path}")
        raise SystemExit(1)

    log(f"读取基础数据集: {path}")
    df = pd.read_parquet(path)

    date_col, df = detect_date_col(df)
    _, df = detect_code_col(df)

    # 统一转成 datetime 类型，后续 merge / groupby 更稳
    df[date_col] = pd.to_datetime(df[date_col])
    return df, date_col


# ==============================
# 打分文件加载
# ==============================

def _parse_as_of_from_filename(path: Path) -> pd.Timestamp:
    """
    从文件名里尝试解析类似 *_20251031_full.csv 的 as_of 日期。
    """
    m = re.search(r"_(\d{8})_full\.csv$", path.name)
    if not m:
        raise ValueError(f"无法从文件名中解析 as_of 日期: {path}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def load_scores(job_id: str, tag: str) -> pd.DataFrame:
    """
    加载 reports 目录下的所有 u1_scores_<job_id>_<tag>_*_full.csv，
    并统一标准列名：
        - 日期列: as_of (datetime)
        - 代码列: code
        - 打分列: u1_score
    """
    pattern = PROJECT_ROOT / "reports" / f"u1_scores_{job_id}_{tag}_*_full.csv"
    paths = sorted(pattern.parent.glob(pattern.name))

    if not paths:
        error(f"在 reports 目录下未找到任何打分文件: 模式 {pattern.name}")
        raise SystemExit(1)

    log(f"发现打分文件 {len(paths)} 个。")

    all_dfs = []
    total_rows = 0

    for p in paths:
        df = pd.read_csv(p)

        # 代码列
        _, df = detect_code_col(df)

        # 打分列
        score_col = None
        for cand in ["u1_score", "score", "pred"]:
            if cand in df.columns:
                score_col = cand
                break
        if score_col is None:
            error(f"打分文件缺少打分列（期望列名之一：u1_score / score / pred）：{p}")
            raise SystemExit(1)
        if score_col != "u1_score":
            df = df.rename(columns={score_col: "u1_score"})

        # as_of 日期列
        if "as_of" in df.columns:
            df["as_of"] = pd.to_datetime(df["as_of"])
        else:
            # 文件名里推
            as_of = _parse_as_of_from_filename(p)
            df["as_of"] = as_of

        all_dfs.append(df[["as_of", "code", "u1_score"]])
        total_rows += df.shape[0]

    scores_df = pd.concat(all_dfs, ignore_index=True)
    log(f"打分样本总行数: {total_rows}")
    return scores_df


# ==============================
# 合并打分与真实收益
# ==============================

def join_scores_with_returns(
    base_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    date_col: str,
    ret_col: str,
) -> pd.DataFrame:
    """
    将打分样本与基础数据中的真实收益 ret_col 对齐。

    假设：
        scores_df 中 as_of 与基础数据中的 {date_col} 对齐，
        即“信号产生日”的标签就是该日的 ret_1 / ret_5 / ...。
    """
    if ret_col not in base_df.columns:
        error(f"基础数据集中不存在收益列: {ret_col}")
        raise SystemExit(1)

    merged = scores_df.merge(
        base_df[[date_col, "code", ret_col]],
        left_on=["as_of", "code"],
        right_on=[date_col, "code"],
        how="left",
    )

    n_missing = merged[ret_col].isna().sum()
    if n_missing > 0:
        warn(f"在合并后样本中，有 {n_missing} 条记录缺少 {ret_col}，"
             f"这部分将被排除在统计之外。")

    merged = merged.dropna(subset=[ret_col, "u1_score"]).copy()
    merged = merged.drop(columns=[date_col])
    return merged


# ==============================
# 性能统计：top_k 策略 + 分桶
# ==============================

def max_drawdown_from_returns(rets: np.ndarray) -> float:
    """
    根据日度收益序列计算最大回撤（基于净值曲线）。
    """
    if len(rets) == 0:
        return float("nan")

    curve = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak) - 1.0
    return float(dd.min())


def calc_topk_stats(
    merged: pd.DataFrame,
    ret_col: str,
    top_k_list: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    以“每天等权买入 top_k 组合”的方式，计算收益表现。
    返回：
        {k: {n_days, cum_ret, ann_ret, ann_vol, sharpe, max_dd}}
    """
    results: Dict[int, Dict[str, float]] = {}
    if merged.empty:
        for k in top_k_list:
            results[k] = {
                "n_days": 0,
                "cum_ret": float("nan"),
                "ann_ret": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_dd": float("nan"),
            }
        return results

    grouped = merged.groupby("as_of", sort=True)

    for k in top_k_list:
        daily_rets = []

        for _, g in grouped:
            g_sorted = g.sort_values("u1_score", ascending=False)
            if g_sorted.shape[0] < k:
                continue
            topk = g_sorted.head(k)
            daily_rets.append(topk[ret_col].mean())

        daily_rets = np.asarray(daily_rets, dtype=float)
        n_days = int(daily_rets.shape[0])

        if n_days == 0:
            results[k] = {
                "n_days": 0,
                "cum_ret": float("nan"),
                "ann_ret": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_dd": float("nan"),
            }
            continue

        cum_ret = float(np.prod(1.0 + daily_rets) - 1.0)
        ann_factor = 252.0 / n_days
        ann_ret = float((1.0 + cum_ret) ** ann_factor - 1.0)
        ann_vol = float(daily_rets.std(ddof=1) * math.sqrt(252.0))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-8 else float("nan")
        max_dd = max_drawdown_from_returns(daily_rets)

        results[k] = {
            "n_days": n_days,
            "cum_ret": cum_ret,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
        }

    return results


def calc_bucket_stats(
    merged: pd.DataFrame,
    ret_col: str,
    n_buckets: int,
) -> pd.DataFrame:
    """
    按打分 u1_score 分桶（默认 10 桶），看不同分数段的平均收益。
    bucket 编号越大，代表打分越高。
    """
    if merged.empty:
        return pd.DataFrame()

    df = merged.copy()
    # qcut 可能会因为分布问题桶数 < n_buckets，因此允许 duplicates="drop"
    df["bucket"] = pd.qcut(
        df["u1_score"],
        q=n_buckets,
        labels=False,
        duplicates="drop",
    )

    stats = (
        df.groupby("bucket")[ret_col]
        .agg(["count", "mean", "std"])
        .sort_index(ascending=False)
        .reset_index()
    )
    stats["bucket"] = stats["bucket"].astype(int)
    return stats


# ==============================
# 报告输出（markdown）
# ==============================

def save_markdown_report(
    job_id: str,
    tag: str,
    ret_col: str,
    merged: pd.DataFrame,
    topk_stats: Dict[int, Dict[str, float]],
    bucket_stats: pd.DataFrame,
) -> Path:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"u1_signal_report_{job_id}_{tag}_{ret_col}.md"

    lines = []
    lines.append(f"# U1 信号体检报告\n")
    lines.append(f"- 任务 ID：`{job_id}`")
    lines.append(f"- 模型标签：`{tag}`")
    lines.append(f"- 收益列：`{ret_col}`")
    lines.append(f"- 有效样本数：{merged.shape[0]}\n")

    # === top_k 组合表现 ===
    lines.append("## 1. 按日等权 top_k 组合表现\n")
    lines.append("| top_k | 参与交易日数 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |")
    lines.append("|------:|------------:|--------:|--------:|--------:|-------:|--------:|")

    for k in sorted(topk_stats.keys()):
        s = topk_stats[k]
        if s["n_days"] == 0:
            line = f"| {k} | 0 | - | - | - | - | - |"
        else:
            line = (
                f"| {k} | {s['n_days']} | "
                f"{s['cum_ret']:.2%} | {s['ann_ret']:.2%} | {s['ann_vol']:.2%} | "
                f"{s['sharpe']:.2f} | {s['max_dd']:.2%} |"
            )
        lines.append(line)

    lines.append("")

    # === 分桶表现 ===
    lines.append("## 2. 按模型打分分桶的收益表现\n")
    if bucket_stats.empty:
        lines.append("> 暂无可用样本，无法进行分桶统计。\n")
    else:
        lines.append("| bucket(高 -> 低) | 样本数 | 平均收益 | 收益标准差 |")
        lines.append("|-----------------:|-------:|--------:|-----------:|")
        for _, row in bucket_stats.iterrows():
            lines.append(
                f"| {row['bucket']} | {int(row['count'])} | "
                f"{row['mean']:.2%} | {row['std']:.2%} |"
            )
        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


# ==============================
# CLI
# ==============================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 信号体检报告生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-id", required=True, help="数据集 / 任务 ID，例如 ultrashort_main")
    parser.add_argument("--tag", required=True, help="模型标签，例如 u1_v1_base_rf")
    parser.add_argument("--ret-col", required=True, help="用于评估的收益列，例如 ret_1 / ret_5 / ret_20")
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="需要评估的 top_k 组合列表，例如: 3 5 10",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=10,
        help="按打分分桶的桶数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log(
        "开始生成信号体检报告："
        f"job_id={args.job_id}, tag={args.tag}, "
        f"ret_col={args.ret_col}, top_k={args.top_k}, n_buckets={args.n_buckets}"
    )
    print("=" * 70)

    base_df, date_col = load_base_dataset(args.job_id)
    scores_df = load_scores(args.job_id, args.tag)
    merged = join_scores_with_returns(base_df, scores_df, date_col, args.ret_col)

    if merged.empty:
        error("合并后样本为空，无法生成统计。请检查打分文件与基础数据是否对齐。")
        raise SystemExit(1)

    topk_stats = calc_topk_stats(merged, args.ret_col, args.top_k)
    bucket_stats = calc_bucket_stats(merged, args.ret_col, args.n_buckets)

    report_path = save_markdown_report(
        args.job_id, args.tag, args.ret_col, merged, topk_stats, bucket_stats
    )
    log(f"信号体检报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
