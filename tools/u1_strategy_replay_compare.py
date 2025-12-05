import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .u1_strategy_replay import ROOT, compute_metrics


def run_replay_for_k(job_id: str, tag: str, ret_col: str, k: int) -> None:
    """调用 u1_strategy_replay，生成单个 top_k 的回放结果。"""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_strategy_replay",
        "--job-id",
        job_id,
        "--tag",
        tag,
        "--ret-col",
        ret_col,
        "--top-k",
        str(k),
    ]
    print(f"[UIStratCmp] INFO: 开始执行 top_k={k} 的策略回放……")
    subprocess.run(cmd, check=True)


def load_daily_result(job_id: str, tag: str, ret_col: str, k: int) -> pd.DataFrame:
    daily_path = ROOT / "reports" / f"u1_strategy_replay_daily_{job_id}_{tag}_{ret_col}_top{k}.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"找不到 daily 回放文件: {daily_path}")
    print(f"[UIStratCmp] 读取 daily 回放文件: {daily_path}")
    df = pd.read_csv(daily_path, parse_dates=["trade_date"])
    return df


def build_compare_report(job_id: str, tag: str, ret_col: str, top_k_list: List[int]) -> Path:
    rows = []
    for k in top_k_list:
        df = load_daily_result(job_id, tag, ret_col, k)
        m = compute_metrics(df)
        m["top_k"] = k
        rows.append(m)

    cols = ["top_k", "n_days", "total_ret", "ann_ret", "ann_vol", "sharpe", "max_dd", "win_ratio"]
    cmp_df = pd.DataFrame(rows)[cols].sort_values("top_k")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ks_str = "_".join(str(k) for k in top_k_list)
    report_path = reports_dir / f"u1_strategy_replay_compare_{job_id}_{tag}_{ret_col}_top{ks_str}.md"

    # 转成百分比方便阅读
    pct_cols = ["total_ret", "ann_ret", "ann_vol", "max_dd", "win_ratio"]
    for c in pct_cols:
        cmp_df[c] = cmp_df[c] * 100.0

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# U1 策略回放 top_k 对比报告\n\n")
        f.write(f"- 任务: job_id=`{job_id}`, tag=`{tag}`, ret_col=`{ret_col}`\n")
        f.write(f"- 对比的 top_k: {', '.join(str(k) for k in top_k_list)}\n\n")

        f.write("## 关键指标对比\n\n")
        f.write("| top_k | 交易日数 | 累计收益(%) | 年化收益(%) | 年化波动(%) | Sharpe | 最大回撤(%) | 胜率(%) |\n")
        f.write("|-------|----------|------------|------------|------------|--------|------------|--------|\n")
        for _, row in cmp_df.iterrows():
            f.write(
                f"| {int(row['top_k'])} | {int(row['n_days'])} | "
                f"{row['total_ret']:.2f} | {row['ann_ret']:.2f} | {row['ann_vol']:.2f} | "
                f"{row['sharpe']:.2f} | {row['max_dd']:.2f} | {row['win_ratio']:.2f} |\n"
            )

    print(f"[UIStratCmp] DONE: top_k={top_k_list} 回放对比完成, 报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="U1 策略回放 top_k 对比脚本")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ret-col", required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        required=True,
        help="需要对比的 top_k 列表，例如: --top-k 3 5 10",
    )
    args = parser.parse_args()

    job_id = args.job_id
    tag = args.tag
    ret_col = args.ret_col
    top_k_list = sorted(set(args.top_k))

    print(
        f"[UIStratCmp] START: 开始执行 top_k 对比: job_id={job_id}, tag={tag}, "
        f"ret_col={ret_col}, top_k={top_k_list}"
    )

    # 逐个回放（会自动生成各自的 daily + trades + 单独报告）
    for k in top_k_list:
        run_replay_for_k(job_id, tag, ret_col, k)

    build_compare_report(job_id, tag, ret_col, top_k_list)


if __name__ == "__main__":
    main()
