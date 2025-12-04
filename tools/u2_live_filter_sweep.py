# tools/u2_live_filter_sweep.py
# 一键对比多组 filter_tag，调用 u2_live_batch_backtest 回测并自动排行

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def run_one_backtest(
    project_root: Path,
    config_path: str,
    start_date: str,
    end_date: str,
    tag: str,
    filter_tag: str,
) -> bool:
    """
    调用 `python -m tools.u2_live_batch_backtest` 跑一遍回测。
    返回 True 表示子进程返回码为 0，认为成功。
    """
    cmd = [
        sys.executable,
        "-m",
        "tools.u2_live_batch_backtest",
        "--config",
        config_path,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--tag",
        tag,
        "--filter-tag",
        filter_tag,
    ]

    print(f"[SWEEP] 命令：{' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print("[SWEEP] 被手工中断，退出。")
        sys.exit(1)
    except Exception as e:
        print(f"[SWEEP][ERROR] 启动回测子进程失败：{e}")
        return False

    if proc.returncode != 0:
        print(
            f"[SWEEP][ERROR] 回测失败，filter_tag={filter_tag}，"
            f"tag={tag}，返回码={proc.returncode}"
        )
        return False

    return True


def collect_metrics_for_tag(
    project_root: Path,
    reports_dir: Path,
    tag: str,
    filter_tag: str,
) -> Optional[Dict[str, float]]:
    """
    读取 u2_live_batch_backtest 产出的 yearly 文件，汇总成便于排序的一行指标。
    若文件不存在或为空，返回 None。
    """
    yearly_path = reports_dir / f"u2_batch_yearly_{tag}.csv"
    if not yearly_path.exists():
        print(
            f"[SWEEP][WARN] 找不到年度结果文件：{yearly_path}，"
            f"filter_tag={filter_tag} 将无法参与排名。"
        )
        return None

    try:
        df = pd.read_csv(yearly_path)
    except Exception as e:
        print(f"[SWEEP][WARN] 读取 {yearly_path} 失败：{e}")
        return None

    if df.empty:
        print(f"[SWEEP][WARN] {yearly_path} 为空，跳过。")
        return None

    # 兼容列名：按 u2_backtest_stats / u2_live_batch_backtest 的风格
    required_cols = ["year", "n_days", "total_return", "ann_return",
                     "ann_vol", "sharpe", "max_drawdown", "win_ratio"]
    for c in required_cols:
        if c not in df.columns:
            print(
                f"[SWEEP][WARN] {yearly_path} 缺少列 {c}，"
                f"filter_tag={filter_tag} 跳过。"
            )
            return None

    # 加权汇总（按 n_days 权重）
    n_days_total = float(df["n_days"].sum())
    if n_days_total <= 0:
        return None

    def wavg(col: str) -> float:
        return float((df[col] * df["n_days"]).sum() / n_days_total)

    total_return_overall = float((1.0 + df["total_return"]).prod() - 1.0)
    ann_return_overall = wavg("ann_return")
    ann_vol_mean = wavg("ann_vol")
    sharpe_mean = wavg("sharpe")
    max_drawdown_min = float(df["max_drawdown"].min())
    win_ratio_mean = wavg("win_ratio")

    return {
        "filter_tag": filter_tag,
        "tag": tag,
        "n_years": int(df.shape[0]),
        "n_days": int(n_days_total),
        "total_return": total_return_overall,
        "ann_return_overall": ann_return_overall,
        "ann_vol_mean": ann_vol_mean,
        "sharpe_mean": sharpe_mean,
        "max_drawdown_min": max_drawdown_min,
        "win_ratio_mean": win_ratio_mean,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.u2_live_filter_sweep",
        description=(
            "一键对比多组 filter_tag，调用 u2_live_batch_backtest 回测并自动排行。"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/u2_live_config.json",
        help="U2 实盘配置文件路径（默认：config/u2_live_config.json）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="回测开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="回测结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--filter-tags",
        type=str,
        required=True,
        help="要对比的 filter_tag 列表，用逗号分隔，例如：base,aggressive,defensive",
    )
    parser.add_argument(
        "--base-tag",
        type=str,
        default="live_abtest",
        help="回测 tag 前缀，每个方案实际 tag = base_tag + '_' + filter_tag",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ann_return_overall",
        choices=["ann_return_overall", "sharpe_mean"],
        help="排序主指标（默认 ann_return_overall）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    print(f"[SWEEP] 项目根目录: {project_root}")

    reports_dir = project_root / "reports" / "u2_live_batch"
    os.makedirs(reports_dir, exist_ok=True)
    print(f"[SWEEP] 报告目录: {reports_dir}")

    # 解析要对比的 filter_tag 列表
    raw_tags = [t.strip() for t in args.filter_tags.split(",") if t.strip()]
    if not raw_tags:
        print("[SWEEP][ERROR] filter-tags 为空，请至少提供一个。")
        return

    print(f"[SWEEP] 将对比的 filter_tag: {', '.join(raw_tags)}")

    results: List[Dict[str, float]] = []

    # 逐个跑 backtest
    for ft in raw_tags:
        tag = f"{args.base_tag}_{ft}"
        print("\n" + "=" * 60)
        print(f"[SWEEP] 开始回测 filter_tag = {ft} (tag={tag})")

        ok = run_one_backtest(
            project_root=project_root,
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            tag=tag,
            filter_tag=ft,
        )
        if not ok:
            print(f"[SWEEP][ERROR] 回测失败，filter_tag={ft}，返回码非 0。")
            continue

        metrics = collect_metrics_for_tag(
            project_root=project_root,
            reports_dir=reports_dir,
            tag=tag,
            filter_tag=ft,
        )
        if metrics is None:
            print(f"[SWEEP][WARN] 未能收集到 {ft} 的统计指标，跳过。")
            continue

        results.append(metrics)

    # 排名 & 输出
    if not results:
        print("[SWEEP][ERROR] 没有任何成功的回测结果，无法排序。")
        return

    df = pd.DataFrame(results)

    # 主排序指标：metric，辅排序：sharpe 高、回撤小
    sort_cols = [args.metric, "sharpe_mean", "max_drawdown_min"]
    ascending = [False, False, True]
    df_sorted = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    print("\n[SWEEP] 按整体表现排序（Top 10）")
    cols_to_show = [
        "filter_tag",
        "tag",
        "n_years",
        "n_days",
        "total_return",
        "ann_return_overall",
        "ann_vol_mean",
        "sharpe_mean",
        "max_drawdown_min",
        "win_ratio_mean",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_sorted.columns]

    # 格式化打印前 10 条
    with pd.option_context(
        "display.max_rows",
        20,
        "display.max_columns",
        None,
        "display.width",
        140,
        "display.float_format",
        lambda x: f"{x: .4f}",
    ):
        print(df_sorted[cols_to_show].head(10).to_string(index=True))

    print("\n[SWEEP] 对比完成。你可以结合排名结果 + 体检报告，挑选更合适的 filter_tag 方案。")


if __name__ == "__main__":
    main()
