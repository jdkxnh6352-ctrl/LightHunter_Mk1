# -*- coding: utf-8 -*-
"""
tools/u2_live_batch_health_report.py

读取 u2_live_batch_backtest 产出的 equity/yearly/summary，
生成一份更直观的中文体检报告。

示例：

    python -m tools.u2_live_batch_health_report ^
        --tag live_check_base ^
        --filter-tag default

    python -m tools.u2_live_batch_health_report ^
        --tag live_check_cons ^
        --filter-tag conservative
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.config_manager import get_paths_config


def _calc_basic_stats(equity_df: pd.DataFrame) -> Dict[str, float]:
    if equity_df.empty:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_ratio": 0.0,
        }

    n_days = len(equity_df)
    equity = equity_df["equity"].astype(float)
    daily_ret = equity_df["daily_return"].astype(float)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    if n_days > 0:
        ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    else:
        ann_return = 0.0

    if n_days > 1:
        ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    else:
        ann_vol = 0.0

    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    win_ratio = float((daily_ret > 0).mean()) if n_days > 0 else 0.0

    return {
        "n_days": int(n_days),
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_ratio": float(win_ratio),
    }


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U2 Live 批量回测体检报告（支持过滤方案 tag 标注）"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="回测 tag（与 u2_live_batch_backtest 的 --tag 一致）。",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="报告目录（默认：paths.reports_dir / 'u2_live_batch'）。",
    )
    parser.add_argument(
        "--filter-tag",
        type=str,
        default=None,
        help="可选：过滤方案 tag，仅用于在体检报告标题中标注。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    paths_cfg = get_paths_config()
    project_root = Path(paths_cfg.get("project_root", ".")).resolve()
    reports_root = (
        Path(args.reports_dir)
        if args.reports_dir
        else project_root / paths_cfg.get("reports_dir", "reports") / "u2_live_batch"
    )

    tag = args.tag
    equity_file = reports_root / f"u2_batch_equity_{tag}.csv"
    yearly_file = reports_root / f"u2_batch_yearly_{tag}.csv"
    summary_file = reports_root / f"u2_batch_summary_{tag}.txt"
    health_file = reports_root / f"u2_batch_health_{tag}.txt"

    print(f"[HEALTH] 读取 equity : {equity_file}")
    print(f"[HEALTH] 读取 yearly : {yearly_file}")
    if summary_file.exists():
        print(f"[HEALTH] 读取 summary: {summary_file}")
    else:
        print(f"[HEALTH] 未找到 summary 文件，跳过: {summary_file}")

    if not equity_file.exists() or not yearly_file.exists():
        print("[HEALTH] equity/yearly 文件缺失，无法生成体检报告。")
        return

    equity = pd.read_csv(equity_file)
    yearly = pd.read_csv(yearly_file)

    stats_all = _calc_basic_stats(equity)

    # 年度概览（只用 annual_return）
    year_rows = []
    if not yearly.empty:
        for _, row in yearly.iterrows():
            year_rows.append(
                f"{int(row['year'])}({row['ann_return']:.2%})"
            )
    worst_year = ""
    best_year = ""
    if not yearly.empty:
        idx_min = yearly["ann_return"].idxmin()
        idx_max = yearly["ann_return"].idxmax()
        worst_year = (
            f"{int(yearly.loc[idx_min, 'year'])} 年化 {yearly.loc[idx_min, 'ann_return']:.2%}"
        )
        best_year = (
            f"{int(yearly.loc[idx_max, 'year'])} 年化 {yearly.loc[idx_max, 'ann_return']:.2%}"
        )

    # 风格 / 风险评价（很粗的 heuristics，主要是给你一个直观提示）
    max_dd = stats_all["max_drawdown"]
    ann_ret = stats_all["ann_return"]
    win_ratio = stats_all["win_ratio"]

    if ann_ret <= 0 or max_dd <= -0.6:
        risk_judgement = "当前整体亏损，策略暂不合格（建议停用或大幅缩仓）。"
        risk_level = "极高风险"
    elif max_dd <= -0.45:
        risk_judgement = "回撤偏大，波动较猛，适合小仓位试验，不宜重仓。"
        risk_level = "高风险"
    elif max_dd <= -0.30:
        risk_judgement = "中等回撤，属于偏进攻型。如果年化为正，可考虑小仓位实盘。"
        risk_level = "中等风险"
    else:
        risk_judgement = "回撤相对可控，风格偏稳健。结合收益情况，可作为底仓方案。"
        risk_level = "偏保守 / 稳健"

    # 综合打个分：这里简单用（年化 - |回撤|）+ 胜率微调，你看个乐呵
    score = ann_ret - abs(max_dd) + (win_ratio - 0.5) * 0.5

    # 拼接报告文本
    lines = []
    title = f"U2 Live 体检报告 (tag={tag}"
    if args.filter_tag:
        title += f", filter_tag={args.filter_tag}"
    title += ")"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("【一】整体统计")
    lines.append(f"- 回测交易日数：{stats_all['n_days']}")
    lines.append(f"- 累计收益     ：{stats_all['total_return']:.2%}")
    lines.append(f"- 年化收益     ：{stats_all['ann_return']:.2%}")
    lines.append(f"- 年化波动     ：{stats_all['ann_vol']:.2%}")
    lines.append(f"- Sharpe       ：{stats_all['sharpe']:.2f}")
    lines.append(f"- 最大回撤     ：{stats_all['max_drawdown']:.2%}")
    lines.append(f"- 日度胜率     ：{stats_all['win_ratio']:.2%}")
    lines.append("")
    lines.append("【二】年度拆分概览（按 annual_return）")
    if year_rows:
        lines.append("- 历年表现     ：" + "， ".join(year_rows))
        lines.append(f"- 最好年份     ：{best_year}")
        lines.append(f"- 最差年份     ：{worst_year}")
    else:
        lines.append("- 年度数据为空，可能回测区间太短。")
    lines.append("")
    lines.append("【三】风格与风险评估")
    lines.append(f"- 风险评级     ：{risk_level}")
    lines.append(f"- 风格判断     ：{risk_judgement}")
    lines.append(f"- 综合评分（仅供参考）：{score:.3f}")
    lines.append("")
    lines.append("【提示】本体检报告只基于历史回测结果：")
    lines.append("  1) 请结合 U2 训练 / Walk-Forward 结果一起看；")
    lines.append("  2) 真正上实盘前建议先用极小仓位试运行一段时间；")
    lines.append("  3) 定期用 u2_live_batch_backtest + 本脚本重新体检。")

    report_text = "\n".join(lines)
    print("")
    print(report_text)

    health_file.parent.mkdir(parents=True, exist_ok=True)
    with open(health_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[HEALTH] 体检报告已保存：{health_file}")


if __name__ == "__main__":
    main()
