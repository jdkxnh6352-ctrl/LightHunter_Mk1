import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .u1_strategy_replay import compute_metrics
from .u1_risk_replay import RiskReplayConfig, run_risk_replay

# 项目根目录：.../LightHunter_Mk1
ROOT = Path(__file__).resolve().parents[1]

# 我帮你定一个“可接受”的最大回撤上限：40%
# 在 sweep 的结果里，会优先选 |max_dd| <= 40% 且年化收益最高的那组
MAX_DD_LIMIT = 0.40


@dataclass
class SweepConfig:
    job_id: str
    tag: str
    ret_col: str
    top_k: int
    dd_grid: List[float]
    cool_grid: List[int]


def run_sweep(cfg: SweepConfig) -> Tuple[pd.DataFrame, dict]:
    """
    扫描一圈 (dd_limit, cool_down) 组合，返回：
    - sweep_df：每组参数对应的风险控制后指标
    - base_metrics：不加风控时的整体指标（只算一次）
    """
    rows = []
    base_metrics = None

    for dd_limit in cfg.dd_grid:
        for cool_down in cfg.cool_grid:
            print(
                f"[UISweep] INFO: 回放参数: dd_limit={dd_limit:.0%}, "
                f"cool_down={cool_down}"
            )

            # RiskReplayConfig 的字段顺序是
            # (job_id, tag, ret_col, top_k, dd_limit, cool_down)
            risk_cfg = RiskReplayConfig(
                cfg.job_id,
                cfg.tag,
                cfg.ret_col,
                cfg.top_k,
                dd_limit,
                cool_down,
            )

            result = run_risk_replay(risk_cfg)
            # 兼容 2 返回值或 3 返回值的情况
            if isinstance(result, tuple):
                if len(result) == 3:
                    base_daily, risk_daily, _ = result
                elif len(result) == 2:
                    base_daily, risk_daily = result
                else:
                    raise RuntimeError("run_risk_replay 返回的结果长度异常")
            else:
                raise RuntimeError("run_risk_replay 返回的类型异常")

            m_base = compute_metrics(base_daily)
            m_risk = compute_metrics(risk_daily)

            if base_metrics is None:
                base_metrics = m_base

            rows.append(
                {
                    "dd_limit": dd_limit,
                    "cool_down": cool_down,
                    "n_days": m_risk["n_days"],
                    "total_ret": m_risk["total_ret"],
                    "ann_ret": m_risk["ann_ret"],
                    "ann_vol": m_risk["ann_vol"],
                    "sharpe": m_risk["sharpe"],
                    "max_dd": m_risk["max_dd"],
                    "win_ratio": m_risk["win_ratio"],
                }
            )

    sweep_df = pd.DataFrame(rows)
    return sweep_df, base_metrics


def pick_best_combo(sweep_df: pd.DataFrame) -> pd.Series:
    """
    在 sweep 结果里自动选一组“推荐参数”：
    1. 先过滤出 |max_dd| <= MAX_DD_LIMIT 的组合（例如 <= 40%）
    2. 在这些组合里选 ann_ret（年化收益）最高的那组
    3. 如果一个都不满足，就在全部组合里选 ann_ret 最高的
    """
    if sweep_df.empty:
        raise RuntimeError("sweep 结果为空，无法选择推荐参数。")

    candidates = sweep_df[sweep_df["max_dd"].abs() <= MAX_DD_LIMIT]
    if candidates.empty:
        candidates = sweep_df

    best_idx = candidates["ann_ret"].idxmax()
    return candidates.loc[best_idx]


def save_sweep_report(
    cfg: SweepConfig,
    sweep_df: pd.DataFrame,
    base_metrics: dict,
    best_row: pd.Series,
) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        reports_dir
        / f"u1_risk_replay_sweep_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.csv"
    )
    sweep_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report_path = (
        reports_dir
        / f"u1_risk_replay_sweep_report_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.md"
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# U1 风控参数 sweep 报告\n\n")
        f.write(
            f"- 任务: job_id=`{cfg.job_id}`, tag=`{cfg.tag}`, "
            f"ret_col=`{cfg.ret_col}`, top_k={cfg.top_k}\n"
        )
        f.write(
            f"- 扫描 dd_limit: {[f'{x*100:.0f}%' for x in cfg.dd_grid]}, "
            f"冷静期: {cfg.cool_grid} 个交易日\n"
        )
        f.write(
            f"- 推荐规则: 在 |max_dd| <= {MAX_DD_LIMIT*100:.0f}% 的组合中，"
            "选择年化收益最高的一组。\n\n"
        )

        # 1) baseline
        f.write("## 1. baseline 表现（无风控）\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 累计收益 | {base_metrics['total_ret']*100:.2f}% |\n")
        f.write(f"| 年化收益 | {base_metrics['ann_ret']*100:.2f}% |\n")
        f.write(f"| 年化波动 | {base_metrics['ann_vol']*100:.2f}% |\n")
        f.write(f"| Sharpe | {base_metrics['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | {base_metrics['max_dd']*100:.2f}% |\n")
        f.write(f"| 胜率(按日) | {base_metrics['win_ratio']*100:.2f}% |\n\n")

        # 2) sweep 结果表
        f.write("## 2. 不同风控参数下的整体表现\n\n")

        tmp = sweep_df.copy()
        tmp["total_ret"] *= 100.0
        tmp["ann_ret"] *= 100.0
        tmp["ann_vol"] *= 100.0
        tmp["max_dd"] *= 100.0
        tmp["win_ratio"] *= 100.0

        f.write(
            "| dd_limit | cool_down | 累计收益(%) | 年化收益(%) | "
            "年化波动(%) | Sharpe | 最大回撤(%) | 胜率(%) |\n"
        )
        f.write(
            "|----------|-----------|------------|------------|"
            "------------|--------|------------|--------|\n"
        )

        for _, row in tmp.sort_values(["dd_limit", "cool_down"]).iterrows():
            f.write(
                f"| {row['dd_limit']*100:.1f}% | {int(row['cool_down'])} | "
                f"{row['total_ret']:.2f} | {row['ann_ret']:.2f} | "
                f"{row['ann_vol']:.2f} | {row['sharpe']:.2f} | "
                f"{row['max_dd']:.2f} | {row['win_ratio']:.2f} |\n"
            )

        # 3) 推荐参数
        f.write("\n## 3. 推荐参数（自动选择）\n\n")
        f.write("| dd_limit | cool_down | 年化收益(%) | 最大回撤(%) | Sharpe |\n")
        f.write("|----------|-----------|------------|------------|--------|\n")
        f.write(
            f"| {best_row['dd_limit']*100:.1f}% | {int(best_row['cool_down'])} | "
            f"{best_row['ann_ret']*100:.2f} | {best_row['max_dd']*100:.2f} | "
            f"{best_row['sharpe']:.2f} |\n"
        )

        f.write(
            "\n> 说明：推荐参数是在 `|max_dd|` 不超过约 "
            f"{MAX_DD_LIMIT*100:.0f}% 的前提下，"
            "选择年化收益最高的一组。你可以把它视为当前版本的“默认风控参数”。\n"
        )

    print(f"[UISweep] OK: sweep 结果 CSV 已保存到: {csv_path}")
    print(f"[UISweep] OK: sweep 报告已保存到: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="U1 风控参数 sweep 脚本")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ret-col", required=True)
    parser.add_argument("--top-k", type=int, required=True)

    parser.add_argument(
        "--dd-grid",
        type=float,
        nargs="*",
        default=[0.30, 0.35, 0.40],
        help="最大回撤阈值列表（小数），例如: 0.30 0.35 0.40",
    )
    parser.add_argument(
        "--cool-grid",
        type=int,
        nargs="*",
        default=[10, 20, 30],
        help="冷静期长度列表（交易日），例如: 10 20 30",
    )

    args = parser.parse_args()

    cfg = SweepConfig(
        job_id=args.job_id,
        tag=args.tag,
        ret_col=args.ret_col,
        top_k=args.top_k,
        dd_grid=args.dd_grid,
        cool_grid=args.cool_grid,
    )

    print("=" * 60)
    print(
        f"[UISweep] START: 风控参数 sweep, job_id={cfg.job_id}, "
        f"tag={cfg.tag}, ret_col={cfg.ret_col}, top_k={cfg.top_k}"
    )
    print("=" * 60)

    sweep_df, base_metrics = run_sweep(cfg)
    best_row = pick_best_combo(sweep_df)
    save_sweep_report(cfg, sweep_df, base_metrics, best_row)


if __name__ == "__main__":
    main()
