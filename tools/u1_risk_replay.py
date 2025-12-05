import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# 复用策略回放里的指标计算函数
from .u1_strategy_replay import compute_metrics, compute_yearly_metrics

# 项目根目录： .../LightHunter_Mk1
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RiskReplayConfig:
    job_id: str
    tag: str
    ret_col: str
    top_k: int
    dd_stop: float      # 触发清仓的最大回撤阈值，例如 0.30 表示 -30%
    cool_down: int      # 触发一次后，空仓冷静多少个交易日


def load_baseline_daily(cfg: RiskReplayConfig) -> pd.DataFrame:
    """
    读取 baseline 的策略回放结果（不带风险控制），即：u1_strategy_replay_daily_*.csv
    """
    daily_path = (
        ROOT
        / "reports"
        / f"u1_strategy_replay_daily_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.csv"
    )
    if not daily_path.exists():
        raise FileNotFoundError(
            f"找不到 baseline 日度回放文件: {daily_path}\n"
            f"请先运行：python -m tools.u1_strategy_replay "
            f"--job-id {cfg.job_id} --tag {cfg.tag} --ret-col {cfg.ret_col} --top-k {cfg.top_k}"
        )

    print(f"[UIRisk] INFO: 读取 baseline 回放结果: {daily_path}")
    df = pd.read_csv(daily_path, encoding="utf-8-sig")
    if "trade_date" not in df.columns:
        raise RuntimeError("baseline CSV 中找不到 trade_date 列")
    if "daily_ret" not in df.columns:
        raise RuntimeError("baseline CSV 中找不到 daily_ret 列")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 为了方便，对 baseline 自己再算一条 equity
    df["equity_baseline"] = (1.0 + df["daily_ret"].fillna(0.0)).cumprod()
    return df


def apply_drawdown_risk_control(
    baseline_df: pd.DataFrame, dd_stop: float, cool_down: int
) -> pd.DataFrame:
    """
    在 baseline 日收益序列上叠加一个简单的“最大回撤止损 + 冷静期”仓位控制：

    - 初始仓位 pos = 1（满仓按 baseline 走）；
    - 每天根据 pos 乘以 baseline 的 daily_ret 得到实际收益；
    - 如果当前 equity 相对历史峰值的回撤 <= -dd_stop：
        -> 触发风险控制：接下来 cool_down 天 pos = 0（空仓观望）；
    - 冷静期结束后，pos 自动恢复为 1，继续跟随 baseline。
    """

    df = baseline_df.copy().reset_index(drop=True)
    rets = df["daily_ret"].fillna(0.0).values
    n = len(rets)

    pos = np.ones(n, dtype=float)  # 每日实际仓位
    eq = np.zeros(n, dtype=float)  # 风险控制后的权益曲线

    equity = 1.0
    peak = 1.0
    cool_remaining = 0

    for i in range(n):
        # 当前是否在冷静期
        if cool_remaining > 0:
            pos[i] = 0.0
            cool_remaining -= 1
        else:
            pos[i] = 1.0

        # 根据仓位更新当天权益
        equity *= 1.0 + pos[i] * rets[i]
        eq[i] = equity

        # 更新历史峰值 & 回撤
        if equity > peak:
            peak = equity
        dd = equity / peak - 1.0

        # 如果回撤超过阈值，并且当前不在冷静期，触发新的冷静期
        if dd <= -dd_stop and cool_remaining == 0:
            cool_remaining = cool_down

    # 把结果写回 DataFrame
    df["position"] = pos
    df["daily_ret_with_risk"] = df["daily_ret"] * df["position"]
    df["equity_with_risk"] = eq

    return df


def build_risk_daily_df(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    从带有 position / daily_ret_with_risk 的 DataFrame 中，
    提炼出一个“新的日度序列”，供 compute_metrics 使用。
    """
    out = risk_df[["trade_date", "daily_ret_with_risk"]].copy()
    out = out.rename(columns={"daily_ret_with_risk": "daily_ret"})
    # 再补一条自己的 equity 曲线，方便检查
    out["equity_curve"] = (1.0 + out["daily_ret"].fillna(0.0)).cumprod()
    return out


def save_risk_outputs(
    cfg: RiskReplayConfig,
    baseline_daily: pd.DataFrame,
    risk_daily: pd.DataFrame,
    risk_df_full: pd.DataFrame,
) -> Path:
    """
    写出：
    - 日度风险控制后的序列 CSV
    - 对比型 markdown 报告（baseline vs 风险控制）
    """
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    daily_path = (
        reports_dir
        / f"u1_risk_replay_daily_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}_"
          f"dd{int(cfg.dd_stop*100):02d}_cool{cfg.cool_down}.csv"
    )
    risk_df_full.to_csv(daily_path, index=False, encoding="utf-8-sig")
    print(f"[UIRisk] OK: 风险控制后的日度序列已保存到: {daily_path}")

    # 计算 baseline & 风险控制后 的整体 / 分年指标
    metrics_base = compute_metrics(baseline_daily)
    metrics_risk = compute_metrics(risk_daily)

    yearly_base = compute_yearly_metrics(baseline_daily)
    yearly_risk = compute_yearly_metrics(risk_daily)

    report_path = (
        reports_dir
        / f"u1_risk_replay_report_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}_"
          f"dd{int(cfg.dd_stop*100):02d}_cool{cfg.cool_down}.md"
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# U1 风险控制回放报告\n\n")
        f.write(
            f"- 任务: job_id=`{cfg.job_id}`, tag=`{cfg.tag}`, "
            f"ret_col=`{cfg.ret_col}`, top_k={cfg.top_k}\n"
        )
        f.write(
            f"- 规则: 最大回撤阈值 = {cfg.dd_stop*100:.1f}%，"
            f"触发后空仓冷静 {cfg.cool_down} 个交易日\n\n"
        )

        # 整体表现对比
        f.write("## 整体表现对比\n\n")
        f.write("| 指标 | baseline | 加风险控制 |\n")
        f.write("|------|----------|------------|\n")
        f.write(
            f"| 累计收益 | {metrics_base['total_ret']*100:.2f}% "
            f"| {metrics_risk['total_ret']*100:.2f}% |\n"
        )
        f.write(
            f"| 年化收益 | {metrics_base['ann_ret']*100:.2f}% "
            f"| {metrics_risk['ann_ret']*100:.2f}% |\n"
        )
        f.write(
            f"| 年化波动 | {metrics_base['ann_vol']*100:.2f}% "
            f"| {metrics_risk['ann_vol']*100:.2f}% |\n"
        )
        f.write(
            f"| Sharpe | {metrics_base['sharpe']:.2f} "
            f"| {metrics_risk['sharpe']:.2f} |\n"
        )
        f.write(
            f"| 最大回撤 | {metrics_base['max_dd']*100:.2f}% "
            f"| {metrics_risk['max_dd']*100:.2f}% |\n"
        )
        f.write(
            f"| 胜率(按日) | {metrics_base['win_ratio']*100:.2f}% "
            f"| {metrics_risk['win_ratio']*100:.2f}% |\n\n"
        )

        # 按年份拆分表现对比
        f.write("## 按年份拆分表现对比\n\n")
        f.write(
            "| 年份 | base_年化收益(%) | base_最大回撤(%) | "
            "risk_年化收益(%) | risk_最大回撤(%) |\n"
        )
        f.write(
            "|------|-----------------|-----------------|"
            "-----------------|-----------------|\n"
        )

        yr_base = yearly_base.set_index("year")
        yr_risk = yearly_risk.set_index("year")
        all_years = sorted(set(yr_base.index) | set(yr_risk.index))

        for y in all_years:
            mb = yr_base.loc[y] if y in yr_base.index else None
            mr = yr_risk.loc[y] if y in yr_risk.index else None

            def fmt(row, key):
                if row is None:
                    return "NA"
                if key in ("ann_ret", "max_dd"):
                    return f"{row[key]*100:.2f}"
                return f"{row[key]:.2f}"

            f.write(
                f"| {y} | {fmt(mb, 'ann_ret')} | {fmt(mb, 'max_dd')} | "
                f"{fmt(mr, 'ann_ret')} | {fmt(mr, 'max_dd')} |\n"
            )

    print(f"[UIRisk] DONE: 风险控制回放完成, 报告: {report_path}")
    return report_path


def run_risk_replay(cfg: RiskReplayConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    主流程：
    1. 读取 baseline 日度回放结果；
    2. 套入“最大回撤止损 + 冷静期”规则，得到带 position 的 DataFrame；
    3. 提炼风险控制后的日度序列，用于算指标。
    """
    baseline_daily = load_baseline_daily(cfg)
    risk_df_full = apply_drawdown_risk_control(
        baseline_daily, dd_stop=cfg.dd_stop, cool_down=cfg.cool_down
    )
    risk_daily = build_risk_daily_df(risk_df_full)
    return baseline_daily, risk_daily, risk_df_full


def main():
    parser = argparse.ArgumentParser(description="U1 风险控制回放脚本")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ret-col", required=True)
    parser.add_argument("--top-k", type=int, required=True)

    parser.add_argument(
        "--dd-stop",
        type=float,
        default=0.30,
        help="最大回撤止损阈值，例如 0.30 表示 -30%% (默认 0.30)",
    )
    parser.add_argument(
        "--cool-down",
        type=int,
        default=20,
        help="触发一次止损后，空仓冷静的交易日数量，默认 20",
    )

    args = parser.parse_args()

    cfg = RiskReplayConfig(
        job_id=args.job_id,
        tag=args.tag,
        ret_col=args.ret_col,
        top_k=args.top_k,
        dd_stop=args.dd_stop,
        cool_down=args.cool_down,
    )

    print("=" * 60)
    print(
        f"[UIRisk] START: 风险控制回放: job_id={cfg.job_id}, tag={cfg.tag}, "
        f"ret_col={cfg.ret_col}, top_k={cfg.top_k}, "
        f"dd_stop={cfg.dd_stop}, cool_down={cfg.cool_down}"
    )
    print("=" * 60)

    baseline_daily, risk_daily, risk_df_full = run_risk_replay(cfg)
    save_risk_outputs(cfg, baseline_daily, risk_daily, risk_df_full)


if __name__ == "__main__":
    main()
