import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# 项目根目录： .../LightHunter_Mk1
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ReplayConfig:
    job_id: str
    tag: str
    ret_col: str
    top_k: int


def load_dataset(job_id: str, ret_col: str) -> pd.DataFrame:
    """读取基础数据集，并统一成 multi-index: [trade_date, code]."""
    dataset_path = ROOT / "data" / "datasets" / f"{job_id}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到基础数据集: {dataset_path}")
    print(f"[UIStrat] INFO: 读取基础数据集: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    # 统一日期列名为 trade_date
    if "trade_date" not in df.columns:
        date_cols = [c for c in df.columns if c in ("trading_date", "date", "as_of")]
        if not date_cols:
            raise RuntimeError("在基础数据集中找不到日期列（期待列名之一: trade_date / trading_date / date / as_of）")
        df = df.rename(columns={date_cols[0]: "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    if "code" not in df.columns:
        raise RuntimeError("在基础数据集中找不到股票代码列 code")

    if ret_col not in df.columns:
        raise RuntimeError(f"在基础数据集中找不到收益列 {ret_col}")

    df = df[["trade_date", "code", ret_col]].copy()
    df = df.set_index(["trade_date", "code"]).sort_index()
    return df


def find_score_files(job_id: str, tag: str) -> List[Path]:
    """
    使用完整打分文件 (_full.csv)，方便对不同 top_k 做回放。
    例如: reports/u1_scores_ultrashort_main_u1_v1_base_rf_20251031_full.csv
    """
    pattern = ROOT / "reports" / f"u1_scores_{job_id}_{tag}_*_full.csv"
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise RuntimeError(f"没有找到任何历史打分文件 ({pattern}) ，无法进行策略回放。")
    print(f"[UIStrat] INFO: 找到历史打分文件 {len(files)} 个（full）")
    return files


def parse_as_of_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"_(\d{8})_full$", path.stem)
    if not m:
        raise ValueError(f"无法从文件名解析日期: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def select_top_k(scores_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """从完整打分文件里选出 top_k 只股票。"""
    if "u1_score" not in scores_df.columns:
        score_cols = [c for c in scores_df.columns if c.endswith("_score") or c.endswith("_pred")]
        if not score_cols:
            raise RuntimeError("在打分文件中找不到模型得分列（例如 u1_score）")
        score_col = score_cols[0]
    else:
        score_col = "u1_score"

    df = scores_df.dropna(subset=[score_col])
    df = df.sort_values(score_col, ascending=False)
    return df.head(top_k).copy()


def compute_metrics(daily_df: pd.DataFrame) -> Dict[str, float]:
    """根据日收益序列计算年化收益、波动率、Sharpe 等指标。"""
    if daily_df.empty:
        raise RuntimeError("日收益序列为空，无法计算指标。")

    rets = daily_df["daily_ret"].fillna(0.0).values
    n = len(rets)
    equity = (1.0 + rets).cumprod()
    total_ret = float(equity[-1] - 1.0)

    # 年化假设 252 个交易日
    ann_ret = (1.0 + total_ret) ** (252.0 / n) - 1.0 if n > 0 else np.nan
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    ann_vol = std * np.sqrt(252.0)
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else np.nan

    dd = equity / np.maximum.accumulate(equity) - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    win_ratio = float((rets > 0).mean()) if n > 0 else np.nan

    return {
        "n_days": int(n),
        "total_ret": total_ret,
        "ann_ret": float(ann_ret),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_ratio": win_ratio,
    }


def compute_yearly_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    """按年份拆分指标。"""
    df = daily_df.copy()
    df["year"] = df["trade_date"].dt.year
    rows = []
    for year, g in df.groupby("year"):
        m = compute_metrics(g)
        m["year"] = int(year)
        rows.append(m)
    cols = ["year", "n_days", "total_ret", "ann_ret", "ann_vol", "sharpe", "max_dd", "win_ratio"]
    out = pd.DataFrame(rows)[cols].sort_values("year")
    return out


def run_replay(cfg: ReplayConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """核心回放逻辑：生成 daily_df + trades_df。"""
    base_df = load_dataset(cfg.job_id, cfg.ret_col)
    score_files = find_score_files(cfg.job_id, cfg.tag)

    daily_rows = []
    trade_rows = []

    for path in score_files:
        as_of = parse_as_of_from_filename(path)
        print(f"[UIStrat] Replay: 模拟交易日 {as_of.date()} ……")

        scores = pd.read_csv(path)
        if "code" not in scores.columns:
            raise RuntimeError(f"{path} 中找不到 code 列")

        top_df = select_top_k(scores, cfg.top_k)
        n_selected = len(top_df)

        if n_selected == 0:
            daily_ret = 0.0
        else:
            idx = pd.MultiIndex.from_product([[as_of], top_df["code"]])
            rets = base_df.reindex(idx)[cfg.ret_col].fillna(0.0).values
            daily_ret = float(np.mean(rets))

        daily_rows.append(
            {
                "trade_date": as_of,
                "n_selected": n_selected,
                "daily_ret": daily_ret,
            }
        )

        for rank, (code, row) in enumerate(top_df.iterrows(), start=1):
            trade_rows.append(
                {
                    "trade_date": as_of,
                    "code": code,
                    "rank": rank,
                    "u1_score": float(row.get("u1_score", np.nan)),
                    cfg.ret_col: float(base_df.loc[(as_of, code), cfg.ret_col])
                    if (as_of, code) in base_df.index
                    else np.nan,
                }
            )

    daily_df = pd.DataFrame(daily_rows).sort_values("trade_date")
    daily_df["equity_curve"] = (1.0 + daily_df["daily_ret"].fillna(0.0)).cumprod()

    trades_df = pd.DataFrame(trade_rows).sort_values(["trade_date", "rank"])
    return daily_df, trades_df


def save_outputs(cfg: ReplayConfig, daily_df: pd.DataFrame, trades_df: pd.DataFrame) -> Path:
    """把结果写出到 CSV + Markdown 报告。"""
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    daily_path = reports_dir / f"u1_strategy_replay_daily_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.csv"
    trades_path = reports_dir / f"u1_strategy_replay_trades_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.csv"
    report_path = reports_dir / f"u1_strategy_replay_report_{cfg.job_id}_{cfg.tag}_{cfg.ret_col}_top{cfg.top_k}.md"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print(f"[UIStrat] OK: 每日收益曲线已保存到: {daily_path}")
    print(f"[UIStrat] OK: 交易明细已保存到: {trades_path}")

    metrics_all = compute_metrics(daily_df)
    yearly_df = compute_yearly_metrics(daily_df)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# U1 策略级历史回放报告 (top{cfg.top_k})\n\n")
        f.write(f"- 任务: job_id=`{cfg.job_id}`, tag=`{cfg.tag}`, ret_col=`{cfg.ret_col}`, top_k={cfg.top_k}\n")
        f.write(f"- 回放日期数: {metrics_all['n_days']}\n\n")

        f.write("## 整体表现\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 累计收益 | {metrics_all['total_ret']*100:.2f}% |\n")
        f.write(f"| 年化收益 | {metrics_all['ann_ret']*100:.2f}% |\n")
        f.write(f"| 年化波动 | {metrics_all['ann_vol']*100:.2f}% |\n")
        f.write(f"| Sharpe | {metrics_all['sharpe']:.2f} |\n")
        f.write(f"| 最大回撤 | {metrics_all['max_dd']*100:.2f}% |\n")
        f.write(f"| 胜率(按日) | {metrics_all['win_ratio']*100:.2f}% |\n\n")

        f.write("## 按年份拆分表现\n\n")
        yearly_df_out = yearly_df.copy()
        yearly_df_out["total_ret"] = yearly_df_out["total_ret"] * 100.0
        yearly_df_out["ann_ret"] = yearly_df_out["ann_ret"] * 100.0
        yearly_df_out["ann_vol"] = yearly_df_out["ann_vol"] * 100.0
        yearly_df_out["max_dd"] = yearly_df_out["max_dd"] * 100.0
        yearly_df_out["win_ratio"] = yearly_df_out["win_ratio"] * 100.0

        f.write("| 年份 | 交易日数 | 累计收益(%) | 年化收益(%) | 年化波动(%) | Sharpe | 最大回撤(%) | 胜率(%) |\n")
        f.write("|------|----------|------------|------------|------------|--------|------------|--------|\n")
        for _, row in yearly_df_out.iterrows():
            f.write(
                f"| {int(row['year'])} | {int(row['n_days'])} | "
                f"{row['total_ret']:.2f} | {row['ann_ret']:.2f} | {row['ann_vol']:.2f} | "
                f"{row['sharpe']:.2f} | {row['max_dd']:.2f} | {row['win_ratio']:.2f} |\n"
            )

    print(f"[UIStrat] DONE: 策略级历史回放完成, 报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="U1 策略级历史回放脚本")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ret-col", required=True)
    parser.add_argument("--top-k", type=int, required=True, help="每日持仓数量 (例如 3、5、10)")
    args = parser.parse_args()

    cfg = ReplayConfig(job_id=args.job_id, tag=args.tag, ret_col=args.ret_col, top_k=args.top_k)

    print("=" * 60)
    print(
        f"[UIStrat] START: 开始策略级历史回放: job_id={cfg.job_id}, tag={cfg.tag}, "
        f"ret_col={cfg.ret_col}, top_k={cfg.top_k}"
    )
    print("=" * 60)

    daily_df, trades_df = run_replay(cfg)
    save_outputs(cfg, daily_df, trades_df)


if __name__ == "__main__":
    main()
