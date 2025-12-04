from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class U2LiveParams:
    top_k: int
    min_prob: float
    min_price: float
    max_price: float
    min_amount: float
    position_weight: float
    max_positions: int
    stop_loss: float
    take_profit: float
    ret_col: str
    input_path: Path
    output_dir: Path


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_params(cfg: Dict, project_root: Path) -> U2LiveParams:
    """
    从 u2_live_config.json 里抽出实盘参数。

    支持两种结构：
    1）{"u2_live": {...}}   # 推荐
    2）{...}               # 直接平铺在根节点
    """
    live_cfg = cfg.get("u2_live", cfg)

    # 回测输入文件（之前生成好的 U2 日常打分候选股数据）
    dataset_rel = live_cfg.get("daily_backtest_input", "reports/u2_daily_backtest_demo.csv")
    input_path = (project_root / dataset_rel).resolve()

    # 体检报告输出目录
    output_dir_rel = live_cfg.get("healthcheck_output_dir", "reports/u2_live_healthcheck")
    output_dir = (project_root / output_dir_rel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = U2LiveParams(
        top_k=int(live_cfg["top_k"]),
        min_prob=float(live_cfg["min_prob"]),
        min_price=float(live_cfg.get("min_price", 3.0)),
        max_price=float(live_cfg.get("max_price", 80.0)),
        min_amount=float(live_cfg.get("min_amount", 2e7)),
        position_weight=float(live_cfg.get("position_weight", 0.3)),
        max_positions=int(live_cfg.get("max_positions", 3)),
        stop_loss=float(live_cfg.get("stop_loss", 0.05)),
        take_profit=float(live_cfg.get("take_profit", 0.08)),
        ret_col=live_cfg.get("ret_col", "ret_1"),
        input_path=input_path,
        output_dir=output_dir,
    )
    return params


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_windows(df: pd.DataFrame, window_years: int) -> List[Dict]:
    """
    根据 trade_date 自动生成滑动时间窗口。

    比如 window_years=5，数据是 2020~2025 年：
    会生成：
      2020_2024
      2021_2025
    再加一个 all_history。
    """
    dates = pd.to_datetime(df["trade_date"]).dt.date
    first_year = dates.min().year
    last_year = dates.max().year

    windows: List[Dict] = []
    for start_year in range(first_year, last_year - window_years + 2):
        end_year = start_year + window_years - 1
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        if end_date < dates.min() or start_date > dates.max():
            continue
        label = f"{start_year}_{end_year}"
        windows.append(
            {
                "name": label,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    # 最后加一个“全历史”
    windows.append(
        {
            "name": "all_history",
            "start_date": dates.min(),
            "end_date": dates.max(),
        }
    )
    return windows


def filter_by_window(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    trade_dates = pd.to_datetime(df["trade_date"]).dt.date
    mask = (trade_dates >= start_date) & (trade_dates <= end_date)
    return df.loc[mask].copy()


def parse_metrics_from_stdout(text: str) -> Dict[str, float]:
    """
    从 tools.u2_daily_backtest_v2 的 stdout 里解析关键指标。
    只依赖中文行里的数字，格式稍微变一下也能兼容。
    """
    metrics: Dict[str, float] = {}

    def parse_pct(line: str) -> Optional[float]:
        m = re.search(r"([-+]?\d*\.\d+|\d+)%", line)
        if not m:
            return None
        return float(m.group(1)) / 100.0

    for raw in text.splitlines():
        line = raw.strip()
        if "累计收益" in line:
            v = parse_pct(line)
            if v is not None:
                metrics["total_return"] = v
        elif "年化收益" in line:
            v = parse_pct(line)
            if v is not None:
                metrics["ann_return"] = v
        elif "年化波动" in line:
            v = parse_pct(line)
            if v is not None:
                metrics["ann_vol"] = v
        elif line.startswith("Sharpe"):
            # 例：Sharpe      :  0.30
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["sharpe"] = float(parts[1])
                except ValueError:
                    pass
        elif "最大回撤" in line:
            v = parse_pct(line)
            if v is not None:
                metrics["max_drawdown"] = v
        elif "胜率" in line:
            v = parse_pct(line)
            if v is not None:
                metrics["win_ratio"] = v

    return metrics


def run_healthcheck(
    params: U2LiveParams,
    window_years: int = 5,
) -> None:
    df_all = pd.read_csv(params.input_path)
    if "trade_date" not in df_all.columns:
        raise RuntimeError(f"输入文件 {params.input_path} 里没有 trade_date 列，无法按时间切片。")

    df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])

    windows = build_windows(df_all, window_years)
    summary_rows: List[Dict] = []

    print(f"[HC] 使用输入文件: {params.input_path}")
    print(f"[HC] 将按 {window_years} 年滑动窗口 + 全历史 进行回测。")
    print(f"[HC] 共生成 {len(windows)} 个窗口。\n")

    for idx, w in enumerate(windows, start=1):
        name = w["name"]
        start_date = w["start_date"]
        end_date = w["end_date"]

        df_win = filter_by_window(df_all, start_date, end_date)
        if df_win.empty:
            print(f"[HC][WARN] 窗口 {name} ({start_date}~{end_date}) 没有样本，跳过。")
            continue

        tmp_input = params.output_dir / f"u2_live_hc_{name}_input.csv"
        equity_path = params.output_dir / f"u2_live_hc_{name}_equity.csv"
        yearly_path = params.output_dir / f"u2_live_hc_{name}_yearly.csv"

        df_win.to_csv(tmp_input, index=False)

        cmd = [
            sys.executable,
            "-m",
            "tools.u2_daily_backtest_v2",
            "--input",
            str(tmp_input),
            "--ret-col",
            params.ret_col,
            "--top-k",
            str(params.top_k),
            "--min-prob",
            str(params.min_prob),
            "--min-price",
            str(params.min_price),
            "--max-price",
            str(params.max_price),
            "--min-amount",
            str(params.min_amount),
            "--position-weight",
            str(params.position_weight),
            "--max-positions",
            str(params.max_positions),
            "--stop-loss",
            str(params.stop_loss),
            "--take-profit",
            str(params.take_profit),
            "--output-equity",
            str(equity_path),
            "--output-yearly",
            str(yearly_path),
        ]

        print(
            f"[HC] 开始回测窗口 {idx}/{len(windows)}: {name}  "
            f"({start_date} ~ {end_date}), 样本数={len(df_win)}"
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"[HC][ERROR] 回测窗口 {name} 失败，返回码={proc.returncode}")
            print(proc.stderr)
            continue

        metrics = parse_metrics_from_stdout(proc.stdout)
        metrics.update(
            {
                "window": name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "n_samples": int(len(df_win)),
                "equity_path": os.path.relpath(equity_path, params.output_dir),
                "yearly_path": os.path.relpath(yearly_path, params.output_dir),
            }
        )
        summary_rows.append(metrics)

    if not summary_rows:
        print("[HC][ERROR] 没有任何窗口回测成功，无法生成汇总。")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = params.output_dir / "u2_live_healthcheck_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n[HC] 回测完成，窗口汇总结果已保存到:")
    print(f"     {summary_path}")
    print("\n[HC] 建议优先关注以下几个指标:")
    print("  - ann_return: 年化收益 (越高越好)")
    print("  - max_drawdown: 最大回撤 (越小越好，负数)")
    print("  - sharpe: 夏普比率 (大于 0 才有研究价值)")
    print("  - win_ratio: 胜率 (结合 max_drawdown 一起看)")
    print("\n[HC] 你可以用 Excel 打开 summary 和各个 *_yearly.csv，")
    print("     看看不同时间窗口里，这套 U2 实盘参数的表现是否稳定。")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="一键批量回测 U2 实盘参数（从 u2_live_config.json 读取）并输出体检报告。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/u2_live_config.json",
        help="U2 实盘配置文件路径，默认 config/u2_live_config.json",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=5,
        help="滑动窗口的年数，默认 5 年。",
    )

    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    params = extract_params(cfg, project_root)

    print("[HC] 从配置中读取到的 U2 实盘参数：")
    print(f"  top_k           = {params.top_k}")
    print(f"  min_prob        = {params.min_prob}")
    print(f"  min_price       = {params.min_price}")
    print(f"  max_price       = {params.max_price}")
    print(f"  min_amount      = {params.min_amount}")
    print(f"  position_weight = {params.position_weight}")
    print(f"  max_positions   = {params.max_positions}")
    print(f"  stop_loss       = {params.stop_loss}")
    print(f"  take_profit     = {params.take_profit}")
    print(f"  ret_col         = {params.ret_col}")
    print(f"  input_path      = {params.input_path}")
    print(f"  output_dir      = {params.output_dir}")
    print()

    run_healthcheck(params, window_years=args.window_years)


if __name__ == "__main__":
    main()
