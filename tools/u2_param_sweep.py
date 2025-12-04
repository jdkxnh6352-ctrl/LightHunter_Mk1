# -*- coding: utf-8 -*-
"""
U2 Param Sweep 工具

作用：
- 在一份 U2 日常打分回测输入 CSV 上，批量扫描一组参数网格；
- 每组参数都调用一次 tools.u2_daily_backtest_v2 做完整回测；
- 读取每组回测生成的年度统计 CSV，归纳成一张参数表现排行榜。

使用方式（示例）：
    (.venv) G:\LightHunter_Mk1>python -m tools.u2_param_sweep ^
        --job-id ultrashort_main ^
        --input reports/u2_daily_backtest_demo.csv ^
        --ret-col ret_1 ^
        --output reports/u2_param_sweep_results.csv
"""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import subprocess


# ================================
# 1) 参数网格：在这里调
# ================================
PARAM_GRID: Dict[str, List[Any]] = {
    # 每天最多同时持仓的打分股数量
    "top_k": [2, 3, 4],
    # U2 打分置信度下限（0~1）
    "min_prob": [0.55, 0.60, 0.65],
    # 价格过滤（单位：元）
    "min_price": [3.0],
    "max_price": [80.0],
    # 成交额过滤（单位：元）
    "min_amount": [20_000_000.0],
    # 单只股票仓位（如 0.3 = 30%）
    "position_weight": [0.30, 0.35],
    # 止损 & 止盈（相对买入价的比例，0.03 = -3%，0.06 = +6%）
    "stop_loss": [0.03, 0.05],
    "take_profit": [0.06, 0.08],
}


# ================================
# 2) CLI 解析
# ================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="u2_param_sweep",
        description="批量扫描 U2 日常打分回测参数（调用 u2_daily_backtest_v2）",
    )

    parser.add_argument(
        "--job-id",
        type=str,
        default="ultrashort_main",
        help="任务 ID，仅用于结果里打标签（默认 ultrashort_main）",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="U2 日常打分回测用的原始 CSV（通常是 u2_daily_backtest_demo 的输入）",
    )
    parser.add_argument(
        "--ret-col",
        type=str,
        default="ret_1",
        help="收益列名，默认 ret_1",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="参数扫描结果输出 CSV 路径，例如 reports/u2_param_sweep_results.csv",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="reports/u2_param_sweep_tmp",
        help="中间文件临时目录（默认 reports/u2_param_sweep_tmp）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="最后打印表现最好的前 N 组参数（默认 10）",
    )

    return parser.parse_args()


# ================================
# 3) 工具函数
# ================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_param_grid(grid: Dict[str, List[Any]]):
    """把 PARAM_GRID 转成逐个组合的 dict。"""
    keys = list(grid.keys())
    values_product = itertools.product(*(grid[k] for k in keys))
    for values in values_product:
        yield dict(zip(keys, values))


def run_backtest_once(
    params: Dict[str, Any],
    args: argparse.Namespace,
    tmp_dir: Path,
) -> Dict[str, Any]:
    """
    用一组参数调用一次 u2_daily_backtest_v2，并返回这组参数的汇总指标。
    """
    tmp_equity = tmp_dir / "_tmp_equity.csv"
    tmp_yearly = tmp_dir / "_tmp_yearly.csv"

    # 确保旧的临时文件不会干扰
    if tmp_equity.exists():
        tmp_equity.unlink()
    if tmp_yearly.exists():
        tmp_yearly.unlink()

    cmd = [
        sys.executable,
        "-m",
        "tools.u2_daily_backtest_v2",
        "--input",
        args.input,
        "--ret-col",
        args.ret_col,
        "--top-k",
        str(params["top_k"]),
        "--min-prob",
        str(params["min_prob"]),
        "--min-price",
        str(params["min_price"]),
        "--max-price",
        str(params["max_price"]),
        "--min-amount",
        str(params["min_amount"]),
        "--position-weight",
        str(params["position_weight"]),
        "--stop-loss",
        str(params["stop_loss"]),
        "--take-profit",
        str(params["take_profit"]),
        "--output-equity",
        str(tmp_equity),
        "--output-yearly",
        str(tmp_yearly),
    ]

    print(
        f"[SWEEP] 运行命令： {' '.join(cmd)}",
        flush=True,
    )

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        # 把 stdout / stderr 打出来方便排查
        print("[SWEEP][ERROR] backtest 进程返回非零：", proc.returncode)
        if proc.stdout:
            print("[SWEEP][STDOUT]")
            print(proc.stdout)
        if proc.stderr:
            print("[SWEEP][STDERR]")
            print(proc.stderr)
        raise RuntimeError(
            "u2_daily_backtest_v2 运行失败，请先单独检查该命令。"
        )

    if not tmp_yearly.exists():
        raise RuntimeError(f"未找到年度统计文件：{tmp_yearly}")

    # 只依赖年度统计 CSV，不强制读取 equity
    yearly = pd.read_csv(tmp_yearly)

    if yearly.empty:
        raise RuntimeError("年度统计结果为空，请检查回测输入数据。")

    # 汇总指标的计算逻辑，和 stats 脚本保持一致口径
    n_years = int(yearly.shape[0])
    n_days = int(yearly["n_days"].sum())

    # total_return：整个样本期的总收益
    total_return = float((1.0 + yearly["total_return"]).prod() - 1.0)

    # overall 年化（把整个样本视作一条长时间序列）
    if n_days > 0:
        ann_return_overall = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    else:
        ann_return_overall = float("nan")

    # 按年简单平均的统计
    ann_return_mean = float(yearly["ann_return"].mean())
    ann_vol_mean = float(yearly["ann_vol"].mean())
    sharpe_mean = float(yearly["sharpe"].mean())
    max_drawdown_min = float(yearly["max_drawdown"].min())
    win_ratio_mean = float(yearly["win_ratio"].mean())

    result = dict(params)
    result.update(
        {
            "n_years": n_years,
            "n_days": n_days,
            "total_return": total_return,
            "ann_return_overall": ann_return_overall,
            "ann_return_mean": ann_return_mean,
            "ann_vol_mean": ann_vol_mean,
            "sharpe_mean": sharpe_mean,
            "max_drawdown_min": max_drawdown_min,
            "win_ratio_mean": win_ratio_mean,
        }
    )
    return result


# ================================
# 4) 主流程
# ================================
def main() -> None:
    args = parse_args()

    print("[SWEEP] 当前使用输入文件：", args.input)
    print("[SWEEP] 结果输出文件：", args.output)
    print("[SWEEP] 临时目录：", args.tmp_dir)

    tmp_dir = ensure_dir(Path(args.tmp_dir))

    print("[SWEEP] 当前参数网格：")
    for k, v in PARAM_GRID.items():
        print(f"  {k}: {v}")
    print("[SWEEP] ==============================")

    grid_list = list(iter_param_grid(PARAM_GRID))
    total_combos = len(grid_list)
    print(f"[SWEEP] 一共 {total_combos} 组参数组合，将逐一回测……")

    all_results: List[Dict[str, Any]] = []

    for idx, params in enumerate(grid_list, start=1):
        print(
            f"[SWEEP] 开始第 {idx}/{total_combos} 组参数：{params}",
            flush=True,
        )
        try:
            res = run_backtest_once(params, args, tmp_dir)
        except Exception as e:
            print("[SWEEP][ERROR] 本组参数回测失败：", e)
            raise

        all_results.append(res)

    if not all_results:
        print("[SWEEP][WARN] 没有任何成功的回测结果。")
        return

    df = pd.DataFrame(all_results)

    # 按“整体年化收益”从高到低排序
    df_sorted = df.sort_values("ann_return_overall", ascending=False).reset_index(
        drop=True
    )

    # 写结果到 CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[SWEEP] 回测完成，结果已保存到：", out_path)

    # 打印 Top N
    top_n = min(args.top_n, len(df_sorted))
    print()
    print(f"[SWEEP] 按整体年化收益排序 (Top {top_n})")
    print(df_sorted.head(top_n).to_string(index=False))


if __name__ == "__main__":
    main()
