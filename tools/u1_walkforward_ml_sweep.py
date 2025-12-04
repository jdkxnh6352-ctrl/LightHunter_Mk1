# tools/u1_walkforward_ml_sweep.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Sequence

import pandas as pd


# ---------------------------
# CLI 解析
# ---------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 Walk-Forward 参数 Sweep（train_days / min_train_days）"
    )

    parser.add_argument("--job-id", type=str, required=True, help="数据集 job_id，例如 ultrashort_main")
    parser.add_argument("--start-date", type=str, required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="结束日期 YYYY-MM-DD")

    parser.add_argument("--top-k", type=int, default=30, help="每天持仓标的数量")
    parser.add_argument("--min-price", type=float, default=3.0, help="最小股价过滤")
    parser.add_argument("--max-price", type=float, default=80.0, help="最大股价过滤")
    parser.add_argument("--min-amount", type=float, default=20_000_000.0, help="最小成交额过滤")
    parser.add_argument("--ret-col", type=str, default="ret_1", help="用来回测的收益列")

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="逗号分隔的特征列表，例如: amount,vol_20,ret_1,ret_5,ret_20",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        help="模型名称（传给 u1_walkforward_ml_backtest，例如 rf / gb / ridge）",
    )

    parser.add_argument(
        "--train-days-list",
        type=str,
        default="240",
        help="要扫描的 train_days 列表，逗号分隔，例如: 160,200,240,280",
    )
    parser.add_argument(
        "--min-train-days-list",
        type=str,
        default="120",
        help="要扫描的 min_train_days 列表，逗号分隔，例如: 80,120",
    )

    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="u1_v1_base_rf",
        help="结果 tag 的前缀，用来区分不同版本，例如 u1_v1_base_rf",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# ---------------------------
# 工具函数
# ---------------------------

def _parse_int_list(s: str) -> List[int]:
    items: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _parse_summary_file(path: Path) -> Dict[str, Any]:
    """
    解析 u1_walkforward_summary_xxx.txt

    目前格式类似：

        ==== U1 Walk-Forward 回测统计 ====
        n_days: 1272
        total_return: 0.4341
        ann_return: 0.0740
        ann_vol: 0.2358
        sharpe: 0.3141
        max_drawdown: -0.4627
        win_ratio: 0.4937
        n_trade_days: 1272
        n_total_trades: 3815
    """
    metrics: Dict[str, Any] = {}
    if not path.exists():
        print(f"[SWEEP][WARN] 找不到 summary 文件: {path}")
        return metrics

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("===="):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                metrics[k] = float(v)
            except ValueError:
                metrics[k] = v
    return metrics


def _run_single_combo(
    base_args: argparse.Namespace,
    train_days: int,
    min_train_days: int,
    tag_prefix: str,
) -> Dict[str, Any] | None:
    """
    调一次 tools.u1_walkforward_ml_backtest，返回该组合的指标 dict。
    """
    tag = f"{tag_prefix}_T{train_days}_min{min_train_days}"

    # features 统一去掉空格
    feature_str = ",".join(
        [c.strip() for c in base_args.features.split(",") if c.strip()]
    )

    cmd = [
        sys.executable,
        "-m",
        "tools.u1_walkforward_ml_backtest",
        "--job-id",
        base_args.job_id,
        "--start-date",
        base_args.start_date,
        "--end-date",
        base_args.end_date,
        "--top-k",
        str(base_args.top_k),
        "--min-price",
        str(base_args.min_price),
        "--max-price",
        str(base_args.max_price),
        "--min-amount",
        str(base_args.min_amount),
        "--ret-col",
        base_args.ret_col,
        "--features",
        feature_str,
        "--model",
        base_args.model,
        "--train-days",
        str(train_days),
        "--min-train-days",
        str(min_train_days),
        "--tag",
        tag,
    ]

    print(
        f"\n[SWEEP] 开始组合: train_days={train_days}, "
        f"min_train_days={min_train_days}, tag={tag}"
    )
    print("[SWEEP] CMD:", " ".join(cmd))

    try:
        # 直接调用子进程跑回测
        ret = __import__("subprocess").run(cmd, check=True)
        if ret.returncode != 0:
            print(f"[SWEEP][WARN] 子进程返回码={ret.returncode}，跳过该组合。")
            return None
    except Exception as e:
        print(f"[SWEEP][ERROR] 运行 u1_walkforward_ml_backtest 失败: {e}")
        return None

    # 回测脚本会写入 summary 文件：reports/u1_walkforward_summary_<job-id>_<tag>.txt
    reports_dir = Path("reports")
    summary_path = reports_dir / f"u1_walkforward_summary_{base_args.job_id}_{tag}.txt"

    metrics = _parse_summary_file(summary_path)
    if not metrics:
        print(f"[SWEEP][WARN] 未能解析 summary 文件: {summary_path}")
        return None

    record: Dict[str, Any] = {
        "tag": tag,
        "job_id": base_args.job_id,
        "model": base_args.model,
        "features": feature_str,
        "train_days": train_days,
        "min_train_days": min_train_days,
    }
    record.update(metrics)
    return record


# ---------------------------
# 主逻辑
# ---------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    train_days_list = _parse_int_list(args.train_days_list)
    min_train_days_list = _parse_int_list(args.min_train_days_list)

    if not train_days_list or not min_train_days_list:
        print("[SWEEP][ERROR] train_days_list 或 min_train_days_list 为空，请检查参数。")
        sys.exit(1)

    print("\n[SWEEP] ==============================")
    print("[SWEEP] U1 Walk-Forward 参数扫描启动")
    print(f"[SWEEP] job-id      : {args.job_id}")
    print(f"[SWEEP] 日期区间    : {args.start_date} ~ {args.end_date}")
    print(f"[SWEEP] 特征        : {args.features}")
    print(f"[SWEEP] 模型        : {args.model}")
    print(f"[SWEEP] train_days  : {train_days_list}")
    print(f"[SWEEP] min_train   : {min_train_days_list}")
    print(f"[SWEEP] tag 前缀    : {args.tag_prefix}")
    print("[SWEEP] ==============================\n")

    records: List[Dict[str, Any]] = []

    for td in train_days_list:
        for mtd in min_train_days_list:
            rec = _run_single_combo(args, td, mtd, args.tag_prefix)
            if rec is not None:
                records.append(rec)

    if not records:
        print("[SWEEP][ERROR] 所有组合都失败，未得到任何结果。")
        return

    df = pd.DataFrame(records)

    # 排序：先看 Sharpe，再看 annual_return
    sort_cols = []
    if "sharpe" in df.columns:
        sort_cols.append(("sharpe", False))
    if "ann_return" in df.columns:
        sort_cols.append(("ann_return", False))

    if sort_cols:
        by = [c for c, _ in sort_cols]
        ascending = [a for _, a in sort_cols]
        df = df.sort_values(by=by, ascending=ascending)
    else:
        df = df.sort_values("tag")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    out_csv = reports_dir / f"u1_walkforward_param_sweep_{args.job_id}_{args.tag_prefix}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==== U1 Walk‑Forward 参数 Sweep 结果（按 Sharpe 排序） ====\n")
    float_cols = [c for c in df.columns if df[c].dtype.kind in "fc"]
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}",
            columns=[
                "tag",
                "train_days",
                "min_train_days",
                "total_return",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "win_ratio",
                "n_days",
                "n_trade_days",
                "n_total_trades",
            ]
            if set(
                [
                    "tag",
                    "train_days",
                    "min_train_days",
                    "total_return",
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                    "win_ratio",
                    "n_days",
                    "n_trade_days",
                    "n_total_trades",
                ]
            ).issubset(df.columns)
            else None,
        )
    )

    print(f"\n[SWEEP] 排名结果已保存到：{out_csv}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
