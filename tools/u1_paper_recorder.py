# tools/u1_paper_recorder.py
# -*- coding: utf-8 -*-
"""
U1 纸上实盘记录脚本

功能：
- 读取某一天的下单列表 CSV
- 自动加上日期 / 策略标签 / job_id 等元信息
- 追加写入一份长期的纸上实盘交易流水 CSV

用法示例：
python -m tools.u1_paper_recorder ^
  --job-id ultrashort_main ^
  --tag u1_v1_base_rf ^
  --as-of 2025-12-04

如果不传 --orders-csv，会自动去找：
  reports/u1_orders_{job_id}_{tag}_{yyyymmdd}.csv
"""

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 纸上实盘记录脚本（把每日下单列表汇总到一份长期流水里）"
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="任务 / 数据集 ID，例如 ultrashort_main",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="策略标签，例如 u1_v1_base_rf",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="信号日期，格式 YYYY-MM-DD（例如 2025-12-04）",
    )
    parser.add_argument(
        "--orders-csv",
        default=None,
        help=(
            "当日下单列表 CSV 路径。"
            "不填则默认：reports/u1_orders_{job_id}_{tag}_{yyyymmdd}.csv"
        ),
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help=(
            "纸上实盘总流水 CSV 路径。"
            "不填则默认：reports/u1_paper_trades_{job_id}_{tag}.csv"
        ),
    )
    return parser.parse_args()


def _infer_orders_path(job_id: str, tag: str, as_of: str) -> Path:
    as_of_str = as_of.replace("-", "")
    return Path("reports") / f"u1_orders_{job_id}_{tag}_{as_of_str}.csv"


def _infer_log_path(job_id: str, tag: str) -> Path:
    return Path("reports") / f"u1_paper_trades_{job_id}_{tag}.csv"


def main() -> None:
    args = parse_args()

    # 解析日期，顺便校验格式
    try:
        trade_date = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"[U1Paper] as-of 日期格式不对：{args.as_of}（应为 YYYY-MM-DD）")

    # 确定下单列表路径
    if args.orders_csv:
        orders_path = Path(args.orders_csv)
    else:
        orders_path = _infer_orders_path(args.job_id, args.tag, args.as_of)

    # 确定总流水路径
    if args.log_csv:
        log_path = Path(args.log_csv)
    else:
        log_path = _infer_log_path(args.job_id, args.tag)

    print(f"[U1Paper] 使用 job-id: {args.job_id}")
    print(f"[U1Paper] 使用策略标签: {args.tag}")
    print(f"[U1Paper] 信号日期: {trade_date.isoformat()}")
    print(f"[U1Paper] 读取下单列表: {orders_path}")

    if not orders_path.exists():
        raise SystemExit(f"[U1Paper] 未找到当日下单列表文件：{orders_path}")

    orders_df = pd.read_csv(orders_path)

    if orders_df.empty:
        print("[U1Paper] 下单列表为空，本次不记录任何交易。")
        return

    # 给这一批订单加上元信息
    enriched = orders_df.copy()
    enriched.insert(0, "trade_date", trade_date.isoformat())
    enriched.insert(1, "strategy_tag", args.tag)
    enriched.insert(2, "job_id", args.job_id)

    print(f"[U1Paper] 本次记录订单数量：{len(enriched)} 条")

    # 如果已有历史流水，则读出来追加；否则直接保存
    if log_path.exists():
        print(f"[U1Paper] 发现已有历史流水：{log_path}，准备追加写入...")
        history_df = pd.read_csv(log_path)

        combined = pd.concat([history_df, enriched], ignore_index=True)

        # 去重逻辑：如果存在 trade_date + strategy_tag + code + side 这些列，就用它们做主键
        dedup_cols = ["trade_date", "strategy_tag"]
        for col in ["code", "side", "action"]:
            if col in combined.columns:
                dedup_cols.append(col)

        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

        combined.to_csv(log_path, index=False)
        print(
            f"[U1Paper] 已追加写入，当前总记录数：{len(combined)} 条\n"
            f"[U1Paper] 流水文件：{log_path}"
        )
    else:
        # 首次创建
        log_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(log_path, index=False)
        print(
            f"[U1Paper] 首次创建纸上实盘流水文件：{log_path}\n"
            f"[U1Paper] 当前记录数：{len(enriched)} 条"
        )

    print("[U1Paper] 纸上实盘记录完成。")


if __name__ == "__main__":
    main()
