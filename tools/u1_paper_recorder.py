# -*- coding: utf-8 -*-
"""
u1_paper_recorder.py

U1 纸上实盘记录脚本

功能：
- 读取某一天的 U1 下单列表（orders csv）
- 自动推断 / 或使用用户指定路径
- 追加写入一份长期的“纸上实盘日志” csv
- 同时输出当天的 markdown 预览，方便人工复盘

典型用法：

python tools/u1_paper_recorder.py ^
    --job-id ultrashort_main ^
    --tag u1_v1_base_rf ^
    --as-of 2025-12-04

可选参数：
    --orders-csv  手工指定下单列表路径
    --log-csv     手工指定纸上实盘日志路径

注意：
- --as-of 支持两种写法：
    * 20251204
    * 2025-12-04
  内部会统一转成 8 位数字用于文件名后缀。
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Tuple

import pandas as pd


# ----------------------------------------------------------------------
# 参数解析 & 工具函数
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 纸上实盘记录脚本（从当日下单列表生成/追加纸上实盘日志）"
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="任务 ID，例如 ultrashort_main",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="模型 / 配置标签，例如 u1_v1_base_rf",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="打分/下单日期，格式 YYYYMMDD 或 YYYY-MM-DD，例如 20251204 或 2025-12-04",
    )
    parser.add_argument(
        "--orders-csv",
        default=None,
        help=(
            "下单列表 csv 路径。"
            "若不指定，则默认使用 reports/u1_orders_{job_id}_{tag}_{as_of}.csv"
        ),
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help=(
            "纸上实盘日志 csv 路径。"
            "若不指定，则默认使用 data/live/paper_trades_{job_id}.csv"
        ),
    )
    return parser.parse_args()


def infer_orders_path(job_id: str, tag: str, as_of_key: str) -> str:
    """
    按约定的命名规则推断下单列表路径：
    reports/u1_orders_{job_id}_{tag}_{as_of}.csv

    这里的 as_of_key 一定是 8 位数字（YYYYMMDD）。
    """
    fname = f"u1_orders_{job_id}_{tag}_{as_of_key}.csv"
    return os.path.join("reports", fname)


def infer_log_path(job_id: str) -> str:
    """
    推断纸上实盘总日志路径：
    data/live/paper_trades_{job_id}.csv
    """
    base_dir = os.path.join("data", "live")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"paper_trades_{job_id}.csv")


def normalize_as_of(as_of_raw: str) -> Tuple[str, str]:
    """
    归一化 as_of：

    输入：
        as_of_raw: "20251031" 或 "2025-10-31"

    返回：
        as_of_key   : "20251031"  （用于文件名、csv 中的 as_of 字段）
        as_of_label : "2025-10-31"（用于打印和 markdown 展示）
    """
    s = as_of_raw.strip()
    if not s:
        raise SystemExit("[ERROR] as_of 不能为空")

    compact = s.replace("-", "")
    if len(compact) != 8 or not compact.isdigit():
        raise SystemExit(
            f"[ERROR] 无法解析 as_of={as_of_raw}，请使用 YYYYMMDD 或 YYYY-MM-DD 形式"
        )

    year = compact[0:4]
    month = compact[4:6]
    day = compact[6:8]
    label = f"{year}-{month}-{day}"
    return compact, label


# ----------------------------------------------------------------------
# markdown 预览
# ----------------------------------------------------------------------


def save_markdown_daily(
    orders_df: pd.DataFrame,
    job_id: str,
    tag: str,
    as_of_label: str,
    md_path: str,
) -> None:
    """
    保存当日纸上实盘 markdown 预览。

    只依赖基础字段：code / close / amount / u1_score / target_weight / rank（存在才用）。
    """
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    candidate_cols = ["code", "close", "amount", "u1_score", "target_weight", "rank"]
    cols = [c for c in candidate_cols if c in orders_df.columns]

    # 一定要有 code
    if "code" not in cols and "code" in orders_df.columns:
        cols.insert(0, "code")

    lines = []
    lines.append(f"# U1 纸上实盘记录 - {job_id} / {tag} / {as_of_label}")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 记录数量: {len(orders_df)}")
    lines.append("")

    if not cols:
        # 没有可展示列，就简单列出 code
        if "code" in orders_df.columns:
            cols = ["code"]
        else:
            # 实在不行就直接退出
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

    # markdown 表头
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)

    for _, row in orders_df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                # 浮点数限制一下长度，防止太长
                v = f"{v:.6g}"
            vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    job_id = args.job_id
    tag = args.tag

    # 归一化 as_of：得到 8 位数字 key + 漂亮的 label
    as_of_key, as_of_label = normalize_as_of(args.as_of)

    orders_path = args.orders_csv or infer_orders_path(job_id, tag, as_of_key)
    log_path = args.log_csv or infer_log_path(job_id)

    print(f"[INFO] 任务信息: job_id={job_id}, tag={tag}, as_of={as_of_label} ({as_of_key})")
    print(f"[INFO] 预期下单列表: {orders_path}")

    if not os.path.exists(orders_path):
        print(f"[ERROR] 找不到下单列表文件: {orders_path}")
        print("        请检查 --orders-csv 参数或确保先跑完 u1_daily_pipeline。")
        return 1

    # 读取当天下单列表
    orders_df = pd.read_csv(orders_path, encoding="utf-8")
    if orders_df.empty:
        print("[WARN] 下单列表为空，本次不记录纸上实盘。")
        return 0

    if "code" not in orders_df.columns:
        print("[ERROR] 下单列表中缺少必需列: code")
        print(f"       实际列名: {list(orders_df.columns)}")
        return 1

    print(f"[INFO] 当日下单记录数: {len(orders_df)}")

    # 构造要追加到日志中的记录
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_rows = orders_df.copy()

    # 在前面插入 / 覆盖几列固定字段：job_id, tag, as_of
    # 先覆盖已有列的值，再补充缺失列，避免 "cannot insert ... already exists"
    for col_name, value in [("job_id", job_id), ("tag", tag), ("as_of", as_of_key)]:
        if col_name in new_rows.columns:
            new_rows[col_name] = value
        else:
            new_rows.insert(0, col_name, value)

    new_rows["recorded_at"] = recorded_at

    # 调整列顺序：job_id, tag, as_of, code, recorded_at, ...
    cols = new_rows.columns.tolist()
    preferred = ["job_id", "tag", "as_of", "code", "recorded_at"]
    ordered = preferred + [c for c in cols if c not in preferred]
    new_rows = new_rows[ordered]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        # 已有历史日志 -> 追加并去重
        old_df = pd.read_csv(log_path, encoding="utf-8")
        before = len(old_df)

        combined = pd.concat([old_df, new_rows], ignore_index=True)

        # 按 (job_id, tag, as_of, code) 去重，保留最后一次记录
        subset_cols = [c for c in ["job_id", "tag", "as_of", "code"] if c in combined.columns]
        if subset_cols:
            combined.drop_duplicates(subset=subset_cols, keep="last", inplace=True)

        after = len(combined)
        added = max(after - before, 0)

        combined.to_csv(log_path, index=False, encoding="utf-8-sig")
    else:
        # 首次创建日志
        combined = new_rows
        added = len(new_rows)
        combined.to_csv(log_path, index=False, encoding="utf-8-sig")

    total = len(combined)
    print(f"[INFO] 纸上实盘日志文件: {log_path}")
    print(f"[OK] 本次新增记录 {added} 条，去重后总记录数 {total} 条。")

    # 生成当天的 markdown 预览
    md_path = os.path.join("reports", f"u1_paper_{job_id}_{tag}_{as_of_key}.md")
    try:
        save_markdown_daily(orders_df, job_id, tag, as_of_label, md_path)
        print(f"[OK] 当日纸上实盘预览已保存为 markdown: {md_path}")
    except Exception as e:
        print(f"[WARN] 写入 markdown 预览失败: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
