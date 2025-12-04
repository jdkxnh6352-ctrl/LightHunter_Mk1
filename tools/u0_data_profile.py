# -*- coding: utf-8 -*-
"""
U0 Data Profile / 数据体检脚本

功能：
- 对 data/datasets/<job-id>.parquet 做基本体检；
- 输出：列名、数据类型、缺失值统计、样本数统计、
        数值列分布（mean/std/分位数）、类别列 TopN 分布；
- 报告以 Markdown / 纯文本形式写到 reports/u0_data_profile_<job-id>.md。

用法示例（在项目根目录下）：

python -m tools.u0_data_profile ^
  --job-id ultrashort_main ^
  --start-date 2020-01-01 ^
  --end-date 2025-10-31

也可以手动指定输入输出：

python -m tools.u0_data_profile ^
  --input data/datasets/ultrashort_main.parquet ^
  --output reports/u0_data_profile_ultrashort_main.md
"""

import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


# --------------------------
# 路径与日志工具
# --------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")


def log(msg: str) -> None:
    """简单日志输出。"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{now} [U0Profile] {msg}")


# --------------------------
# 核心体检逻辑
# --------------------------

def profile_dataframe(df: pd.DataFrame, max_cat_values: int = 5) -> str:
    """
    对 DataFrame 做体检，返回 Markdown 文本。
    """
    n_rows, n_cols = df.shape
    report_lines: List[str] = []

    # ---- 标题 & 总览 ----
    report_lines.append("# U0 数据体检报告")
    report_lines.append("")
    report_lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"- 样本数（行）：**{n_rows}**")
    report_lines.append(f"- 特征数（列）：**{n_cols}**")
    report_lines.append("")

    # trade_date / trading_date 识别
    trade_date_col = None
    if "trade_date" in df.columns:
        trade_date_col = "trade_date"
    elif "trading_date" in df.columns:
        trade_date_col = "trading_date"

    if trade_date_col is not None:
        try:
            # 尝试转成日期，失败就当普通字符串
            dates = pd.to_datetime(df[trade_date_col], errors="coerce")
            d_min = dates.min()
            d_max = dates.max()
            if pd.notna(d_min) and pd.notna(d_max):
                report_lines.append(
                    f"- 交易日期范围：**{d_min.date()} ~ {d_max.date()}**  "
                    f"(列名：`{trade_date_col}`)"
                )
                report_lines.append("")
        except Exception:
            # 安全起见，任何错误都忽略
            pass

    # ---- 列概览表：dtype / 缺失 / 唯一值 ----
    report_lines.append("## 1. 列概览（数据类型 & 缺失情况）")
    report_lines.append("")
    report_lines.append(
        "| 列名 | dtype | 非缺失数量 | 缺失数量 | 缺失比例 | 唯一值数量 |"
    )
    report_lines.append("| --- | --- | ---: | ---: | ---: | ---: |")

    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        missing = s.isna().sum()
        missing_pct = (missing / n_rows * 100) if n_rows > 0 else np.nan
        n_unique = s.nunique(dropna=True)

        report_lines.append(
            f"| `{col}` | {s.dtype} | {non_null} | {missing} | "
            f"{missing_pct:.2f}% | {n_unique} |"
        )

    report_lines.append("")

    # ---- 数值列分布 ----
    report_lines.append("## 2. 数值列分布统计")
    report_lines.append("")
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

    if not numeric_cols:
        report_lines.append("_没有检测到数值型列。_")
        report_lines.append("")
    else:
        report_lines.append(
            "| 列名 | 非缺失数 | 均值 | 标准差 | 最小值 | 25%分位 | 中位数 | 75%分位 | 最大值 |"
        )
        report_lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )

        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                report_lines.append(
                    f"| `{col}` | 0 | NaN | NaN | NaN | NaN | NaN | NaN | NaN |"
                )
                continue

            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            mean = desc.get("mean", np.nan)
            std = desc.get("std", np.nan)
            vmin = desc.get("min", np.nan)
            q25 = desc.get("25%", np.nan)
            med = desc.get("50%", np.nan)
            q75 = desc.get("75%", np.nan)
            vmax = desc.get("max", np.nan)

            report_lines.append(
                f"| `{col}` | {int(desc.get('count', 0))} | "
                f"{mean:.4f} | {std:.4f} | {vmin:.4f} | {q25:.4f} | "
                f"{med:.4f} | {q75:.4f} | {vmax:.4f} |"
            )

        report_lines.append("")

    # ---- 类别/文本列分布 ----
    report_lines.append("## 3. 类别/文本列 TopN 分布")
    report_lines.append("")
    report_lines.append(
        f"下面只列出非数值列的前 **{max_cat_values}** 个取值及其占比。"
    )
    report_lines.append("")

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if not non_numeric_cols:
        report_lines.append("_没有检测到非数值型列。_")
        report_lines.append("")
    else:
        for col in non_numeric_cols:
            s = df[col]
            report_lines.append(f"### 列 `{col}`")
            report_lines.append("")
            non_null = s.notna().sum()
            missing = s.isna().sum()
            missing_pct = (missing / n_rows * 100) if n_rows > 0 else np.nan
            n_unique = s.nunique(dropna=True)

            report_lines.append(
                f"- 非缺失数量：**{non_null}** / {n_rows}  "
                f"(缺失：{missing}, {missing_pct:.2f}%)"
            )
            report_lines.append(f"- 唯一值数量：**{n_unique}**")
            report_lines.append("")

            if non_null == 0:
                report_lines.append("_全列缺失，无取值统计。_")
                report_lines.append("")
                continue

            vc = (
                s.value_counts(dropna=True)
                .head(max_cat_values)
            )

            report_lines.append(
                "| 值 | 计数 | 占比 |"
            )
            report_lines.append(
                "| --- | ---: | ---: |"
            )

            for val, cnt in vc.items():
                pct = cnt / non_null * 100
                # 把值转成字符串，避免换行太夸张
                val_str = str(val)
                if len(val_str) > 40:
                    val_str = val_str[:37] + "..."
                report_lines.append(
                    f"| `{val_str}` | {cnt} | {pct:.2f}% |"
                )

            report_lines.append("")

    # ---- 总结 ----
    report_lines.append("## 4. 简要小结")
    report_lines.append("")
    report_lines.append(
        "- 可以重点关注缺失比例较高的列，考虑补全 / 删除 / 用模型预测；"
    )
    report_lines.append(
        "- 数值列波动过大的，可能需要做标准化或截尾；"
    )
    report_lines.append(
        "- 类别列如果唯一值很多（接近样本数），更适合作为 ID 而不是特征。"
    )
    report_lines.append("")

    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U0 数据体检脚本：对 parquet 数据集做列名、缺失值、分布统计，并输出 Markdown 报告。"
    )

    parser.add_argument(
        "--job-id",
        type=str,
        default="ultrashort_main",
        help="数据集 ID，默认会读 data/datasets/<job-id>.parquet",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 parquet 路径（可选，优先于 job-id 默认路径）。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告路径，默认为 reports/u0_data_profile_<job-id>.md",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="数据区间起始日期（仅写到报告里做说明，不做过滤）。",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="数据区间结束日期（仅写到报告里做说明，不做过滤）。",
    )
    parser.add_argument(
        "--max-category-values",
        type=int,
        default=5,
        help="类别列展示的 TopN 取值数量，默认 5。",
    )

    args = parser.parse_args()

    # 解析输入输出路径
    if args.input is not None:
        dataset_path = args.input
    else:
        dataset_path = os.path.join(DATA_DIR, f"{args.job_id}.parquet")

    if args.output is not None:
        report_path = args.output
    else:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(
            REPORT_DIR, f"u0_data_profile_{args.job_id}.md"
        )

    log(f"读取数据集：{dataset_path}")
    if not os.path.exists(dataset_path):
        log(f"[ERROR] 找不到数据集文件：{dataset_path}")
        raise SystemExit(1)

    df = pd.read_parquet(dataset_path)

    log("开始生成体检报告……")
    report_md = profile_dataframe(df, max_cat_values=args.max_category_values)

    # 在报告头部补充一下“理论区间说明”
    if args.start_date or args.end_date:
        header_lines = []
        if args.start_date and args.end_date:
            header_lines.append(
                f"> 本次体检理论覆盖区间：{args.start_date} ~ {args.end_date}"
            )
        elif args.start_date:
            header_lines.append(
                f"> 本次体检理论覆盖区间：自 {args.start_date} 之后所有数据"
            )
        elif args.end_date:
            header_lines.append(
                f"> 本次体检理论覆盖区间：截止到 {args.end_date} 之前所有数据"
            )
        header_lines.append("")
        report_md = "\n".join(header_lines) + "\n" + report_md

    # 写文件
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    log(f"体检报告已保存到：{report_path}")
    log("数据体检完成。")


if __name__ == "__main__":
    main()
