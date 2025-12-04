# -*- coding: utf-8 -*-
"""
U1 多因子 + ML Sweep 脚本

一键扫多组特征 + 多个模型，并按测试集 Sharpe 排名。

用法示例：

python -m tools.u1_multifactor_ml_sweep ^
    --job-id ultrashort_main ^
    --start-date 2020-01-01 ^
    --end-date 2025-10-31 ^
    --top-k 30 ^
    --min-price 3 ^
    --max-price 80 ^
    --min-amount 20000000 ^
    --ret-col ret_1 ^
    --train-end-date 2022-12-31 ^
    --feature-tags base,base_nomom,amt_only ^
    --models rf,gb,ridge
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .u1_multifactor_ml_research import run_experiment, _get_logger


logger = _get_logger("U1")
sweep_logger = _get_logger("U1SWEEP")


# 预定义特征组合（标签 -> 列名列表）
# 注意：会在 run_experiment 内部自动忽略数据集中不存在的列。
FEATURE_TAG_MAP: Dict[str, List[str]] = {
    # 原始设计里的 log_* / *_mean_* 继续保留，如果数据里暂时没有，会被自动忽略
    "base": [
        "log_amount",
        "log_volume",
        "log_amt_mean_20",
        "amt_to_mean_20",
        "amount",
        "vol_20",
        "ret_1",
        "ret_5",
        "ret_20",
        "rev_1",
    ],
    "base_nomom": [
        "log_amount",
        "log_volume",
        "log_amt_mean_20",
        "amt_to_mean_20",
        "amount",
        "vol_20",
    ],
    "amt_only": [
        "log_amount",
        "log_volume",
        "log_amt_mean_20",
        "amt_to_mean_20",
        "amount",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("u1_multifactor_ml_sweep.py")

    p.add_argument("--job-id", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--train-end-date", required=True)

    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--max-price", type=float, default=80.0)
    p.add_argument("--min-amount", type=float, default=20_000_000.0)
    p.add_argument("--ret-col", default="ret_1")

    p.add_argument(
        "--feature-tags",
        required=True,
        help="逗号分隔的特征标签，如: base,base_nomom,amt_only",
    )
    p.add_argument(
        "--models",
        required=True,
        help="逗号分隔的模型名称，如: rf,gb,ridge",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    feature_tags = [s.strip() for s in args.feature_tags.split(",") if s.strip()]
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]

    # 检查标签合法性
    unknown_tags = [t for t in feature_tags if t not in FEATURE_TAG_MAP]
    if unknown_tags:
        raise ValueError(f"存在未知 feature_tag: {unknown_tags}，当前可选: {sorted(FEATURE_TAG_MAP.keys())}")

    sweep_logger.info("使用特征标签: %s", feature_tags)
    sweep_logger.info("使用模型集 : %s", model_names)

    rows = []

    for tag in feature_tags:
        feature_cols = FEATURE_TAG_MAP[tag]
        for model_name in model_names:
            sweep_logger.info("开始组合: feature_tag=%s, model=%s", tag, model_name)

            try:
                stats_full, stats_train, stats_test, equity_df, yearly_df, summary = run_experiment(
                    job_id=args.job_id,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    train_end_date=args.train_end_date,
                    top_k=args.top_k,
                    min_price=args.min_price,
                    max_price=args.max_price,
                    min_amount=args.min_amount,
                    ret_col=args.ret_col,
                    feature_cols=feature_cols,
                    model_name=model_name,
                )

                row = dict(
                    feature_tag=tag,
                    model=model_name,
                    n_features=summary["n_features"],
                    n_train=summary["n_train"],
                    n_test=summary["n_test"],
                    test_ann_return=summary["test_ann_return"],
                    test_sharpe=summary["test_sharpe"],
                    test_max_dd=summary["test_max_dd"],
                    full_ann_return=summary["full_ann_return"],
                    full_sharpe=summary["full_sharpe"],
                    full_max_dd=summary["full_max_dd"],
                    train_sharpe=summary["train_sharpe"],
                )
                rows.append(row)

            except ValueError as e:
                msg = str(e)
                # 专门处理“有效特征列为空”的情况：该组合记为无效，不中断整个 sweep
                if msg.startswith("有效特征列为空"):
                    sweep_logger.warning(
                        "组合 feature_tag=%s, model=%s 无有效特征，已跳过：%s", tag, model_name, msg
                    )
                    rows.append(
                        dict(
                            feature_tag=tag,
                            model=model_name,
                            n_features=0,
                            n_train=0,
                            n_test=0,
                            test_ann_return=np.nan,
                            test_sharpe=np.nan,
                            test_max_dd=np.nan,
                            full_ann_return=np.nan,
                            full_sharpe=np.nan,
                            full_max_dd=np.nan,
                            train_sharpe=np.nan,
                        )
                    )
                    continue
                else:
                    # 其它错误直接抛出，方便你排查
                    raise

    # 汇总结果
    summary_df = pd.DataFrame(rows)

    # 按测试集 Sharpe 排序（从高到低）
    if "test_sharpe" in summary_df.columns:
        summary_df = summary_df.sort_values("test_sharpe", ascending=False).reset_index(drop=True)

    print()
    print("==== U1 多因子 + ML Sweep 结果（按测试集 Sharpe 排序） ====")
    if not summary_df.empty:
        # 为了在终端好看一点，百分比列统一 *100
        def fmt_pct(x: float) -> str:
            if pd.isna(x):
                return "nan"
            return f"{x * 100:.2f}%"

        display_df = summary_df.copy()
        for col in [
            "test_ann_return",
            "test_max_dd",
            "full_ann_return",
            "full_max_dd",
            "full_sharpe",  # 这里 Sharpe 不乘 100，保持原样
            "train_sharpe",
        ]:
            if col in display_df.columns and col.endswith("_return") or col.endswith("_dd"):
                display_df[col] = display_df[col].apply(lambda v: fmt_pct(v) if pd.notna(v) else "nan")

        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("没有任何有效组合。")

    # 保存结果
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "reports" / "u1_multifactor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"u1_sweep_summary_{args.job_id}.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8")
    sweep_logger.info("排名结果已保存到: %s", out_path)


if __name__ == "__main__":
    main()
