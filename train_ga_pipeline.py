# -*- coding: utf-8 -*-
"""
train_ga_pipeline.py

训练入口（基于 DatasetBuilder-S + ModelZoo + TrainingPipelines）：

- 通过 DatasetBuilder 构建 [start, end] 区间的特征 + 标签数据集；
- 使用 TrainingPipelines.run_single_label_training 完成
  train/val/test 拆分 + 模型训练 + 指标评估；
- 默认用 sklearn 模型作为 baseline，你可以在 ModelZoo 中挂上自己的模型，
  或在 GA 中调用 TrainingPipelines 做大规模参数搜索。

说明：
- 文件名仍然叫 train_ga_pipeline.py，但内部逻辑已经是通用训练入口；
- GA 搜索时，可以把这里面的训练过程封装成“个体评估函数”。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.logging_utils import get_logger
from alpha.training_pipelines import TrainingConfig, TrainingResult, run_single_label_training

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter - GA / 模型训练入口（基于 DatasetBuilder + ModelZoo）"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="训练样本开始日期（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="训练样本结束日期（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="y_ret_1d_close",
        help="主标签名（需在 config/label_spec.json 中定义），默认 y_ret_1d_close。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        help="模型名：rf / gbdt / linear 等，默认 rf。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集按日期划分比例（0~1），默认 0.2。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="测试集按日期划分比例（0~1），默认 0（不开测试集）。",
    )
    parser.add_argument(
        "--min-data-score",
        type=float,
        default=80.0,
        help="依据 data_calendar 过滤数据质量分数阈值，默认 80。",
    )
    parser.add_argument(
        "--no-dropna",
        action="store_true",
        help="不丢弃含 NaN 的样本（默认会丢弃）。",
    )
    parser.add_argument(
        "--codes",
        type=str,
        nargs="*",
        default=None,
        help="可选，仅在指定股票池上训练，例如：--codes 000001 600000。",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="可选，将训练好的模型保存到此路径（Joblib 格式）。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = TrainingConfig(
        start_date=args.start,
        end_date=args.end,
        label_name=args.label,
        model_name=args.model,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_data_score=args.min_data_score,
        dropna=not args.no_dropna,
        codes=args.codes,
    )

    try:
        result: TrainingResult = run_single_label_training(cfg)
    except Exception:
        logger.exception("训练流程发生异常。")
        print("训练失败，请查看日志。")
        return

    # 打印指标
    print("=== Training Metrics ===")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Data Info ===")
    print(f"label        : {result.label_name}")
    print(f"features     : {len(result.feature_cols)} 个")
    print(f"train samples: {result.train_samples}, dates={result.train_date_range}")
    print(f"val samples  : {result.val_samples}, dates={result.val_date_range}")
    print(f"test samples : {result.test_samples}, dates={result.test_date_range}")

    # 可选：保存模型
    if args.model_out:
        try:
            import joblib

            out_path = Path(args.model_out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result.model, out_path)
            logger.info("模型已保存到：%s", out_path)
            print(f"\n模型已保存到：{out_path}")
        except Exception:
            logger.exception("保存模型时出错。")
            print("模型保存失败，请查看日志。")


if __name__ == "__main__":
    main()
