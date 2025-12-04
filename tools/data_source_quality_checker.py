# -*- coding: utf-8 -*-
"""
tools/data_source_quality_checker.py

LightHunter Mk4 - 多源字段质量统计与日报输出
============================================

用途
----
对同一逻辑字段在多个数据源上的取值进行质量评估，输出每日数据源质量报告：

- 覆盖率 Coverage: 每个字段/源的非缺失比例
- 偏差率 Bias: 相对于“参考值”（主源或多源中位数）的平均相对偏差
- 异常率 Anomaly rate: 相对于本源中位数的极端偏离比例
- 时效性 Timeliness: 如有时间戳列，计算相对“预期更新时间”的平均延迟
- 综合质量分 Quality Score (0-100)

本模块只做“统计 + 报表”，不负责修复数据，也不改变存储。

运行方式示例
------------
# 默认检查“今天”（或最近一个自然日）的质量
python -m tools.data_source_quality_checker

# 指定日期（YYYYMMDD）
python -m tools.data_source_quality_checker --date 20250314

# 调整取样股票数量（0 或负数 = 不采样，用全市场）
python -m tools.data_source_quality_checker --date 20250314 --sample-size 500

配置依赖
--------
1. config/system_config.json 中的字段：
   - paths.duckdb_path: DuckDB 文件路径
   - data_quality: 质量检查配置（见本文件下方配置示例）

2. config/data_schema.json (可选增强)：
   - logical_fields: 字段定义
   - source_mappings: 字段的多源映射（用于确定主源 / 源ID）
   - quality_weights: 默认权重，如果 data_quality 中未覆盖

数据表假设
----------
默认假设存在一张“多源日线长表”，结构类似：

    multi_source_daily_quotes (
        trade_date   VARCHAR,   -- 日期，如 '2025-03-14'
        ts_code      VARCHAR,   -- 内部代码，如 '600000.SH'
        source_id    VARCHAR,   -- 数据源ID，如 'ths', 'eastmoney'
        field        VARCHAR,   -- 逻辑字段名，如 'close', 'pct_chg'
        value        DOUBLE,    -- 字段值
        ingest_ts    TIMESTAMP  -- (可选) 入库时间，用于时效性指标
    )

如你的实际表/字段不同，可在 system_config.data_quality 中配置：
    table / date_column / code_column / source_column / field_column / value_column / timestamp_column

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from core.logging_utils import get_logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str):
        return logging.getLogger(name)


log = get_logger(__name__)

try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        log.warning("config.config_center.get_system_config 未找到，使用空配置")
        return {}


# ----------------------------------------------------------------------
# 配置与数据结构
# ----------------------------------------------------------------------


@dataclass
class QualityWeights:
    coverage: float = 0.35
    bias: float = 0.25
    anomaly: float = 0.20
    latency: float = 0.15
    schema: float = 0.05  # 目前尚未使用，可保留扩展

    def normalize(self) -> "QualityWeights":
        s = self.coverage + self.bias + self.anomaly + self.latency + self.schema
        if s <= 0:
            return self
        return QualityWeights(
            coverage=self.coverage / s,
            bias=self.bias / s,
            anomaly=self.anomaly / s,
            latency=self.latency / s,
            schema=self.schema / s,
        )


@dataclass
class Thresholds:
    bias: float = 0.02           # 2% 相对偏差视为“满处罚”，常规偏差小于这个不严重
    anomaly_z: float = 5.0       # 相对中位数偏离超过 5 倍视为异常
    latency_seconds: float = 300 # 超过 5 分钟视为“满处罚”


@dataclass
class FieldSourceMetrics:
    coverage: float
    bias: Optional[float]
    anomaly_rate: Optional[float]
    latency_mean: Optional[float]
    score: float


# ----------------------------------------------------------------------
# 核心质量检查类
# ----------------------------------------------------------------------


class DataSourceQualityChecker:
    """
    多源字段质量统计器
    """

    def __init__(self, system_config: Dict[str, Any]) -> None:
        self.system_config = system_config or {}
        self.paths_cfg = self.system_config.get("paths", {}) or {}
        self.dq_cfg = self.system_config.get("data_quality", {}) or {}

        self.db_path = (
            self.dq_cfg.get("duckdb_path")
            or self.paths_cfg.get("duckdb_path")
            or "data/lighthunter.duckdb"
        )

        self.table = self.dq_cfg.get("table", "multi_source_daily_quotes")
        self.date_col = self.dq_cfg.get("date_column", "trade_date")
        self.code_col = self.dq_cfg.get("code_column", "ts_code")
        self.source_col = self.dq_cfg.get("source_column", "source_id")
        self.field_col = self.dq_cfg.get("field_column", "field")
        self.value_col = self.dq_cfg.get("value_column", "value")
        self.ts_col = self.dq_cfg.get("timestamp_column", "t_ingest")  # 可不存在

        self.sample_size = int(self.dq_cfg.get("sample_size_per_day", 0) or 0)
        self.bias_reference = self.dq_cfg.get("bias_reference", "primary")  # 或 "median"

        self.output_dir = self.dq_cfg.get("output_dir", "reports/data_quality")

        # 默认权重
        qw = self._load_quality_weights()
        self.weights = qw.normalize()

        # 阈值
        t_cfg = self.dq_cfg.get("thresholds", {}) or {}
        self.thresholds = Thresholds(
            bias=float(t_cfg.get("bias", 0.02)),
            anomaly_z=float(t_cfg.get("anomaly_z", 5.0)),
            latency_seconds=float(t_cfg.get("latency_seconds", 300.0)),
        )

        # 尝试加载 data_schema（可选）
        self.data_schema = self._load_data_schema()

    # ------------------------------------------------------------------
    # 外部入口
    # ------------------------------------------------------------------

    def run_for_date(self, date: dt.date) -> Dict[str, Any]:
        """
        对某一天的数据进行质量评估，并返回报告字典。
        """
        if duckdb is None:
            raise RuntimeError("duckdb 未安装，请先安装 duckdb 库。")

        if pd is None:
            raise RuntimeError("pandas 未安装，请先安装 pandas 库。")

        date_str = date.strftime("%Y-%m-%d")
        log.info("开始进行多源质量评估：date=%s, db=%s, table=%s", date_str, self.db_path, self.table)

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 加载数据
        df = self._load_data_for_date(date_str)
        if df.empty:
            log.warning("指定日期 %s 在表 %s 中没有数据，输出空报告。", date_str, self.table)
            report = {
                "date": date_str,
                "db_path": self.db_path,
                "table": self.table,
                "status": "empty",
                "message": "no data found for this date",
            }
            self._save_report(report)
            return report

        # 2. 如需采样，随机抽取部分标的
        df = self._maybe_sample_codes(df)

        # 3. 计算指标
        report = self._compute_metrics(df, date_str)

        # 4. 保存报告
        self._save_report(report)

        return report

    # ------------------------------------------------------------------
    # 加载配置 / 数据
    # ------------------------------------------------------------------

    def _load_quality_weights(self) -> QualityWeights:
        """
        优先从 data_schema.quality_weights 读取，其次从 system_config.data_quality.weights 读取，
        最后使用默认值。
        """
        # system_config 覆盖
        w_cfg = self.dq_cfg.get("weights")
        if isinstance(w_cfg, dict):
            return QualityWeights(
                coverage=float(w_cfg.get("coverage", 0.35)),
                bias=float(w_cfg.get("bias", 0.25)),
                anomaly=float(w_cfg.get("anomaly", 0.20)),
                latency=float(w_cfg.get("latency", 0.15)),
                schema=float(w_cfg.get("schema", 0.05)),
            )

        # 尝试从 data_schema.json 读取
        schema = self._load_data_schema()
        if schema:
            qw = schema.get("quality_weights") or {}
            if isinstance(qw, dict):
                return QualityWeights(
                    coverage=float(qw.get("coverage", 0.35)),
                    bias=float(qw.get("bias", 0.25)),
                    anomaly=float(qw.get("anomaly", 0.20)),
                    latency=float(qw.get("latency", 0.15)),
                    schema=float(qw.get("schema", 0.05)),
                )

        # 默认
        return QualityWeights()

    def _load_data_schema(self) -> Optional[Dict[str, Any]]:
        """
        尝试加载 config/data_schema.json，可选。
        """
        # 先从 config_center 找
        try:
            from config.config_center import get_data_schema  # type: ignore

            schema = get_data_schema()  # type: ignore[call-arg]
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

        # 退而求其次，直接读 config/data_schema.json
        try:
            # 推断 config 目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
            cfg_dir = os.path.join(base_dir, "config")
            schema_path = os.path.join(cfg_dir, "data_schema.json")
            if os.path.isfile(schema_path):
                with open(schema_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            log.warning("加载 data_schema.json 失败: %s", e)

        return None

    def _load_data_for_date(self, date_str: str) -> "pd.DataFrame":
        """
        从 DuckDB 加载指定日期的多源长表数据。
        """
        con = duckdb.connect(self.db_path, read_only=True)  # type: ignore[arg-type]
        cols = [
            self.date_col,
            self.code_col,
            self.source_col,
            self.field_col,
            self.value_col,
        ]

        # 如果存在时间戳列则一并加载
        with_ts = False
        try:
            # 简单判断列是否在表中：通过 PRAGMA 或 information_schema
            # 这里偷懒，直接尝试查询，如果失败再降级移除 ts 列
            test_sql = f"""
                SELECT {self.ts_col}
                FROM {self.table}
                WHERE 1 = 0
            """
            con.execute(test_sql)
            cols.append(self.ts_col)
            with_ts = True
        except Exception:
            with_ts = False

        cols_sql = ", ".join(cols)
        sql = f"""
            SELECT {cols_sql}
            FROM {self.table}
            WHERE {self.date_col} = ?
        """
        log.info("DuckDB SQL: %s", sql.replace("\n", " "))
        try:
            df = con.execute(sql, [date_str]).fetch_df()
        finally:
            con.close()

        if df.empty:
            return df

        # 标准列名对齐
        df = df.rename(
            columns={
                self.date_col: "trade_date",
                self.code_col: "ts_code",
                self.source_col: "source_id",
                self.field_col: "field",
                self.value_col: "value",
                self.ts_col: "ingest_ts" if with_ts else self.ts_col,
            }
        )

        # 如果时间戳列存在，转换为 pandas datetime
        if with_ts and "ingest_ts" in df.columns:
            try:
                df["ingest_ts"] = pd.to_datetime(df["ingest_ts"])
            except Exception:
                log.warning("无法将 ingest_ts 转换为 datetime，将忽略时效性指标。")
                df.drop(columns=["ingest_ts"], inplace=True, errors="ignore")

        return df

    def _maybe_sample_codes(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """
        如果 sample_size > 0，则随机抽样部分标的进行评估。
        """
        if self.sample_size <= 0:
            return df

        codes = df["ts_code"].unique().tolist()
        if len(codes) <= self.sample_size:
            return df

        sampled = pd.Series(codes).sample(self.sample_size, random_state=42).tolist()
        log.info("从 %d 个标的中随机抽取 %d 个进行质量评估。", len(codes), len(sampled))
        return df[df["ts_code"].isin(sampled)].copy()

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def _compute_metrics(self, df: "pd.DataFrame", date_str: str) -> Dict[str, Any]:
        """
        核心质量评估逻辑。
        返回结构化报告字典。
        """
        fields = sorted(df["field"].unique().tolist())
        sources = sorted(df["source_id"].unique().tolist())
        log.info("本次评估涉及字段 %d 个，数据源 %d 个。", len(fields), len(sources))

        # 汇总结果: field -> source -> metrics
        field_results: Dict[str, Dict[str, FieldSourceMetrics]] = {}

        for field in fields:
            df_f = df[df["field"] == field].copy()
            if df_f.empty:
                continue

            metrics_for_field: Dict[str, FieldSourceMetrics] = {}
            # 每个 ts_code 一天应该有一次取值
            codes = sorted(df_f["ts_code"].unique().tolist())
            expected_count = len(codes)
            if expected_count == 0:
                continue

            # pivot 成宽表: index = ts_code, columns = source_id, values = value
            pivot = df_f.pivot_table(
                index="ts_code",
                columns="source_id",
                values="value",
                aggfunc="first",
            )

            # 参考值: primary 或 median
            v_ref = self._compute_reference_series(field, pivot)

            # 为每个 source 计算 coverage / bias / anomaly / latency / score
            for source_id in pivot.columns:
                series = pivot[source_id]

                coverage = float(series.notna().sum()) / float(expected_count)

                bias = self._compute_bias(series, v_ref)
                anomaly_rate = self._compute_anomaly_rate(series)
                latency_mean = self._compute_latency_mean(df_f, source_id)

                score = self._compute_quality_score(
                    coverage=coverage,
                    bias=bias,
                    anomaly_rate=anomaly_rate,
                    latency_mean=latency_mean,
                )

                metrics_for_field[source_id] = FieldSourceMetrics(
                    coverage=coverage,
                    bias=bias,
                    anomaly_rate=anomaly_rate,
                    latency_mean=latency_mean,
                    score=score,
                )

            field_results[field] = metrics_for_field

        # 汇总 summary：找出“最差源/最差字段”
        summary = self._build_summary(field_results)

        report: Dict[str, Any] = {
            "date": date_str,
            "db_path": self.db_path,
            "table": self.table,
            "sampling": {
                "sample_size_per_day": self.sample_size,
                "actual_codes": int(df["ts_code"].nunique()),
            },
            "weights": {
                "coverage": self.weights.coverage,
                "bias": self.weights.bias,
                "anomaly": self.weights.anomaly,
                "latency": self.weights.latency,
                "schema": self.weights.schema,
            },
            "thresholds": {
                "bias": self.thresholds.bias,
                "anomaly_z": self.thresholds.anomaly_z,
                "latency_seconds": self.thresholds.latency_seconds,
            },
            "fields": {},
            "summary": summary,
        }

        # 序列化 field_results
        for field, src_metrics in field_results.items():
            report["fields"][field] = {
                "sources": {
                    src: {
                        "coverage": m.coverage,
                        "bias": m.bias,
                        "anomaly_rate": m.anomaly_rate,
                        "latency_mean": m.latency_mean,
                        "score": m.score,
                    }
                    for src, m in src_metrics.items()
                }
            }

        return report

    def _compute_reference_series(self, field: str, pivot: "pd.DataFrame") -> "pd.Series":
        """
        计算参考值序列 v_ref (index=ts_code)。优先使用主源，否则用多源中位数。
        """
        if pivot.empty:
            return pd.Series(dtype=float)

        # 候选主源列表
        primary_sources: List[str] = self._get_primary_sources_for_field(field)

        if self.bias_reference == "primary" and primary_sources:
            # 在 pivot 的列中，保留存在的主源
            avail_primary = [s for s in primary_sources if s in pivot.columns]
            if avail_primary:
                # 对主源列按列优先级进行横向填充，选取第一非空
                v_ref = pivot[avail_primary].bfill(axis=1).iloc[:, 0]
                # 仍有整行全 NaN 的，用多源中位数补
                median_series = pivot.median(axis=1, skipna=True)
                v_ref = v_ref.fillna(median_series)
                return v_ref

        # 默认用多源中位数
        return pivot.median(axis=1, skipna=True)

    def _get_primary_sources_for_field(self, field: str) -> List[str]:
        """
        从 data_schema.source_mappings 中解析该字段的主源 source_id 列表。
        """
        schema = self.data_schema
        if not schema:
            return []

        try:
            sm = schema.get("source_mappings", {}).get(field, {})
            src_cfg = sm.get("sources", {}) or {}
        except Exception:
            return []

        primary: List[str] = []
        for _, cfg in src_cfg.items():
            try:
                if cfg.get("is_primary"):
                    sid = cfg.get("source_id")
                    if sid:
                        primary.append(str(sid))
            except Exception:
                continue

        return primary

    def _compute_bias(self, series: "pd.Series", v_ref: "pd.Series") -> Optional[float]:
        """
        计算该 source 对应字段相对于参考值的平均相对偏差。
        """
        if series is None or v_ref is None:
            return None

        mask = series.notna() & v_ref.notna()
        if not mask.any():
            return None

        v = series[mask].astype(float)
        ref = v_ref[mask].astype(float)

        eps = 1e-8
        rel_diff = (v - ref).abs() / (ref.abs() + eps)
        if rel_diff.empty:
            return None
        return float(rel_diff.mean())

    def _compute_anomaly_rate(self, series: "pd.Series") -> Optional[float]:
        """
        对该 source 的该字段，基于“相对于本源中位数的偏离”粗略估计异常率。
        """
        if series is None:
            return None

        v = series.dropna().astype(float)
        if len(v) < 3:
            return None

        median = v.median()
        eps = 1e-8
        rel_dev = (v - median).abs() / (abs(median) + eps)

        # 偏离超过 anomaly_z（倍数）的视为异常
        mask = rel_dev > self.thresholds.anomaly_z
        rate = float(mask.sum()) / float(len(v))
        return rate

    def _compute_latency_mean(
        self,
        df_field: "pd.DataFrame",
        source_id: str,
    ) -> Optional[float]:
        """
        粗略计算该字段在指定 source 的平均延迟秒数（如 ingest_ts 存在）。
        """
        if "ingest_ts" not in df_field.columns:
            return None

        df_src = df_field[(df_field["source_id"] == source_id) & df_field["ingest_ts"].notna()]
        if df_src.empty:
            return None

        # 预期更新时间（每日固定）配置：data_quality.expected_update_time = "HH:MM"
        exp_time_str = self.dq_cfg.get("expected_update_time", "09:10")
        try:
            hh, mm = [int(x) for x in exp_time_str.split(":")]
        except Exception:
            hh, mm = 9, 10

        # 使用 trade_date 列构造 expected_ts
        # 由于本模块只针对某一天，这里直接用该日期
        trade_date = df_src["trade_date"].iloc[0]
        try:
            d = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except Exception:
            # 退而求其次，使用今天
            d = dt.date.today()

        expected_ts = dt.datetime(d.year, d.month, d.day, hh, mm, 0)

        # 计算延迟
        latency = (df_src["ingest_ts"] - expected_ts).dt.total_seconds()
        latency = latency[latency.notna()]
        if latency.empty:
            return None

        # 只关心正向延迟，负数（提前到达）当作 0
        latency_clipped = latency.clip(lower=0.0)
        return float(latency_clipped.mean())

    def _compute_quality_score(
        self,
        coverage: float,
        bias: Optional[float],
        anomaly_rate: Optional[float],
        latency_mean: Optional[float],
    ) -> float:
        """
        根据各项指标和权重计算综合质量分（0-100）。
        """
        w = self.weights
        thr = self.thresholds

        # 缺失处罚
        missing_penalty = max(0.0, 1.0 - coverage) * 100.0

        # 偏差处罚
        if bias is None:
            bias_penalty = 0.0
        else:
            ratio = min(abs(bias) / thr.bias, 1.0)  # 相对阈值的比例，最多 1
            bias_penalty = ratio * 100.0

        # 异常率处罚
        if anomaly_rate is None:
            anomaly_penalty = 0.0
        else:
            anomaly_penalty = max(0.0, min(anomaly_rate, 1.0)) * 100.0

        # 时延处罚
        if latency_mean is None:
            latency_penalty = 0.0
        else:
            ratio = min(max(latency_mean, 0.0) / thr.latency_seconds, 1.0)
            latency_penalty = ratio * 100.0

        total_penalty = (
            w.coverage * missing_penalty
            + w.bias * bias_penalty
            + w.anomaly * anomaly_penalty
            + w.latency * latency_penalty
        )

        score = 100.0 - total_penalty
        if score < 0.0:
            score = 0.0
        if score > 100.0:
            score = 100.0
        return score

    def _build_summary(
        self,
        field_results: Dict[str, Dict[str, FieldSourceMetrics]],
    ) -> Dict[str, Any]:
        """
        构造整体 summary，用于快速定位问题源/字段。
        """
        records: List[Tuple[str, str, float]] = []  # (field, source, score)
        for field, src_metrics in field_results.items():
            for source_id, m in src_metrics.items():
                records.append((field, source_id, m.score))

        if not records:
            return {}

        # 找出得分最低的若干组合
        records_sorted = sorted(records, key=lambda x: x[2])
        worst_pairs = [
            {
                "field": f,
                "source_id": s,
                "score": sc,
            }
            for (f, s, sc) in records_sorted[:10]
        ]

        # 计算每个 source 的平均得分
        source_scores: Dict[str, List[float]] = {}
        for f, s, sc in records:
            source_scores.setdefault(s, []).append(sc)
        source_avg = [
            {"source_id": s, "avg_score": float(sum(vals) / len(vals))}
            for s, vals in source_scores.items()
            if vals
        ]
        source_avg_sorted = sorted(source_avg, key=lambda x: x["avg_score"])

        return {
            "worst_field_source_pairs": worst_pairs,
            "source_avg_scores": source_avg_sorted,
        }

    # ------------------------------------------------------------------
    # 报表输出
    # ------------------------------------------------------------------

    def _save_report(self, report: Dict[str, Any]) -> None:
        """
        保存 JSON 报告，并打印简要摘要。
        """
        date_str = report.get("date") or dt.date.today().strftime("%Y-%m-%d")
        filename = f"data_quality_{date_str.replace('-', '')}.json"
        path = os.path.join(self.output_dir, filename)

        # 将 numpy 类型转换为原生 Python 类型，避免 json dump 报错
        def _to_native(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_native(v) for v in obj]
            try:
                import numpy as np  # type: ignore

                if isinstance(obj, (np.floating, np.integer)):  # type: ignore[attr-defined]
                    return obj.item()
            except Exception:
                pass
            if isinstance(obj, (float, int, str)) or obj is None:
                return obj
            return obj

        native_report = _to_native(report)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(native_report, f, ensure_ascii=False, indent=2)

        log.info("数据源质量报告已保存：%s", path)

        # 打印 Top 摘要信息
        summary = native_report.get("summary") or {}
        src_avg = summary.get("source_avg_scores") or []
        worst_pairs = summary.get("worst_field_source_pairs") or []

        if src_avg:
            log.info("各源平均得分（前若干）：")
            for item in src_avg[:5]:
                log.info("  source=%s, avg_score=%.2f", item["source_id"], item["avg_score"])

        if worst_pairs:
            log.info("问题最严重的字段-源组合：")
            for item in worst_pairs[:5]:
                log.info(
                    "  field=%s, source=%s, score=%.2f",
                    item["field"],
                    item["source_id"],
                    item["score"],
                )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter Mk4 - 多源字段质量统计与日报输出",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="评估日期，格式 YYYYMMDD（默认=今天）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="每日日标的采样数量（None 或 <=0 则使用全市场）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if args.date:
        try:
            date = dt.datetime.strptime(args.date, "%Y%m%d").date()
        except Exception as e:
            log.error("日期格式错误，应为 YYYYMMDD：%s", e)
            sys.exit(1)
    else:
        # 默认用“今天”的日期；如果你想用“昨天”，可在调用时自行减一天
        date = dt.date.today()

    cfg = get_system_config(refresh=True)
    checker = DataSourceQualityChecker(cfg)

    # 覆盖采样数（如果 cli 提供）
    if args.sample_size is not None:
        checker.sample_size = int(args.sample_size or 0)

    try:
        report = checker.run_for_date(date)
        status = report.get("status", "ok")
        if status != "ok" and status != "empty":
            sys.exit(1)
    except Exception as e:
        log.error("数据源质量评估失败: %s", e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
