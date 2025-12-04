# -*- coding: utf-8 -*-
"""
tools/factor_label_report.py

LightHunter Mk4 - Factor & Label Report
=======================================

用途：
- 对主战因子与标签做 IC / RankIC / 行情分段分析
- 生成 JSON + Markdown 体检报告，服务于科研和因子筛选

依赖：
- pandas
- numpy
- （可选）duckdb：当从 DuckDB 表中直接读取数据时需要

配置入口：
- config/system_config.json 中可选字段：factor_label_report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # 可选 duckdb 支持
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

# 日志工具：优先使用项目内 logging_utils
try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover

    def get_logger(name: str) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
        return logging.getLogger(name)


# 配置中心：优先使用 config_center
try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        cfg_path = os.path.join("config", "system_config.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)


log = get_logger("FactorLabelReport")


def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


class FactorLabelReport:
    """
    对因子-标签组合进行：
    - 日度横截面 IC / RankIC 统计
    - 因子分组收益（Quantile 分桶）
    - 按行情阶段 / regime 分段的表现分析
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        job: Optional[str] = None,
        dataset_path: Optional[str] = None,
        duckdb_table: Optional[str] = None,
        label_cols: Optional[List[str]] = None,
        factor_cols: Optional[List[str]] = None,
        regime_col: Optional[str] = None,
        n_quantiles: Optional[int] = None,
    ) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)
        self.cfg = system_config or {}

        paths_cfg = self.cfg.get("paths", {}) or {}
        self.project_root = paths_cfg.get("project_root", ".") or "."
        try:
            os.chdir(self.project_root)
        except Exception as e:  # pragma: no cover
            log.warning("切换到 project_root=%s 失败：%s", self.project_root, e)

        # 报告目录
        self.reports_dir = paths_cfg.get("reports_dir", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        fl_cfg = self.cfg.get("factor_label_report", {}) or {}

        # job / 数据集配置
        if job is None:
            job = fl_cfg.get("default_job") or (self.cfg.get("alpha") or {}).get(
                "default_job", "ultrashort_main"
            )
        self.job = job

        ds_cfg_all = fl_cfg.get("datasets", {}) or {}
        ds_cfg = ds_cfg_all.get(job, {})

        self.dataset_type: str = ds_cfg.get("type", "parquet")  # parquet / duckdb
        # 数据源路径/表名优先用 CLI 传入，其次用配置
        self.dataset_path = dataset_path or ds_cfg.get("path")
        self.duckdb_table = duckdb_table or ds_cfg.get("table")

        # 基本列名
        self.date_col: str = ds_cfg.get("date_col", "trade_date")
        self.symbol_col: str = ds_cfg.get("symbol_col", "symbol")

        # 标签列
        if label_cols is None:
            cfg_labels = ds_cfg.get("label_cols")
            self.label_cols: List[str] = list(cfg_labels) if cfg_labels else []
        else:
            self.label_cols = list(label_cols)

        # 因子列：分为“显式列表”和“自动匹配前缀”两种
        if factor_cols is None:
            cfg_factors = ds_cfg.get("factor_cols")
            self.factor_cols_cfg: List[str] = list(cfg_factors) if cfg_factors else []
        else:
            self.factor_cols_cfg = list(factor_cols)

        self.factor_prefixes: List[str] = ds_cfg.get("factor_prefixes", ["f_"])

        # 行情分段列（如 y_market_sentiment_phase 或 f_mkt_index_ret_bucket）
        if regime_col is None:
            regime_col = ds_cfg.get("regime_col")
        self.regime_col: Optional[str] = regime_col

        # IC 配置
        ic_cfg = fl_cfg.get("ic", {}) or {}
        self.min_sample_per_day: int = int(ic_cfg.get("min_sample_per_day", 30))

        # 分桶配置
        q_cfg = fl_cfg.get("quantile", {}) or {}
        self.n_quantiles: int = int(n_quantiles or q_cfg.get("num_quantiles", 5))
        self.min_sample_per_bucket: int = int(q_cfg.get("min_sample_per_bucket", 5))

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _resolve_duckdb_path(self) -> str:
        duck_cfg = self.cfg.get("duckdb", {}) or {}
        db_path = duck_cfg.get("db_path")
        if not db_path:
            storage_ts = (self.cfg.get("storage") or {}).get("ts") or {}
            db_path = storage_ts.get("duckdb_path", "data/lighthunter.duckdb")
        return db_path

    def _load_from_parquet(self) -> pd.DataFrame:
        if not self.dataset_path:
            raise ValueError("未配置 dataset_path（Parquet 文件路径）。")
        path = self.dataset_path
        if not os.path.isabs(path):
            path = os.path.join(self.project_root, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件：{path}")
        log.info("从 Parquet 载入因子/标签数据：%s", path)
        df = pd.read_parquet(path)
        return df

    def _load_from_duckdb(
        self, start_date: Optional[dt.date], end_date: Optional[dt.date]
    ) -> pd.DataFrame:
        if duckdb is None:  # pragma: no cover
            raise RuntimeError("未安装 duckdb，无法从 DuckDB 表读取数据。")

        if not self.duckdb_table:
            raise ValueError("未配置 duckdb_table 名称。")

        db_path = self._resolve_duckdb_path()
        log.info("从 DuckDB(%s) 表 %s 载入数据", db_path, self.duckdb_table)
        con = duckdb.connect(db_path, read_only=False)
        try:
            sql = f"SELECT * FROM {self.duckdb_table}"
            params: List[Any] = []
            conds: List[str] = []
            if start_date is not None:
                conds.append(f"{self.date_col} >= ?")
                params.append(start_date.strftime("%Y-%m-%d"))
            if end_date is not None:
                conds.append(f"{self.date_col} <= ?")
                params.append(end_date.strftime("%Y-%m-%d"))
            if conds:
                sql += " WHERE " + " AND ".join(conds)
            df = con.execute(sql, params).df()
        finally:
            con.close()
        return df

    def load_dataset(
        self,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if self.dataset_type == "duckdb":
            df = self._load_from_duckdb(start_date, end_date)
        else:
            # parquet / default
            df = self._load_from_parquet()

        if self.date_col in df.columns:
            # 统一为 date 类型
            df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.date

        if start_date is not None:
            df = df[df[self.date_col] >= start_date]
        if end_date is not None:
            df = df[df[self.date_col] <= end_date]

        # 丢弃标签全为空的行
        if self.label_cols:
            df = df.dropna(subset=self.label_cols, how="any")

        return df

    # ------------------------------------------------------------------
    # 列名推断
    # ------------------------------------------------------------------

    def _infer_label_cols(self, df: pd.DataFrame) -> List[str]:
        if self.label_cols:
            return self.label_cols
        # 自动：选择前缀 y_ 的数值字段
        cand: List[str] = []
        for col, dtype in df.dtypes.items():
            if not str(col).startswith("y_"):
                continue
            if not np.issubdtype(dtype, np.number):
                continue
            cand.append(str(col))
        if not cand:
            raise ValueError("无法自动推断标签列（未找到 y_* 数值列），请在配置或命令行中显式指定 label_cols。")
        log.info("自动识别到标签列：%s", ", ".join(cand))
        self.label_cols = cand
        return cand

    def _infer_factor_cols(self, df: pd.DataFrame) -> List[str]:
        if self.factor_cols_cfg:
            # 只保留在 df 中真实存在的列
            cols = [c for c in self.factor_cols_cfg if c in df.columns]
            if not cols:
                raise ValueError("配置中的 factor_cols 在数据集中均不存在，请检查。")
            return cols

        exclude = set([self.date_col, self.symbol_col] + self.label_cols)
        if self.regime_col:
            exclude.add(self.regime_col)

        cand: List[str] = []
        for col, dtype in df.dtypes.items():
            c = str(col)
            if c in exclude:
                continue
            if not np.issubdtype(dtype, np.number):
                continue
            if any(c.startswith(p) for p in self.factor_prefixes):
                cand.append(c)

        if not cand:
            raise ValueError("无法自动推断因子列（未找到匹配前缀的数值列），请在配置或命令行中显式指定 factor_cols。")

        log.info("自动识别到因子列（前缀 %s）：%s", ",".join(self.factor_prefixes), ", ".join(cand))
        return cand

    # ------------------------------------------------------------------
    # 统计计算：IC / RankIC / 分桶
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_corr(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {
                "mean": None,
                "std": None,
                "t": None,
                "n": 0,
                "pos_ratio": None,
            }
        arr = np.asarray(values, dtype=float)
        n = arr.size
        mean = float(arr.mean())
        if n > 1:
            std = float(arr.std(ddof=1))
        else:
            std = 0.0
        if n > 1 and std > 1e-12:
            t_val = float(mean / (std / np.sqrt(n)))
        else:
            t_val = 0.0
        pos_ratio = float((arr > 0).sum() / n)
        return {
            "mean": mean,
            "std": std,
            "t": t_val,
            "n": int(n),
            "pos_ratio": pos_ratio,
        }

    def _compute_ic_rankic(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_cols: List[str],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        返回结构：
            { (factor, label): { 'ic': {...}, 'rank_ic': {...} } }
        """
        if self.date_col not in df.columns:
            raise ValueError(f"数据中不存在日期列 {self.date_col}")

        grouped = df.groupby(self.date_col)
        results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for f in factor_cols:
            for y in label_cols:
                ic_vals: List[float] = []
                ric_vals: List[float] = []
                for trade_date, g in grouped:
                    sub = g[[f, y]].dropna()
                    n = len(sub)
                    if n < self.min_sample_per_day:
                        continue
                    x = pd.to_numeric(sub[f], errors="coerce")
                    t = pd.to_numeric(sub[y], errors="coerce")
                    sub2 = pd.concat([x, t], axis=1).dropna()
                    if sub2.shape[0] < self.min_sample_per_day:
                        continue
                    x = sub2.iloc[:, 0]
                    t = sub2.iloc[:, 1]
                    if x.nunique() < 2 or t.nunique() < 2:
                        continue
                    ic = float(x.corr(t, method="pearson"))
                    if np.isfinite(ic):
                        ic_vals.append(ic)
                    # RankIC：先做秩，再做 Pearson
                    xr = x.rank(method="average")
                    tr = t.rank(method="average")
                    ric = float(xr.corr(tr, method="pearson"))
                    if np.isfinite(ric):
                        ric_vals.append(ric)

                ic_summary = self._summary_corr(ic_vals)
                ric_summary = self._summary_corr(ric_vals)
                results[(f, y)] = {"ic": ic_summary, "rank_ic": ric_summary}

        return results

    def _compute_quantile_analysis(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_cols: List[str],
        n_quantiles: int,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        按因子做 Quantile 分组（截面上分桶），统计各桶的平均标签收益等。
        结果结构：
            { (factor, label): {
                "avg_label_per_quantile": [q1, q2, ..., qN],
                "long_short": qN - q1,
                "bucket_counts": [n1, ..., nN]
            }}
        """
        grouped = df.groupby(self.date_col)
        results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for f in factor_cols:
            for y in label_cols:
                bucket_sum = np.zeros(n_quantiles, dtype=float)
                bucket_count = np.zeros(n_quantiles, dtype=float)

                for trade_date, g in grouped:
                    sub = g[[f, y]].dropna()
                    n = len(sub)
                    if n < n_quantiles * self.min_sample_per_bucket:
                        continue

                    x = pd.to_numeric(sub[f], errors="coerce")
                    t = pd.to_numeric(sub[y], errors="coerce")
                    sub2 = pd.concat([x, t], axis=1).dropna()
                    if sub2.shape[0] < n_quantiles * self.min_sample_per_bucket:
                        continue

                    x = sub2.iloc[:, 0]
                    t = sub2.iloc[:, 1]

                    # 先按秩再切分，避免极端值干扰
                    try:
                        ranks = x.rank(method="first")
                        q = pd.qcut(
                            ranks,
                            q=n_quantiles,
                            labels=False,
                            duplicates="drop",
                        )
                    except ValueError:
                        # 数据过少或分位不齐，跳过该日
                        continue

                    # 如果实际桶数少于期望值，则跳过（保证不同日的桶定义一致）
                    if q.nunique() < n_quantiles:
                        continue

                    for b in range(n_quantiles):
                        mask = q == b
                        if not mask.any():
                            continue
                        bucket_sum[b] += float(t[mask].mean())
                        bucket_count[b] += 1.0

                avg_per_q: List[Optional[float]] = []
                for b in range(n_quantiles):
                    if bucket_count[b] > 0:
                        avg_per_q.append(float(bucket_sum[b] / bucket_count[b]))
                    else:
                        avg_per_q.append(None)

                long_short: Optional[float]
                if avg_per_q[0] is not None and avg_per_q[-1] is not None:
                    long_short = float(avg_per_q[-1] - avg_per_q[0])
                else:
                    long_short = None

                results[(f, y)] = {
                    "avg_label_per_quantile": avg_per_q,
                    "long_short": long_short,
                    "bucket_counts": [int(x) for x in bucket_count],
                }

        return results

    def _compute_regime_analysis(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_cols: List[str],
    ) -> Dict[str, Any]:
        """
        按行情分段列（regime_col）进行子样本分析：
        - 每个 regime 内单独算 IC / RankIC / 分桶
        """
        if not self.regime_col or self.regime_col not in df.columns:
            return {}

        regimes = sorted(df[self.regime_col].dropna().unique().tolist())
        regime_result: Dict[str, Any] = {}

        for reg in regimes:
            sub = df[df[self.regime_col] == reg]
            if sub.empty:
                continue
            label = str(reg)
            log.info(
                "按行情分段分析：regime=%s, 样本行数=%d", label, len(sub)
            )
            ic_stats = self._compute_ic_rankic(sub, factor_cols, label_cols)
            q_stats = self._compute_quantile_analysis(
                sub, factor_cols, label_cols, self.n_quantiles
            )
            regime_result[label] = {
                "sample_size": int(len(sub)),
                "ic": ic_stats,
                "quantiles": q_stats,
            }

        return regime_result

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def _build_summary_dict(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        label_cols: List[str],
        ic_stats: Dict[Tuple[str, str], Dict[str, Any]],
        q_stats: Dict[Tuple[str, str], Dict[str, Any]],
        regime_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        dates = df[self.date_col].dropna()
        date_min = dates.min()
        date_max = dates.max()

        meta = {
            "job": self.job,
            "dataset_type": self.dataset_type,
            "dataset_path": self.dataset_path,
            "duckdb_table": self.duckdb_table,
            "date_col": self.date_col,
            "symbol_col": self.symbol_col,
            "date_min": str(date_min) if pd.notna(date_min) else None,
            "date_max": str(date_max) if pd.notna(date_max) else None,
            "num_rows": int(len(df)),
            "num_symbols": int(df[self.symbol_col].nunique())
            if self.symbol_col in df.columns
            else None,
            "label_cols": label_cols,
            "factor_cols": factor_cols,
            "regime_col": self.regime_col,
            "n_quantiles": self.n_quantiles,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }

        factors_block: Dict[str, Any] = {}
        for f in factor_cols:
            label_block: Dict[str, Any] = {}
            for y in label_cols:
                key = (f, y)
                stats = ic_stats.get(key, {})
                label_block[y] = {
                    "ic": stats.get("ic"),
                    "rank_ic": stats.get("rank_ic"),
                    "quantiles": q_stats.get(key, {}),
                }
            factors_block[f] = {"labels": label_block}

        summary = {
            "meta": meta,
            "factors": factors_block,
            "regimes": regime_stats,
        }
        return summary

    def _write_reports(
        self,
        summary: Dict[str, Any],
        primary_label: str,
        ic_stats: Dict[Tuple[str, str], Dict[str, Any]],
        q_stats: Dict[Tuple[str, str], Dict[str, Any]],
        out_name: Optional[str] = None,
    ) -> None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_name:
            base_name = out_name
        else:
            base_name = f"factor_label_report_{self.job}_{primary_label}_{timestamp}"

        json_path = os.path.join(self.reports_dir, base_name + ".json")
        md_path = os.path.join(self.reports_dir, base_name + ".md")

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info("因子-标签 JSON 报告已写入：%s", json_path)

        # Markdown 摘要：主要看 primary_label 的结果
        meta = summary["meta"]
        factor_cols: List[str] = meta["factor_cols"]
        label_cols: List[str] = meta["label_cols"]

        # 选取主标签下 IC 绝对值排名前若干的因子
        rows: List[Tuple[str, float, float, int]] = []
        for f in factor_cols:
            key = (f, primary_label)
            stats = ic_stats.get(key, {})
            ic_info = stats.get("ic")
            if ic_info is None:
                ic_mean = 0.0
                t_val = 0.0
                n_days = 0
            else:
                ic_mean = ic_info.get("mean") or 0.0
                t_val = ic_info.get("t") or 0.0
                n_days = ic_info.get("n") or 0
            rows.append((f, float(ic_mean), float(t_val), int(n_days)))

        rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
        top_n = rows_sorted[:30]

        lines: List[str] = []
        lines.append(f"# Factor-Label 报告 —— job: `{self.job}`, label: `{primary_label}`")
        lines.append("")
        lines.append("## 1. 数据集概览")
        lines.append("")
        lines.append(f"- 数据集类型：`{meta['dataset_type']}`")
        lines.append(f"- 数据路径/表：`{meta['dataset_path'] or meta['duckdb_table']}`")
        lines.append(f"- 日期列：`{meta['date_col']}`，范围：{meta['date_min']} ~ {meta['date_max']}")
        lines.append(f"- 样本行数：{meta['num_rows']}")
        lines.append(f"- 标的数量：{meta['num_symbols']}")
        lines.append(f"- 标签列集合：{', '.join(label_cols)}")
        lines.append(f"- 因子列数量：{len(factor_cols)}")
        lines.append("")

        lines.append("## 2. 主标签下的因子 IC 排行（按 |IC| 排序，Top 30）")
        lines.append("")
        lines.append("| 因子 | 平均 IC | t 值 | 有效交易日数 |")
        lines.append("|------|---------|------|--------------|")
        for f, mean_ic, t_val, n_days in top_n:
            lines.append(f"| `{f}` | {mean_ic:.4f} | {t_val:.2f} | {n_days} |")
        lines.append("")

        # 展示前 3 个因子的分桶收益
        lines.append("## 3. 因子分桶收益（示例：前 3 个因子）")
        lines.append("")
        qn = summary["meta"]["n_quantiles"]
        for f, _, _, _ in top_n[:3]:
            key = (f, primary_label)
            qinfo = q_stats.get(key, {})
            avg_vals: List[Optional[float]] = qinfo.get("avg_label_per_quantile") or []
            long_short = qinfo.get("long_short")
            lines.append(f"### 因子 `{f}` 分桶表现（标签：`{primary_label}`）")
            lines.append("")
            lines.append("| 分桶 | 平均标签值 |")
            lines.append("|------|------------|")
            if avg_vals and len(avg_vals) == qn:
                for i, v in enumerate(avg_vals):
                    if v is None:
                        lines.append(f"| Q{i+1} | - |")
                    else:
                        lines.append(f"| Q{i+1} | {v:.4f} |")
            else:
                lines.append("| (无数据) | |")
            lines.append("")
            lines.append(
                f"> Long-Short（Q{qn} - Q1）≈ {long_short:.4f}"
                if isinstance(long_short, (int, float))
                else "> Long-Short（Qn - Q1）数据不足"
            )
            lines.append("")

        # 行情分段简单概览
        if summary.get("regimes"):
            lines.append("## 4. 行情分段表现概览")
            lines.append("")
            lines.append(
                "下表给出不同行情 regime 下，主标签与部分因子的平均 IC（仅示例前 5 个因子）："
            )
            lines.append("")
            regimes = sorted(summary["regimes"].keys())
            header = "| 因子 | " + " | ".join([f"regime {r} IC" for r in regimes]) + " |"
            sep = "|------|" + "|".join(["-----------" for _ in regimes]) + "|"
            lines.append(header)
            lines.append(sep)
            for f, _, _, _ in top_n[:5]:
                row = [f"`{f}`"]
                for r in regimes:
                    reg_ic_stats = summary["regimes"][r]["ic"]
                    stats = reg_ic_stats.get((f, primary_label))
                    if stats and stats.get("ic"):
                        mean_ic = stats["ic"].get("mean")
                        if mean_ic is None:
                            row.append("-")
                        else:
                            row.append(f"{mean_ic:.4f}")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("因子-标签 Markdown 报告已写入：%s", md_path)

    # ------------------------------------------------------------------
    # 外部调用入口
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        out_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.load_dataset(start_date, end_date)
        if df.empty:
            raise RuntimeError("数据集为空，无法进行因子-标签分析。")

        # 推断标签 & 因子列
        label_cols = self._infer_label_cols(df)
        factor_cols = self._infer_factor_cols(df)

        log.info(
            "开始因子-标签分析：labels=%s, factors=%d 行数=%d",
            ",".join(label_cols),
            len(factor_cols),
            len(df),
        )

        ic_stats = self._compute_ic_rankic(df, factor_cols, label_cols)
        q_stats = self._compute_quantile_analysis(
            df, factor_cols, label_cols, self.n_quantiles
        )
        regime_stats = self._compute_regime_analysis(df, factor_cols, label_cols)

        summary = self._build_summary_dict(df, factor_cols, label_cols, ic_stats, q_stats, regime_stats)

        # primary_label：默认第一个标签
        primary_label = label_cols[0]
        self._write_reports(summary, primary_label, ic_stats, q_stats, out_name)

        return summary


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter 因子-标签体检工具（IC/RankIC/行情分段分析）"
    )
    parser.add_argument(
        "--job",
        type=str,
        default=None,
        help="数据集 job 名称（默认使用 factor_label_report.default_job 或 alpha.default_job）",
    )
    parser.add_argument(
        "--dataset-parquet",
        type=str,
        default=None,
        help="数据集 Parquet 路径（如不提供则使用配置中的 datasets[job].path）",
    )
    parser.add_argument(
        "--duckdb-table",
        type=str,
        default=None,
        help="DuckDB 表名（如不提供则使用配置中的 datasets[job].table）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期（YYYY-MM-DD，可选）",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期（YYYY-MM-DD，可选）",
    )
    parser.add_argument(
        "--label-cols",
        type=str,
        default=None,
        help="逗号分隔的标签列列表（如不提供则自动扫描 y_* 列）",
    )
    parser.add_argument(
        "--factor-cols",
        type=str,
        default=None,
        help="逗号分隔的因子列列表（如不提供则按前缀 f_* 自动识别）",
    )
    parser.add_argument(
        "--regime-col",
        type=str,
        default=None,
        help="行情分段列名（如 y_market_sentiment_phase），不提供则不做分段分析",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=None,
        help="因子分桶数量（默认来自配置，通常为 5 或 10）",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="输出报告文件名基础（不含扩展名），不提供则自动生成",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    cfg = get_system_config(refresh=False)

    label_cols = (
        [s.strip() for s in args.label_cols.split(",") if s.strip()]
        if args.label_cols
        else None
    )
    factor_cols = (
        [s.strip() for s in args.factor_cols.split(",") if s.strip()]
        if args.factor_cols
        else None
    )

    reporter = FactorLabelReport(
        system_config=cfg,
        job=args.job,
        dataset_path=args.dataset_parquet,
        duckdb_table=args.duckdb_table,
        label_cols=label_cols,
        factor_cols=factor_cols,
        regime_col=args.regime_col,
        n_quantiles=args.n_quantiles,
    )

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    reporter.run(start_date=start_date, end_date=end_date, out_name=args.out_name)


if __name__ == "__main__":  # pragma: no cover
    main()
