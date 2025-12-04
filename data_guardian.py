# -*- coding: utf-8 -*-
"""
data_guardian.py

LightHunter Mk4 - DataGuardian Ω
================================

目标：
- 把“数据体检”升级成接近实战需求的「数据健康评分系统」。
- 综合使用：
    1）多源日线表的字段质量统计（缺失/非法值/源间偏差）
    2）Mk4-Step-07 的历史回放/采集完整性检查结果
- 对每只标的 + 全局给出 0~100 的数据健康评分，并输出报告文件。

默认从 system_config.json 中读取：
- duckdb.db_path 或 storage.ts.duckdb_path
- data_quality.daily_table / fields / sources / reference_source / invalid_rules / diff_threshold
- paths.reports_dir

可选配置（如果你想调权重，可以在 system_config.json 里增加）：
    "data_guardian": {
      "weight_coverage": 0.35,
      "weight_source": 0.25,
      "weight_value": 0.20,
      "weight_consistency": 0.20
    }

命令行用法：
    # 默认：检查“昨天”的数据
    python -m data_guardian

    # 指定日期区间 + 标的列表
    python -m data_guardian --start-date 2024-01-01 --end-date 2024-01-10 \
        --symbols 000001.SZ,000002.SZ

    # 同时打印简单“历史回放视图”
    python -m data_guardian --date 2024-01-10 --replay
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

try:
    import duckdb  # type: ignore
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("DataGuardian 需要 duckdb + pandas 支持，请先安装。") from e

# 日志适配：优先核心 logging_utils
try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    def get_logger(name: str) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
        return logging.getLogger(name)


# 配置中心适配
try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover
    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        # 简易兜底：直接读 config/system_config.json
        cfg_path = os.path.join("config", "system_config.json")
        if not os.path.exists(cfg_path):
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)


# Replay & HealthCheck 工具（Mk4-Step-07）
try:
    from tools import replay_and_healthcheck as rh  # type: ignore
except Exception:  # pragma: no cover
    rh = None  # 没有该模块时，只做本地多源统计


log = get_logger("DataGuardianOmega")


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _date_to_str(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


class DataGuardianOmega:
    """
    Ω 级 DataGuardian：
    - 从 DuckDB 读取 multi_source_daily（或自定义表）
    - 计算字段合法性 + 多源偏差
    - 调用 replay_and_healthcheck 得到采集完整性和多源覆盖情况
    - 输出 0~100 评分 + 报告
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)
        self.cfg = system_config or {}

        paths_cfg = self.cfg.get("paths", {}) or {}
        self.project_root = paths_cfg.get("project_root", ".") or "."

        # 尽量切目录到 project_root，方便相对路径
        try:
            os.chdir(self.project_root)
        except Exception as e:  # pragma: no cover
            log.warning("切换到 project_root=%s 失败：%s", self.project_root, e)

        dq_cfg = self.cfg.get("data_quality", {}) or {}

        # DuckDB 路径：优先 data_quality.duckdb_path，其次 duckdb.db_path，再次 storage.ts.duckdb_path
        duckdb_path = dq_cfg.get("duckdb_path")
        if not duckdb_path:
            duck_cfg = self.cfg.get("duckdb", {}) or {}
            duckdb_path = duck_cfg.get("db_path")
        if not duckdb_path:
            storage_ts = (self.cfg.get("storage") or {}).get("ts") or {}
            duckdb_path = storage_ts.get("duckdb_path", "data/lighthunter.duckdb")

        self.duckdb_path = duckdb_path
        self.daily_table = dq_cfg.get("daily_table", "multi_source_daily")
        self.fields: List[str] = dq_cfg.get(
            "fields", ["close_price", "chg_pct", "volume", "turnover"]
        )
        self.sources: List[str] = dq_cfg.get("sources", []) or []
        self.ref_source: Optional[str] = dq_cfg.get("reference_source") or None
        self.invalid_rules: Dict[str, Dict[str, float]] = dq_cfg.get("invalid_rules", {}) or {}
        self.diff_threshold: float = float(dq_cfg.get("diff_threshold", 0.001))

        # 报告目录
        reports_dir = dq_cfg.get("report_dir") or paths_cfg.get("reports_dir") or "reports"
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        # 可选：自定义权重
        dg_cfg = self.cfg.get("data_guardian", {}) or {}
        self.weights = {
            "coverage": float(dg_cfg.get("weight_coverage", 0.35)),
            "source": float(dg_cfg.get("weight_source", 0.25)),
            "value": float(dg_cfg.get("weight_value", 0.20)),
            "consistency": float(dg_cfg.get("weight_consistency", 0.20)),
        }
        total_w = sum(self.weights.values())
        if total_w <= 0:
            self.weights = {
                "coverage": 0.35,
                "source": 0.25,
                "value": 0.20,
                "consistency": 0.20,
            }
        else:
            # 归一化
            for k in self.weights:
                self.weights[k] = self.weights[k] / total_w

        log.info(
            "DataGuardian Ω 初始化完成：duckdb=%s, table=%s, fields=%s, sources=%s",
            self.duckdb_path,
            self.daily_table,
            ",".join(self.fields),
            ",".join(self.sources),
        )

    # ------------------------------------------------------------------
    # DuckDB 查询 & 多源质量统计
    # ------------------------------------------------------------------

    def _connect_duckdb(self) -> "duckdb.DuckDBPyConnection":
        log.info("连接 DuckDB：%s", self.duckdb_path)
        return duckdb.connect(self.duckdb_path, read_only=False)

    def _load_multi_source_df(
        self,
        start_date: dt.date,
        end_date: dt.date,
        symbols: Optional[Sequence[str]],
    ) -> "pd.DataFrame":
        start_str = _date_to_str(start_date)
        end_str = _date_to_str(end_date)

        cols = ["trade_date", "symbol", "source"] + self.fields
        select_cols = ", ".join(cols)
        sql = f"""
            SELECT {select_cols}
            FROM {self.daily_table}
            WHERE trade_date BETWEEN ? AND ?
        """
        params: List[Any] = [start_str, end_str]
        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            sql += f" AND symbol IN ({placeholders})"
            params.extend(list(symbols))

        con = self._connect_duckdb()
        try:
            log.info("从 %s 载入多源日线数据，区间 [%s, %s] ...", self.daily_table, start_str, end_str)
            df = con.execute(sql, params).df()
        finally:
            con.close()

        if df.empty:
            log.warning("在表 %s 中，区间 [%s, %s] 无数据。", self.daily_table, start_str, end_str)
            return df

        # trade_date -> date 类型
        if df["trade_date"].dtype == object:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        return df

    def _is_invalid_row(self, row: "pd.Series") -> bool:
        """
        检查单行是否存在非法字段（根据 data_quality.invalid_rules）。
        """
        for field, cond in self.invalid_rules.items():
            if field not in row:
                continue
            val = row[field]
            if val is None:
                return True
            try:
                v = float(val)
            except Exception:
                return True
            ge = cond.get("ge")
            gt = cond.get("gt")
            le = cond.get("le")
            lt = cond.get("lt")
            if ge is not None and v < ge:
                return True
            if gt is not None and v <= gt:
                return True
            if le is not None and v > le:
                return True
            if lt is not None and v >= lt:
                return True
        return False

    def _compute_quality_stats(
        self,
        df: "pd.DataFrame",
    ) -> Dict[str, Dict[str, int]]:
        """
        对 multi_source_daily 做逐 symbol / trade_date 质量统计：
        - total_days        : 存在记录的交易日数
        - invalid_days      : 存在非法字段的交易日数
        - mismatch_days     : 多源字段偏差超过 diff_threshold 的交易日数
        """
        stats: Dict[str, Dict[str, int]] = {}

        if df.empty:
            return stats

        grouped = df.groupby(["symbol", "trade_date"])

        for (sym, trade_date), g in grouped:
            sym = str(sym)
            invalid_day = False
            mismatch_day = False

            # 1) 非法值检查（行级）
            for _, row in g.iterrows():
                if self._is_invalid_row(row):
                    invalid_day = True
                    break

            # 2) 多源偏差检查（列级）
            unique_sources = g["source"].dropna().unique().tolist()
            if len(unique_sources) > 1:
                for field in self.fields:
                    if field not in g.columns:
                        continue
                    s = pd.to_numeric(g[field], errors="coerce").dropna()
                    if s.empty:
                        continue
                    max_v = s.max()
                    min_v = s.min()
                    diff = max_v - min_v
                    base = max(abs(max_v), abs(min_v), 1e-6)
                    rel_diff = abs(diff) / base
                    if rel_diff > self.diff_threshold:
                        mismatch_day = True
                        break

            st = stats.setdefault(sym, {"total_days": 0, "invalid_days": 0, "mismatch_days": 0})
            st["total_days"] += 1
            if invalid_day:
                st["invalid_days"] += 1
            if mismatch_day:
                st["mismatch_days"] += 1

        return stats

    # ------------------------------------------------------------------
    # 采集完整性 / 回放检查结果（来自 tools.replay_and_healthcheck）
    # ------------------------------------------------------------------

    def _get_coverage_from_replay(
        self,
        start_date: dt.date,
        end_date: dt.date,
        symbols: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        """
        通过 tools.replay_and_healthcheck 获得采集完整性信息：
        - total_trading_days
        - available_days
        - missing_days
        - partial_source_days
        """
        if rh is None:
            log.warning("未找到 tools.replay_and_healthcheck 模块，只用本地多源统计。")
            return {}

        try:
            coverage = rh.run_replay_and_healthcheck(
                cfg=self.cfg,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                replay=False,
                write_report=False,
                table=self.daily_table,
            )
        except Exception as e:  # pragma: no cover
            log.warning("调用 replay_and_healthcheck 失败，将仅使用本地多源统计：%s", e)
            return {}

        return coverage or {}

    # ------------------------------------------------------------------
    # 评分计算逻辑
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        coverage: Dict[str, Any],
        quality_stats: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """
        综合 coverage + quality_stats 计算每只标的的 0~100 分数。
        返回：
            {
              "meta": {...},
              "global": {...},
              "symbols": {
                "000001.SZ": {...},
                ...
              },
              "top_worst": {
                "by_score": [ {...}, ... ]
              }
            }
        """
        all_symbols = sorted(set(coverage.keys()) | set(quality_stats.keys()))
        per_symbol: Dict[str, Any] = {}

        total_days_sum = 0.0
        agg_cov = agg_src = agg_val = agg_cons = 0.0

        for sym in all_symbols:
            cov_info = coverage.get(sym)
            q_info = quality_stats.get(sym, {"total_days": 0, "invalid_days": 0, "mismatch_days": 0})

            # coverage 侧
            if cov_info is not None:
                # dataclass CoverageInfo: 属性名 total_trading_days / available_days / missing_days / partial_source_days
                try:
                    total_days = int(cov_info.total_trading_days)
                    available_days = int(cov_info.available_days)
                    missing_days = int(cov_info.missing_days)
                    partial_source_days = int(cov_info.partial_source_days)
                except AttributeError:
                    # 防御性处理：如果结构不是 dataclass，就按 dict 读
                    total_days = int(cov_info.get("total_trading_days", 0))
                    available_days = int(cov_info.get("available_days", 0))
                    missing_days = int(cov_info.get("missing_days", 0))
                    partial_source_days = int(cov_info.get("partial_source_days", 0))
            else:
                total_days = int(q_info.get("total_days", 0))
                available_days = total_days
                missing_days = 0
                partial_source_days = 0

            if total_days <= 0:
                # 完全没有数据，不参与评分
                continue

            invalid_days = int(q_info.get("invalid_days", 0))
            mismatch_days = int(q_info.get("mismatch_days", 0))

            # 四个子评分（0~1）
            coverage_score = max(0.0, min(1.0, available_days / total_days))
            source_score = max(
                0.0,
                min(1.0, (available_days - partial_source_days) / max(available_days, 1)),
            )
            value_score = max(
                0.0,
                min(1.0, (available_days - invalid_days) / max(available_days, 1)),
            )
            consistency_score = max(
                0.0,
                min(1.0, (available_days - mismatch_days) / max(available_days, 1)),
            )

            total_score = (
                coverage_score * self.weights["coverage"]
                + source_score * self.weights["source"]
                + value_score * self.weights["value"]
                + consistency_score * self.weights["consistency"]
            ) * 100.0

            per_symbol[sym] = {
                "symbol": sym,
                "total_trading_days": total_days,
                "available_days": available_days,
                "missing_days": missing_days,
                "partial_source_days": partial_source_days,
                "invalid_days": invalid_days,
                "mismatch_days": mismatch_days,
                "coverage_score": round(coverage_score * 100.0, 2),
                "source_score": round(source_score * 100.0, 2),
                "value_score": round(value_score * 100.0, 2),
                "consistency_score": round(consistency_score * 100.0, 2),
                "total_score": round(total_score, 2),
            }

            # 全局加权统计（按 total_days 加权）
            total_days_sum += float(total_days)
            agg_cov += coverage_score * total_days
            agg_src += source_score * total_days
            agg_val += value_score * total_days
            agg_cons += consistency_score * total_days

        if total_days_sum > 0:
            g_cov = agg_cov / total_days_sum
            g_src = agg_src / total_days_sum
            g_val = agg_val / total_days_sum
            g_cons = agg_cons / total_days_sum
            g_score = (
                g_cov * self.weights["coverage"]
                + g_src * self.weights["source"]
                + g_val * self.weights["value"]
                + g_cons * self.weights["consistency"]
            ) * 100.0
        else:
            g_cov = g_src = g_val = g_cons = g_score = 0.0

        # 找出若干“最不健康”的标的（最低分前 30）
        worst = sorted(per_symbol.values(), key=lambda x: x["total_score"])[:30]

        report = {
            "global": {
                "coverage_score": round(g_cov * 100.0, 2),
                "source_score": round(g_src * 100.0, 2),
                "value_score": round(g_val * 100.0, 2),
                "consistency_score": round(g_cons * 100.0, 2),
                "total_score": round(g_score, 2),
                "symbols_count": len(per_symbol),
            },
            "symbols": per_symbol,
            "top_worst": {
                "by_score": worst,
            },
        }
        return report

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def _write_reports(
        self,
        start_date: dt.date,
        end_date: dt.date,
        report: Dict[str, Any],
    ) -> None:
        base_name = f"data_guardian_omega_{_date_to_str(start_date)}_{_date_to_str(end_date)}"
        json_path = os.path.join(self.reports_dir, base_name + ".json")
        md_path = os.path.join(self.reports_dir, base_name + ".md")

        meta = {
            "start_date": _date_to_str(start_date),
            "end_date": _date_to_str(end_date),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "daily_table": self.daily_table,
            "sources": self.sources,
            "weights": self.weights,
        }
        full = {"meta": meta}
        full.update(report)

        # JSON 报告
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        log.info("DataGuardian Ω JSON 报告已写入：%s", json_path)

        # 简易 Markdown 摘要
        lines: List[str] = []
        g = report["global"]
        lines.append(f"# DataGuardian Ω 数据健康报告")
        lines.append("")
        lines.append(f"- 日期区间：{meta['start_date']} ~ {meta['end_date']}")
        lines.append(f"- 多源表：`{self.daily_table}`")
        lines.append(f"- 数据源：{', '.join(self.sources) if self.sources else '(未配置)'}")
        lines.append("")
        lines.append("## 1. 全局评分")
        lines.append("")
        lines.append(f"- 覆盖度评分：{g['coverage_score']:.2f}")
        lines.append(f"- 源完整度评分：{g['source_score']:.2f}")
        lines.append(f"- 字段合法性评分：{g['value_score']:.2f}")
        lines.append(f"- 多源一致性评分：{g['consistency_score']:.2f}")
        lines.append(f"- **综合数据健康评分：{g['total_score']:.2f} / 100**")
        lines.append("")
        lines.append("## 2. 最高风险标的 Top 30（按总分从低到高）")
        lines.append("")
        lines.append("| symbol | total_score | coverage | source | value | consistency | missing_days | partial_source_days | invalid_days | mismatch_days |")
        lines.append("|--------|-------------|----------|--------|-------|-------------|--------------|---------------------|--------------|---------------|")
        for item in report["top_worst"]["by_score"]:
            lines.append(
                f"| {item['symbol']} | {item['total_score']:.2f} | "
                f"{item['coverage_score']:.2f} | {item['source_score']:.2f} | "
                f"{item['value_score']:.2f} | {item['consistency_score']:.2f} | "
                f"{item['missing_days']} | {item['partial_source_days']} | "
                f"{item['invalid_days']} | {item['mismatch_days']} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("DataGuardian Ω Markdown 摘要已写入：%s", md_path)

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: dt.date,
        end_date: dt.date,
        symbols: Optional[Sequence[str]] = None,
        replay: bool = False,
    ) -> Dict[str, Any]:
        """
        对指定日期区间 / 标的集合执行 Ω 级数据体检：
        - 从 multi_source_daily 读取字段数据
        - 调用 replay_and_healthcheck 计算采集完整性
        - 综合打分 & 输出报告
        """
        log.info(
            "DataGuardian Ω 开始运行：[%s ~ %s], symbols=%s",
            _date_to_str(start_date),
            _date_to_str(end_date),
            ",".join(symbols) if symbols else "ALL",
        )

        df = self._load_multi_source_df(start_date, end_date, symbols)
        quality_stats = self._compute_quality_stats(df)

        # 采集完整性 / 多源覆盖情况
        coverage = self._get_coverage_from_replay(start_date, end_date, symbols)

        # 评分
        report = self._compute_scores(coverage, quality_stats)

        # 输出报告文件
        self._write_reports(start_date, end_date, report)

        # 如需要，可以顺带让 replay 打印回放视图（主要给人肉盯数据用）
        if replay and rh is not None:
            try:
                rh.run_replay_and_healthcheck(
                    cfg=self.cfg,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    replay=True,
                    write_report=False,
                    table=self.daily_table,
                )
            except Exception as e:  # pragma: no cover
                log.warning("调用 replay_and_healthcheck 打印回放视图失败：%s", e)

        log.info(
            "DataGuardian Ω 完成：综合健康评分=%.2f（全局），symbol 数=%d",
            report["global"]["total_score"],
            report["global"]["symbols_count"],
        )
        return report


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightHunter DataGuardian Ω")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="system_config.json 路径（默认使用 config_center）",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="单个日期 YYYY-MM-DD（与 --start-date/--end-date 互斥）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期 YYYY-MM-DD（默认等于 start-date）",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="以逗号分隔的标的列表，如 000001.SZ,000002.SZ；为空则表示全部标的",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="是否同时打印简易历史回放视图（依赖 tools.replay_and_healthcheck）",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    # 配置加载：优先 config_center，只有在 __init__ 中会真正读取
    cfg = None
    if args.config:
        # 如果你想指定配置文件路径，可以简单修改 get_system_config 实现；
        # 这里先保持由 config_center 决定。
        log.info("注意：--config 参数当前未单独解析，仍由 config_center 决定配置来源。")

    today = dt.date.today()
    if args.date:
        start_date = _parse_date(args.date)
        end_date = start_date
    elif args.start_date:
        start_date = _parse_date(args.start_date)
        end_date = _parse_date(args.end_date) if args.end_date else start_date
    else:
        # 默认：检查“昨天”的数据
        start_date = today - dt.timedelta(days=1)
        end_date = start_date
        log.info("未提供日期参数，默认检查昨天：%s", _date_to_str(start_date))

    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    guardian = DataGuardianOmega(system_config=cfg)
    guardian.run(start_date=start_date, end_date=end_date, symbols=symbols, replay=args.replay)


if __name__ == "__main__":
    main()
