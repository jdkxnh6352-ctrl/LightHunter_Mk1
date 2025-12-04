# -*- coding: utf-8 -*-
"""
tools/replay_and_healthcheck.py

LightHunter Mk4 - 历史回放 & 采集完整性检查工具
==============================================

功能：
- 对 DuckDB 中的日线多源表（默认为 data_quality.daily_table，即 multi_source_daily）
  在指定日期区间 / 标的范围内做：
  1）采集完整性检查：哪些源缺失；
  2）字段有效性检查：价格/成交量等是否符合 data_quality.invalid_rules；
  3）与参考源（reference_source，如 em）的价差统计；
- 生成 Markdown 日报（写入 paths.reports_dir）；
- 可选进行“日线级历史回放”，在终端按日期/标的顺序打印关键行情字段。

依赖：
- config/system_config.json：
    duckdb.db_path 或 storage.ts.duckdb_path
    paths.reports_dir / paths.project_root
    data_quality.daily_table / sources / reference_source / invalid_rules / diff_threshold
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import duckdb  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: 需要安装 duckdb 库：pip install duckdb", file=sys.stderr)
    raise

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: 需要安装 pandas 库：pip install pandas", file=sys.stderr)
    raise

try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str):
        return logging.getLogger(name)


try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        return {}


log = get_logger(__name__)


# ----------------------------------------------------------------------
# 基础工具函数
# ----------------------------------------------------------------------


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _date_range(start: dt.date, end: dt.date) -> List[dt.date]:
    if end < start:
        raise ValueError("end_date 不能早于 start_date")
    days = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(days + 1)]


def _ensure_reports_dir(cfg: Dict[str, Any]) -> str:
    paths = cfg.get("paths", {}) or {}
    reports_dir = paths.get("reports_dir") or "reports"
    if not os.path.isabs(reports_dir):
        project_root = paths.get("project_root", ".") or "."
        reports_dir = os.path.join(project_root, reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def _get_duckdb_path(cfg: Dict[str, Any]) -> str:
    duck_cfg = cfg.get("duckdb", {}) or {}
    path = duck_cfg.get("db_path")
    if not path:
        ts_cfg = (cfg.get("storage") or {}).get("ts") or {}
        path = ts_cfg.get("duckdb_path", "data/lighthunter.duckdb")
    return path


# ----------------------------------------------------------------------
# 数据加载与质量检查
# ----------------------------------------------------------------------


def _load_daily_data(
    con: "duckdb.DuckDBPyConnection",
    table: str,
    start: dt.date,
    end: dt.date,
    symbols: Optional[Sequence[str]],
) -> "pd.DataFrame":
    """
    从 DuckDB 载入 multi_source_daily 风格的日线多源数据：
    需要字段：
        trade_date, symbol, source,
        close_price, chg_pct, volume, turnover
    """
    sql = f"""
        SELECT
            trade_date,
            symbol,
            source,
            close_price,
            chg_pct,
            volume,
            turnover
        FROM {table}
        WHERE trade_date BETWEEN ? AND ?
    """
    params: List[Any] = [start.isoformat(), end.isoformat()]
    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        sql += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    log.info("从 DuckDB 载入 multi_source_daily 数据：table=%s, start=%s, end=%s, symbols=%s",
             table, start, end, "ALL" if not symbols else ",".join(symbols))

    df = con.execute(sql, params).df()
    if df.empty:
        log.warning("指定区间内 multi_source_daily 查询结果为空。")
    return df


def _check_invalid_fields(row: "pd.Series", rules: Dict[str, Dict[str, float]]) -> List[str]:
    """
    对单条记录按 data_quality.invalid_rules 检查字段是否在合法范围内。
    规则格式示例：
        "close_price": {"gt": 0.0}
        "chg_pct": {"ge": -30.0, "le": 30.0}
    """
    invalid_fields: List[str] = []
    for field, cond in rules.items():
        if field not in row:
            continue
        val = row[field]
        if val is None:
            invalid_fields.append(f"{field}=None")
            continue
        try:
            v = float(val)
        except Exception:
            invalid_fields.append(f"{field}=NaN")
            continue
        ge = cond.get("ge")
        gt = cond.get("gt")
        le = cond.get("le")
        lt = cond.get("lt")
        if ge is not None and v < ge:
            invalid_fields.append(f"{field}<{ge}")
        if gt is not None and v <= gt:
            invalid_fields.append(f"{field}<={gt}")
        if le is not None and v > le:
            invalid_fields.append(f"{field}>{le}")
        if lt is not None and v >= lt:
            invalid_fields.append(f"{field}>={lt}")
    return invalid_fields


def compute_health_summary(
    df: "pd.DataFrame",
    dates: Sequence[dt.date],
    symbols: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    对 multi_source_daily 进行完整性统计：
    - 缺失源数量
    - 无效字段数量
    - 与参考源的绝对/相对偏差
    返回：
        summary_df: 按 (trade_date, symbol) 聚合的摘要
        pairs_df  : 参考源对比的样本明细
        stats     : 总体统计字典
    """
    if df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "total_records": 0,
                "total_missing_points": 0,
                "total_invalid_points": 0,
                "total_pairs_checked": 0,
            },
        )

    dq_cfg = cfg.get("data_quality", {}) or {}
    expected_sources: List[str] = dq_cfg.get("sources", []) or []
    ref_source: str = dq_cfg.get("reference_source", "") or ""
    invalid_rules: Dict[str, Dict[str, float]] = dq_cfg.get("invalid_rules", {}) or {}
    diff_threshold: float = float(dq_cfg.get("diff_threshold", 0.001))  # 当前只统计，不做硬过滤

    if not symbols:
        symbols = sorted(df["symbol"].dropna().unique().tolist())

    # 统一 trade_date 为 date 类型
    if df["trade_date"].dtype == "object":
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    key_cols = ["trade_date", "symbol"]
    result_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    total_missing = 0
    total_invalid = 0
    total_pairs = 0

    grouped = df.groupby(key_cols)

    for d in dates:
        for sym in symbols:
            key = (d, sym)
            if key not in grouped.groups:
                # 整个标的当天缺失
                result_rows.append(
                    {
                        "trade_date": d,
                        "symbol": sym,
                        "present_sources": 0,
                        "missing_sources": len(expected_sources),
                        "invalid_points": 0,
                        "max_abs_diff": None,
                        "max_rel_diff": None,
                    }
                )
                total_missing += len(expected_sources)
                continue

            g = grouped.get_group(key)
            present_sources = set(g["source"].tolist())
            missing_sources = [s for s in expected_sources if s not in present_sources]
            total_missing += len(missing_sources)

            invalid_points = 0
            for _, row in g.iterrows():
                bad_fields = _check_invalid_fields(row, invalid_rules)
                if bad_fields:
                    invalid_points += 1
                    total_invalid += 1

            # 与参考源的偏差
            max_abs_diff = None
            max_rel_diff = None

            if ref_source and ref_source in present_sources:
                ref_row = g[g["source"] == ref_source].iloc[0]
                ref_close = float(ref_row["close_price"]) if ref_row["close_price"] is not None else None

                for _, row in g.iterrows():
                    src = row["source"]
                    if src == ref_source:
                        continue
                    if row["close_price"] is None or ref_close in (None, 0.0):
                        continue
                    v = float(row["close_price"])
                    abs_diff = abs(v - ref_close)
                    rel_diff = abs_diff / max(abs(ref_close), 1e-6)

                    total_pairs += 1
                    pair_rows.append(
                        {
                            "trade_date": d,
                            "symbol": sym,
                            "source": src,
                            "ref_source": ref_source,
                            "close_price": v,
                            "ref_close_price": ref_close,
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff,
                        }
                    )

                    if max_abs_diff is None or abs_diff > max_abs_diff:
                        max_abs_diff = abs_diff
                    if max_rel_diff is None or rel_diff > max_rel_diff:
                        max_rel_diff = rel_diff

            result_rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "present_sources": len(present_sources),
                    "missing_sources": len(missing_sources),
                    "invalid_points": invalid_points,
                    "max_abs_diff": max_abs_diff,
                    "max_rel_diff": max_rel_diff,
                }
            )

    summary_df = pd.DataFrame(result_rows)
    pairs_df = pd.DataFrame(pair_rows)

    summary_stats = {
        "total_records": int(len(df)),
        "total_missing_points": int(total_missing),
        "total_invalid_points": int(total_invalid),
        "total_pairs_checked": int(total_pairs),
    }
    return summary_df, pairs_df, summary_stats


# ----------------------------------------------------------------------
# 报告输出 & 回放
# ----------------------------------------------------------------------


def render_markdown_report(
    out_path: str,
    start: dt.date,
    end: dt.date,
    symbols: Optional[Sequence[str]],
    summary_df: "pd.DataFrame",
    pairs_df: "pd.DataFrame",
    stats: Dict[str, Any],
) -> None:
    """
    输出一个简单的 Markdown 日报，方便浏览。
    """
    lines: List[str] = []
    lines.append(f"# 数据回放与采集完整性日报")
    lines.append("")
    lines.append(f"- 统计区间：{start.isoformat()} ~ {end.isoformat()}")
    if symbols:
        lines.append(f"- 标的范围：{', '.join(symbols)}")
    else:
        lines.append(f"- 标的范围：全部在表中出现的标的")
    lines.append(f"- 总记录数：{stats.get('total_records', 0)}")
    lines.append(f"- 总缺失点数：{stats.get('total_missing_points', 0)}")
    lines.append(f"- 总无效点数：{stats.get('total_invalid_points', 0)}")
    lines.append(f"- 参考源对比样本数：{stats.get('total_pairs_checked', 0)}")
    lines.append("")
    lines.append("## 1. 按标的/日期的完整性统计（缺失源 + 无效点）")
    lines.append("")
    if summary_df.empty:
        lines.append("（无数据）")
    else:
        # 只展示问题较多的 Top N
        top_n = (
            summary_df.sort_values(
                ["missing_sources", "invalid_points", "max_rel_diff"],
                ascending=[False, False, False],
            )
            .head(50)
            .copy()
        )
        lines.append("以下为缺失/异常较多的前 50 个 (trade_date, symbol)：")
        lines.append("")
        lines.append("| trade_date | symbol | present_sources | missing_sources | invalid_points | max_abs_diff | max_rel_diff |")
        lines.append("|------------|--------|-----------------|-----------------|----------------|--------------|--------------|")
        for _, row in top_n.iterrows():
            lines.append(
                f"| {row['trade_date']} | {row['symbol']} | {row['present_sources']} | "
                f"{row['missing_sources']} | {row['invalid_points']} | "
                f"{row.get('max_abs_diff', '')} | {row.get('max_rel_diff', '')} |"
            )

    lines.append("")
    lines.append("## 2. 与参考源的偏差样本（按相对偏差降序前 50）")
    lines.append("")
    if pairs_df.empty:
        lines.append("（无对比样本）")
    else:
        top_pairs = pairs_df.sort_values("rel_diff", ascending=False).head(50)
        lines.append("| trade_date | symbol | source | ref_source | close_price | ref_close_price | abs_diff | rel_diff |")
        lines.append("|------------|--------|--------|------------|-------------|-----------------|----------|----------|")
        for _, row in top_pairs.iterrows():
            lines.append(
                f"| {row['trade_date']} | {row['symbol']} | {row['source']} | {row['ref_source']} | "
                f"{row['close_price']} | {row['ref_close_price']} | {row['abs_diff']} | {row['rel_diff']} |"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("已生成 Markdown 报告：%s", out_path)


def replay_daily(
    df: "pd.DataFrame",
    start: dt.date,
    end: dt.date,
    symbols: Optional[Sequence[str]],
    speed: float = 0.0,
) -> None:
    """
    简单的日线级“回放”：按日期、标的顺序输出关键行情字段。
    speed > 0 时，可以适当 sleep 模拟播放节奏（单位：秒）。
    """
    import time as _time

    if df.empty:
        log.info("回放数据为空，跳过。")
        return

    if df["trade_date"].dtype == "object":
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    if not symbols:
        symbols = sorted(df["symbol"].dropna().unique().tolist())

    mask = (df["trade_date"] >= start) & (df["trade_date"] <= end)
    df = df.loc[mask].copy()

    # 只展示参考源或某个默认源（例如 em）
    if "source" in df.columns:
        if (df["source"] == "em").any():
            df = df[df["source"] == "em"]
        else:
            df = (
                df.sort_values(["trade_date", "symbol"])
                .groupby(["trade_date", "symbol"])
                .head(1)
            )

    df = df.sort_values(["trade_date", "symbol"])

    log.info("开始历史回放：%s ~ %s，标的数=%d，记录条数=%d",
             start, end, len(symbols), len(df))

    for _, row in df.iterrows():
        d = row["trade_date"]
        sym = row["symbol"]
        close = row.get("close_price")
        chg = row.get("chg_pct")
        vol = row.get("volume")
        turnover = row.get("turnover")
        print(
            f"[REPLAY] {d} {sym}: close={close}, chg_pct={chg}, volume={vol}, turnover={turnover}"
        )
        if speed and speed > 0:
            _time.sleep(speed)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="LightHunter - 历史回放与采集完整性检查"
    )
    parser.add_argument("--start-date", type=str, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="结束日期 YYYY-MM-DD（默认同起始）")
    parser.add_argument("--symbols", type=str, default="", help="标的列表，逗号分隔，如 000001.SZ,000002.SZ")
    parser.add_argument(
        "--mode",
        type=str,
        default="health",
        choices=["health", "replay", "both"],
        help="运行模式：health=只做完整性检查；replay=只做回放；both=先检查再回放",
    )
    parser.add_argument(
        "--daily-table",
        type=str,
        default=None,
        help="日线多源表名（默认读取 system_config.data_quality.daily_table 或 multi_source_daily）",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="回放速度控制（秒），仅在 mode=replay/both 时生效，0 表示不 sleep",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="完整性日报输出路径（默认写入 reports/replay_health_<start>_<end>.md）",
    )

    args = parser.parse_args(argv)

    cfg = get_system_config(refresh=False)
    if not args.start_date:
        # 默认跑“昨天”的数据
        today = dt.date.today()
        start = today - dt.timedelta(days=1)
        end = start
        log.info("未指定 start-date，默认使用昨天：%s", start)
    else:
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date) if args.end_date else start

    if end < start:
        raise SystemExit("end-date 不能早于 start-date")

    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    dates = _date_range(start, end)

    dq_cfg = cfg.get("data_quality", {}) or {}
    table = args.daily_table or dq_cfg.get("daily_table") or "multi_source_daily"

    db_path = _get_duckdb_path(cfg)
    log.info("连接 DuckDB 数据库：%s", db_path)
    con = duckdb.connect(database=db_path, read_only=True)

    df = _load_daily_data(con, table, start, end, symbols)

    if args.mode in ("health", "both"):
        summary_df, pairs_df, stats = compute_health_summary(df, dates, symbols, cfg)
        reports_dir = _ensure_reports_dir(cfg)
        if args.output:
            out_path = args.output
            if not os.path.isabs(out_path):
                out_path = os.path.join(reports_dir, out_path)
        else:
            out_path = os.path.join(
                reports_dir,
                f"replay_health_{start.isoformat()}_{end.isoformat()}.md",
            )
        render_markdown_report(out_path, start, end, symbols, summary_df, pairs_df, stats)

    if args.mode in ("replay", "both"):
        replay_daily(df, start, end, symbols, speed=args.speed)


if __name__ == "__main__":
    main()
