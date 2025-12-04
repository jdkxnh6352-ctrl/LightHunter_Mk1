# -*- coding: utf-8 -*-
"""
模块名称：TSBuilder Mk-Fusion
版本：Mk-Fusion R10 (1-Min Aggregator)
路径: G:/LightHunter_Mk1/ts_builder.py

功能：
- 从 ts_data.db.snapshots 聚合高频分时为 1 分钟级别快照；
- 写入 market_ts.db.snapshot_1m：
    * 同时提供英文字段: code / price / pct ...（给 FactorEngine 用）
    * 和中文字段: 代码 / 现价 / 涨幅 ...（给 RiskBrain 用）
- 为：
    * RiskBrain (market_ts.db.snapshot_1m) 训练 10m 回撤模型提供输入；
    * FactorEngine (market_ts.db.snapshot_1m) 训练 10m 收益标签提供输入。
"""

import os
import sqlite3
import datetime

import pandas as pd
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)


class TSBuilder:
    def __init__(self, src_db: str = "ts_data.db", dst_db: str = "market_ts.db"):
        """
        :param src_db: 高频分时库 (ts_collector / ts_recorder 写入)
                       需要有 snapshots 表: ts, code, price, pct, amount, turnover_rate
        :param dst_db: 分钟级训练库 (本脚本生成 snapshot_1m 表)
        """
        self.src_db = src_db
        self.dst_db = dst_db
        self.src_conn = None
        self.dst_conn = None

        self._connect_dbs()
        if self.dst_conn is not None:
            self._init_dst_schema()

    # ----------------------------- 基础连接 & 关闭 ----------------------------- #

    def _connect_dbs(self):
        if not os.path.exists(self.src_db):
            print(
                Fore.RED
                + f"[TSB] 源 DB 不存在: {self.src_db}，先跑 ts_collector / Commander 采集分时再来。"
                + Style.RESET_ALL
            )
            return

        try:
            self.src_conn = sqlite3.connect(self.src_db, check_same_thread=False)
            self.dst_conn = sqlite3.connect(self.dst_db, check_same_thread=False)
            print(
                Fore.CYAN
                + f"[TSB] Connected src={self.src_db}, dst={self.dst_db}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[TSB] 连接数据库失败: {e}"
                + Style.RESET_ALL
            )
            self.src_conn = None
            self.dst_conn = None

    def close(self):
        if self.src_conn is not None:
            self.src_conn.close()
        if self.dst_conn is not None:
            self.dst_conn.close()
        print(Fore.BLUE + "[TSB] Connections closed." + Style.RESET_ALL)

    # ----------------------------- 目标库结构 ----------------------------- #

    def _init_dst_schema(self):
        cur = self.dst_conn.cursor()
        # snapshot_1m：中英文字段并存，方便 RiskBrain / FactorEngine 直接读取
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_1m (
                ts TEXT NOT NULL,             -- 分钟时间戳 'YYYY-MM-DD HH:MM:00'
                trade_date TEXT NOT NULL,     -- 交易日 'YYYY-MM-DD'

                -- 英文字段：FactorEngine 使用
                code TEXT NOT NULL,
                name TEXT,
                price REAL,
                pct REAL,
                turnover_rate REAL,
                amount REAL,
                float_mkt REAL,
                force REAL,
                vwap REAL,
                ba_ratio REAL,

                -- 中文字段：RiskBrain 使用
                代码 TEXT,
                名称 TEXT,
                现价 REAL,
                涨幅 REAL,
                换手率 REAL,
                成交额 REAL,
                主力攻击系数 REAL,

                PRIMARY KEY (ts, code)
            );
            """
        )
        # 常用索引：按 code+ts 查询
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshot1m_code_ts
            ON snapshot_1m (code, ts);
            """
        )
        # 按 trade_date 查询
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshot1m_date
            ON snapshot_1m (trade_date);
            """
        )
        self.dst_conn.commit()
        print(
            Fore.GREEN
            + "[TSB] snapshot_1m schema ready in market_ts.db."
            + Style.RESET_ALL
        )

    # ----------------------------- 可用交易日 ----------------------------- #

    def list_src_dates(self):
        """
        从 ts_data.db.snapshots 中列出所有交易日
        """
        if self.src_conn is None:
            return []

        cur = self.src_conn.cursor()
        try:
            cur.execute(
                "SELECT substr(ts, 1, 10) AS d, COUNT(*) FROM snapshots "
                "GROUP BY d ORDER BY d;"
            )
            rows = cur.fetchall()
        except Exception as e:
            print(
                Fore.RED
                + f"[TSB] 查询 snapshots 日期失败: {e}"
                + Style.RESET_ALL
            )
            return []

        dates = []
        for d, cnt in rows:
            if cnt and d:
                dates.append(d)
        return dates

    def list_built_dates(self):
        """
        查询 market_ts.db.snapshot_1m 中已经构建过的交易日（方便增量构建）
        """
        if self.dst_conn is None:
            return []

        cur = self.dst_conn.cursor()
        try:
            cur.execute(
                "SELECT DISTINCT trade_date FROM snapshot_1m ORDER BY trade_date;"
            )
            rows = cur.fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            return []

    # ----------------------------- 核心：构建单日 ----------------------------- #

    def build_for_date(self, trade_date: str, overwrite: bool = True) -> int:
        """
        为指定交易日 trade_date 构建 1 分钟聚合数据：
        - 从 ts_data.db.snapshots 读取该日所有分时；
        - 按 code + minute 聚合（每分钟取最后一笔）；
        - 写入 market_ts.db.snapshot_1m。

        :param trade_date: 'YYYY-MM-DD'
        :param overwrite: 若为 True，则先删除该日原有数据再重建。
        :return: 写入行数
        """
        if self.src_conn is None or self.dst_conn is None:
            print(
                Fore.RED
                + "[TSB] DB 未连接，无法构建。"
                + Style.RESET_ALL
            )
            return 0

        print(
            Fore.CYAN
            + f"[TSB] Building 1-min snapshot for {trade_date} ..."
            + Style.RESET_ALL
        )

        try:
            df = pd.read_sql_query(
                """
                SELECT ts, code, price, pct, amount, turnover_rate
                FROM snapshots
                WHERE substr(ts, 1, 10) = ?
                """,
                self.src_conn,
                params=(trade_date,),
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[TSB] 读取 snapshots 失败: {e}"
                + Style.RESET_ALL
            )
            return 0

        if df.empty:
            print(
                Fore.YELLOW
                + f"[TSB] {trade_date} 当日无分时数据。"
                + Style.RESET_ALL
            )
            return 0

        # 类型 & 清洗
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception as e:
            print(
                Fore.RED
                + f"[TSB] ts 字段解析失败: {e}"
                + Style.RESET_ALL
            )
            return 0

        df = df.dropna(subset=["ts", "code"])

        # 聚合到分钟：每个 code 每分钟取“最后一条快照”
        df["minute"] = df["ts"].dt.floor("T")  # 向下取整到分钟
        df = df.sort_values(["code", "ts"])

        grouped = (
            df.groupby(["code", "minute"], as_index=False)
            .last()
            .copy()
        )

        if grouped.empty:
            print(
                Fore.YELLOW
                + f"[TSB] {trade_date} 聚合后无有效数据。"
                + Style.RESET_ALL
            )
            return 0

        # 构造目标字段
        grouped.rename(columns={"minute": "ts"}, inplace=True)
        grouped["trade_date"] = grouped["ts"].dt.strftime("%Y-%m-%d")

        # 英文字段（FactorEngine / 其他模块）
        grouped["name"] = ""  # 暂无名称，后续可从日线或证券基础表补
        grouped["price"] = grouped["price"].astype(float)
        grouped["pct"] = grouped["pct"].astype(float)
        grouped["turnover_rate"] = grouped["turnover_rate"].astype(float)
        grouped["amount"] = grouped["amount"].astype(float)

        # 以下为暂时默认值，可后续升级：
        grouped["float_mkt"] = 0.0          # 流通市值暂缺
        grouped["force"] = 0.0              # 主力攻击系数（可后面用高频盘口重算）
        grouped["vwap"] = grouped["price"]  # 分钟价近似 VWAP
        grouped["ba_ratio"] = 1.0           # 买卖盘量比中性

        # 中文字段（RiskBrain）
        grouped["代码"] = grouped["code"]
        grouped["名称"] = grouped["name"]
        grouped["现价"] = grouped["price"]
        grouped["涨幅"] = grouped["pct"]
        grouped["换手率"] = grouped["turnover_rate"]
        grouped["成交额"] = grouped["amount"]
        grouped["主力攻击系数"] = grouped["force"]

        # 排列列顺序
        cols = [
            "ts",
            "trade_date",
            "code",
            "name",
            "price",
            "pct",
            "turnover_rate",
            "amount",
            "float_mkt",
            "force",
            "vwap",
            "ba_ratio",
            "代码",
            "名称",
            "现价",
            "涨幅",
            "换手率",
            "成交额",
            "主力攻击系数",
        ]
        grouped = grouped[cols]

        # 写入 DB
        cur = self.dst_conn.cursor()
        if overwrite:
            try:
                cur.execute(
                    "DELETE FROM snapshot_1m WHERE trade_date = ?;",
                    (trade_date,),
                )
            except Exception:
                pass

        rows = [
            tuple(row[col] for col in cols)
            for _, row in grouped.iterrows()
        ]

        try:
            cur.executemany(
                """
                INSERT OR REPLACE INTO snapshot_1m
                (ts, trade_date, code, name, price, pct, turnover_rate, amount,
                 float_mkt, force, vwap, ba_ratio,
                 代码, 名称, 现价, 涨幅, 换手率, 成交额, 主力攻击系数)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            self.dst_conn.commit()
        except Exception as e:
            print(
                Fore.RED
                + f"[TSB] 写入 snapshot_1m 失败: {e}"
                + Style.RESET_ALL
            )
            return 0

        print(
            Fore.GREEN
            + f"[TSB] {trade_date} 聚合完成，写入 {len(rows)} 行 1-min 记录。"
            + Style.RESET_ALL
        )
        return len(rows)

    # ----------------------------- 批量构建：最近 N 天 / 全部 ----------------------------- #

    def build_recent(self, days: int = 3, overwrite: bool = True):
        dates = self.list_src_dates()
        if not dates:
            print(
                Fore.RED
                + "[TSB] 源库中没有任何交易日。"
                + Style.RESET_ALL
            )
            return

        target_dates = dates[-days:]
        print(
            Fore.CYAN
            + f"[TSB] 准备构建最近 {len(target_dates)} 个交易日：{', '.join(target_dates)}"
            + Style.RESET_ALL
        )
        total_rows = 0
        for d in target_dates:
            total_rows += self.build_for_date(d, overwrite=overwrite)
        print(
            Fore.CYAN
            + f"[TSB] 最近 {len(target_dates)} 日构建完成，总写入 {total_rows} 行。"
            + Style.RESET_ALL
        )

    def build_all(self, overwrite: bool = False):
        dates = self.list_src_dates()
        if not dates:
            print(
                Fore.RED
                + "[TSB] 源库中没有任何交易日。"
                + Style.RESET_ALL
            )
            return

        print(
            Fore.CYAN
            + f"[TSB] 准备构建全部 {len(dates)} 个交易日：{dates[0]} ~ {dates[-1]}"
            + Style.RESET_ALL
        )
        total_rows = 0
        for d in dates:
            total_rows += self.build_for_date(d, overwrite=overwrite)
        print(
            Fore.CYAN
            + f"[TSB] 全历史构建完成，总写入 {total_rows} 行。"
            + Style.RESET_ALL
        )


def _print_banner():
    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
########################################################
#           LIGHT HUNTER - TS BUILDER Mk-Fusion        #
#          ts_data.db  ->  market_ts.db (1m)           #
########################################################
"""
        + Style.RESET_ALL
    )


if __name__ == "__main__":
    _print_banner()
    builder = TSBuilder()

    if builder.src_conn is None or builder.dst_conn is None:
        # 无法连接 DB，直接退出
        raise SystemExit(1)

    dates = builder.list_src_dates()
    if not dates:
        print(
            Fore.RED
            + "[TSB] ts_data.db.snapshots 中没有数据，先跑一段采集。"
            + Style.RESET_ALL
        )
        builder.close()
        raise SystemExit(0)

    last_date = dates[-1]
    print(
        Fore.YELLOW
        + f"[TSB] 源库可用交易日：{', '.join(dates[-5:])} ... （最新: {last_date}）"
        + Style.RESET_ALL
    )

    print(
        Fore.YELLOW
        + "\n  [1] 构建最近 1 个交易日"
        + "\n  [2] 构建最近 N 个交易日"
        + "\n  [3] 构建全部历史"
        + "\n  [4] 退出"
        + Style.RESET_ALL
    )

    choice = input(
        Fore.YELLOW + "\n  >>> 请选择 [1-4]: " + Style.RESET_ALL
    ).strip() or "1"

    if choice == "1":
        builder.build_for_date(last_date, overwrite=True)
    elif choice == "2":
        n_str = input(
            Fore.YELLOW
            + "  输入 N（最近 N 日，默认 5）: "
            + Style.RESET_ALL
        ).strip()
        try:
            n = int(n_str) if n_str else 5
            builder.build_recent(days=n, overwrite=True)
        except ValueError:
            print(
                Fore.RED
                + "[TSB] N 必须是整数。"
                + Style.RESET_ALL
            )
    elif choice == "3":
        builder.build_all(overwrite=False)
    else:
        print(Fore.GREEN + "[TSB] 已退出，不进行构建。" + Style.RESET_ALL)

    builder.close()
