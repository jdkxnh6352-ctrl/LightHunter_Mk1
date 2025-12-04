# -*- coding: utf-8 -*-
"""
模块名称：TSMinuteBuilder Mk-Fusion
版本：Mk-Fusion R10 (1m Aggregator)
路径: G:/LightHunter_Mk1/ts_minute_builder.py

功能：
- 从 ts_data.db.snapshots 读取 Tick/秒级快照；
- 按 code + 每分钟最后一笔 聚合成 1 分钟级快照；
- 写入 market_ts.db.snapshot_1m，供 RiskBrain / FactorEngine 使用。
"""

import os
import sqlite3
import datetime
from typing import List

import pandas as pd
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)


class TSMinuteBuilder:
    def __init__(
        self,
        src_db: str = "ts_data.db",
        dst_db: str = "market_ts.db",
    ):
        self.src_db = src_db
        self.dst_db = dst_db
        self._init_dst_db()

    # --------------------------- 目标库初始化 --------------------------- #
    def _init_dst_db(self):
        conn = sqlite3.connect(self.dst_db)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_1m (
                trade_date TEXT NOT NULL,
                ts TEXT NOT NULL,
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
                代码 TEXT,
                名称 TEXT,
                现价 REAL,
                涨幅 REAL,
                换手率 REAL,
                成交额 REAL,
                主力攻击系数 REAL,
                PRIMARY KEY (trade_date, ts, code)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_s1m_code_ts
            ON snapshot_1m (code, ts);
            """
        )
        conn.commit()
        conn.close()
        print(
            Fore.GREEN
            + f"[1M] target DB ready -> {self.dst_db} (snapshot_1m)"
            + Style.RESET_ALL
        )

    # --------------------------- 工具：列出交易日 --------------------------- #
    def _list_dates(self) -> List[str]:
        if not os.path.exists(self.src_db):
            print(
                Fore.RED
                + f"[1M] source DB not found: {self.src_db}"
                + Style.RESET_ALL
            )
            return []
        conn = sqlite3.connect(self.src_db)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT substr(ts,1,10) AS d FROM snapshots ORDER BY d;"
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    # --------------------------- 核心：单日聚合 --------------------------- #
    def build_for_date(self, date_str: str):
        """
        :param date_str: 'YYYY-MM-DD'
        """
        if not os.path.exists(self.src_db):
            print(
                Fore.RED
                + f"[1M] source DB not found: {self.src_db}"
                + Style.RESET_ALL
            )
            return

        print(
            Fore.CYAN
            + f"[1M] Aggregating {date_str} ..."
            + Style.RESET_ALL
        )

        conn_src = sqlite3.connect(self.src_db)
        try:
            df = pd.read_sql_query(
                """
                SELECT ts, code, name, price, pct, amount, turnover_rate,
                       float_mkt, force, vwap, ba_ratio
                FROM snapshots
                WHERE substr(ts,1,10) = ?
                ORDER BY code, ts;
                """,
                conn_src,
                params=(date_str,),
            )
        except Exception as e:
            conn_src.close()
            print(
                Fore.RED
                + f"[1M] read snapshots failed: {e}"
                + Style.RESET_ALL
            )
            return
        finally:
            conn_src.close()

        if df.empty:
            print(
                Fore.YELLOW
                + f"[1M] No snapshots for {date_str}."
                + Style.RESET_ALL
            )
            return

        # 字段兜底
        for col, default in [
            ("name", ""),
            ("price", 0.0),
            ("pct", 0.0),
            ("amount", 0.0),
            ("turnover_rate", 0.0),
            ("float_mkt", 0.0),
            ("force", 0.0),
            ("vwap", 0.0),
            ("ba_ratio", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

        df["ts"] = pd.to_datetime(df["ts"])

        # 按 code 分组，每分钟最后一笔
        all_rows = []
        for (code, name), g in df.groupby(["code", "name"]):
            g = g.sort_values("ts").reset_index(drop=True)
            if g.empty:
                continue
            g["minute"] = g["ts"].dt.floor("T")
            # 每分钟取最后一行
            g_min = (
                g.sort_values("ts")
                .drop_duplicates(subset=["minute"], keep="last")
                .reset_index(drop=True)
            )

            sub = pd.DataFrame(
                {
                    "trade_date": g_min["minute"]
                    .dt.date.astype(str),
                    "ts": g_min["minute"].dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "code": code,
                    "name": name,
                    "price": g_min["price"].astype(float),
                    "pct": g_min["pct"].astype(float),
                    "turnover_rate": g_min["turnover_rate"].astype(
                        float
                    ),
                    "amount": g_min["amount"].astype(float),
                    "float_mkt": g_min["float_mkt"].astype(float),
                    "force": g_min["force"].astype(float),
                    "vwap": g_min["vwap"].astype(float),
                    "ba_ratio": g_min["ba_ratio"].astype(float),
                }
            )
            all_rows.append(sub)

        if not all_rows:
            print(
                Fore.YELLOW
                + f"[1M] After grouping, no rows for {date_str}."
                + Style.RESET_ALL
            )
            return

        out_df = pd.concat(all_rows, ignore_index=True)

        # 写入目标 DB
        conn_dst = sqlite3.connect(self.dst_db)
        cur = conn_dst.cursor()
        try:
            # 先删掉该交易日旧记录
            cur.execute(
                "DELETE FROM snapshot_1m WHERE trade_date = ?;",
                (date_str,),
            )
            conn_dst.commit()

            rows = [
                (
                    r.trade_date,
                    r.ts,
                    r.code,
                    r.name,
                    float(r.price),
                    float(r.pct),
                    float(r.turnover_rate),
                    float(r.amount),
                    float(r.float_mkt),
                    float(r.force),
                    float(r.vwap),
                    float(r.ba_ratio),
                    r.code,             # 代码
                    r.name,             # 名称
                    float(r.price),     # 现价
                    float(r.pct),       # 涨幅
                    float(r.turnover_rate),  # 换手率
                    float(r.amount),    # 成交额
                    float(r.force),     # 主力攻击系数
                )
                for r in out_df.itertuples(index=False)
            ]

            cur.executemany(
                """
                INSERT INTO snapshot_1m (
                    trade_date, ts, code, name, price, pct,
                    turnover_rate, amount, float_mkt, force,
                    vwap, ba_ratio,
                    代码, 名称, 现价, 涨幅, 换手率, 成交额, 主力攻击系数
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
                """,
                rows,
            )
            conn_dst.commit()
            print(
                Fore.GREEN
                + f"[1M] Saved {len(rows)} rows -> {self.dst_db}.snapshot_1m ({date_str})"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[1M] write snapshot_1m failed: {e}"
                + Style.RESET_ALL
            )
        finally:
            conn_dst.close()

    # --------------------------- 批量接口 --------------------------- #
    def build_recent(self, days: int = 3):
        dates = self._list_dates()
        if not dates:
            return
        target = dates[-days:]
        for d in target:
            self.build_for_date(d)

    def build_all(self):
        dates = self._list_dates()
        for d in dates:
            self.build_for_date(d)


if __name__ == "__main__":
    builder = TSMinuteBuilder()
    # 初次可以 build_all()，之后每天收盘跑 build_recent(1) 即可
    builder.build_recent(days=3)
