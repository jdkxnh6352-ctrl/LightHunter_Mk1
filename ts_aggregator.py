# -*- coding: utf-8 -*-
"""
模块名称：TSAggregator Mk-Minute
版本：Mk-Minute R60 (Time Fusion Builder)
路径: G:/LightHunter_Mk1/ts_aggregator.py

功能：
- 从 ts_data.db.snapshots 聚合生成分钟级行情库 market_ts.db.snapshot_1m；
- 每个 (code, 1分钟) 保留那一分钟内最后一条快照作为“收盘价”；
- 自动反推 float_mkt（流通市值近似）和 主力攻击系数（force）；
- 同时写入中英文两套字段，兼容 RiskBrain / FactorEngine 等模块。
"""

import os
import sqlite3
import datetime

import numpy as np
import pandas as pd
from colorama import init, Fore, Style

init(autoreset=True)

SRC_DB = "ts_data.db"
DST_DB = "market_ts.db"


class TS1mBuilder:
    def __init__(self, src_db: str = SRC_DB, dst_db: str = DST_DB):
        self.src_db = src_db
        self.dst_db = dst_db

    # --------------------------------------------------
    # 工具：列出已有交易日
    # --------------------------------------------------
    def _list_trade_dates(self):
        if not os.path.exists(self.src_db):
            print(
                Fore.RED
                + f"[TS1M] 源库不存在: {self.src_db}，先跑 ts_collector / Commander 采集分时。"
                + Style.RESET_ALL
            )
            return []
        conn = sqlite3.connect(self.src_db)
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT substr(ts,1,10) AS d FROM snapshots ORDER BY d;",
                conn,
            )
        finally:
            conn.close()
        dates = df["d"].dropna().astype(str).tolist()
        return dates

    # --------------------------------------------------
    # 工具：加载某个交易日的全部分时
    # --------------------------------------------------
    def _load_day(self, trade_date: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.src_db)
        try:
            df = pd.read_sql_query(
                """
                SELECT ts, code, price, pct, amount, turnover_rate
                FROM snapshots
                WHERE substr(ts,1,10) = ?
                ORDER BY code, ts;
                """,
                conn,
                params=(trade_date,),
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    # --------------------------------------------------
    # 目标库初始化
    # --------------------------------------------------
    def _init_dst(self):
        need_init = not os.path.exists(self.dst_db)
        conn = sqlite3.connect(self.dst_db)
        cur = conn.cursor()
        # 英文 + 中文两套字段，方便后续 SQL 复用
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_1m (
                ts TEXT NOT NULL,             -- 该分钟收盘时间，如 '2025-01-02 09:31:00'
                trade_date TEXT NOT NULL,     -- 交易日 'YYYY-MM-DD'
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
                PRIMARY KEY (ts, code)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshot1m_code_ts
            ON snapshot_1m (code, ts);
            """
        )
        conn.commit()
        conn.close()

        if need_init:
            print(
                Fore.GREEN
                + f"[TS1M] New {self.dst_db} created (snapshot_1m ready)."
                + Style.RESET_ALL
            )
        else:
            print(
                Fore.GREEN
                + f"[TS1M] Connected to {self.dst_db} (snapshot_1m ready)."
                + Style.RESET_ALL
            )

    # --------------------------------------------------
    # 为单个交易日构建分钟级数据
    # --------------------------------------------------
    def build_for_date(self, trade_date: str):
        print(
            Fore.CYAN
            + f"[TS1M] Building 1-min snapshot for {trade_date} ..."
            + Style.RESET_ALL
        )

        df = self._load_day(trade_date)
        if df.empty:
            print(
                Fore.YELLOW
                + f"[TS1M] {trade_date} 在 ts_data.db 中没有分时记录。"
                + Style.RESET_ALL
            )
            return

        # 计算分钟粒度
        df = df.copy()
        df["minute"] = df["ts"].dt.floor("T")

        # 按 code + minute 聚合，保留这一分钟内的“最后一笔”为代表
        df = df.sort_values(["code", "ts"])
        agg = (
            df.groupby(["code", "minute"])
            .agg(
                price=("price", "last"),
                pct=("pct", "last"),
                amount=("amount", "last"),
                turnover_rate=("turnover_rate", "last"),
            )
            .reset_index()
        )

        if agg.empty:
            print(
                Fore.YELLOW
                + f"[TS1M] {trade_date} 聚合后为空。"
                + Style.RESET_ALL
            )
            return

        agg.rename(columns={"minute": "ts"}, inplace=True)
        agg["trade_date"] = agg["ts"].dt.strftime("%Y-%m-%d")

        # 反推流通市值：float_mkt ≈ amount / (turnover_rate% / 100)
        tr = agg["turnover_rate"].astype(float).replace(0, np.nan)
        float_mkt = np.where(
            np.isfinite(tr),
            agg["amount"].astype(float) * 100.0 / tr,
            np.nan,
        )

        # 简化版主力攻击系数：force ≈ 涨幅 * 换手率 / 10
        force = agg["pct"].astype(float) * agg["turnover_rate"].astype(float) / 10.0

        out = pd.DataFrame(
            {
                "ts": agg["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "trade_date": agg["trade_date"].astype(str),
                "code": agg["code"].astype(str),
                "name": "",
                "price": agg["price"].astype(float),
                "pct": agg["pct"].astype(float),
                "turnover_rate": agg["turnover_rate"].astype(float),
                "amount": agg["amount"].astype(float),
                "float_mkt": float_mkt,
                "force": force,
                "vwap": agg["price"].astype(float),
                "ba_ratio": np.ones(len(agg), dtype=float),
                # 中文别名字段（给 RiskBrain 等 SQL 使用）
                "代码": agg["code"].astype(str),
                "名称": "",
                "现价": agg["price"].astype(float),
                "涨幅": agg["pct"].astype(float),
                "换手率": agg["turnover_rate"].astype(float),
                "成交额": agg["amount"].astype(float),
                "主力攻击系数": force,
            }
        )

        self._init_dst()
        conn = sqlite3.connect(self.dst_db)
        try:
            cur = conn.cursor()
            # 避免重复：先删掉该交易日旧数据
            cur.execute(
                "DELETE FROM snapshot_1m WHERE trade_date = ?;",
                (trade_date,),
            )
            conn.commit()

            out.to_sql(
                "snapshot_1m",
                conn,
                if_exists="append",
                index=False,
            )
            conn.commit()
        finally:
            conn.close()

        print(
            Fore.GREEN
            + f"[TS1M] {trade_date} -> 写入 {len(out)} 行到 {self.dst_db}.snapshot_1m"
            + Style.RESET_ALL
        )

    # --------------------------------------------------
    # 构建最近 N 个交易日
    # --------------------------------------------------
    def build_recent(self, days: int = 3):
        dates = self._list_trade_dates()
        if not dates:
            return
        target = dates[-days:]
        for d in target:
            self.build_for_date(d)

    # --------------------------------------------------
    # 为全部历史交易日构建
    # --------------------------------------------------
    def build_all(self):
        dates = self._list_trade_dates()
        for d in dates:
            self.build_for_date(d)


if __name__ == "__main__":
    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
################################################
#      LIGHT HUNTER - TS 1-Min Builder        #
#           Mk-Minute R60                     #
################################################
"""
        + Style.RESET_ALL
    )
    builder = TS1mBuilder()
    # 首次可用 build_all()，之后每天收盘跑 build_recent(1) 即可
    builder.build_recent(days=3)
