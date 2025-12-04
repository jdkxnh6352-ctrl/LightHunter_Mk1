# -*- coding: utf-8 -*-
"""
模块名称：MarketTSBuilder Mk-Fusion
版本：Mk-Fusion R40 (TS → 1-Min Snapshot)
路径: G:/LightHunter_Mk1/market_ts_builder.py

功能：
- 从 ts_data.db.snapshots 聚合生成分钟级行情库 market_ts.db.snapshot_1m；
- 每个 (trade_date, code, 1分钟) 取该分钟最后一条快照作为“收盘价”；
- 反推流通市值 float_mkt、主力攻击系数 force，近似估算；
- 同时写入中英文字段，兼容 FactorEngine / RiskBrain / TSBacktest 等模块。
"""

import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from colorama import init, Fore, Style

init(autoreset=True)

TS_DB_FILE = "ts_data.db"
MKT_DB_FILE = "market_ts.db"


class MarketTSBuilder:
    def __init__(
        self,
        ts_db: str = TS_DB_FILE,
        out_db: str = MKT_DB_FILE,
    ):
        self.ts_db = ts_db
        self.out_db = out_db
        self._init_out_db()

    # -----------------------------
    # 输出库初始化：snapshot_1m 表
    # -----------------------------
    def _init_out_db(self):
        need_init = not os.path.exists(self.out_db)
        conn = sqlite3.connect(self.out_db)
        cur = conn.cursor()
        # 主表：分钟级快照（中英文字段同存）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_1m (
                trade_date TEXT NOT NULL,  -- 交易日 YYYY-MM-DD
                ts         TEXT NOT NULL,  -- 分钟时间戳 YYYY-MM-DD HH:MM:00

                -- 英文字段：FactorEngine / 回测 使用
                code       TEXT NOT NULL,
                name       TEXT,
                price      REAL,
                pct        REAL,
                turnover_rate REAL,
                amount     REAL,
                float_mkt  REAL,
                force      REAL,
                vwap       REAL,
                ba_ratio   REAL,

                -- 中文字段：RiskBrain / SQL 分析使用
                "代码"     TEXT,
                "名称"     TEXT,
                "现价"     REAL,
                "涨幅"     REAL,
                "换手率"   REAL,
                "成交额"   REAL,
                "主力攻击系数" REAL,

                PRIMARY KEY (trade_date, ts, code)
            );
            """
        )
        # 常用查询索引
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshot1m_date_code_ts
            ON snapshot_1m (trade_date, code, ts);
            """
        )
        conn.commit()
        conn.close()

        if need_init:
            print(
                Fore.GREEN
                + f"[MTS] New {self.out_db} created (snapshot_1m ready)."
                + Style.RESET_ALL
            )
        else:
            print(
                Fore.GREEN
                + f"[MTS] {self.out_db} connected (snapshot_1m ready)."
                + Style.RESET_ALL
            )

    # -----------------------------
    # 工具：连接 ts_data.db
    # -----------------------------
    def _connect_ts(self) -> sqlite3.Connection:
        if not os.path.exists(self.ts_db):
            raise FileNotFoundError(
                f"[MTS] ts DB not found: {self.ts_db}, "
                f"请先运行 ts_collector.py 采集分时数据。"
            )
        return sqlite3.connect(self.ts_db)

    # -----------------------------
    # 工具：列出 ts_data 中所有交易日
    # -----------------------------
    def list_ts_dates(self) -> List[str]:
        conn = self._connect_ts()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT substr(ts,1,10) AS d "
            "FROM snapshots ORDER BY d;"
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    # -----------------------------
    # 核心：聚合某一交易日的 1 分钟数据
    # -----------------------------
    def build_for_date(self, trade_date: str) -> int:
        """
        从 ts_data.db 中聚合某天数据，写入 market_ts.db.snapshot_1m
        :return: 写入的行数
        """
        print(
            Fore.CYAN
            + f"[MTS] Building 1-min snapshot for {trade_date}."
            + Style.RESET_ALL
        )

        # 1) 读取该交易日的所有秒级快照
        conn_ts = self._connect_ts()
        try:
            df = pd.read_sql_query(
                """
                SELECT ts, code, price, pct, amount, turnover_rate
                FROM snapshots
                WHERE substr(ts,1,10) = ?
                ORDER BY code, ts;
                """,
                conn_ts,
                params=(trade_date,),
            )
        except Exception as e:
            conn_ts.close()
            print(
                Fore.RED
                + f"[MTS] Failed to read snapshots for {trade_date}: {e}"
                + Style.RESET_ALL
            )
            return 0
        finally:
            conn_ts.close()

        if df.empty:
            print(
                Fore.YELLOW
                + f"[MTS] No TS data for {trade_date} in ts_data.db."
                + Style.RESET_ALL
            )
            return 0

        # 2) 时间列解析
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception as e:
            print(
                Fore.RED
                + f"[MTS] Failed to parse ts column for {trade_date}: {e}"
                + Style.RESET_ALL
            )
            return 0

        # 3) 数值清洗
        for col in ["price", "pct", "amount", "turnover_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["price", "pct", "amount", "turnover_rate"])
        df = df[df["price"] > 0]
        if df.empty:
            print(
                Fore.YELLOW
                + f"[MTS] After cleaning, no valid rows for {trade_date}."
                + Style.RESET_ALL
            )
            return 0

        # 4) 取每分钟最后一条：相当于「该分钟收盘快照」
        df["minute"] = df["ts"].dt.floor("T")  # 向下取整到分钟
        df = df.sort_values(["code", "minute", "ts"])
        last_rows = df.groupby(["code", "minute"]).tail(1).copy()

        # trade_date & 标准化 ts 字符串
        last_rows["trade_date"] = trade_date
        last_rows["ts_minute"] = last_rows["minute"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ---------- 反推出流通市值 float_mkt ----------
        # 理论上：换手率(%) = 成交额 / 流通市值 * 100
        # => 流通市值 ≈ 成交额 * 100 / 换手率
        tr = last_rows["turnover_rate"].replace(0, np.nan)
        float_mkt = last_rows["amount"] * 100.0 / tr
        float_mkt = float_mkt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        last_rows["float_mkt"] = float_mkt

        # ---------- 主力攻击系数 force ----------
        # 尽量贴近 data_core 中的计算逻辑：
        #   cap_factor = 1.0 (小盘) / 0.8 (大盘)
        #   force = turnover / mkt_cap * pct * 10 * cap_factor
        mkt_cap = last_rows["float_mkt"]
        turnover = last_rows["amount"]
        pct = last_rows["pct"]

        cap_factor = np.where(mkt_cap < 1e11, 1.0, 0.8)
        denom = mkt_cap.replace(0, np.nan)
        force = turnover / denom * pct * 10.0 * cap_factor
        force = force.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        last_rows["force"] = force

        # ---------- VWAP & BA_Ratio ----------
        # 没有逐笔成交量 / 委托簿，只能近似：
        # - vwap ~ price
        # - ba_ratio ~ 1.0（中性）
        last_rows["vwap"] = last_rows["price"]
        last_rows["ba_ratio"] = 1.0

        # 名称目前没有离线来源，先留空（不影响训练）
        last_rows["name"] = ""

        # 中文别名字段（兼容 RiskBrain 的 SQL）
        last_rows["代码"] = last_rows["code"]
        last_rows["名称"] = last_rows["name"]
        last_rows["现价"] = last_rows["price"]
        last_rows["涨幅"] = last_rows["pct"]
        last_rows["换手率"] = last_rows["turnover_rate"]
        last_rows["成交额"] = last_rows["amount"]
        last_rows["主力攻击系数"] = last_rows["force"]

        # 最终写入列
        out_cols = [
            "trade_date",
            "ts_minute",
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
        out_df = last_rows[out_cols].copy()
        out_df = out_df.rename(columns={"ts_minute": "ts"})

        # 5) 写入 market_ts.db
        conn_out = sqlite3.connect(self.out_db)
        cur = conn_out.cursor()

        # 同一交易日先清空再重建，避免重复
        cur.execute(
            "DELETE FROM snapshot_1m WHERE trade_date = ?;",
            (trade_date,),
        )
        conn_out.commit()

        records = out_df.to_records(index=False)
        cur.executemany(
            """
            INSERT OR REPLACE INTO snapshot_1m (
                trade_date, ts, code, name, price, pct, turnover_rate,
                amount, float_mkt, force, vwap, ba_ratio,
                "代码", "名称", "现价", "涨幅", "换手率", "成交额", "主力攻击系数"
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            list(records),
        )
        conn_out.commit()
        n_rows = len(out_df)
        conn_out.close()

        print(
            Fore.GREEN
            + f"[MTS] {trade_date}: wrote {n_rows} rows -> {self.out_db}.snapshot_1m"
            + Style.RESET_ALL
        )
        return n_rows

    # -----------------------------
    # 最近 N 天批量构建
    # -----------------------------
    def build_recent(self, days: int = 3):
        dates = self.list_ts_dates()
        if not dates:
            print(
                Fore.RED
                + "[MTS] No dates found in ts_data.db, "
                + "请先运行 ts_collector.py 采集分时数据。"
                + Style.RESET_ALL
            )
            return

        target_dates = dates[-days:]
        print(
            Fore.CYAN
            + f"[MTS] Building recent {len(target_dates)} days: "
            + ", ".join(target_dates)
            + Style.RESET_ALL
        )
        for d in target_dates:
            self.build_for_date(d)

    # -----------------------------
    # 全部交易日一次性构建
    # -----------------------------
    def build_all(self):
        dates = self.list_ts_dates()
        if not dates:
            print(
                Fore.RED
                + "[MTS] No dates found in ts_data.db."
                + Style.RESET_ALL
            )
            return
        print(
            Fore.CYAN
            + f"[MTS] Building all {len(dates)} trade dates..."
            + Style.RESET_ALL
        )
        for d in dates:
            self.build_for_date(d)


if __name__ == "__main__":
    builder = MarketTSBuilder()
    # 初次使用建议先 build_recent(3) 看看效果，再按需改成 build_all()
    builder.build_recent(days=3)
