# -*- coding: utf-8 -*-
"""
market_ts_collector.py

LightHunter Mk1 - Market TS Collector 1m
=======================================

功能：
- 定期调用 MarketHunter.fetch_snapshot() 抓一帧全市场实时快照；
- 把整理好的字段写入本地 SQLite 数据库 market_ts.db 的 snapshot_1m 表；
- （可选）把每一帧按股票拆成 1m bar 事件，通过 ZeroMQ 总线发布。

说明：
- 为了“新手安全测试”，只把最核心的字段当成必选：
    代码 / 名称 / 现价 / 涨幅 / 换手率 / 成交额
- 其它字段（流通市值 / 主力攻击系数 / VWAP / BA_Ratio）都视为**可选字段**：
    缺了就自动补默认值，不会影响整帧写入。
"""

from __future__ import annotations

import os
import time
import sqlite3
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from colorama import init, Fore, Style

from config.config_center import get_system_config
from data_core import MarketHunter
from bus.zmq_bus import get_zmq_bus
from bus.event_schema import EventTopic, make_market_bar_1m_event

# 颜色初始化
init(autoreset=True)

DB_FILE = "market_ts.db"


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    """尽量把 val 转成 float，失败或 NaN 时返回 default。"""
    try:
        x = float(val)
        if x != x:  # NaN
            return default
        return x
    except Exception:
        return default


# ----------------------------------------------------------------------
# 主类
# ----------------------------------------------------------------------


class MarketTSCollector:
    def __init__(
        self,
        db_path: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        enable_bus: Optional[bool] = None,
        max_rows_per_frame: Optional[int] = None,
        system_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param db_path: SQLite 数据库路径，默认 ./market_ts.db
        :param interval_seconds: 采样间隔（秒），默认从 system_config 读取，缺省为 60 秒
        :param enable_bus: 是否启用 ZMQ 总线，默认跟随 system_config["event_bus"]["enabled"]
        :param max_rows_per_frame: 每帧最多写入多少只股票（按成交额排序），None 表示不限
        """
        sys_cfg = system_config or get_system_config()
        col_cfg = (sys_cfg.get("collector") or {}).get("market_ts") or {}

        # DB 路径
        if db_path is None:
            db_path = (
                col_cfg.get("db_path")
                or col_cfg.get("ts_db_path")
                or DB_FILE
            )
        self.db_path: str = db_path

        # 周期
        if interval_seconds is None:
            interval_seconds = int(col_cfg.get("interval_seconds", 60))
        self.interval_seconds: int = int(interval_seconds)

        # 是否启用总线
        if enable_bus is None:
            bus_enabled_cfg = (sys_cfg.get("event_bus") or {}).get("enabled")
            enable_bus = True if bus_enabled_cfg is None else bool(bus_enabled_cfg)
        self.enable_bus: bool = bool(enable_bus)

        # 每帧最多写多少行
        if max_rows_per_frame is None:
            max_rows_per_frame = int(col_cfg.get("max_rows_per_frame", 200))
        self.max_rows_per_frame: Optional[int] = int(max_rows_per_frame) if max_rows_per_frame > 0 else None

        # 依赖
        self.hunter = MarketHunter()
        self.conn: Optional[sqlite3.Connection] = None
        self.bus = get_zmq_bus(sys_cfg) if self.enable_bus else None

        self.source_name: str = "market_ts_collector"
        # 记录已经提示过“可选字段缺失”的字段名，只提示一次
        self._missing_optional_logged: set[str] = set()

        self._init_db()

    # ------------------------------------------------------------------
    # DB 初始化
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()

        # 简单的 1 分钟快照表
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_1m (
                trade_date      TEXT NOT NULL,
                ts              TEXT NOT NULL,
                code            TEXT NOT NULL,
                name            TEXT NOT NULL,
                price           REAL NOT NULL,
                pct             REAL NOT NULL,
                turnover_rate   REAL NOT NULL,
                amount          REAL NOT NULL,
                float_mkt       REAL NOT NULL,
                force           REAL NOT NULL,
                vwap            REAL NOT NULL,
                ba_ratio        REAL NOT NULL,
                PRIMARY KEY (trade_date, ts, code)
            );
            """
        )
        self.conn.commit()
        print(
            Fore.GREEN
            + f"[MTS] market_ts.db connected (snapshot_1m ready)."
            + Style.RESET_ALL
        )

    # ------------------------------------------------------------------
    # 交易时段判断（主循环里用，测试脚本可以直接调 _capture_once 不走这里）
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trading_time() -> bool:
        """
        简单判断是否在 A 股交易时段：
        - 09:30 - 11:30
        - 13:00 - 15:00
        """
        now = datetime.datetime.now().time()
        if datetime.time(9, 30) <= now <= datetime.time(11, 30):
            return True
        if datetime.time(13, 0) <= now <= datetime.time(15, 0):
            return True
        return False

    # ------------------------------------------------------------------
    # 把 df 中的一帧快照写入 snapshot_1m
    # ------------------------------------------------------------------

    def _capture_once(self) -> int:
        """
        抓一帧行情快照并写入 snapshot_1m。
        返回成功写入的行数。
        """
        if self.conn is None:
            return 0

        now = datetime.datetime.now()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        trade_date = ts_str.split(" ")[0]

        # 1) 抓一帧
        df: pd.DataFrame = self.hunter.fetch_snapshot()
        if df is None or df.empty:
            print(
                Fore.YELLOW
                + f"[MTS] {ts_str} snapshot empty, skip."
                + Style.RESET_ALL
            )
            return 0

        cols = set(df.columns)

        # 2) 必选字段（抓不到就整帧跳过）
        required_cols: Sequence[str] = [
            "代码",
            "名称",
            "现价",
            "涨幅",
            "换手率",
            "成交额",
        ]
        missing_required = [c for c in required_cols if c not in cols]
        if missing_required:
            print(
                Fore.RED
                + "[MTS][WARN] required column(s) missing: "
                + ",".join(missing_required)
                + " , skip this frame."
                + Style.RESET_ALL
            )
            return 0

        # 3) 可选字段：缺了就补默认值，不影响写入
        optional_defaults: Dict[str, float] = {
            "流通市值": 0.0,
            "主力攻击系数": 0.0,
            "VWAP": None,      # 用现价兜底
            "BA_Ratio": 1.0,
        }

        for col, default in optional_defaults.items():
            if col not in cols:
                if col not in self._missing_optional_logged:
                    msg_default = default if default is not None else "price"
                    print(
                        Fore.YELLOW
                        + f"[MTS][WARN] optional column missing: {col}, fill with {msg_default}."
                        + Style.RESET_ALL
                    )
                    self._missing_optional_logged.add(col)
                # 真正补列：VWAP 用现价兜底，其它用常数
                if col == "VWAP":
                    df[col] = df["现价"]
                else:
                    df[col] = default

        # 4) 整理 / 选前 N 名
        df_use = df.copy()
        if "成交额" in df_use.columns:
            df_use = df_use.sort_values(by="成交额", ascending=False)
        if self.max_rows_per_frame:
            df_use = df_use.head(self.max_rows_per_frame)

        rows: List[Tuple[Any, ...]] = []
        for _, row in df_use.iterrows():
            try:
                code = str(row["代码"]).strip()
                name = str(row["名称"]).strip()
                if not code:
                    continue

                price = _safe_float(row["现价"], 0.0)
                pct = _safe_float(row["涨幅"], 0.0)
                turnover_rate = _safe_float(row["换手率"], 0.0)
                amount = _safe_float(row["成交额"], 0.0)

                float_mkt = _safe_float(row.get("流通市值", 0.0), 0.0)
                force = _safe_float(row.get("主力攻击系数", 0.0), 0.0)
                vwap = _safe_float(row.get("VWAP", price), price or 0.0)
                ba_ratio = _safe_float(row.get("BA_Ratio", 1.0), 1.0)

                rows.append(
                    (
                        trade_date,
                        ts_str,
                        code,
                        name,
                        price,
                        pct,
                        turnover_rate,
                        amount,
                        float_mkt,
                        force,
                        vwap,
                        ba_ratio,
                    )
                )
            except Exception:
                # 单行有问题就跳过，不影响其它行
                continue

        if not rows:
            print(
                Fore.YELLOW
                + f"[MTS] {ts_str} snapshot has no valid rows after cleaning, skip commit."
                + Style.RESET_ALL
            )
            return 0

        # 5) 批量写入 SQLite
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO snapshot_1m (
                trade_date, ts, code, name,
                price, pct, turnover_rate, amount,
                float_mkt, force, vwap, ba_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        self.conn.commit()

        print(
            Fore.GREEN
            + f"[MTS] {ts_str} captured {len(rows)} rows -> {self.db_path}.snapshot_1m"
            + Style.RESET_ALL
        )

        # 6) 可选：发布 1m bar 事件
        if self.enable_bus:
            try:
                self._publish_bar_events(now, trade_date, df_use)
            except Exception:
                # 总线只是增强功能，异常不影响采集
                pass

        return len(rows)

    # ------------------------------------------------------------------
    # ZMQ Bus 发布 1m bar 事件（可选）
    # ------------------------------------------------------------------

    def _publish_bar_events(
        self,
        now: datetime.datetime,
        trade_date: str,
        df_use: pd.DataFrame,
    ) -> None:
        if self.bus is None or df_use.empty:
            return

        for _, row in df_use.iterrows():
            try:
                code = str(row["代码"]).strip()
                name = str(row["名称"]).strip()
                if not code:
                    continue

                price = _safe_float(row["现价"], 0.0)
                pct = _safe_float(row["涨幅"], 0.0)
                turnover_rate = _safe_float(row["换手率"], 0.0)
                amount = _safe_float(row["成交额"], 0.0)

                float_mkt = _safe_float(row.get("流通市值", 0.0), 0.0)
                force = _safe_float(row.get("主力攻击系数", 0.0), 0.0)
                vwap = _safe_float(row.get("VWAP", price), price or 0.0)
                ba_ratio = _safe_float(row.get("BA_Ratio", 1.0), 1.0)

                evt = make_market_bar_1m_event(
                    source=self.source_name,
                    symbol=code,
                    name=name,
                    bar_ts=now,
                    trade_date=trade_date,
                    price=price,
                    pct=pct,
                    turnover_rate=turnover_rate,
                    amount=amount,
                    float_mkt=float_mkt,
                    force=force,
                    vwap=vwap,
                    ba_ratio=ba_ratio,
                )
                self.bus.publish(EventTopic.MARKET_BAR_1M, evt)
            except Exception:
                # 单个事件出错不影响其它
                continue

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(
            Fore.CYAN
            + "############################################\n"
            + "#   LIGHT HUNTER - Market TS Collector 1m  #\n"
            + f"#   Target DB : {self.db_path}              #\n"
            + "############################################"
            + Style.RESET_ALL
        )
        print(
            Fore.GREEN
            + f"[MTS] Interval = {self.interval_seconds}s, ZMQ = {'ON' if self.enable_bus else 'OFF'}"
            + Style.RESET_ALL
        )

        try:
            while True:
                if not self._is_trading_time():
                    print(
                        Fore.YELLOW
                        + "[MTS] Outside trading hours, sleeping 60s..."
                        + Style.RESET_ALL
                    )
                    time.sleep(self.interval_seconds)
                    continue

                self._capture_once()
                time.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            print(
                Fore.RED
                + "[MTS] Collector stopped by user."
                + Style.RESET_ALL
            )


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------


def main() -> None:
    collector = MarketTSCollector()
    collector.run()


if __name__ == "__main__":
    main()
