# -*- coding: utf-8 -*-
"""
模块名称：TSCollector Mk-Stream
版本：Mk-Stream Chronolog Mk2 (Time Vault + ZMQ Bus)
路径: ts_collector.py

功能：
- 定期调用 MarketHunter.fetch_snapshot() 抓取全市场实时快照；
- 通过 TSRecorder 将 [时间戳, 代码, 名称, 现价, 涨幅, 成交额, 换手率, 流通市值, 主力攻击系数, VWAP, BA_Ratio]
  写入本地 SQLite / TSStorage；
- 同时将每一帧快照按 symbol 拆分为逐股事件，通过 ZeroMQ 事件总线发布：
    * topic = "market.tick"
    * payload = {symbol, name, ts, price, pct, turnover_rate, amount, float_mkt,
                 force, vwap, ba_ratio}
- 为 SequenceBrain / RiskBrain / 在线策略提供“推送式”的实时行情流。
"""

from __future__ import annotations

import time
import datetime
from typing import Optional, Dict, Any

from colorama import init, Fore, Style

from config.config_center import get_system_config
from data_core import MarketHunter  # 数据猎手
from ts_recorder import TSRecorder  # 分时写入统一入口
from bus.zmq_bus import get_zmq_bus
from bus.event_schema import EventTopic


DB_FILE = "ts_data.db"


class TSCollector:
    def __init__(self, interval: Optional[int] = None, enable_bus: Optional[bool] = None):
        """
        :param interval:
            采样间隔（秒），默认为 20 秒一拍。
            - 若传入 None，则优先从 system_config["collector"]["ts"]["interval_seconds"] 读取；
              如果没有该配置，回退为 20。
        :param enable_bus:
            是否启用 ZeroMQ 事件发布：
            - None: 跟随 system_config["event_bus"]["enabled"]；
            - True/False: 强制开启/关闭。
        """
        sys_cfg = get_system_config()
        col_cfg = (sys_cfg.get("collector") or {}).get("ts") or {}

        if interval is None:
            interval = int(col_cfg.get("interval_seconds", 20))
        self.interval: int = int(interval)

        if enable_bus is None:
            bus_enabled_cfg = (sys_cfg.get("event_bus") or {}).get("enabled")
            enable_bus = True if bus_enabled_cfg is None else bool(bus_enabled_cfg)
        self.enable_bus: bool = bool(enable_bus)

        self.hunter = MarketHunter()
        # TSRecorder 内部已接入 TSStorage 双写，不需要在这里关心 DuckDB
        self.recorder = TSRecorder(db_path=DB_FILE)

        self.bus = get_zmq_bus(sys_cfg) if self.enable_bus else None
        self.source_name = "ts_collector"

    # ----------------------------- 交易时段判断 ----------------------------- #
    @staticmethod
    def _is_trading_time() -> bool:
        """
        简单判断是否在 A 股交易时段：
        - 09:30 - 11:30
        - 13:00 - 15:00
        """
        now = datetime.datetime.now().time()
        if datetime.time(9, 30) <= now < datetime.time(11, 30):
            return True
        if datetime.time(13, 0) <= now < datetime.time(15, 5):
            return True
        return False

    # ----------------------------- 发布 ZMQ 行情事件 ----------------------------- #
    def _publish_snapshot_events(self, now: datetime.datetime, df) -> None:
        """
        将当前快照按股票拆成逐股事件，通过 ZMQ 总线发布。

        Event 规范：
            topic   : "market.tick"
            payload : {
                "symbol": "sz000001",
                "name": "平安银行",
                "ts": "YYYY-MM-DD HH:MM:SS",
                "price": float,
                "pct": float,
                "turnover_rate": float,
                "amount": float,
                "float_mkt": float,
                "force": float,
                "vwap": float,
                "ba_ratio": float,
            }
        """
        if self.bus is None:
            return
        if df is None or df.empty:
            return

        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        cols = set(df.columns)

        def safe_float(row, col: str, default: float = 0.0) -> float:
            if col not in row or row[col] is None:
                return default
            try:
                v = float(row[col])
                if v != v:  # NaN
                    return default
                return v
            except Exception:
                return default

        for _, row in df.iterrows():
            try:
                code = str(row["代码"])
                if not code or code.lower() == "nan":
                    continue
            except Exception:
                continue

            name = str(row["名称"]) if "名称" in cols else ""

            payload: Dict[str, Any] = {
                "symbol": code,
                "name": name,
                "ts": ts_str,
                "price": safe_float(row, "现价", 0.0),
                "pct": safe_float(row, "涨幅", 0.0),
                "turnover_rate": safe_float(row, "换手率", 0.0),
                "amount": safe_float(row, "成交额", 0.0),
                "float_mkt": safe_float(row, "流通市值", 0.0),
                "force": safe_float(row, "主力攻击系数", 0.0),
                "vwap": safe_float(row, "VWAP", 0.0),
                "ba_ratio": safe_float(row, "BA_Ratio", 0.0),
            }

            try:
                self.bus.publish(
                    topic=EventTopic.MARKET_TICK.value,
                    payload=payload,
                    source=self.source_name,
                )
            except Exception as e:
                # 事件总线异常不要影响主流程写盘
                print(
                    Fore.RED
                    + f"[TS][WARN] publish tick event failed for {code}: {e}"
                    + Style.RESET_ALL
                )
                # 不 break，继续其他股票

    # ----------------------------- 抓取一拍并写入 DB ----------------------------- #
    def _capture_once(self) -> int:
        now = datetime.datetime.now()

        df = self.hunter.fetch_snapshot()
        if df is None or df.empty:
            print(
                Fore.RED
                + f"[TS] {now.strftime('%Y-%m-%d %H:%M:%S')} snapshot empty, will retry..."
                + Style.RESET_ALL
            )
            return 0

        # 必要列检查
        required_cols = ["代码", "名称", "现价", "涨幅", "成交额", "换手率"]
        for col in required_cols:
            if col not in df.columns:
                print(
                    Fore.RED
                    + f"[TS][WARN] column missing: {col}, skip this frame."
                    + Style.RESET_ALL
                )
                return 0

        # 交给 TSRecorder 写入
        try:
            n = self.recorder.save_frame(now, df, limit=None)
        except Exception as e:
            print(
                Fore.RED
                + f"[TS][ERROR] write snapshots failed: {e}"
                + Style.RESET_ALL
            )
            n = 0

        # 发布 ZMQ 行情事件（不影响写盘）
        if n > 0:
            self._publish_snapshot_events(now, df)

        return n

    # ----------------------------- 主循环 ----------------------------- #
    def run(self) -> None:
        print(
            Fore.CYAN
            + Style.BRIGHT
            + """
################################################
#        LIGHT HUNTER - TS Collector Mk2       #
#        Stream Snapshot + ZMQ Tick Bus        #
################################################
"""
            + Style.RESET_ALL
        )
        print(
            Fore.GREEN
            + f"[TS] Interval = {self.interval} seconds, DB = {DB_FILE}, ZMQ = {'ON' if self.enable_bus else 'OFF'}"
            + Style.RESET_ALL
        )

        try:
            while True:
                if not self._is_trading_time():
                    print(
                        Fore.YELLOW
                        + "[TS] Outside trading hours, sleeping 60s..."
                        + Style.RESET_ALL
                    )
                    time.sleep(60)
                    continue

                n = self._capture_once()
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    Fore.CYAN
                    + f"[TS] {now} captured {n} rows -> {DB_FILE}"
                    + Style.RESET_ALL
                )

                time.sleep(self.interval)
        except KeyboardInterrupt:
            print(
                Fore.RED
                + "\n[TS] Collector stopped by user."
                + Style.RESET_ALL
            )
        finally:
            try:
                self.recorder.close()
            except Exception:
                pass


if __name__ == "__main__":
    init(autoreset=True)
    collector = TSCollector(interval=20)  # 默认 20 秒一拍，可调
    collector.run()
