# -*- coding: utf-8 -*-
"""
ops/monitor_daemon.py

LightHunter Mk4 - MonitorDaemon (战役版)
======================================

在 Mk3 版本的基础上，保留原有功能，同时新增：

1. 持续汇总以下监控信息：
   - 关键 Topic 的数据延迟（latency）
   - 各组件错误计数（errors）
   - ZeroMQ 队列堆积情况（bus_queues）
   - RiskBrain 推送的账户/标的风险（risk_accounts / risk_symbols）

2. 输出形式：
   - JSON Lines 指标文件：
       system_config["monitor"]["metrics_file"]
       默认：<monitor_dir>/metrics.jsonl
   - HUD 专用 summary JSON：
       system_config["monitor"]["summary_json"]
       默认：<monitor_dir>/state.json
     Web HUD 会定期读取该文件，在两周战役期间展示核心健康指标。

EventBus 约定：
---------------
订阅消息结构（示例）：

{
  "topic": "market.tick",
  "ts": 1672531200000,
  "payload": {
    ...
  }
}

risk.alert 结构参考 RiskBrain：
- risk.account_alert
- risk.symbol_alert
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # pragma: no cover
    from core.logging_utils import get_logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)


try:  # pragma: no cover
    from config.config_center import get_system_config
except Exception:  # pragma: no cover
    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        return {}


log = get_logger(__name__)


# 可选：ZeroMQ 总线客户端
try:  # pragma: no cover
    from bus.zmq_bus import ZmqBus  # type: ignore

    HAS_ZMQ_BUS = True
except Exception:  # pragma: no cover
    ZmqBus = None  # type: ignore
    HAS_ZMQ_BUS = False


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class TopicLatencyStats:
    topic: str
    count: int = 0
    last_latency: float = 0.0
    ewma_latency: float = 0.0
    max_latency: float = 0.0
    last_event_ts: float = 0.0  # 监控进程收到时间


@dataclass
class ErrorStats:
    component: str
    count: int = 0
    last_ts: float = 0.0
    last_message: str = ""


@dataclass
class BusQueueStats:
    endpoint: str
    last_ts: float = 0.0
    in_queue: int = 0
    out_queue: int = 0
    dropped: int = 0


@dataclass
class AccountRiskStats:
    account_id: str
    last_ts: float = 0.0
    last_score: float = 0.0
    last_level: str = "normal"
    alert_count: int = 0


@dataclass
class SymbolRiskStats:
    symbol: str
    account_id: str
    last_ts: float = 0.0
    last_score: float = 0.0
    last_level: str = "normal"
    alert_count: int = 0


# ----------------------------------------------------------------------
# MonitorDaemon 主体
# ----------------------------------------------------------------------


class MonitorDaemon:
    """
    监控守护进程。

    一般作为独立进程/脚本运行，通过 ZeroMQ EventBus 订阅：
    - metrics.* / market.tick / collector.*
    - risk.alert
    等 Topic。

    Mk4 升级点：
    ------------
    - 继续写 metrics.jsonl（用于战后复盘）
    - 新增写 state.json（HUD 直接读取，用于两周战役实时大屏）
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self.system_config = system_config or get_system_config()
        monitor_cfg = self.system_config.get("monitor", {}) or {}
        paths_cfg = self.system_config.get("paths", {}) or {}

        # monitor 目录（通过 paths.monitor_dir 配置）
        self.monitor_dir: str = paths_cfg.get(
            "monitor_dir",
            os.path.join("logs", "monitor"),
        )

        # 需要监控延迟的 topic 前缀
        self.latency_topics: List[str] = monitor_cfg.get(
            "latency_topics",
            ["market.tick", "collector.", "metrics.data.latency"],
        )

        # 错误相关 topic
        self.error_topics: List[str] = monitor_cfg.get(
            "error_topics",
            ["log.error", "metrics.error", "error.", "exception."],
        )

        # 队列 metrics topic
        self.queue_topics: List[str] = monitor_cfg.get(
            "queue_topics",
            ["metrics.bus.queue", "bus.queue_metrics"],
        )

        # 风险告警 topic
        self.risk_topic: str = monitor_cfg.get("risk_topic", "risk.alert")

        # 汇总频率（秒）
        self.summary_interval: float = float(
            monitor_cfg.get("summary_interval_sec", 5.0)
        )

        # 指标文件（JSONL）
        default_metrics_file = os.path.join(self.monitor_dir, "metrics.jsonl")
        self.metrics_file: str = monitor_cfg.get("metrics_file", default_metrics_file)

        # HUD summary JSON（state.json）
        default_summary_json = os.path.join(self.monitor_dir, "state.json")
        self.summary_json: str = monitor_cfg.get(
            "summary_json",
            default_summary_json,
        )

        # 日志前缀
        self.log_prefix: str = monitor_cfg.get("log_prefix", "MONITOR SUMMARY")

        # 内部状态
        self._latency_stats: Dict[str, TopicLatencyStats] = {}
        self._error_stats: Dict[str, ErrorStats] = {}
        self._queue_stats: Dict[str, BusQueueStats] = {}
        self._risk_accounts: Dict[str, AccountRiskStats] = {}
        self._risk_symbols: Dict[str, SymbolRiskStats] = {}

        self._lock = threading.Lock()

        self.event_bus = event_bus

        log.info(
            "MonitorDaemon 初始化完成：latency_topics=%s, error_topics=%s, "
            "queue_topics=%s, risk_topic=%s, metrics_file=%s, summary_json=%s",
            self.latency_topics,
            self.error_topics,
            self.queue_topics,
            self.risk_topic,
            self.metrics_file,
            self.summary_json,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def attach_bus(self, event_bus: Any) -> None:
        """
        附着到一个 EventBus。要求：
        - 支持 subscribe(topic, handler) 接口；
        - handler 接收一个 dict 消息。
        """
        self.event_bus = event_bus

    def start(self) -> None:
        """
        启动监控守护进程：
        - 注册总线订阅；
        - 启动汇总线程；
        - 当前线程阻塞（简单 sleep）。
        """
        self._ensure_dirs()

        if self.event_bus is not None and hasattr(self.event_bus, "subscribe"):
            # 订阅风险告警
            self.event_bus.subscribe(self.risk_topic, self._on_bus_message)

            # 订阅 queue metrics
            for t in self.queue_topics:
                self.event_bus.subscribe(t, self._on_bus_message)

            # 订阅数据延迟 / 错误相关（用前缀匹配，具体组件可以在 bus 层做 fanout）
            for t in self.latency_topics:
                self.event_bus.subscribe(t, self._on_bus_message)
            for t in self.error_topics:
                self.event_bus.subscribe(t, self._on_bus_message)

            log.info("MonitorDaemon: 已注册 EventBus 订阅。")
        else:
            log.warning("MonitorDaemon: 未绑定 event_bus，无法订阅实时事件，只能输出空汇总。")

        # 启动汇总线程
        t = threading.Thread(
            target=self._summary_loop,
            name="MonitorSummary",
            daemon=True,
        )
        t.start()

        # 阻塞主线程（守护）
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            log.info("MonitorDaemon: 收到 KeyboardInterrupt，退出。")

    # ------------------------------------------------------------------
    # EventBus 消息处理
    # ------------------------------------------------------------------

    def _on_bus_message(self, msg: Dict[str, Any]) -> None:
        """
        总线消息统一入口。

        约定消息结构：
        {
          "topic": "market.tick",
          "ts": 1672531200000,
          "payload": {...}
        }
        也支持直接 payload 即消息本身（没有外层 topic）。
        """
        try:
            topic = str(msg.get("topic") or msg.get("type") or "")
            payload = msg.get("payload", msg)

            # 数据延迟统计
            if self._is_latency_topic(topic):
                self._update_latency(topic, payload)

            # 错误率统计
            if self._is_error_topic(topic, payload):
                self._update_error(payload)

            # 队列堆积
            if self._is_queue_topic(topic):
                self._update_queue_metrics(payload)

            # 风险指标
            if self._is_risk_topic(topic, payload):
                self._update_risk_metrics(payload)

        except Exception as e:  # pragma: no cover
            log.warning("MonitorDaemon._on_bus_message 异常: %s, msg=%s", e, msg)

    # ------------------------------------------------------------------
    # 判定函数
    # ------------------------------------------------------------------

    def _is_latency_topic(self, topic: str) -> bool:
        for prefix in self.latency_topics:
            if topic.startswith(prefix):
                return True
        return False

    def _is_error_topic(self, topic: str, payload: Dict[str, Any]) -> bool:
        if str(payload.get("level", "")).lower() == "error":
            return True
        if "exception" in str(payload.get("type", "")).lower():
            return True
        for prefix in self.error_topics:
            if topic.startswith(prefix):
                return True
        return False

    def _is_queue_topic(self, topic: str) -> bool:
        for t in self.queue_topics:
            if topic.startswith(t):
                return True
        return False

    def _is_risk_topic(self, topic: str, payload: Dict[str, Any]) -> bool:
        if topic == self.risk_topic:
            return True
        tp = str(payload.get("type", ""))
        if tp.startswith("risk."):
            return True
        return False

    # ------------------------------------------------------------------
    # 指标更新
    # ------------------------------------------------------------------

    def _update_latency(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        基于 payload 中的 ts / timestamp 计算事件延迟。
        支持：
        - 毫秒时间戳（> 1e12）
        - 秒时间戳
        - 字符串时间（交给 pandas 解析）
        """
        now = time.time()
        ts_raw = payload.get("ts") or payload.get("timestamp") or payload.get("event_ts")
        if ts_raw is None:
            # 没有时间戳就无法算延迟
            return

        event_ts: float
        if isinstance(ts_raw, (int, float)):
            event_ts = float(ts_raw) / 1000.0 if ts_raw > 1e12 else float(ts_raw)
        else:
            try:
                event_ts = pd.to_datetime(ts_raw).timestamp()
            except Exception:
                return

        latency = max(0.0, now - event_ts)

        with self._lock:
            st = self._latency_stats.get(topic)
            if st is None:
                st = TopicLatencyStats(topic=topic)
                self._latency_stats[topic] = st

            st.count += 1
            st.last_latency = latency
            st.last_event_ts = now
            if st.count == 1:
                st.ewma_latency = latency
                st.max_latency = latency
            else:
                # EWMA, alpha=0.2
                st.ewma_latency = 0.8 * st.ewma_latency + 0.2 * latency
                st.max_latency = max(st.max_latency, latency)

    def _update_error(self, payload: Dict[str, Any]) -> None:
        comp = str(payload.get("component") or payload.get("source") or "unknown")
        msg = str(payload.get("message") or payload.get("error") or "")[:200]
        now = time.time()
        with self._lock:
            st = self._error_stats.get(comp)
            if st is None:
                st = ErrorStats(component=comp)
                self._error_stats[comp] = st
            st.count += 1
            st.last_ts = now
            st.last_message = msg

    def _update_queue_metrics(self, payload: Dict[str, Any]) -> None:
        """
        预期 payload 结构示例：
        {
          "endpoint": "tcp://127.0.0.1:5555",
          "in_queue": 100,
          "out_queue": 50,
          "dropped": 3
        }
        """
        endpoint = str(payload.get("endpoint") or payload.get("socket") or "unknown")
        in_q = int(payload.get("in_queue", 0))
        out_q = int(payload.get("out_queue", 0))
        dropped = int(payload.get("dropped", 0))
        now = time.time()

        with self._lock:
            st = self._queue_stats.get(endpoint)
            if st is None:
                st = BusQueueStats(endpoint=endpoint)
                self._queue_stats[endpoint] = st
            st.last_ts = now
            st.in_queue = in_q
            st.out_queue = out_q
            st.dropped = dropped

    def _update_risk_metrics(self, payload: Dict[str, Any]) -> None:
        """
        预期 risk.alert payload 结构（来自 RiskBrain._publish_alert）：

        - 标的告警：
          {
            "type": "risk.symbol_alert",
            "account_id": "...",
            "symbol": "...",
            "level": "warning"/"critical",
            "score": 0.83,
            ...
          }

        - 账户告警：
          {
            "type": "risk.account_alert",
            "account_id": "...",
            "level": "...",
            "score": 0.7,
            ...
          }
        """
        tp = str(payload.get("type", ""))
        now = time.time()

        if tp == "risk.account_alert":
            account_id = str(payload.get("account_id") or "unknown")
            score = float(payload.get("score", 0.0))
            level = str(payload.get("level", "normal"))

            with self._lock:
                st = self._risk_accounts.get(account_id)
                if st is None:
                    st = AccountRiskStats(account_id=account_id)
                    self._risk_accounts[account_id] = st
                st.last_ts = now
                st.last_score = score
                st.last_level = level
                st.alert_count += 1

        elif tp == "risk.symbol_alert":
            account_id = str(payload.get("account_id") or "unknown")
            symbol = str(payload.get("symbol") or "unknown")
            score = float(payload.get("score", 0.0))
            level = str(payload.get("level", "normal"))

            key = f"{account_id}/{symbol}"
            with self._lock:
                st = self._risk_symbols.get(key)
                if st is None:
                    st = SymbolRiskStats(symbol=symbol, account_id=account_id)
                    self._risk_symbols[key] = st
                st.last_ts = now
                st.last_score = score
                st.last_level = level
                st.alert_count += 1

    # ------------------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------------------

    def _summary_loop(self) -> None:
        while True:
            try:
                time.sleep(self.summary_interval)
                summary = self._build_summary()
                self._log_summary(summary)
                self._append_metrics_file(summary)
                self._write_summary_json(summary)
            except Exception as e:  # pragma: no cover
                log.warning("MonitorDaemon._summary_loop 异常: %s", e)

    def _build_summary(self) -> Dict[str, Any]:
        now = time.time()
        ts_iso = pd.to_datetime(now, unit="s").isoformat()

        with self._lock:
            latency_dict = {
                topic: {
                    "count": st.count,
                    "last_latency_ms": round(st.last_latency * 1000, 1),
                    "ewma_latency_ms": round(st.ewma_latency * 1000, 1),
                    "max_latency_ms": round(st.max_latency * 1000, 1),
                }
                for topic, st in self._latency_stats.items()
            }

            error_dict = {
                comp: {
                    "count": st.count,
                    "last_ts": pd.to_datetime(st.last_ts, unit="s").isoformat()
                    if st.last_ts
                    else None,
                    "last_message": st.last_message,
                }
                for comp, st in self._error_stats.items()
            }

            queue_dict = {
                ep: {
                    "last_ts": pd.to_datetime(st.last_ts, unit="s").isoformat()
                    if st.last_ts
                    else None,
                    "in_queue": st.in_queue,
                    "out_queue": st.out_queue,
                    "dropped": st.dropped,
                }
                for ep, st in self._queue_stats.items()
            }

            risk_accounts = {
                aid: {
                    "last_ts": pd.to_datetime(st.last_ts, unit="s").isoformat()
                    if st.last_ts
                    else None,
                    "last_score": round(st.last_score, 3),
                    "last_level": st.last_level,
                    "alert_count": st.alert_count,
                }
                for aid, st in self._risk_accounts.items()
            }

            # 找出风险最高的若干标的
            symbol_items = sorted(
                self._risk_symbols.values(),
                key=lambda x: x.last_score,
                reverse=True,
            )
            top_symbols = symbol_items[:10]
            risk_symbols = [
                {
                    "account_id": st.account_id,
                    "symbol": st.symbol,
                    "last_ts": pd.to_datetime(st.last_ts, unit="s").isoformat()
                    if st.last_ts
                    else None,
                    "last_score": round(st.last_score, 3),
                    "last_level": st.last_level,
                    "alert_count": st.alert_count,
                }
                for st in top_symbols
            ]

        return {
            "ts": ts_iso,
            "kind": "summary",
            "latency": latency_dict,
            "errors": error_dict,
            "bus_queues": queue_dict,
            "risk_accounts": risk_accounts,
            "risk_symbols": risk_symbols,
        }

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        latency_brief = {
            k: v["ewma_latency_ms"] for k, v in summary.get("latency", {}).items()
        }
        error_brief = {k: v["count"] for k, v in summary.get("errors", {}).items()}
        queue_brief = {
            k: {
                "in": v["in_queue"],
                "out": v["out_queue"],
                "drop": v["dropped"],
            }
            for k, v in summary.get("bus_queues", {}).items()
        }
        risk_acct_brief = {
            k: {"score": v["last_score"], "level": v["last_level"]}
            for k, v in summary.get("risk_accounts", {}).items()
        }

        log.info(
            "%s | t=%s | latency=%s | errors=%s | bus=%s | risk=%s",
            self.log_prefix,
            summary.get("ts"),
            latency_brief,
            error_brief,
            queue_brief,
            risk_acct_brief,
        )

    def _append_metrics_file(self, summary: Dict[str, Any]) -> None:
        try:
            line = json.dumps(summary, ensure_ascii=False)
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # pragma: no cover
            log.warning("MonitorDaemon._append_metrics_file 写文件失败: %s", e)

    def _write_summary_json(self, summary: Dict[str, Any]) -> None:
        """
        将最新一次 summary 写入 HUD 用的 state.json。
        """
        try:
            d = os.path.dirname(self.summary_json)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.summary_json + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.summary_json)
        except Exception as e:  # pragma: no cover
            log.warning(
                "MonitorDaemon._write_summary_json 写 summary_json 失败: %s", e
            )

    def _ensure_dirs(self) -> None:
        paths = set()
        d1 = os.path.dirname(self.metrics_file)
        if d1:
            paths.add(d1)
        d2 = os.path.dirname(self.summary_json)
        if d2:
            paths.add(d2)
        for d in paths:
            os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------


def main() -> None:  # pragma: no cover - CLI 不在单元测试覆盖范围
    cfg = get_system_config()
    bus = None

    if HAS_ZMQ_BUS and ZmqBus is not None:
        try:
            bus = ZmqBus.from_system_config(cfg)  # type: ignore[attr-defined]
            log.info(
                "MonitorDaemon.main: 已通过 ZmqBus.from_system_config 初始化 EventBus。"
            )
        except Exception as e:
            log.warning("MonitorDaemon.main: 初始化 ZmqBus 失败: %s", e)
            bus = None
    else:
        log.warning(
            "MonitorDaemon.main: 未找到 ZmqBus，MonitorDaemon 将在无总线模式下运行。"
        )

    daemon = MonitorDaemon(system_config=cfg, event_bus=bus)
    daemon.start()


if __name__ == "__main__":  # pragma: no cover
    main()
