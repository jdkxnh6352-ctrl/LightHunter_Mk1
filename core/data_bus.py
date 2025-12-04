# -*- coding: utf-8 -*-
"""
core/data_bus.py

LightHunter DataBus - 统一数据总线 & Schema 定义。

设计目标：
- 为整个系统提供一个统一的数据总线（Event Bus）抽象；
- 对实时行情、订单流、风险告警、交易指令等流式事件定义标准化消息结构；
- 默认使用进程内总线（InProcessBus），可选通过 ZeroMQ 做跨进程桥接。

核心概念：

1. BusTopic
   - 数据总线上的“频道”，如 SNAPSHOT_L1 / ORDER_FLOW / FACTOR_SIGNAL / RISK_EVENT 等。

2. BusMessage
   - 单条消息的标准结构：topic / symbol / ts / payload / source / meta。

3. DataBus
   - 统一 API：
        - publish(topic, payload, **kwargs)
        - subscribe(topic, callback)
        - unsubscribe(subscription_id)
   - 后端实现：
        - InProcessBus: 纯 Python，在同一进程内用回调分发事件；
        - 可选 ZMQ Bridge: 如果配置开启且安装了 pyzmq，则 publish 时同时往 ZeroMQ PUB 口广播。

注意：
- ZeroMQ 为“增强选项”，如果本地未安装 pyzmq，会自动降级为纯内存实现；
- DataBus 只提供“控制面”与 Schema，不负责高频 tick-by-tick 的 nanosecond 级通讯，
  但对 A 股超短线策略通常是足够的。
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from core.logging_utils import get_logger
from config.config_center import get_system_config

logger = get_logger(__name__)

# 尝试导入 ZeroMQ，用作可选桥接
try:  # pragma: no cover
    import zmq  # type: ignore
    import zmq.asyncio  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore


class BusTopic(str, Enum):
    """数据总线 Topic 定义。

    可以根据需要继续扩展，但尽量保持前缀语义清晰：
      - SNAPSHOT_* : 行情快照相关
      - OF_*       : 订单流相关
      - FACTOR_*   : 因子/信号相关
      - RISK_*     : 风险管理相关
      - TRADE_*    : 交易指令/成交回报
      - SYS_*      : 系统级事件
      - SENT_*     : 舆情 / 情绪相关
    """

    # 行情 / 订单流
    SNAPSHOT_L1 = "SNAPSHOT_L1"          # L1 行情快照（tick 或秒级）
    SNAPSHOT_BAR = "SNAPSHOT_BAR"        # 聚合 K 线（1m / 5m 等）
    OF_EVENT = "OF_EVENT"                # 订单流事件（大单 / 冲击）

    # 因子 / 标签 / 信号
    FACTOR_ROW = "FACTOR_ROW"            # 单标的单时刻因子向量
    FACTOR_SIGNAL = "FACTOR_SIGNAL"      # 实盘打分信号
    LABEL_EVENT = "LABEL_EVENT"          # 训练用标签（可选）

    # 风控 / 交易
    RISK_EVENT = "RISK_EVENT"            # 风险告警，如“炸板预警”、“风控熔断”
    TRADE_SIGNAL = "TRADE_SIGNAL"        # 策略发出的交易意图（买入/卖出）
    TRADE_ORDER = "TRADE_ORDER"          # 已下单的指令
    TRADE_FILL = "TRADE_FILL"            # 成交回报

    # 舆情 / 情绪
    SENT_EVENT = "SENT_EVENT"            # 舆情事件（某股热度暴增/情绪极端）

    # 系统
    SYS_HEARTBEAT = "SYS_HEARTBEAT"      # 心跳
    SYS_LOG = "SYS_LOG"                  # 系统级日志事件


@dataclass
class BusMessage:
    """数据总线上的标准消息结构。"""

    topic: str
    symbol: Optional[str]
    ts: datetime
    payload: Dict[str, Any]
    source: str = "unknown"
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(data: Union[str, bytes]) -> "BusMessage":
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        obj = json.loads(data)
        ts = obj.get("ts")
        if isinstance(ts, str):
            obj["ts"] = datetime.fromisoformat(ts)
        return BusMessage(
            topic=obj.get("topic"),
            symbol=obj.get("symbol"),
            ts=obj.get("ts"),
            payload=obj.get("payload") or {},
            source=obj.get("source", "unknown"),
            meta=obj.get("meta") or {},
        )


# 订阅回调类型
SyncCallback = Callable[[BusMessage], None]
AsyncCallback = Callable[[BusMessage], Awaitable[None]]


@dataclass
class _Subscription:
    topic: str
    callback: Union[SyncCallback, AsyncCallback]
    is_async: bool
    name: str
    id: int


class InProcessBus:
    """进程内数据总线实现（默认后端）。

    特点：
    - 订阅是回调式的：publish 时立即在当前线程触发回调；
    - 支持同步 & 异步回调（异步回调会通过 asyncio.create_task 调度）；
    - 适合轻量的事件广播，重 CPU 逻辑建议在回调里再丢到线程池/协程处理。
    """

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._subs: Dict[str, List[_Subscription]] = {}
        self._lock = threading.RLock()
        self._next_id = 1

    # 订阅 / 退订 ---------------------------------------------------------
    def subscribe(
        self,
        topic: Union[str, BusTopic],
        callback: Union[SyncCallback, AsyncCallback],
        *,
        name: Optional[str] = None,
    ) -> _Subscription:
        topic_str = topic.value if isinstance(topic, BusTopic) else str(topic)
        with self._lock:
            sub = _Subscription(
                topic=topic_str,
                callback=callback,
                is_async=asyncio.iscoroutinefunction(callback),
                name=name or f"sub-{self._next_id}",
                id=self._next_id,
            )
            self._next_id += 1
            self._subs.setdefault(topic_str, []).append(sub)

        self._log.info(
            "DataBus subscribe: topic=%s, name=%s, id=%d",
            topic_str,
            sub.name,
            sub.id,
        )
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        with self._lock:
            lst = self._subs.get(sub.topic, [])
            self._subs[sub.topic] = [s for s in lst if s.id != sub.id]
        self._log.info(
            "DataBus unsubscribe: topic=%s, name=%s, id=%d",
            sub.topic,
            sub.name,
            sub.id,
        )

    # 发布 ---------------------------------------------------------------
    def publish(self, msg: BusMessage) -> None:
        """在当前线程里依次调用所有订阅者回调。"""
        topic = msg.topic
        with self._lock:
            subs = list(self._subs.get(topic, []))

        if not subs:
            # 这里不算错误，经常会有“广播没人听”的场景
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for sub in subs:
            cb = sub.callback
            try:
                if sub.is_async:
                    if loop is not None:
                        loop.create_task(cb(msg))  # type: ignore[arg-type]
                    else:
                        # 没有 event loop，退化为同步执行
                        asyncio.run(cb(msg))  # type: ignore[arg-type]
                else:
                    cb(msg)  # type: ignore[arg-type]
            except Exception:
                self._log.exception(
                    "DataBus subscriber 回调异常: topic=%s, name=%s, id=%d",
                    sub.topic,
                    sub.name,
                    sub.id,
                )


class ZmqBridge:
    """ZeroMQ 桥接器（可选）。

    模式：
    - publish: DataBus 本地 publish 时，同时往 PUB Socket 广播；
    - subscribe: 后台线程从 SUB Socket 收消息，转发到本地 InProcessBus。
    """

    def __init__(self, config: Dict[str, Any], inproc_bus: InProcessBus) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._enabled = False
        self._inproc_bus = inproc_bus

        if zmq is None:
            self._log.warning("未安装 pyzmq，ZmqBridge 自动禁用。")
            self._ctx = None
            self._pub_socket = None
            self._sub_socket = None
            self._sub_thread = None
            self._stop_event = threading.Event()
            return

        bus_cfg = config.get("data_bus", {}) or {}
        zmq_cfg = bus_cfg.get("zmq", {}) or {}
        self._pub_addr: str = zmq_cfg.get("pub_addr", "tcp://127.0.0.1:5555")
        self._sub_addr: str = zmq_cfg.get("sub_addr", "tcp://127.0.0.1:5556")
        self._mode: str = zmq_cfg.get("mode", "pub")  # pub / sub / both

        ctx = zmq.Context.instance()
        self._ctx = ctx
        self._pub_socket = None
        self._sub_socket = None
        self._sub_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self._mode in ("pub", "both"):
            self._pub_socket = ctx.socket(zmq.PUB)
            self._pub_socket.bind(self._pub_addr)
            self._log.info("ZmqBridge PUB 绑定地址: %s", self._pub_addr)

        if self._mode in ("sub", "both"):
            self._sub_socket = ctx.socket(zmq.SUB)
            self._sub_socket.connect(self._sub_addr)
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._log.info("ZmqBridge SUB 连接地址: %s", self._sub_addr)
            self._start_sub_thread()

        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    # PUB ----------------------------------------------------------------
    def publish(self, msg: BusMessage) -> None:
        if not self._enabled or self._pub_socket is None:
            return
        try:
            topic = msg.topic
            payload = msg.to_json().encode("utf-8")
            # 使用 multipart: [topic, json]
            self._pub_socket.send_multipart([topic.encode("utf-8"), payload])
        except Exception:
            self._log.exception("ZmqBridge.publish 发送失败。")

    # SUB ----------------------------------------------------------------
    def _start_sub_thread(self) -> None:
        if self._sub_socket is None:
            return

        def _worker() -> None:
            self._log.info("ZmqBridge SUB 线程启动。")
            while not self._stop_event.is_set():
                try:
                    parts = self._sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:  # type: ignore[attr-defined]
                    self._stop_event.wait(0.01)
                    continue
                except Exception:
                    self._log.exception("ZmqBridge SUB 接收失败。")
                    continue

                if len(parts) != 2:
                    continue
                topic_b, payload_b = parts
                try:
                    msg = BusMessage.from_json(payload_b)
                    # ZMQ 过来的消息统一标记 source
                    if not msg.source:
                        msg.source = "zmq"
                    self._inproc_bus.publish(msg)
                except Exception:
                    self._log.exception("ZmqBridge SUB 消息处理失败。")

            self._log.info("ZmqBridge SUB 线程退出。")

        th = threading.Thread(target=_worker, name="ZmqBridge-SUB", daemon=True)
        th.start()
        self._sub_thread = th

    def close(self) -> None:
        self._stop_event.set()
        if getattr(self, "_sub_thread", None) is not None:
            self._sub_thread.join(timeout=1.0)
        try:
            if self._pub_socket is not None:
                self._pub_socket.close(0)
            if self._sub_socket is not None:
                self._sub_socket.close(0)
        except Exception:
            pass
        if getattr(self, "_ctx", None) is not None:
            try:
                self._ctx.term()
            except Exception:
                pass


class DataBus:
    """数据总线统一入口（单例）。

    - 默认使用 InProcessBus；
    - 如果配置 data_bus.zmq.enabled=true 且安装了 pyzmq，则启用 ZmqBridge。
    """

    _instance: Optional["DataBus"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._cfg = get_system_config()
        self._bus = InProcessBus()

        bus_cfg = self._cfg.get("data_bus", {}) or {}
        zmq_enabled = bool(bus_cfg.get("zmq", {}).get("enabled", False))
        self._zmq_bridge: Optional[ZmqBridge] = None
        if zmq_enabled:
            try:
                self._zmq_bridge = ZmqBridge(self._cfg, self._bus)
                if not self._zmq_bridge.enabled:
                    self._zmq_bridge = None
            except Exception:
                self._log.exception("初始化 ZmqBridge 失败，自动降级为纯内存总线。")

        self._log.info(
            "DataBus 初始化完成。backend=inproc, zmq_bridge=%s",
            bool(self._zmq_bridge),
        )

    # 单例 ---------------------------------------------------------------
    @classmethod
    def instance(cls) -> "DataBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DataBus()
        return cls._instance

    # API 封装 -----------------------------------------------------------
    def subscribe(
        self,
        topic: Union[str, BusTopic],
        callback: Union[SyncCallback, AsyncCallback],
        *,
        name: Optional[str] = None,
    ) -> _Subscription:
        return self._bus.subscribe(topic, callback, name=name)

    def unsubscribe(self, sub: _Subscription) -> None:
        self._bus.unsubscribe(sub)

    def publish(
        self,
        topic: Union[str, BusTopic],
        payload: Dict[str, Any],
        *,
        symbol: Optional[str] = None,
        ts: Optional[datetime] = None,
        source: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
    ) -> BusMessage:
        """构造 BusMessage 并广播到总线（本地 + 可选 ZMQ）。"""
        topic_str = topic.value if isinstance(topic, BusTopic) else str(topic)
        msg = BusMessage(
            topic=topic_str,
            symbol=symbol,
            ts=ts or datetime.utcnow(),
            payload=payload,
            source=source,
            meta=meta or {},
        )

        # 本地 in-process 分发
        self._bus.publish(msg)

        # 可选 ZMQ 广播
        if self._zmq_bridge is not None:
            self._zmq_bridge.publish(msg)

        return msg

    def close(self) -> None:
        if self._zmq_bridge is not None:
            self._zmq_bridge.close()


# 便捷函数 ---------------------------------------------------------------
def get_data_bus() -> DataBus:
    """获取全局 DataBus 实例。"""
    return DataBus.instance()


if __name__ == "__main__":  # pragma: no cover
    # 简单自测：启动一个订阅者，然后发两条消息
    bus = get_data_bus()

    def on_snapshot(msg: BusMessage) -> None:
        print("收到 SNAPSHOT_L1:", msg.symbol, msg.payload.get("last"))

    bus.subscribe(BusTopic.SNAPSHOT_L1, on_snapshot, name="demo-snapshot")

    bus.publish(
        BusTopic.SNAPSHOT_L1,
        payload={"last": 10.25, "volume": 1000},
        symbol="000001.SZ",
        source="self-test",
    )
