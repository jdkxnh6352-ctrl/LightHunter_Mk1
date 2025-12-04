# -*- coding: utf-8 -*-
"""
strategy/online_listener.py

LightHunter Mk3 - 在线信号监听器
================================

职责：
------
- 从 ZeroMQ 总线订阅 alpha.signal
- 调用 PortfolioManager 将 SignalEvent 转换为若干 OrderEvent
- 调用 TradeCore 执行订单（进而通过 Broker 下单）

设计注意：
----------
- 不直接依赖具体 Broker，实现层只依赖：
    - PortfolioManager 接口（duck typing）
    - TradeCore.handle_order_event(OrderEvent)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional

from config.config_center import get_system_config
from core.logging_utils import get_logger

from bus.event_schema import (
    TOPIC_SIGNAL,
    SignalEvent,
    OrderEvent,
    event_from_dict,
)
from bus.zmq_bus import get_zmq_bus

log = get_logger(__name__)


class OnlineListener:
    """
    在线信号监听与路由器。

    参数：
        cfg                : system_config
        portfolio_manager  : 组合管理器实例（需要实现 on_signal / process_signal / generate_orders 等方法之一）
        trade_core         : TradeCore 实例（需要实现 handle_order_event(order_event)）
        bus                : ZMQBus 实例（可选，不传则自动 get_zmq_bus）

    使用方式：
        listener = OnlineListener(cfg, pm, trade_core)
        listener.run_forever()
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        portfolio_manager: Any = None,
        trade_core: Any = None,
        bus: Any = None,
        topics: Optional[Iterable[str]] = None,
    ) -> None:
        self.cfg = cfg or get_system_config()
        self.bus = bus or get_zmq_bus(self.cfg)
        self.pm = portfolio_manager
        self.trade_core = trade_core

        if self.pm is None:
            raise RuntimeError("OnlineListener 需要传入 portfolio_manager 实例。")
        if self.trade_core is None:
            raise RuntimeError("OnlineListener 需要传入 trade_core 实例。")

        self.topics = list(topics) if topics is not None else [TOPIC_SIGNAL]
        self.sub = self.bus.create_subscriber(self.topics)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run_forever(self, sleep_on_error: float = 1.0) -> None:
        log.info(
            "OnlineListener: 启动，订阅 topics=%s",
            ",".join(self.topics),
        )
        while True:
            try:
                topic_bytes, payload_bytes = self.sub.recv_multipart()
                topic = topic_bytes.decode("utf-8")
                data = json.loads(payload_bytes.decode("utf-8"))
                event = event_from_dict(data)

                if isinstance(event, SignalEvent):
                    self._handle_signal(event)
                else:
                    log.debug("OnlineListener: 收到非 SignalEvent，忽略: %r", event)
            except Exception:
                log.exception("OnlineListener: 处理消息时出现异常，稍后重试。")
                time.sleep(sleep_on_error)

    # ------------------------------------------------------------------
    # 信号处理
    # ------------------------------------------------------------------

    def _handle_signal(self, signal: SignalEvent) -> None:
        """
        处理单条 SignalEvent：
        - 调用 PortfolioManager 生成 OrderEvent 列表
        - 将每个 OrderEvent 交给 TradeCore 执行
        """
        log.info(
            "OnlineListener: 收到信号 symbol=%s dir=%s score=%.4f",
            signal.symbol,
            signal.direction,
            signal.score,
        )

        orders = self._generate_orders_from_pm(signal)
        if not orders:
            log.info(
                "OnlineListener: PortfolioManager 对该信号未生成订单，symbol=%s",
                signal.symbol,
            )
            return

        for od in orders:
            try:
                self.trade_core.handle_order_event(od)
            except Exception:
                log.exception(
                    "OnlineListener: TradeCore 处理订单失败 symbol=%s side=%s qty=%s",
                    od.symbol,
                    od.side,
                    od.quantity,
                )

    def _generate_orders_from_pm(self, signal: SignalEvent) -> List[OrderEvent]:
        """
        尝试调用 PortfolioManager 的各种可能接口，返回 OrderEvent 列表。
        约定优先级：
            1) on_signal(SignalEvent) -> OrderEvent/list[OrderEvent]/None
            2) process_signal(SignalEvent) -> 同上
            3) generate_orders(SignalEvent) -> 同上
        """
        pm = self.pm

        if hasattr(pm, "on_signal"):
            res = pm.on_signal(signal)  # type: ignore
        elif hasattr(pm, "process_signal"):
            res = pm.process_signal(signal)  # type: ignore
        elif hasattr(pm, "generate_orders"):
            res = pm.generate_orders(signal)  # type: ignore
        else:
            raise RuntimeError(
                "PortfolioManager 未实现 on_signal / process_signal / generate_orders 任一方法，"
                "请在 strategy/portfolio_manager.py 中补充。"
            )

        if res is None:
            return []

        if isinstance(res, OrderEvent):
            return [res]

        if isinstance(res, list):
            # 允许 PM 返回 dict 列表，这里自动转 OrderEvent
            orders: List[OrderEvent] = []
            for item in res:
                if isinstance(item, OrderEvent):
                    orders.append(item)
                elif isinstance(item, dict):
                    orders.append(OrderEvent.from_dict({**item, "__event__": "OrderEvent"}))
                else:
                    raise TypeError(f"PortfolioManager 返回了未知订单类型: {item!r}")
            return orders

        if isinstance(res, dict):
            # 单个 dict
            return [OrderEvent.from_dict({**res, "__event__": "OrderEvent"})]

        raise TypeError(f"PortfolioManager 返回了未知类型: {type(res)!r}")


__all__ = ["OnlineListener"]
