# -*- coding: utf-8 -*-
"""
bus/zmq_bus.py

LightHunter Mk3 - ZeroMQ 总线封装
================================

统一封装：
- 发布事件：publish_event(event)
- 订阅事件：create_subscriber(topics) -> zmq.Socket

配置（system_config.json）中的 event_bus 段示例：
------------------------------------------------
"event_bus": {
  "backend": "zmq",
  "pub_endpoint": "tcp://127.0.0.1:5555",
  "sub_endpoint": "tcp://127.0.0.1:5555",
  "pub_mode": "connect",   // 或 "bind"，视你谁作为 server
  "sub_mode": "connect"
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from config.config_center import get_system_config
from core.logging_utils import get_logger

from bus.event_schema import event_to_dict, topic_for_event

log = get_logger(__name__)

try:
    import zmq  # type: ignore
except Exception as e:  # pragma: no cover
    zmq = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class ZMQBus:
    """
    ZeroMQ 事件总线的轻量封装。

    - 单例 Context
    - 惰性创建 PUB socket
    - 按需创建 SUB socket
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        if zmq is None:
            raise RuntimeError(
                f"未安装 pyzmq，无法使用 ZMQBus。原始错误: {_IMPORT_ERROR!r}\n"
                f"请先: pip install pyzmq"
            )

        self.cfg = cfg or get_system_config()
        eb_cfg = self.cfg.get("event_bus") or {}

        self.pub_endpoint: str = eb_cfg.get("pub_endpoint", "tcp://127.0.0.1:5555")
        self.sub_endpoint: str = eb_cfg.get("sub_endpoint", self.pub_endpoint)
        self.pub_mode: str = eb_cfg.get("pub_mode", "connect")  # "bind" / "connect"
        self.sub_mode: str = eb_cfg.get("sub_mode", "connect")  # "bind" / "connect"

        self._ctx = zmq.Context.instance()
        self._pub_socket: Optional["zmq.Socket"] = None

    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------

    def _get_pub_socket(self) -> "zmq.Socket":
        if self._pub_socket is not None:
            return self._pub_socket

        s = self._ctx.socket(zmq.PUB)
        if self.pub_mode == "bind":
            log.info("ZMQBus: PUB bind at %s", self.pub_endpoint)
            s.bind(self.pub_endpoint)
        else:
            log.info("ZMQBus: PUB connect to %s", self.pub_endpoint)
            s.connect(self.pub_endpoint)
        self._pub_socket = s
        return s

    def publish_raw(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        低级接口：直接发 topic + JSON dict。
        """
        sock = self._get_pub_socket()
        msg = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sock.send_multipart([topic.encode("utf-8"), msg])

    def publish_event(self, event: Any) -> None:
        """
        高级接口：自动根据事件类型选择 topic，并完成序列化。
        """
        topic = topic_for_event(event)
        data = event_to_dict(event)
        self.publish_raw(topic, data)

    # ------------------------------------------------------------------
    # Subscriber
    # ------------------------------------------------------------------

    def create_subscriber(self, topics: Iterable[str]) -> "zmq.Socket":
        """
        创建一个 SUB socket，并订阅给定 topics。

        使用方法：
            sub = bus.create_subscriber([TOPIC_SIGNAL])
            while True:
                topic, raw = sub.recv_multipart()
        """
        s = self._ctx.socket(zmq.SUB)
        if self.sub_mode == "bind":
            log.info("ZMQBus: SUB bind at %s", self.sub_endpoint)
            s.bind(self.sub_endpoint)
        else:
            log.info("ZMQBus: SUB connect to %s", self.sub_endpoint)
            s.connect(self.sub_endpoint)

        for t in topics:
            s.setsockopt_string(zmq.SUBSCRIBE, t)

        return s


# ----------------------------------------------------------------------
# 单例封装
# ----------------------------------------------------------------------

_BUS_SINGLETON: Optional[ZMQBus] = None


def get_zmq_bus(cfg: Optional[Dict[str, Any]] = None) -> ZMQBus:
    global _BUS_SINGLETON
    if _BUS_SINGLETON is None:
        _BUS_SINGLETON = ZMQBus(cfg)
    return _BUS_SINGLETON


__all__ = [
    "ZMQBus",
    "get_zmq_bus",
]
