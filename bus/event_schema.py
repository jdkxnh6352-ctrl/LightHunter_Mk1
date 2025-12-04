# -*- coding: utf-8 -*-
"""
bus/event_schema.py

LightHunter Mk 系列 - 事件模型定义

统一定义在消息总线(ZMQ / 内存总线)上流转的核心事件：
- SignalEvent        : alpha / 模型信号
- OrderEvent         : 下单请求
- ExecutionEvent     : 成交流 / 订单状态
- MarketBar1mEvent   : 1 分钟行情 Bar 事件

并提供：
- 事件 <-> dict 的编解码工具
- 根据事件推导默认 topic 的工具
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

# ----------------------------------------------------------------------
# Topic 常量（字符串形式，给 ZMQ SUB / 配置文件用）
# ----------------------------------------------------------------------

TOPIC_SIGNAL = "alpha.signal"
TOPIC_ORDER = "trade.order"
TOPIC_EXECUTION = "trade.execution"
TOPIC_RISK = "risk.alert"

# 行情相关 topic（新加的）
TOPIC_MARKET_SNAPSHOT = "market.snapshot"   # 多标的快照（一般是原始 dict）
TOPIC_MARKET_TICK = "market.tick"          # 单股票 tick/snapshot 推送
TOPIC_MARKET_BAR_1M = "market.bar.1m"      # 单股票 1 分钟 bar 推送


class EventTopic(str, Enum):
    """
    统一的事件 Topic 枚举，方便 IDE 补全 & 避免手写字符串出错。

    外部用法：
        EventTopic.MARKET_TICK.value   ->  "market.tick"
    """

    SIGNAL = TOPIC_SIGNAL
    ORDER = TOPIC_ORDER
    EXECUTION = TOPIC_EXECUTION
    RISK = TOPIC_RISK

    MARKET_SNAPSHOT = TOPIC_MARKET_SNAPSHOT
    MARKET_TICK = TOPIC_MARKET_TICK
    MARKET_BAR_1M = TOPIC_MARKET_BAR_1M


# ----------------------------------------------------------------------
# 事件定义
# ----------------------------------------------------------------------


@dataclass
class SignalEvent:
    """
    模型/Alpha 产生的选股信号。
    """

    symbol: str                 # 股票代码，如 "000001.SZ"
    ts: str                     # ISO 时间戳字符串
    direction: str              # "BUY" / "SELL" / "FLAT"
    score: float                # 模型打分，越大越看多
    weight: float = 1.0         # 在组合中的权重建议（0~1）
    strategy_id: str = "default_ultrashort"
    model_id: Optional[str] = None
    job_id: Optional[str] = None
    horizon: Optional[str] = None   # 持有周期，如 "T+1", "intraday"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["__event__"] = "SignalEvent"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalEvent":
        data = dict(data)
        data.pop("__event__", None)
        return cls(**data)


@dataclass
class OrderEvent:
    """
    组合层输出的下单请求。
    """

    symbol: str                 # 股票代码
    side: str                   # "BUY" / "SELL"
    quantity: float             # 数量（股数/手数）
    order_type: str = "MKT"     # "MKT" 市价, "LMT" 限价
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"  # "DAY", "IOC", "FOK" ...
    account_id: Optional[str] = None
    strategy_id: str = "default_ultrashort"
    client_order_id: Optional[str] = None  # 业务自定义 ID
    ts: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["__event__"] = "OrderEvent"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderEvent":
        data = dict(data)
        data.pop("__event__", None)
        return cls(**data)


@dataclass
class ExecutionEvent:
    """
    成交 / 订单状态变更事件。
    """

    order_id: Optional[str]
    client_order_id: Optional[str]
    exec_id: Optional[str]
    symbol: str
    side: str
    filled_qty: float
    avg_price: float
    status: str                 # "NEW", "FILLED", "CANCELED", ...
    account_id: Optional[str] = None
    strategy_id: Optional[str] = None
    ts: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)  # Broker 原始返回，用于 debug

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["__event__"] = "ExecutionEvent"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionEvent":
        data = dict(data)
        data.pop("__event__", None)
        return cls(**data)


@dataclass
class MarketBar1mEvent:
    """
    单股票 1 分钟 Bar 事件。

    典型来源：MarketTSCollector 把分钟级行情写入本地库的同时，
    在总线上按 symbol 推送 1 分钟母流。
    """

    source: str                 # 事件来源标识，如 "MarketTSCollector"
    symbol: str                 # 股票代码，如 "000001.SZ"
    trading_date: str           # 交易日期 "YYYY-MM-DD"
    bar_ts: str                 # bar 时间戳，通常是该分钟起始时间
    open: float
    high: float
    low: float
    close: float
    volume: float               # 成交量
    amount: float               # 成交额
    payload: Dict[str, Any] = field(default_factory=dict)  # 额外字段：name/vwap 等

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["__event__"] = "MarketBar1mEvent"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketBar1mEvent":
        data = dict(data)
        data.pop("__event__", None)
        return cls(**data)


# ----------------------------------------------------------------------
# 工具函数：构造行情事件
# ----------------------------------------------------------------------


def make_market_bar_1m_event(
    *,
    source: str,
    symbol: str,
    trading_date: str,
    bar_ts: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    amount: float,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> MarketBar1mEvent:
    """
    方便从原始分钟线记录构造 MarketBar1mEvent。

    注意：用 open_ 参数避免和关键字冲突，内部会映射到字段 open。
    """
    evt = MarketBar1mEvent(
        source=str(source),
        symbol=str(symbol),
        trading_date=str(trading_date),
        bar_ts=str(bar_ts),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume),
        amount=float(amount),
    )
    if extra_payload:
        evt.payload.update(extra_payload)
    return evt


# ----------------------------------------------------------------------
# 编解码工具
# ----------------------------------------------------------------------

_EVENT_REGISTRY: Dict[str, Type[Any]] = {
    "SignalEvent": SignalEvent,
    "OrderEvent": OrderEvent,
    "ExecutionEvent": ExecutionEvent,
    "MarketBar1mEvent": MarketBar1mEvent,
}

T = TypeVar("T")


def event_to_dict(event: Any) -> Dict[str, Any]:
    """
    将事件对象转成 dict（附加 __event__ 字段）。
    """
    if hasattr(event, "to_dict"):
        return event.to_dict()  # type: ignore[no-any-return]
    raise TypeError(f"对象 {event!r} 不支持 to_dict()，不是合法事件")


def event_from_dict(data: Dict[str, Any]) -> Any:
    """
    从 dict 恢复为具体事件对象。
    """
    etype = data.get("__event__")
    if not etype:
        raise ValueError(f"dict 中缺少 __event__ 字段，无法识别事件类型: {data}")
    cls = _EVENT_REGISTRY.get(str(etype))
    if cls is None:
        raise ValueError(f"未知事件类型 __event__={etype!r}")
    return cls.from_dict(data)  # type: ignore[no-any-return]


def topic_for_event(event: Any) -> str:
    """
    根据事件类型选择默认 topic。
    ZMQBus.publish_event() 会用到这个函数。
    """
    if isinstance(event, SignalEvent):
        return EventTopic.SIGNAL.value
    if isinstance(event, OrderEvent):
        return EventTopic.ORDER.value
    if isinstance(event, ExecutionEvent):
        return EventTopic.EXECUTION.value
    if isinstance(event, MarketBar1mEvent):
        return EventTopic.MARKET_BAR_1M.value
    # 默认走风控 / 杂项
    return EventTopic.RISK.value


__all__ = [
    # topics
    "TOPIC_SIGNAL",
    "TOPIC_ORDER",
    "TOPIC_EXECUTION",
    "TOPIC_RISK",
    "TOPIC_MARKET_SNAPSHOT",
    "TOPIC_MARKET_TICK",
    "TOPIC_MARKET_BAR_1M",
    "EventTopic",
    # events
    "SignalEvent",
    "OrderEvent",
    "ExecutionEvent",
    "MarketBar1mEvent",
    # helpers
    "make_market_bar_1m_event",
    "event_to_dict",
    "event_from_dict",
    "topic_for_event",
]
