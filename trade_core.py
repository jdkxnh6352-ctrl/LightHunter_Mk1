# -*- coding: utf-8 -*-
"""
trade_core.py

LightHunter Mk3/Mk4 - TradeCore 交易执行核心
==========================================

职责：
------
- 接收 OrderEvent（策略/组合层产出）
- 调用 Broker 抽象接口（PaperBroker / ShadowBroker / 实盘 Broker）
- 根据 Broker 返回构造 ExecutionEvent
- 将 ExecutionEvent 发布到 ZMQ 总线，供 HUD / 风控 / 监控使用
"""

from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from config.config_center import get_system_config
from core.logging_utils import get_logger

from bus.event_schema import OrderEvent, ExecutionEvent
from bus.zmq_bus import get_zmq_bus
from broker.broker_api import get_default_broker

log = get_logger(__name__)


class TradeCore:
    """
    TradeCore 封装下单与成交回报逻辑。

    期望 Broker 抽象支持（duck typing 即可）：
        - snapshot_all_accounts()                          # DayOps/NightOps 用
        - submit_order(.) 或 place_order(.) 等方法        # 下单
          - 允许两种风格：
              a) submit_order(order_event: OrderEvent)
              b) submit_order(symbol=. side=. quantity=. .)

    你可以在 broker/broker_api.py 中调整 PaperBroker/ShadowBroker/实盘 Broker 以适配此接口。
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        broker: Any = None,
        bus: Any = None,
    ) -> None:
        self.cfg = cfg or get_system_config()
        self.bus = bus or get_zmq_bus(self.cfg)
        self.broker = broker or get_default_broker(self.cfg)

    # ------------------------------------------------------------------
    # 对外入口：处理单条订单事件
    # ------------------------------------------------------------------

    def handle_order_event(self, order: OrderEvent) -> None:
        """
        处理 PortfolioManager 产生的 OrderEvent，完成下单与回报。

        - 自动填充 client_order_id（如果为空）
        - 调用 Broker 提交订单
        - 构造 ExecutionEvent（如果 Broker 有返回）
        - 将 ExecutionEvent 广播到总线
        """
        if order.client_order_id is None:
            order.client_order_id = self._make_client_order_id(order)

        if order.ts is None:
            order.ts = datetime.utcnow().isoformat()

        log.info(
            "TradeCore: 收到订单 symbol=%s side=%s qty=%s type=%s limit=%s account=%s",
            order.symbol,
            order.side,
            order.quantity,
            order.order_type,
            order.limit_price,
            order.account_id,
        )

        # 调 Broker 下单
        result: Any = None
        try:
            result = self._submit_order_to_broker(order)
        except Exception:
            log.exception("TradeCore: 提交订单到 Broker 失败。")

        # 构造 ExecutionEvent，如果能拿到有效信息
        exec_evt = self._build_execution_event(order, result)
        if exec_evt is not None:
            try:
                self.bus.publish_event(exec_evt)
            except Exception:
                log.exception("TradeCore: 发布 ExecutionEvent 失败。")

    # ------------------------------------------------------------------
    # Broker 适配层
    # ------------------------------------------------------------------

    def _submit_order_to_broker(self, order: OrderEvent) -> Any:
        """
        Broker 接口适配层，尽量兼容多种实现风格。
        """
        b = self.broker

        # 1) submit_order(order_event)
        if hasattr(b, "submit_order"):
            try:
                return b.submit_order(order)  # type: ignore
            except TypeError:
                # 可能是关键字参数风格
                try:
                    return b.submit_order(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        limit_price=order.limit_price,
                        time_in_force=order.time_in_force,
                        account_id=order.account_id,
                        client_order_id=order.client_order_id,
                        strategy_id=order.strategy_id,
                        extra=order.extra,
                    )  # type: ignore
                except TypeError:
                    log.warning("Broker.submit_order 的签名不兼容，尝试下一种风格。")

        # 2) place_order(.)
        if hasattr(b, "place_order"):
            try:
                return b.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    time_in_force=order.time_in_force,
                    account_id=order.account_id,
                    client_order_id=order.client_order_id,
                    strategy_id=order.strategy_id,
                    extra=order.extra,
                )  # type: ignore
            except TypeError:
                log.warning("Broker.place_order 的签名不兼容。")

        # 3) submit_order_event(order_event) 之类
        for name in ("submit_order_event", "send_order_event"):
            if hasattr(b, name):
                fn = getattr(b, name)
                try:
                    return fn(order)  # type: ignore
                except Exception:
                    log.warning("Broker.%s(order_event) 调用失败。", name, exc_info=True)

        # 4) execute_order(dict) 兜底（某些 Broker 只实现了 execute_order）
        if hasattr(b, "execute_order"):
            try:
                payload = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": order.quantity,
                    "order_type": order.order_type,
                    "limit_price": order.limit_price,
                    "ref_price": getattr(order, "ref_price", None),
                    "account_id": order.account_id,
                    "strategy_id": order.strategy_id,
                    "order_id": order.client_order_id,
                    "time_in_force": order.time_in_force,
                    "meta": order.extra or {},
                    "ts": order.ts or datetime.utcnow().isoformat(),
                }
                return b.execute_order(payload)  # type: ignore
            except Exception:
                log.warning("Broker.execute_order(dict) 调用失败。", exc_info=True)

        raise RuntimeError(
            "Broker 实例不支持已知的下单接口（submit_order/place_order/.），"
            "请在 broker/broker_api.py 中适配 TradeCore。"
        )

    # ------------------------------------------------------------------
    # ExecutionEvent 构造
    # ------------------------------------------------------------------

    def _build_execution_event(
        self,
        order: OrderEvent,
        result: Any,
    ) -> Optional[ExecutionEvent]:
        """
        根据 Broker 返回尽量构造 ExecutionEvent。
        - 如果 result 为 None，则生成“已接受但未知状态”的占位事件。
        - 如果 result 是 dict 或具备相应属性，则尽量解析。
        """
        ts = datetime.utcnow().isoformat()

        # 默认值：假设订单已被接受但未成交
        status = "NEW"
        filled_qty = 0.0
        avg_price = 0.0
        order_id: Optional[str] = None
        exec_id: Optional[str] = None
        account_id = order.account_id
        strategy_id = order.strategy_id

        raw: Dict[str, Any] = {}

        if result is None:
            raw = {}
        elif isinstance(result, dict):
            raw = dict(result)
            # 状态字段：status 或 state
            status = raw.get("status") or raw.get("state") or status

            # 成交数量：优先 filled_qty，其次 fill_qty / filled_quantity / cum_qty
            if "filled_qty" in raw:
                try:
                    filled_qty = float(raw.get("filled_qty") or 0.0)
                except Exception:
                    filled_qty = 0.0
            elif "fill_qty" in raw:
                try:
                    filled_qty = float(raw.get("fill_qty") or 0.0)
                except Exception:
                    filled_qty = 0.0
            elif "filled_quantity" in raw:
                try:
                    filled_qty = float(raw.get("filled_quantity") or 0.0)
                except Exception:
                    filled_qty = 0.0
            elif "cum_qty" in raw:
                try:
                    filled_qty = float(raw.get("cum_qty") or 0.0)
                except Exception:
                    filled_qty = 0.0

            # 成交价格：优先 avg_price，其次 fill_price / price
            if "avg_price" in raw:
                try:
                    avg_price = float(raw.get("avg_price") or 0.0)
                except Exception:
                    avg_price = 0.0
            elif "fill_price" in raw:
                try:
                    avg_price = float(raw.get("fill_price") or 0.0)
                except Exception:
                    avg_price = 0.0
            elif "price" in raw:
                try:
                    avg_price = float(raw.get("price") or 0.0)
                except Exception:
                    avg_price = 0.0

            order_id = raw.get("order_id") or raw.get("id") or order.client_order_id
            exec_id = raw.get("exec_id")
            account_id = raw.get("account_id", account_id)
            strategy_id = raw.get("strategy_id", strategy_id)
        else:
            # 尝试从对象属性中提取信息
            raw = {"repr": repr(result)}
            for name in ("status", "state"):
                if hasattr(result, name):
                    status = getattr(result, name)
                    break
            for name in ("filled_qty", "fill_qty", "filled_quantity", "cum_qty"):
                if hasattr(result, name):
                    try:
                        filled_qty = float(getattr(result, name))
                    except Exception:
                        pass
                    break
            for name in ("avg_price", "fill_price", "price"):
                if hasattr(result, name):
                    try:
                        avg_price = float(getattr(result, name))
                    except Exception:
                        pass
                    break
            for name in ("order_id", "id"):
                if hasattr(result, name):
                    order_id = getattr(result, name)
                    break
            if hasattr(result, "exec_id"):
                exec_id = getattr(result, "exec_id")
            if hasattr(result, "account_id"):
                account_id = getattr(result, "account_id")
            if hasattr(result, "strategy_id"):
                strategy_id = getattr(result, "strategy_id")

        # 如果连 symbol 都没有，就别发了
        if not order.symbol:
            return None

        return ExecutionEvent(
            order_id=order_id,
            client_order_id=order.client_order_id,
            exec_id=exec_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=filled_qty,
            avg_price=avg_price,
            status=status,
            account_id=account_id,
            strategy_id=strategy_id,
            ts=ts,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _make_client_order_id(self, order: OrderEvent) -> str:
        """
        生成一个可追踪的 client_order_id。
        格式示例：
            ULTRA_000001.SZ_20250101T030000Z_abcd1234
        """
        base = order.strategy_id or "ULTRA"
        sym = (order.symbol or "").replace(".", "_")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        rnd = uuid.uuid4().hex[:8]
        return f"{base}_{sym}_{ts}_{rnd}"


__all__ = ["TradeCore"]
