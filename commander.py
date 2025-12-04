# commander.py

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import zmq  # 依赖 pyzmq

from config.config_center import get_system_config
from strategy.portfolio_manager import PortfolioManager, OrderPlan

logger = logging.getLogger(__name__)


class SignalSubscriber:
    """
    简单的 ZeroMQ 订阅器，订阅 alpha.signal 主题。

    实际工程中，你也可以用已有的 bus/zmq_bus.py 封装；如果你已经有 ZmqBus，
    可以把这里替换成对 ZmqBus 的调用。
    """

    def __init__(self, cfg: Mapping[str, Any], topic: str = "alpha.signal") -> None:
        self._cfg = cfg
        self._topic = topic.encode("utf-8")

        bus_cfg = cfg.get("event_bus", {}).get("zmq", {})
        sub_endpoint = bus_cfg.get("sub_endpoint", "tcp://127.0.0.1:5555")

        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.SUB)
        self._socket.connect(sub_endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)

        logger.info("SignalSubscriber 连接到 ZeroMQ: %s, 订阅主题=%s",
                    sub_endpoint, topic)

    def iter_events(self) -> Iterable[Dict[str, Any]]:
        """
        持续从 ZeroMQ 读消息，按 event(dict) 形式产出。

        约定消息格式为：b"{topic} {json_payload}"
        例如：
            b"alpha.signal {\"strategy_id\":\"U1\",\"signals\":[...]}"

        如果你本地的消息格式不同，可以在这里做适配。
        """
        while True:
            try:
                raw = self._socket.recv()
            except zmq.ZMQError as e:
                logger.error("ZeroMQ 订阅出错: %s", e)
                time.sleep(1)
                continue

            try:
                topic, payload = raw.split(b" ", 1)
            except ValueError:
                logger.warning("收到无法解析的消息（没有空格分隔 topic/payload）：%r", raw)
                continue

            if topic != self._topic:
                # 正常不会走到这里，因为我们已经用 SUBSCRIBE 过滤了
                continue

            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception as e:
                logger.warning("解析信号 JSON 失败: %s, payload=%r", e, payload)
                continue

            if not isinstance(data, dict):
                logger.warning("信号 payload 不是 dict 类型: %r", data)
                continue

            yield data


class Commander:
    """
    LightHunter 总指挥（事件驱动版）

    职责：
    - 从 ZeroMQ 订阅信号事件；
    - 根据 strategy_id 路由到 PortfolioManager；
    - 从 Broker 获取账户快照（权益 + 持仓）；
    - 生成并发送订单。
    """

    def __init__(
        self,
        system_config: Optional[Mapping[str, Any]] = None,
        broker: Optional[Any] = None,
    ) -> None:
        self._cfg = system_config or get_system_config()
        self._pm = PortfolioManager(self._cfg)

        # Broker：目前仍用 PaperBroker 仿真
        self._broker = broker or self._init_broker()

        # 信号订阅器
        self._signal_sub = SignalSubscriber(self._cfg, topic="alpha.signal")

    # ------------------------------------------------------------------
    # Broker 初始化 & 适配
    # ------------------------------------------------------------------

    def _init_broker(self) -> Any:
        """
        初始化 Broker。

        默认逻辑：
        - 尝试从 broker.broker_api 导入 PaperBroker；
        - 如果失败，就返回 None，此时 Commander 只打印订单计划，不真正发单。
        """
        try:
            from broker.broker_api import PaperBroker  # type: ignore
        except Exception as e:
            logger.warning(
                "导入 broker.broker_api.PaperBroker 失败：%s，"
                "Commander 将仅打印订单计划，不实际发单。",
                e,
            )
            return None

        trade_cfg = self._cfg.get("trade_core", {})
        broker_cfg = self._cfg.get("broker", {}).get("paper", {})

        # 这里假设 PaperBroker 可以用 (trade_cfg, broker_cfg) 初始化，
        # 如果你本地接口不同，可以在这里改成实际的构造方式。
        try:
            broker = PaperBroker(trade_cfg=trade_cfg, broker_cfg=broker_cfg)
        except TypeError:
            # 退而求其次，尝试只传一个 dict
            broker = PaperBroker(config=self._cfg)

        logger.info("PaperBroker 初始化完成。")
        return broker

    # ------------------------------------------------------------------
    # 账户快照
    # ------------------------------------------------------------------

    def _get_account_snapshot(
        self, account_id: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        从 Broker 获取账户净值和当前持仓（最小化假设，支持多种 Broker 实现）：

        返回：
        - equity: float
        - positions: Dict[symbol, qty]

        如果无法从 Broker 获取，则返回 (starting_cash, {}) 作为 fallback。
        """
        default_cash = float(
            self._cfg.get("trade_core", {}).get("starting_cash", 1_000_000.0)
        )

        if self._broker is None:
            logger.warning(
                "当前没有可用的 Broker 实例，使用 starting_cash=%.2f 且假定空仓。",
                default_cash,
            )
            return default_cash, {}

        # 优先尝试 get_account_snapshot(account_id=...)
        if hasattr(self._broker, "get_account_snapshot"):
            try:
                snap = self._broker.get_account_snapshot(account_id=account_id)
                equity = float(
                    snap.get("equity", snap.get("net_value", default_cash))
                )
                positions_raw = snap.get("positions", {})
                positions: Dict[str, float] = {}

                if isinstance(positions_raw, Mapping):
                    for sym, pos in positions_raw.items():
                        # 兼容 {symbol: qty} 或 {symbol: {"qty": ...}}
                        if isinstance(pos, Mapping):
                            qty = float(pos.get("qty", 0.0))
                        else:
                            qty = float(pos)
                        if abs(qty) > 0:
                            positions[str(sym)] = qty

                return equity, positions
            except Exception as e:
                logger.error(
                    "调用 broker.get_account_snapshot(account_id=%s) 失败：%s，"
                    "将回退到 starting_cash + 空仓。",
                    account_id,
                    e,
                )
                return default_cash, {}

        # 其次尝试 get_account() 或 account_state 等，尽量聪明一点：
        for attr in ("get_account", "get_account_state", "account_snapshot"):
            if hasattr(self._broker, attr):
                try:
                    snap = getattr(self._broker, attr)(account_id)
                    equity = float(
                        snap.get("equity", snap.get("net_value", default_cash))
                    )
                    positions = {
                        k: float(v)
                        for k, v in snap.get("positions", {}).items()
                    }
                    return equity, positions
                except Exception:
                    continue

        logger.warning(
            "Broker 对象不支持获取账户快照的方法，将使用 starting_cash + 空仓。"
        )
        return default_cash, {}

    # ------------------------------------------------------------------
    # 下单
    # ------------------------------------------------------------------

    def _send_order_plan(self, plan: OrderPlan) -> None:
        """
        把一个 OrderPlan 发给 Broker。如果 Broker 不可用，就只打印日志。
        """
        d = plan.to_dict()
        logger.info(
            "[ORDER PLAN] account=%s strategy=%s symbol=%s side=%s qty=%d",
            d["account_id"],
            d["strategy_id"],
            d["symbol"],
            d["side"],
            d["qty"],
        )

        if self._broker is None:
            # 没有真实 Broker，就只打印计划
            return

        # 尝试调用 submit_order(symbol=..., side=..., qty=..., account_id=..., ...)
        if hasattr(self._broker, "submit_order"):
            try:
                self._broker.submit_order(
                    symbol=d["symbol"],
                    side=d["side"],
                    qty=int(d["qty"]),
                    account_id=d["account_id"],
                    order_type=d.get("order_type", "MKT"),
                    price=d.get("price"),
                    extra=d.get("extra") or {},
                )
                return
            except TypeError as e:
                logger.error(
                    "调用 broker.submit_order(...) 失败（可能是参数不匹配）：%s。"
                    "请根据你本地 PaperBroker 的接口自行调整 Commander 中的调用。",
                    e,
                )
                return
            except Exception as e:
                logger.error("提交订单时发生异常：%s", e)
                return

        logger.warning(
            "Broker 对象没有 submit_order 方法，订单计划仅打印未发送。"
        )

    # ------------------------------------------------------------------
    # 信号处理
    # ------------------------------------------------------------------

    def _extract_batch_from_event(
        self, event: Mapping[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], datetime]:
        """
        从原始 event（通常是 ZeroMQ payload）中，提取：

        - strategy_id
        - signals: List[dict]
        - ts: datetime
        """
        strategy_id = str(event.get("strategy_id", "ultrashort_main"))

        if "signals" in event and isinstance(event["signals"], list):
            signals = list(event["signals"])
        else:
            # 兼容单条信号直接发过来的情况
            signals = [dict(event)]

        ts_raw = event.get("ts") or event.get("timestamp")
        if isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(ts_raw)
        elif isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
            except Exception:
                ts = datetime.utcnow()
        else:
            ts = datetime.utcnow()

        return strategy_id, signals, ts

    def on_signal_event(self, event: Mapping[str, Any]) -> None:
        """
        对外接口：处理一条“信号事件”。

        - 提取 strategy_id / signals；
        - 拉取账户快照；
        - 调用 PortfolioManager 生成订单计划；
        - 调用 Broker 发送订单。
        """
        strategy_id, signals, ts = self._extract_batch_from_event(event)
        cfg = self._pm.get_strategy_config(strategy_id)

        # 获取账户快照
        equity, positions = self._get_account_snapshot(cfg.account_id)

        # 构造价格表（优先从信号中拿 last_price）
        prices: Dict[str, float] = {}
        for s in signals:
            symbol = s.get("symbol") or s.get("code")
            if not symbol:
                continue
            lp = s.get("last_price") or s.get("price")
            if lp is not None:
                try:
                    prices[str(symbol)] = float(lp)
                except Exception:
                    continue

        # 调用组合管理
        plans = self._pm.plan_orders_for_signals(
            strategy_id=strategy_id,
            signals=signals,
            prices=prices,
            current_positions=positions,
            account_equity=equity,
            as_of=ts,
        )

        # 下单
        for p in plans:
            self._send_order_plan(p)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        logger.info("Commander 启动，开始订阅信号并路由订单...")
        for event in self._signal_sub.iter_events():
            try:
                self.on_signal_event(event)
            except Exception as e:
                logger.exception("处理信号事件时发生异常：%s, event=%r", e, event)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="LightHunter Commander - 信号路由 & 下单总控"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只处理一条从 stdin 读入的 JSON 信号（用于调试），而不是订阅 ZeroMQ。",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出更详细的日志。",
    )

    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)

    cfg = get_system_config()
    commander = Commander(system_config=cfg)

    if args.once:
        logger.info("Commander 以 once 模式运行，从 stdin 读取一条信号 JSON。")
        raw = input("请输入一条信号 JSON：\n")
        event = json.loads(raw)
        commander.on_signal_event(event)
    else:
        commander.run_forever()


if __name__ == "__main__":
    main()
