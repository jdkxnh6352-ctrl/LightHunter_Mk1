# -*- coding: utf-8 -*-
"""
tools/lighthunter_dashboard.py

LightHunter Mk3 - 命令行 Dashboard
==================================

功能：
------
- 订阅 ZeroMQ 总线上的关键事件：
    - market.snapshot
    - alpha.signal
    - trade.order
    - trade.execution
- 在终端以表格方式实时展示系统状态：
    - 最近信号
    - 最近订单
    - 最近成交
    - 持仓 & 实现盈亏

用法：
------
    python -m tools.lighthunter_dashboard

依赖：
------
- 建议安装 rich 提升显示效果：
    pip install rich
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from config.config_center import get_system_config
from core.logging_utils import get_logger
from bus.zmq_bus import get_zmq_bus
from bus.event_schema import (
    SignalEvent,
    OrderEvent,
    ExecutionEvent,
    event_from_dict,
    TOPIC_SIGNAL,
    TOPIC_ORDER,
    TOPIC_EXECUTION,
)

log = get_logger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False
    Console = None  # type: ignore


@dataclass
class DashboardState:
    last_market: Dict[str, Any] = field(default_factory=dict)
    signals: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    orders: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    executions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    realized_pnl: float = 0.0
    start_ts: float = field(default_factory=time.time)
    last_update_ts: Optional[float] = None


state = DashboardState()


def _update_positions_from_execution(exec_evt: ExecutionEvent) -> None:
    symbol = exec_evt.symbol
    side = exec_evt.side.upper()
    filled = float(exec_evt.filled_qty)
    px = float(exec_evt.avg_price)

    if filled <= 0:
        return

    pos = state.positions.get(symbol) or {"qty": 0.0, "avg_cost": 0.0}
    qty = float(pos["qty"])
    avg_cost = float(pos["avg_cost"])

    if side == "BUY":
        new_qty = qty + filled
        if new_qty > 0:
            new_cost = (avg_cost * qty + px * filled) / new_qty if qty != 0 else px
        else:
            new_cost = 0.0
        pos["qty"] = new_qty
        pos["avg_cost"] = new_cost
    else:  # SELL
        new_qty = qty - filled
        realized = (px - avg_cost) * filled
        state.realized_pnl += realized
        pos["qty"] = new_qty
        pos["avg_cost"] = avg_cost if new_qty != 0 else 0.0

    state.positions[symbol] = pos


# ----------------------------------------------------------------------
# ZeroMQ 订阅循环
# ----------------------------------------------------------------------


def zmq_loop() -> None:
    cfg = get_system_config()
    bus = get_zmq_bus(cfg)
    if bus is None:
        log.error("Dashboard: ZMQBus 不可用，无法订阅事件。")
        return

    import zmq  # type: ignore

    topics = ["market.snapshot", TOPIC_SIGNAL, TOPIC_ORDER, TOPIC_EXECUTION]
    sub = bus.create_subscriber(topics)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    log.info("Dashboard: 开始订阅 ZeroMQ 事件 topics=%s", topics)

    while True:
        try:
            socks = dict(poller.poll(200))
            if sub not in socks:
                continue

            topic_bytes, payload_bytes = sub.recv_multipart()
            topic = topic_bytes.decode("utf-8")
            data = json.loads(payload_bytes.decode("utf-8"))

            state.last_update_ts = time.time()

            if topic == "market.snapshot":
                state.last_market = data

            elif topic == TOPIC_SIGNAL:
                evt = event_from_dict(data)
                if isinstance(evt, SignalEvent):
                    state.signals.appendleft(
                        {
                            "symbol": evt.symbol,
                            "ts": evt.ts,
                            "direction": evt.direction,
                            "score": evt.score,
                            "strategy_id": evt.strategy_id,
                        }
                    )

            elif topic == TOPIC_ORDER:
                evt = event_from_dict(data)
                if isinstance(evt, OrderEvent):
                    state.orders.appendleft(
                        {
                            "symbol": evt.symbol,
                            "side": evt.side,
                            "qty": evt.quantity,
                            "strategy_id": evt.strategy_id,
                            "ts": evt.ts,
                        }
                    )

            elif topic == TOPIC_EXECUTION:
                evt = event_from_dict(data)
                if isinstance(evt, ExecutionEvent):
                    state.executions.appendleft(
                        {
                            "symbol": evt.symbol,
                            "side": evt.side,
                            "filled_qty": evt.filled_qty,
                            "avg_price": evt.avg_price,
                            "status": evt.status,
                            "ts": evt.ts,
                        }
                    )
                    _update_positions_from_execution(evt)

        except Exception:
            log.exception("Dashboard: 处理 ZeroMQ 消息时发生异常。")
            time.sleep(1.0)


# ----------------------------------------------------------------------
# Rich Dashboard 渲染
# ----------------------------------------------------------------------


def _build_rich_layout() -> "Layout":  # type: ignore
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    return layout


def _render_header() -> "Panel":  # type: ignore
    elapsed = time.time() - state.start_ts
    last_update = (
        time.strftime("%H:%M:%S", time.localtime(state.last_update_ts))
        if state.last_update_ts
        else "N/A"
    )

    txt = (
        f"[bold cyan]LightHunter Mk3 Dashboard[/bold cyan]    "
        f"Uptime: {elapsed:6.1f}s    "
        f"Last Update: {last_update}    "
        f"Realized PnL: [bold {'green' if state.realized_pnl>=0 else 'red'}]{state.realized_pnl:,.2f}[/bold]"
    )
    return Panel(txt, style="white on black")


def _render_signals_table() -> "Table":  # type: ignore
    table = Table(title="Signals (Latest)", expand=True)
    table.add_column("TS", style="dim", width=19)
    table.add_column("Symbol")
    table.add_column("Dir")
    table.add_column("Score", justify="right")
    table.add_column("Strategy")

    for s in list(state.signals)[:15]:
        table.add_row(
            s.get("ts", ""),
            s.get("symbol", ""),
            s.get("direction", ""),
            f"{s.get('score', 0):.2f}",
            s.get("strategy_id", ""),
        )
    return table


def _render_exec_table() -> "Table":  # type: ignore
    table = Table(title="Executions (Latest)", expand=True)
    table.add_column("TS", style="dim", width=19)
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Filled", justify="right")
    table.add_column("AvgPx", justify="right")
    table.add_column("Status")

    for e in list(state.executions)[:15]:
        table.add_row(
            e.get("ts", ""),
            e.get("symbol", ""),
            e.get("side", ""),
            f"{e.get('filled_qty', 0):.0f}",
            f"{e.get('avg_price', 0):.3f}",
            e.get("status", ""),
        )
    return table


def _render_positions_table() -> "Table":  # type: ignore
    table = Table(title="Positions", expand=True)
    table.add_column("Symbol")
    table.add_column("Qty", justify="right")
    table.add_column("AvgCost", justify="right")

    for sym, p in state.positions.items():
        qty = p.get("qty", 0.0)
        if abs(qty) < 1e-6:
            continue
        table.add_row(sym, f"{qty:.0f}", f"{p.get('avg_cost', 0.0):.3f}")
    return table


def _render_orders_table() -> "Table":  # type: ignore
    table = Table(title="Orders (Latest)", expand=True)
    table.add_column("TS", style="dim", width=19)
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Strategy")

    for o in list(state.orders)[:10]:
        table.add_row(
            o.get("ts", ""),
            o.get("symbol", ""),
            o.get("side", ""),
            f"{o.get('qty', 0):.0f}",
            o.get("strategy_id", ""),
        )
    return table


def run_rich_dashboard() -> None:
    console = Console()
    layout = _build_rich_layout()

    with Live(layout, refresh_per_second=4, screen=True):
        while True:
            layout["header"].update(_render_header())
            layout["left"].update(_render_signals_table())
            layout["right"].update(
                Panel(
                    _render_exec_table(),
                    title="Trades / Positions",
                    subtitle="最近成交 & 持仓",
                )
            )
            time.sleep(0.25)


# ----------------------------------------------------------------------
# 简易文本版 Dashboard（无 rich 时）
# ----------------------------------------------------------------------


def run_plain_dashboard() -> None:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=== LightHunter Mk3 Dashboard (plain) ===")
        elapsed = time.time() - state.start_ts
        print(f"Uptime: {elapsed:6.1f}s   Realized PnL: {state.realized_pnl:,.2f}")
        print()

        print("[Signals]")
        for s in list(state.signals)[:10]:
            print(
                f"{s.get('ts','')}  {s.get('symbol',''):>10}  {s.get('direction',''):>4}  "
                f"score={s.get('score',0):6.2f}  strat={s.get('strategy_id','')}"
            )
        print()

        print("[Executions]")
        for e in list(state.executions)[:10]:
            print(
                f"{e.get('ts','')}  {e.get('symbol',''):>10}  {e.get('side',''):>4}  "
                f"filled={e.get('filled_qty',0):6.0f}  px={e.get('avg_price',0):.3f}  "
                f"status={e.get('status','')}"
            )
        print()

        print("[Positions]")
        for sym, p in state.positions.items():
            qty = p.get("qty", 0.0)
            if abs(qty) < 1e-6:
                continue
            print(f"{sym:>10}  qty={qty:8.0f}  avg_cost={p.get('avg_cost',0):.3f}")

        time.sleep(1.0)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    # 先启动 ZMQ 订阅循环线程
    import threading

    t = threading.Thread(target=zmq_loop, daemon=True)
    t.start()

    if _HAS_RICH:
        run_rich_dashboard()
    else:
        log.warning("未安装 rich，使用简易文本版 Dashboard。建议 pip install rich。")
        run_plain_dashboard()


if __name__ == "__main__":  # pragma: no cover
    main()
