# -*- coding: utf-8 -*-
"""
hud_core.py

HUDCore：TradeCore / PaperTrader 的命令行“头显”。

职责：
- 接收 PaperTrader.get_account_snapshot() 的字典快照；
- 可选接收 RiskBrain 的 RiskReport；
- 生成一段人类可读的 HUD 文本，并在终端打印。

使用示例：

    from trade_core import PaperTrader, OrderRequest
    from hud_core import print_hud

    trader = PaperTrader()
    # ... 下单若干笔 ...
    snapshot = trader.get_account_snapshot()
    risk_report = trader.get_risk_report()
    print_hud(snapshot, risk_report)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from core.logging_utils import get_logger

logger = get_logger(__name__)


def _fmt_num(x: Any, prec: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if abs(v) >= 1e8:
        return f"{v/1e8:.{prec}f}E"
    if abs(v) >= 1e4:
        return f"{v/1e4:.{prec}f}W"
    return f"{v:.{prec}f}"


def build_hud_text(
    snapshot: Dict[str, Any],
    risk_report: Optional[Any] = None,
) -> str:
    """根据账户快照 + 风险报告构建 HUD 文本。"""
    lines = []

    account_id = snapshot.get("account_id", "Paper")
    ts = snapshot.get("timestamp", "")
    cash = snapshot.get("cash", 0.0)
    equity = snapshot.get("equity", 0.0)
    realized = snapshot.get("realized_pnl", 0.0)
    unrealized = snapshot.get("unrealized_pnl", 0.0)
    n_pos = snapshot.get("n_positions", 0)

    lines.append(f"[Account] {account_id} @ {ts}")
    lines.append(
        f"  Equity: {_fmt_num(equity)}  Cash: {_fmt_num(cash)}  "
        f"Realized: {_fmt_num(realized)}  Unrealized: {_fmt_num(unrealized)}"
    )
    lines.append(f"  Positions: {n_pos}")

    # 风险信息
    if risk_report is not None:
        try:
            total_score = getattr(risk_report, "total_score", None)
            flags = getattr(risk_report, "flags", None)
            if total_score is None and isinstance(risk_report, dict):
                total_score = risk_report.get("total_score")
                flags = risk_report.get("flags", [])
        except Exception:  # pragma: no cover - 防守
            total_score = None
            flags = None

        if total_score is not None:
            lines.append(f"[RiskBrain] total_score={total_score:.2f}")
        if flags:
            lines.append(f"  flags: {', '.join(flags)}")

    # 持仓列表（最多打印前 10 条）
    positions = snapshot.get("positions", []) or []
    if positions:
        lines.append("")
        lines.append("Code      Qty       Cost       Px        MV        UPNL      RPNL")
        lines.append("-------- -------- -------- -------- ---------- ---------- ----------")
        for pos in positions[:10]:
            code = str(pos.get("code", ""))
            qty = pos.get("qty", 0)
            cost = pos.get("avg_cost", 0.0)
            px = pos.get("market_price", 0.0)
            mv = pos.get("market_value", 0.0)
            upnl = pos.get("unrealized_pnl", 0.0)
            rpnl = pos.get("realized_pnl", 0.0)
            lines.append(
                f"{code:<8} {qty:>8d} {_fmt_num(cost):>8} {_fmt_num(px):>8} "
                f"{_fmt_num(mv):>10} {_fmt_num(upnl):>10} {_fmt_num(rpnl):>10}"
            )

        if len(positions) > 10:
            lines.append(f"... ({len(positions) - 10} more positions)")

    return "\n".join(lines)


def print_hud(
    snapshot: Dict[str, Any],
    risk_report: Optional[Any] = None,
) -> None:
    """在终端打印 HUD 文本。"""
    text = build_hud_text(snapshot, risk_report=risk_report)
    print(text)
