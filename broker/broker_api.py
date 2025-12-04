# -*- coding: utf-8 -*-
"""
broker_api.py

Broker 抽象接口 + PaperBroker + ShadowBroker（影子券商）

统一的订单 dict 规范
-------------------
Order dict（TradeCore -> Broker）：
{
  "order_id": "<可选，client_order_id>",
  "account_id": "paper_main",
  "strategy_id": "ultrashort_main",
  "symbol": "000001.SZ",
  "side": "BUY" | "SELL",
  "qty": 1000.0,
  "order_type": "MKT" | "LIMIT",
  "limit_price": 10.25 或 null,
  "ref_price": 10.20,          # 建议带上，便于 PaperBroker/MKT 定价
  "reason": "portfolio_auto",
  "meta": {.},                 # 任意附加信息
  "ts": "2025-11-26T09:35:01"
}

成交 dict（Broker -> TradeCore -> ZeroMQ -> 其他模块）：
{
  "order_id": "<optional>",
  "account_id": "paper_main",
  "strategy_id": "ultrashort_main",
  "symbol": "000001.SZ",
  "side": "BUY",
  "status": "filled" | "rejected" | "partial",
  "fill_qty": 1000.0,
  "fill_price": 10.26,
  "fee": 12.3,
  "ts": "2025-11-26T09:35:01",
  "reason": "",
  "position_after": {
    "qty": 2000.0,
    "avg_price": 10.20
  },
  "meta": {
    "order": {.},
    "nav_after": 998000.0,
    "cash_after": 500000.0,
    "shadow": {...}    # ShadowBroker 额外打的标记（如果有）
  }
}

配置约定（system_config.json 示例）
----------------------------------
"broker": {
  "type": "paper",          # 也可以为 "shadow"
  "paper": {
    "base_currency": "CNY",
    "starting_cash": 1000000.0,
    "commission_bps": 1.0,
    "slippage_bps": 0.5
  },
  "shadow": {
    "latency_ms_min": 20.0,
    "latency_ms_max": 200.0,
    "enable_latency_sleep": false,
    "reject_prob": 0.01,
    "partial_fill_prob": 0.30,
    "partial_fill_min_ratio": 0.3,
    "partial_fill_max_ratio": 0.9,
    "random_seed": 42
  }
}

如果 broker.paper 未配置，则会退化使用：
  - trade_core.* 段中的 starting_cash / commission_bps / slippage_bps
  - portfolio.accounts.* 段中的 per-account 起始资金
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List

from config.config_center import get_system_config
from core.logging_utils import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 数据结构（PaperBroker 内部使用）
# ----------------------------------------------------------------------


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    def apply_buy(self, qty: float, price: float) -> None:
        if qty <= 0:
            return
        total_cost_old = self.qty * self.avg_price
        total_cost_new = total_cost_old + qty * price
        self.qty += qty
        if self.qty > 0:
            self.avg_price = total_cost_new / self.qty
        else:
            self.avg_price = 0.0

    def apply_sell(self, qty: float) -> None:
        if qty <= 0:
            return
        self.qty -= qty
        if self.qty <= 0:
            self.qty = 0.0
            self.avg_price = 0.0


@dataclass
class AccountState:
    account_id: str
    base_currency: str
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def get_position(self, symbol: str) -> Position:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(symbol=symbol)
            self.positions[symbol] = pos
        return pos

    def nav(self) -> float:
        # 暂不使用市值（市场价），只是现金 + 成本价近似，偏保守
        total_cost = sum(p.qty * p.avg_price for p in self.positions.values())
        return self.cash + total_cost

    def snapshot(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "base_currency": self.base_currency,
            "cash": float(self.cash),
            "nav": float(self.nav()),
            "positions": {
                sym: {"qty": float(p.qty), "avg_price": float(p.avg_price)}
                for sym, p in self.positions.items()
                if abs(p.qty) > 0
            },
        }


# ----------------------------------------------------------------------
# 抽象 Broker 接口
# ----------------------------------------------------------------------


class BaseBroker:
    """
    Broker 抽象基类。

    上层（TradeCore / Commander / RiskBrain）应只依赖这个接口，而不关心具体实现是：
      - PaperBroker（本地回测 / 模拟撮合）
      - ShadowBroker（影子券商，用于实盘前预演）
      - RealBroker（未来接入券商实盘接口）

    未来接入实盘时，只需要：
      - 实现一个新的 Broker 类继承 BaseBroker
      - 在 get_default_broker() 中加一个分支
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        self.sys_cfg = system_config or get_system_config()
        self.log = get_logger(self.__class__.__name__)

    # ---- 核心接口 ---- #

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一笔订单（同步）。

        Args:
            order: 订单 dict（见文件顶部规范）

        Returns:
            execution dict，至少包含字段：
                - status: "filled" / "rejected" / "partial"
                - account_id, symbol, side, fill_qty, fill_price, fee, ts, reason
        """
        raise NotImplementedError

    def get_account_state(self, account_id: str) -> Dict[str, Any]:
        """
        获取单个账户快照。
        """
        raise NotImplementedError

    def snapshot_all_accounts(self) -> Dict[str, Any]:
        """
        获取所有账户快照，返回：
        {
          "accounts": {
            "<account_id>": { . 单账户 snapshot . },
            .
          }
        }
        """
        raise NotImplementedError

    # ---- 统一适配 TradeCore 的 submit_order 接口 ---- #

    def submit_order(self, order: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """
        默认实现：把 OrderEvent / dict / kwargs 归一为订单 dict，
        然后调用 execute_order()。

        子类如果有更复杂需求，可以重写此方法。
        """
        order_dict = self._normalize_order_for_execute(order, **kwargs)
        return self.execute_order(order_dict)

    def _normalize_order_for_execute(self, order: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """
        将各种风格的入参归一为 PaperBroker/RealBroker 能理解的订单 dict。
        """
        od: Dict[str, Any] = {}

        # 1) OrderEvent / 自定义对象
        if order is not None and not isinstance(order, dict):
            for src, dst in [
                ("symbol", "symbol"),
                ("side", "side"),
                ("quantity", "qty"),
                ("qty", "qty"),
                ("order_type", "order_type"),
                ("limit_price", "limit_price"),
                ("ref_price", "ref_price"),
                ("account_id", "account_id"),
                ("strategy_id", "strategy_id"),
                ("client_order_id", "order_id"),
                ("time_in_force", "time_in_force"),
                ("ts", "ts"),
            ]:
                if hasattr(order, src):
                    od[dst] = getattr(order, src)
            if hasattr(order, "extra"):
                extra = getattr(order, "extra") or {}
                if isinstance(extra, dict):
                    meta = dict(extra)
                    if isinstance(od.get("meta"), dict):
                        base_meta = od["meta"]
                        base_meta.update(meta)
                        meta = base_meta
                    od["meta"] = meta

        # 2) dict 形式的订单
        if isinstance(order, dict):
            od.update(order)

        # 3) kwargs 覆盖（优先级最高）
        for k, v in kwargs.items():
            key = k
            if k == "quantity":
                key = "qty"
            elif k == "client_order_id":
                key = "order_id"
            od[key] = v

        # 4) quantity -> qty
        if "qty" not in od and "quantity" in od:
            od["qty"] = od.pop("quantity")

        return od


# ----------------------------------------------------------------------
# PaperBroker 实现（多账户 PaperTrader）
# ----------------------------------------------------------------------


class PaperBroker(BaseBroker):
    """
    纸上交易 Broker，实现本地多账户撮合与资金/持仓维护。

    - 支持多账户（来自 portfolio.accounts），每个账户有独立现金/持仓
    - 支持基础手续费（commission_bps）和滑点（slippage_bps）
    - 不考虑真实盘口 / 深度，只用订单上的价格 / ref_price 来撮合
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(system_config=system_config)

        # 读取配置
        broker_cfg = (self.sys_cfg.get("broker") or {}).get("paper") or {}
        trade_cfg = self.sys_cfg.get("trade_core") or {}
        portfolio_cfg = self.sys_cfg.get("portfolio") or {}

        self.base_currency: str = str(
            broker_cfg.get("base_currency")
            or trade_cfg.get("base_currency")
            or "CNY"
        )
        self.starting_cash: float = float(
            broker_cfg.get("starting_cash") or trade_cfg.get("starting_cash") or 1_000_000.0
        )
        self.commission_bps: float = float(
            broker_cfg.get("commission_bps") or trade_cfg.get("commission_bps") or 1.0
        )
        self.slippage_bps: float = float(
            broker_cfg.get("slippage_bps") or trade_cfg.get("slippage_bps") or 0.5
        )

        accounts_cfg = portfolio_cfg.get("accounts") or {}
        self.accounts: Dict[str, AccountState] = {}

        if isinstance(accounts_cfg, dict) and accounts_cfg:
            for aid, cfg in accounts_cfg.items():
                cash = float(cfg.get("starting_cash", self.starting_cash))
                base_ccy = str(cfg.get("base_currency", self.base_currency))
                self.accounts[aid] = AccountState(
                    account_id=aid,
                    base_currency=base_ccy,
                    cash=cash,
                )
        else:
            # 没有配置 portfolio.accounts 时，提供默认账户
            self.accounts["paper_main"] = AccountState(
                account_id="paper_main",
                base_currency=self.base_currency,
                cash=self.starting_cash,
            )

        self.log.info(
            "PaperBroker 初始化完成: base_currency=%s starting_cash=%.2f accounts=%s "
            "commission_bps=%.3f slippage_bps=%.3f",
            self.base_currency,
            self.starting_cash,
            list(self.accounts.keys()),
            self.commission_bps,
            self.slippage_bps,
        )

    # ---- BaseBroker 接口实现 ---- #

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步执行一笔订单，返回标准 execution dict。
        """
        account_id = str(order.get("account_id") or "paper_main")
        symbol = str(order.get("symbol") or "")
        strategy_id = str(order.get("strategy_id") or "")
        side = str(order.get("side") or "").upper()
        qty = float(order.get("qty") or 0.0)
        order_type = str(order.get("order_type") or "MKT").upper()
        limit_price = order.get("limit_price")
        ts = order.get("ts") or datetime.utcnow().isoformat()

        if not symbol or qty <= 0 or side not in ("BUY", "SELL"):
            return self._reject(
                order=order,
                account=self._ensure_account(account_id),
                reason="invalid_order",
                ts=ts,
            )

        acct = self._ensure_account(account_id)

        # 决定执行价格
        price: Optional[float] = None
        if order_type == "LIMIT" and limit_price is not None:
            try:
                price = float(limit_price)
            except Exception:
                price = None
        if price is None:
            # 用 ref_price / price / last_price 做 MKT 价格
            for key in ("ref_price", "price", "last_price", "limit_price"):
                v = order.get(key)
                if v is not None:
                    try:
                        price = float(v)
                        break
                    except Exception:
                        continue

        if price is None or price <= 0:
            return self._reject(
                order=order,
                account=acct,
                reason="invalid_price",
                ts=ts,
            )

        slip = self.slippage_bps / 10000.0
        fee_bps = self.commission_bps / 10000.0

        if side == "BUY":
            exec_price = price * (1.0 + slip)
        else:
            exec_price = price * (1.0 - slip)

        gross = exec_price * qty
        fee = gross * fee_bps

        if side == "BUY":
            total_cost = gross + fee
            if acct.cash < total_cost:
                # 资金不足，尽量买到能买的最大整数手
                max_qty = math.floor(acct.cash / (exec_price * (1.0 + fee_bps)) / 100.0) * 100.0
                if max_qty <= 0:
                    return self._reject(
                        order=order,
                        account=acct,
                        reason="insufficient_cash",
                        ts=ts,
                    )
                qty = max_qty
                gross = exec_price * qty
                fee = gross * fee_bps
                total_cost = gross + fee

            # 扣现金，增加持仓
            acct.cash -= total_cost
            pos = acct.get_position(symbol)
            pos.apply_buy(qty, exec_price)
        else:
            # SELL
            pos = acct.get_position(symbol)
            if pos.qty < qty:
                # 没仓位 / 仓位不足，最多平到 0
                qty = pos.qty
            if qty <= 0:
                return self._reject(
                    order=order,
                    account=acct,
                    reason="no_position",
                    ts=ts,
                )
            # 减仓，增加现金
            gross = exec_price * qty
            fee = gross * fee_bps
            acct.cash += gross - fee
            pos.apply_sell(qty)

        self.log.info(
            "PaperBroker 执行订单: acct=%s symbol=%s side=%s qty=%.0f price=%.4f fee=%.4f cash=%.2f nav=%.2f",
            acct.account_id,
            symbol,
            side,
            qty,
            exec_price,
            fee,
            acct.cash,
            acct.nav(),
        )

        return {
            "order_id": order.get("order_id"),
            "account_id": acct.account_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side,
            "status": "filled",
            "fill_qty": float(qty),
            "fill_price": float(exec_price),
            "fee": float(fee),
            "ts": ts,
            "reason": "",
            "position_after": {
                "qty": float(acct.get_position(symbol).qty),
                "avg_price": float(acct.get_position(symbol).avg_price),
            },
            "meta": {
                "order": order,
                "nav_after": float(acct.nav()),
                "cash_after": float(acct.cash),
            },
        }

    def get_account_state(self, account_id: str) -> Dict[str, Any]:
        acct = self._ensure_account(account_id)
        snap = acct.snapshot()
        return snap

    def snapshot_all_accounts(self) -> Dict[str, Any]:
        return {
            "accounts": {
                aid: acct.snapshot() for aid, acct in self.accounts.items()
            }
        }

    # ---- 内部工具 ---- #

    def _ensure_account(self, account_id: str) -> AccountState:
        acct = self.accounts.get(account_id)
        if acct is None:
            self.log.warning(
                "PaperBroker: 请求了未知账户 %s，将以默认起始资金创建。",
                account_id,
            )
            acct = AccountState(
                account_id=account_id,
                base_currency=self.base_currency,
                cash=self.starting_cash,
            )
            self.accounts[account_id] = acct
        return acct

    def _reject(
        self,
        order: Dict[str, Any],
        account: AccountState,
        reason: str,
        ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        symbol = str(order.get("symbol") or "")
        side = str(order.get("side") or "").upper()
        strategy_id = str(order.get("strategy_id") or "")
        account_id = account.account_id
        ts = ts or datetime.utcnow().isoformat()

        execution = {
            "order_id": order.get("order_id"),
            "account_id": account_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side,
            "status": "rejected",
            "fill_qty": 0.0,
            "fill_price": 0.0,
            "fee": 0.0,
            "ts": ts,
            "reason": reason,
            "position_after": {
                "qty": float(account.get_position(symbol).qty),
                "avg_price": float(account.get_position(symbol).avg_price),
            },
            "meta": {
                "order": order,
                "nav_after": float(account.nav()),
                "cash_after": float(account.cash),
            },
        }
        self.log.warning(
            "PaperBroker 拒绝订单: acct=%s symbol=%s side=%s reason=%s",
            account_id,
            symbol,
            side,
            reason,
        )
        return execution


# ----------------------------------------------------------------------
# ShadowBroker 实现（影子券商 / 预演模式）
# ----------------------------------------------------------------------


class ShadowBroker(BaseBroker):
    """
    ShadowBroker: 包装一个“真实” broker（当前为 PaperBroker），模拟实盘中的
    延迟 / 拒单 / 部分成交等行为，用于实盘前预演和压力测试。

    行为特征：
      - 可选网络延迟（latency_ms_min ~ latency_ms_max）
      - 随机拒单（reject_prob）
      - 随机部分成交（partial_fill_prob + 成交比例区间）
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(system_config=system_config)

        broker_cfg = (self.sys_cfg.get("broker") or {}).get("shadow") or {}

        self.latency_ms_min: float = float(broker_cfg.get("latency_ms_min", 0.0))
        self.latency_ms_max: float = float(broker_cfg.get("latency_ms_max", 0.0))
        if self.latency_ms_max < self.latency_ms_min:
            self.latency_ms_max = self.latency_ms_min
        self.enable_latency_sleep: bool = bool(broker_cfg.get("enable_latency_sleep", False))

        self.reject_prob: float = float(broker_cfg.get("reject_prob", 0.0))
        self.partial_fill_prob: float = float(broker_cfg.get("partial_fill_prob", 0.0))
        self.partial_fill_min_ratio: float = float(broker_cfg.get("partial_fill_min_ratio", 0.3))
        self.partial_fill_max_ratio: float = float(broker_cfg.get("partial_fill_max_ratio", 0.9))
        if self.partial_fill_max_ratio < self.partial_fill_min_ratio:
            self.partial_fill_max_ratio = self.partial_fill_min_ratio

        seed = broker_cfg.get("random_seed")
        self._rng = random.Random(seed)

        # 内部真实 broker：当前固定为 PaperBroker
        self.base_broker = PaperBroker(system_config=self.sys_cfg)

        self.log.info(
            "ShadowBroker 初始化: latency=[%.1f, %.1f]ms reject_prob=%.3f partial_fill_prob=%.3f",
            self.latency_ms_min,
            self.latency_ms_max,
            self.reject_prob,
            self.partial_fill_prob,
        )

    # ---- BaseBroker 接口实现 ---- #

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # 可能等待模拟网络延迟
        self._maybe_sleep_latency()

        account_id = str(order.get("account_id") or "paper_main")
        symbol = str(order.get("symbol") or "")
        side = str(order.get("side") or "").upper()
        strategy_id = str(order.get("strategy_id") or "")

        # 预先拿一份账户状态（用于拒单时的 after 状态）
        base_snapshot = self.base_broker.get_account_state(account_id)
        base_positions = base_snapshot.get("positions") or {}
        base_pos = base_positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})

        # 随机拒单（不触及真实 broker 资金 / 持仓）
        if self.reject_prob > 0 and self._rng.random() < self.reject_prob:
            ts = order.get("ts") or datetime.utcnow().isoformat()
            exec_dict: Dict[str, Any] = {
                "order_id": order.get("order_id"),
                "account_id": base_snapshot.get("account_id", account_id),
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "status": "rejected",
                "fill_qty": 0.0,
                "fill_price": 0.0,
                "fee": 0.0,
                "ts": ts,
                "reason": "shadow_reject",
                "position_after": {
                    "qty": float(base_pos.get("qty", 0.0)),
                    "avg_price": float(base_pos.get("avg_price", 0.0)),
                },
                "meta": {
                    "order": order,
                    "nav_after": float(base_snapshot.get("nav", 0.0)),
                    "cash_after": float(base_snapshot.get("cash", 0.0)),
                    "shadow": {
                        "mode": "reject",
                        "reject_prob": self.reject_prob,
                    },
                },
            }
            self.log.warning(
                "ShadowBroker: 模拟拒单 account=%s symbol=%s side=%s",
                account_id,
                symbol,
                side,
            )
            return exec_dict

        # 随机部分成交（通过调小 qty，并让 PaperBroker 正常撮合）
        order_for_send = dict(order)
        qty = float(order_for_send.get("qty") or 0.0)
        if (
            qty > 0
            and self.partial_fill_prob > 0
            and self._rng.random() < self.partial_fill_prob
        ):
            ratio = self._rng.uniform(self.partial_fill_min_ratio, self.partial_fill_max_ratio)
            raw_qty = qty * ratio
            part_qty = float(math.floor(raw_qty / 100.0) * 100.0)  # 取整数手
            if 0 < part_qty < qty:
                order_for_send["qty"] = part_qty
                self.log.info(
                    "ShadowBroker: 模拟部分成交 account=%s symbol=%s side=%s qty=%.0f->%.0f",
                    account_id,
                    symbol,
                    side,
                    qty,
                    part_qty,
                )

        base_exec = self.base_broker.execute_order(order_for_send)

        # 打 ShadowBroker 标记
        meta = base_exec.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"_raw_meta": meta}
        shadow_meta = {
            "mode": "passthrough",
            "latency_ms_min": self.latency_ms_min,
            "latency_ms_max": self.latency_ms_max,
            "reject_prob": self.reject_prob,
            "partial_fill_prob": self.partial_fill_prob,
            "partial_fill_min_ratio": self.partial_fill_min_ratio,
            "partial_fill_max_ratio": self.partial_fill_max_ratio,
        }
        meta.setdefault("shadow", shadow_meta)
        base_exec["meta"] = meta

        # 如果我们人为减少了 qty，且状态仍为 filled，则改为 partial
        if order_for_send.get("qty") != order.get("qty") and base_exec.get("status") == "filled":
            base_exec["status"] = "partial"

        return base_exec

    def get_account_state(self, account_id: str) -> Dict[str, Any]:
        return self.base_broker.get_account_state(account_id)

    def snapshot_all_accounts(self) -> Dict[str, Any]:
        return self.base_broker.snapshot_all_accounts()

    # ---- 内部工具 ---- #

    def _maybe_sleep_latency(self) -> None:
        if not self.enable_latency_sleep:
            return
        if self.latency_ms_max <= 0:
            return
        if self.latency_ms_min < 0:
            self.latency_ms_min = 0.0
        delay_ms = self._rng.uniform(self.latency_ms_min, self.latency_ms_max)
        delay_s = max(0.0, delay_ms / 1000.0)
        if delay_s > 0:
            time.sleep(delay_s)


# ----------------------------------------------------------------------
# Broker 工厂
# ----------------------------------------------------------------------


def get_default_broker(system_config: Optional[Dict[str, Any]] = None) -> BaseBroker:
    """
    根据 system_config 中的 "broker.type" 返回默认 Broker 实例。

    当前支持：
      - "paper"（默认）
      - "shadow"（影子券商，内部包装 PaperBroker）
    """
    sys_cfg = system_config or get_system_config()
    broker_cfg = sys_cfg.get("broker") or {}
    broker_type_raw = broker_cfg.get("type", "paper")
    broker_type = str(broker_type_raw).lower() if broker_type_raw is not None else "paper"

    if broker_type in ("paper", "", None):
        return PaperBroker(system_config=sys_cfg)

    if broker_type == "shadow":
        return ShadowBroker(system_config=sys_cfg)

    # 预留其他类型：ctp / xtp / ths / custom 等
    raise NotImplementedError(f"未知 broker.type={broker_type}，当前仅支持 'paper' 和 'shadow'")


__all__ = [
    "BaseBroker",
    "PaperBroker",
    "ShadowBroker",
    "get_default_broker",
    "Position",
    "AccountState",
]
