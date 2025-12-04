# -*- coding: utf-8 -*-
"""
execution/execution_model.py

LightHunter - 统一执行模型 (ExecutionModel)

职责：
- 把“如何从行情推导成交”的逻辑抽出来，形成统一的执行内核；
- 回测 (BacktestCore) 与 实盘/纸交易 (TradeCore) 都使用同一套订单/成交逻辑，
  保证策略在回测与真实执行之间的行为尽量一致。

核心概念：
- OrderIntent      : 策略层下达的“意图”，例如 买入 000001.SZ 1000 股，限价 10.05；
- MarketSnapshot   : 某一刻的行情快照（可以是日线/分钟线 bar，或 L1 快照）；
- ExecutionModel   : 输入 (OrderIntent, MarketSnapshot)，输出 ExecutionResult；
- ExecutionResult  : 成交数量、成交均价、手续费、滑点、订单状态等。

特别说明（简化假设）：
- 目前仅考虑 A 股“普通多头”场景，不支持裸空；
- 默认按 100 股一手的 board lot 交易；
- 回测用 bar 的 OHLCH 来估算能否成交，实盘用 last price 或近似的 bid/ask；
- 滑点以 bps（万分之几）参数形式统一控制，方便以后调参。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


# ----------------------------------------------------------------------
# 枚举定义
# ----------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------


@dataclass
class OrderIntent:
    """策略下达的订单意图。"""

    symbol: str
    side: OrderSide
    quantity: int  # 股数（严格遵守 lot_size 的整数倍）
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None

    # 元信息
    ts: Optional[datetime] = None            # 下单时间（用于回测控制 T+1 之类）
    strategy_name: Optional[str] = None
    tag: Optional[str] = None                # 任意标记字段，方便调试


@dataclass
class ExecutionResult:
    """执行结果（一次 intent 的撮合结果摘要）。"""

    status: OrderStatus
    filled_quantity: int
    avg_price: Optional[float]
    fee: float
    slippage_cost: float
    message: str = ""


@dataclass
class MarketSnapshot:
    """行情快照 / Bar 数据，提供给执行模型使用。

    对于回测：
        - open / high / low / close 都应有值；
        - last 一般用 close 填充即可。
    对于实盘：
        - last 必须有值；
        - open/high/low/close 有则更好。
    """

    symbol: str
    ts: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None


@dataclass
class ExecutionConfig:
    """执行模型参数。"""

    fee_rate: float = 0.0003           # 单边费率，如 0.0003 = 万三
    slippage_bps: float = 5.0          # 滑点，单位 bps（万分之一），例如 5 = 0.05%
    lot_size: int = 100                # 最小交易单位（A 股默认 100 股一手）
    allow_partial: bool = True         # 是否允许部分成交（目前逻辑里都视为一次性全成 或 0 成，保留扩展）


# ----------------------------------------------------------------------
# 执行模型基类
# ----------------------------------------------------------------------


class ExecutionModel:
    """抽象执行模型基类。"""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    # 子类需要实现的核心方法
    def simulate(self, order: OrderIntent, snapshot: MarketSnapshot) -> ExecutionResult:
        raise NotImplementedError

    # 通用工具：计算手续费与滑点
    def _calc_fee(self, traded_value: float) -> float:
        return traded_value * self.config.fee_rate

    def _calc_slippage_price(self, base_price: float, side: OrderSide) -> float:
        """根据滑点 bps 计算调整后的成交价。"""
        if self.config.slippage_bps <= 0:
            return base_price
        slip_ratio = self.config.slippage_bps / 10000.0
        if side == OrderSide.BUY:
            return base_price * (1.0 + slip_ratio)
        else:
            return base_price * (1.0 - slip_ratio)


# ----------------------------------------------------------------------
# 回测执行模型
# ----------------------------------------------------------------------


class BacktestExecutionModel(ExecutionModel):
    """回测用执行模型：使用 bar 的 OHLC 来判断是否能成交以及成交价格。

    典型规则（简化版）：
    - 市价单：
        * BUY : 使用 bar 的 open 价格，加滑点；
        * SELL: 使用 bar 的 open 价格，加滑点；
    - 限价单：
        * BUY : 若 bar.low <= limit_price，则成交；
                成交价取 min(limit_price, bar.open) 后再加滑点；
        * SELL: 若 bar.high >= limit_price，则成交；
                成交价取 max(limit_price, bar.open) 后再加滑点；
    - 若不满足条件，则视为当个 bar 未成交（由上层决定是否带到下一个 bar）。
      这里我们简化为：一次 simulate 只尝试当前 bar，未成交时 status=REJECTED。
    """

    def simulate(self, order: OrderIntent, snapshot: MarketSnapshot) -> ExecutionResult:
        qty = order.quantity
        if qty <= 0:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="数量必须大于 0",
            )

        if snapshot.open is None and snapshot.close is None:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="行情数据缺少 open/close，无法撮合",
            )

        # 决定基准成交价
        base_price: Optional[float] = None

        if order.order_type == OrderType.MARKET:
            base_price = snapshot.open if snapshot.open is not None else snapshot.close
        else:  # LIMIT
            lp = order.limit_price
            if lp is None:
                return ExecutionResult(
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    avg_price=None,
                    fee=0.0,
                    slippage_cost=0.0,
                    message="限价单缺少 limit_price",
                )

            low = snapshot.low if snapshot.low is not None else snapshot.close
            high = snapshot.high if snapshot.high is not None else snapshot.close
            open_ = snapshot.open if snapshot.open is not None else snapshot.close

            if order.side == OrderSide.BUY:
                # 低价触及到限价就认为能成交
                if low is None or open_ is None or lp < 0:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="行情数据异常，无法执行 BUY 限价单",
                    )
                if low <= lp:
                    base_price = min(lp, open_)
                else:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="限价买单价格未被触及，未成交",
                    )
            else:
                # SELL
                if high is None or open_ is None or lp < 0:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="行情数据异常，无法执行 SELL 限价单",
                    )
                if high >= lp:
                    base_price = max(lp, open_)
                else:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="限价卖单价格未被触及，未成交",
                    )

        if base_price is None or base_price <= 0:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="无法确定基准成交价",
            )

        # 加滑点
        trade_price = self._calc_slippage_price(base_price, order.side)

        traded_value = trade_price * qty
        fee = self._calc_fee(traded_value)
        slippage_cost = abs(trade_price - base_price) * qty

        return ExecutionResult(
            status=OrderStatus.FILLED,
            filled_quantity=qty,
            avg_price=trade_price,
            fee=fee,
            slippage_cost=slippage_cost,
            message="回测撮合成功",
        )


# ----------------------------------------------------------------------
# 实盘 / 纸交易 执行模型
# ----------------------------------------------------------------------


class LiveExecutionModel(ExecutionModel):
    """实盘或纸交易用执行模型：基于 L1 快照或最新成交价进行撮合。

    简化规则：
    - 市价单：
        * BUY / SELL : 使用 snapshot.last 或 snapshot.close 作为基准，加滑点；
    - 限价单：
        * BUY : 若 last <= limit_price 则成交（价格取 min(last, limit_price) 再加滑点）；
        * SELL: 若 last >= limit_price 则成交（价格取 max(last, limit_price) 再加滑点）；
    - 若 last 不存在，则回退到 close 或 open 作为近似价格；
    - 未触发条件则直接 REJECT（是否保留到下一 tick 由上层控制）。
    """

    def simulate(self, order: OrderIntent, snapshot: MarketSnapshot) -> ExecutionResult:
        qty = order.quantity
        if qty <= 0:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="数量必须大于 0",
            )

        # 基准价格：优先 last，其次 close，再次 open
        base = snapshot.last or snapshot.close or snapshot.open
        if base is None or base <= 0:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="缺少有效价格 (last/close/open)，无法撮合",
            )

        base_price: Optional[float] = None

        if order.order_type == OrderType.MARKET:
            base_price = base
        else:
            lp = order.limit_price
            if lp is None:
                return ExecutionResult(
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    avg_price=None,
                    fee=0.0,
                    slippage_cost=0.0,
                    message="限价单缺少 limit_price",
                )

            # 简版：用 last 与 limit 比较
            last = base
            if order.side == OrderSide.BUY:
                if last <= lp:
                    base_price = min(last, lp)
                else:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="限价买单价格未到，未成交",
                    )
            else:
                if last >= lp:
                    base_price = max(last, lp)
                else:
                    return ExecutionResult(
                        status=OrderStatus.REJECTED,
                        filled_quantity=0,
                        avg_price=None,
                        fee=0.0,
                        slippage_cost=0.0,
                        message="限价卖单价格未到，未成交",
                    )

        if base_price is None or base_price <= 0:
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_price=None,
                fee=0.0,
                slippage_cost=0.0,
                message="无法确定成交价格",
            )

        trade_price = self._calc_slippage_price(base_price, order.side)
        traded_value = trade_price * qty
        fee = self._calc_fee(traded_value)
        slippage_cost = abs(trade_price - base_price) * qty

        return ExecutionResult(
            status=OrderStatus.FILLED,
            filled_quantity=qty,
            avg_price=trade_price,
            fee=fee,
            slippage_cost=slippage_cost,
            message="实盘/纸交易撮合成功",
        )
