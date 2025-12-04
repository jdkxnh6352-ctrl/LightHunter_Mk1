# strategy/portfolio_manager.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from math import floor
from datetime import datetime

from config.config_center import get_system_config

logger = logging.getLogger(__name__)


LOT_SIZE = 100  # A 股一手 100 股


@dataclass
class StrategyRuntimeConfig:
    """单个策略在 Portfolio 层的运行参数（从 system_config 解析而来）"""

    strategy_id: str
    account_id: str

    max_gross_leverage: float = 1.0          # 总杠杆上限（总市值 / 账户净值）
    max_single_position_pct: float = 0.10    # 单票最大占用净值比例
    max_positions: int = 20                  # 同时持仓股票数量上限

    target_holding_days: int = 1             # 目标持股周期（天），主要用于日志/决策提示
    rebalance_mode: str = "close"            # rebalance 触发时机：close / open / intraday
    description: str = ""                    # 文字说明，方便人类理解


@dataclass
class OrderPlan:
    """组合管理层产出的“下单计划”（还未发送到 Broker）"""

    account_id: str
    strategy_id: str
    symbol: str

    side: str              # "BUY" or "SELL"
    qty: int               # 手数 * LOT_SIZE 之后的股数
    order_type: str = "MKT"
    price: Optional[float] = None
    ts: Optional[datetime] = None

    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "order_type": self.order_type,
            "price": self.price,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else self.ts,
            "extra": self.extra or {},
        }


class PortfolioManager:
    """
    LightHunter 组合管理核心

    职责：
    - 读取 system_config["portfolio"] 下的账户/策略配置
    - 根据策略 id + 信号 + 当前账户状态，产出一组 OrderPlan
    - 显式支持 U1/U2/U3 等超短主战策略（规则主要体现在 config 中）

    约定的输入格式（Commander / OnlineListener 可按此构造）：
    - signals: List[Mapping]，每个元素至少包含：
        {
            "symbol": "SZ000001",
            "score":  0.85,          # 模型打分/信号强度，>0 视为做多意愿
            # 可选：
            "side":   "BUY" / "SELL",
            "last_price": 10.23,
        }
    - prices: Dict[symbol, last_price]，如果 signal 中没有 last_price，会从这里取
    - current_positions: Dict[symbol, qty]，为该策略对应账户的当前持仓
    - account_equity: float，账户净值（现金 + 持仓市值）
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        self._cfg = cfg or get_system_config()
        self._portfolio_cfg = self._cfg.get("portfolio", {})
        self._strategies: Dict[str, StrategyRuntimeConfig] = {}

        self._load_strategy_configs()
        self._apply_builtin_presets_for_ultrashort()

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def _load_strategy_configs(self) -> None:
        accounts_cfg = self._portfolio_cfg.get("accounts", {})
        strategies_cfg = self._portfolio_cfg.get("strategies", {})

        if not strategies_cfg:
            logger.warning("system_config['portfolio']['strategies'] 为空，"
                           "PortfolioManager 将无法正常工作。")

        for sid, s_cfg in strategies_cfg.items():
            account_id = s_cfg.get("account_id")
            if not account_id:
                raise ValueError(f"策略 {sid} 缺少 account_id 配置")

            if account_id not in accounts_cfg:
                logger.warning(
                    "策略 %s 绑定的 account_id=%s 不在 portfolio.accounts 中，"
                    "请检查 system_config.json。",
                    sid,
                    account_id,
                )

            self._strategies[sid] = StrategyRuntimeConfig(
                strategy_id=sid,
                account_id=account_id,
                max_gross_leverage=float(s_cfg.get("max_gross_leverage", 1.0)),
                max_single_position_pct=float(
                    s_cfg.get("max_single_position_pct", 0.10)
                ),
                max_positions=int(s_cfg.get("max_positions", 20)),
                target_holding_days=int(s_cfg.get("target_holding_days", 1)),
                rebalance_mode=str(s_cfg.get("rebalance_mode", "close")),
                description=str(s_cfg.get("description", "")),
            )

        logger.info(
            "PortfolioManager 加载策略配置完成：%s",
            ", ".join(sorted(self._strategies.keys())),
        )

    def _apply_builtin_presets_for_ultrashort(self) -> None:
        """
        针对 U1/U2/U3 的一些“默认风格”，如果 config 里已经显式指定了，就尊重配置。

        （这里主要是为了尽量贴近 Mk4-Step-13 文档中的定位，
         真正的数值你可以在 system_config.json 里自己微调。）
        """
        for sid, cfg in self._strategies.items():
            if sid.upper() == "U1":
                cfg.target_holding_days = cfg.target_holding_days or 1
                # U1：高集中度，少票，单票权重较高
                cfg.max_positions = min(cfg.max_positions or 5, 5)
                cfg.max_single_position_pct = max(
                    cfg.max_single_position_pct, 0.25
                )
            elif sid.upper() == "U2":
                # U2：冰点反弹，分散一些
                cfg.target_holding_days = cfg.target_holding_days or 2
                cfg.max_positions = min(cfg.max_positions or 10, 12)
                cfg.max_single_position_pct = min(
                    cfg.max_single_position_pct, 0.15
                )
            elif sid.upper() == "U3":
                # U3：趋势低吸，仓位更均匀
                cfg.target_holding_days = cfg.target_holding_days or 1
                cfg.max_positions = max(cfg.max_positions, 15)
                cfg.max_single_position_pct = min(
                    cfg.max_single_position_pct, 0.08
                )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    @classmethod
    def from_system_config(cls) -> "PortfolioManager":
        return cls(get_system_config())

    def get_strategy_config(self, strategy_id: str) -> StrategyRuntimeConfig:
        if strategy_id in self._strategies:
            return self._strategies[strategy_id]

        # fallback：如果没找到指定策略，退回 ultrashort_main
        if "ultrashort_main" in self._strategies:
            logger.warning(
                "未找到策略 %s 的配置，回退使用 ultrashort_main 的配置。", strategy_id
            )
            return self._strategies["ultrashort_main"]

        raise KeyError(f"Unknown strategy_id={strategy_id}")

    def plan_orders_for_signals(
        self,
        *,
        strategy_id: str,
        signals: Sequence[Mapping[str, Any]],
        prices: Mapping[str, float],
        current_positions: Mapping[str, float],
        account_equity: float,
        as_of: Optional[datetime] = None,
    ) -> List[OrderPlan]:
        """
        核心入口：把一批信号转成 OrderPlan 列表。

        参数：
        - strategy_id: 当前信号对应的策略（U1/U2/U3/ultrashort_main 等）
        - signals: 模型输出信号列表
        - prices: 最新价（symbol -> price）
        - current_positions: 当前持仓（symbol -> qty）
        - account_equity: 账户净值
        - as_of: 时间戳，用于日志和下单时间记录
        """

        cfg = self.get_strategy_config(strategy_id)
        as_of = as_of or datetime.utcnow()

        if account_equity <= 0:
            logger.warning(
                "账户 %s 净值为 %.2f，无法规划订单。",
                cfg.account_id,
                account_equity,
            )
            return []

        logger.info(
            "开始为策略 %s 规划订单：equity=%.2f, signals=%d",
            strategy_id,
            account_equity,
            len(signals),
        )

        # 1. 过滤和标准化信号
        norm_signals = self._normalize_signals(signals, prices)

        if not norm_signals:
            # 如果没有任何有效做多信号，则默认“平掉所有已有持仓”
            logger.info(
                "策略 %s 当前无有效做多信号，将考虑平掉全部持仓。",
                strategy_id,
            )
            return self._flatten_all_positions(
                cfg, current_positions, as_of=as_of
            )

        # 2. 计算目标权重（0~1 之间），考虑单票限制 + max_positions
        target_weights = self._compute_target_weights(
            cfg, norm_signals
        )  # symbol -> weight

        # 3. 算目标股数，并与当前持仓做差，生成 OrderPlan
        order_plans: List[OrderPlan] = []

        # 3.1 先处理有目标权重的股票
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                logger.debug(
                    "symbol=%s 缺少价格或价格<=0，跳过下单。", symbol
                )
                continue

            target_value = account_equity * cfg.max_gross_leverage * weight
            target_shares = self._value_to_shares(target_value, price)

            current_shares = int(current_positions.get(symbol, 0))

            delta = target_shares - current_shares
            if abs(delta) < LOT_SIZE:
                # 变化太小，不交易
                continue

            side = "BUY" if delta > 0 else "SELL"
            qty = (abs(delta) // LOT_SIZE) * LOT_SIZE
            if qty <= 0:
                continue

            order_plans.append(
                OrderPlan(
                    account_id=cfg.account_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="MKT",
                    price=None,
                    ts=as_of,
                    extra={
                        "weight": weight,
                        "target_shares": target_shares,
                        "current_shares": current_shares,
                    },
                )
            )

        # 3.2 对于当前持仓里有，但 target_weights 中已经没有的股票，需要考虑“平仓”
        symbols_to_flat = [
            s for s in current_positions.keys() if s not in target_weights
        ]
        for symbol in symbols_to_flat:
            qty = int(current_positions.get(symbol, 0))
            if qty <= 0:
                continue
            qty = (qty // LOT_SIZE) * LOT_SIZE
            if qty <= 0:
                continue
            order_plans.append(
                OrderPlan(
                    account_id=cfg.account_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    order_type="MKT",
                    price=None,
                    ts=as_of,
                    extra={"reason": "no_longer_in_signal_set"},
                )
            )

        logger.info(
            "策略 %s 规划完成：生成 %d 条下单计划。",
            strategy_id,
            len(order_plans),
        )
        return order_plans

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_signals(
        signals: Sequence[Mapping[str, Any]],
        prices: Mapping[str, float],
    ) -> List[Dict[str, Any]]:
        """
        把各种格式的输入信号统一成：
            {"symbol": xxx, "score": float, "last_price": float}
        只保留 score > 0 的做多信号。
        """
        norm: List[Dict[str, Any]] = []
        for s in signals:
            symbol = s.get("symbol") or s.get("code")
            if not symbol:
                continue

            raw_score = s.get("score", s.get("value", 0.0))
            try:
                score = float(raw_score)
            except Exception:
                continue

            if score <= 0:
                # 组合层先只处理“做多”，卖出由权重为 0 + 当前持仓决定
                continue

            last_price = s.get("last_price")
            if last_price is None:
                last_price = prices.get(symbol)

            if last_price is None:
                # 如果没有价格，仍然可以算权重，但在下单阶段可能被过滤
                pass

            norm.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "last_price": last_price,
                }
            )

        return norm

    @staticmethod
    def _value_to_shares(value: float, price: float) -> int:
        """给定目标市值和价格，换算成按 LOT_SIZE 对齐的股数。"""
        if price <= 0 or value <= 0:
            return 0
        raw_shares = value / price
        lots = floor(raw_shares / LOT_SIZE)
        return int(lots * LOT_SIZE)

    @staticmethod
    def _flatten_all_positions(
        cfg: StrategyRuntimeConfig,
        current_positions: Mapping[str, float],
        as_of: Optional[datetime] = None,
    ) -> List[OrderPlan]:
        """无任何做多信号时，选择性地“清仓”策略账户内所有持仓。"""
        as_of = as_of or datetime.utcnow()
        plans: List[OrderPlan] = []
        for symbol, qty in current_positions.items():
            qty = int(qty)
            if qty <= 0:
                continue
            qty = (qty // LOT_SIZE) * LOT_SIZE
            if qty <= 0:
                continue
            plans.append(
                OrderPlan(
                    account_id=cfg.account_id,
                    strategy_id=cfg.strategy_id,
                    symbol=symbol,
                    side="SELL",
                    qty=qty,
                    order_type="MKT",
                    price=None,
                    ts=as_of,
                    extra={"reason": "no_long_signals"},
                )
            )
        return plans

    def _compute_target_weights(
        self,
        cfg: StrategyRuntimeConfig,
        signals: Sequence[Mapping[str, Any]],
    ) -> Dict[str, float]:
        """
        根据信号强度计算目标权重：

        - 先按 score 做 softmax-like 分配；
        - 再截断：单票不超过 max_single_position_pct，
          总票数不超过 max_positions。
        """
        if not signals:
            return {}

        # 先按 score 排序
        sorted_sig = sorted(
            signals, key=lambda s: s["score"], reverse=True
        )

        # 限制最大入选股票数量
        sorted_sig = sorted_sig[: cfg.max_positions]

        total_score = sum(s["score"] for s in sorted_sig)
        if total_score <= 0:
            return {}

        base_weights: Dict[str, float] = {}
        for s in sorted_sig:
            w = s["score"] / total_score
            base_weights[s["symbol"]] = w

        # 单票权重上限
        max_single = cfg.max_single_position_pct
        adjusted: Dict[str, float] = {}
        clipped = False
        for symbol, w in base_weights.items():
            if w > max_single:
                adjusted[symbol] = max_single
                clipped = True
            else:
                adjusted[symbol] = w

        if clipped:
            # 如果有截断，剩余权重按比例重新分配
            total = sum(adjusted.values())
            if total > 0:
                for k in list(adjusted.keys()):
                    adjusted[k] = adjusted[k] / total

        return adjusted
