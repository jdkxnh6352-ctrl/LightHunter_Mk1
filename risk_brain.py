# -*- coding: utf-8 -*-
"""
risk_brain.py

LightHunter Mk3 - RiskBrain
===========================

职责
----
1. 作为全局「风险大脑」：
   - 订阅行情/成交/账户事件；
   - 调用 MicrostructureEngine 进行标的级微观风险检测；
   - 跟踪账户净值/回撤/集中度，生成账户级风险评分；
   - 输出统一的风险快照和异常告警。

2. 与 ZeroMQ 总线联动：
   - 可选接入 bus.zmq_bus.ZmqBus：
       - 订阅如 "market.tick", "trade.fill", "account.nav" 等 Topic；
       - 将所有事件统一交给 on_event() 处理。

3. 与其它模块的接口：
   - TradeCore / Commander 可以定期调用：
       - get_symbol_risk(symbol)
       - get_account_risk(account_id)
       - snapshot()
     来决定是否降杠杆、限仓、强制撤退等动作。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Iterable
import math
import time
from collections import defaultdict

from core.logging_utils import get_logger
from config.config_center import get_system_config
from risk.microstructure_engine import (
    MicrostructureEngine,
    MicrostructureConfig,
    MicrostructureRisk,
    MicrostructureAlert,
)

log = get_logger(__name__)

# 可选：ZeroMQ 总线（如果不存在就忽略）
try:  # pragma: no cover
    from bus.zmq_bus import ZmqBus  # type: ignore

    HAS_ZMQ_BUS = True
except Exception:  # pragma: no cover
    ZmqBus = None  # type: ignore
    HAS_ZMQ_BUS = False


# ======================================================================
# 数据结构
# ======================================================================


@dataclass
class AccountRiskConfig:
    """账户层风险参数。"""

    max_dd_intraday: float = 0.05   # 日内最大回撤阈值（例如 5%）
    max_dd_total: float = 0.20      # 总体最大回撤阈值（例如 20%）
    max_leverage: float = 2.0       # 杠杆上限（|总仓位| / 净值）
    concentration_limit: float = 0.2  # 单票仓位占比上限 (20%)

    w_dd_intraday: float = 0.4
    w_dd_total: float = 0.3
    w_leverage: float = 0.2
    w_concentration: float = 0.1


@dataclass
class SymbolRiskState:
    symbol: str
    last_ts: float = 0.0
    risk_micro: float = 0.0
    last_micro: Optional[MicrostructureRisk] = None
    alerts: List[MicrostructureAlert] = field(default_factory=list)


@dataclass
class AccountRiskState:
    account_id: str
    last_ts: float = 0.0
    nav: float = 0.0                     # 当前净值
    nav_peak: float = 0.0                # 历史最高净值
    nav_intraday_peak: float = 0.0       # 当日内最高净值
    dd_total: float = 0.0                # 总体回撤
    dd_intraday: float = 0.0             # 日内回撤
    gross_exposure: float = 0.0          # 总仓位绝对值
    max_single_weight: float = 0.0       # 单票最大仓位占比
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> weight
    risk_score_acct: float = 0.0
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskSnapshot:
    ts: float
    symbols: Dict[str, Dict[str, Any]]
    accounts: Dict[str, Dict[str, Any]]
    alerts: List[Dict[str, Any]]


# ======================================================================
# RiskBrain 实现
# ======================================================================


class RiskBrain:
    """LightHunter Mk3 - 风险大脑。"""

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        micro_cfg: Optional[MicrostructureConfig] = None,
        acct_cfg: Optional[AccountRiskConfig] = None,
    ) -> None:
        self.system_config = system_config or get_system_config()

        # 微观结构引擎
        self.micro_engine = MicrostructureEngine(config=micro_cfg or self._load_micro_cfg())

        # 账户层风险配置
        self.acct_cfg = acct_cfg or self._load_acct_cfg()

        # 状态
        self._symbol_states: Dict[str, SymbolRiskState] = {}
        self._account_states: Dict[str, AccountRiskState] = {}

        # 黑名单/白名单等全局风控配置
        risk_cfg = self.system_config.get("risk", {}) or {}
        self.blacklist_symbols = set(risk_cfg.get("blacklist_symbols", []) or [])
        self.whitelist_symbols = set(risk_cfg.get("whitelist_symbols", []) or [])

        # 订阅总线（可选）
        self._bus: Optional[Any] = None

        log.info(
            "RiskBrain 初始化完成: blacklist=%d, whitelist=%d",
            len(self.blacklist_symbols),
            len(self.whitelist_symbols),
        )

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_system_config(cls, system_config: Optional[Dict[str, Any]] = None) -> "RiskBrain":
        return cls(system_config=system_config)

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def _load_micro_cfg(self) -> MicrostructureConfig:
        cfg = self.system_config.get("risk_micro", {}) or {}
        return MicrostructureConfig(
            window_size=int(cfg.get("window_size", 30)),
            min_ticks_for_eval=int(cfg.get("min_ticks_for_eval", 10)),
            imbalance_threshold=float(cfg.get("imbalance_threshold", 0.6)),
            vol_spike_ratio=float(cfg.get("vol_spike_ratio", 3.0)),
            ret_spike_ratio=float(cfg.get("ret_spike_ratio", 3.0)),
            spoof_min_depth=float(cfg.get("spoof_min_depth", 50_000.0)),
            spoof_drop_ratio=float(cfg.get("spoof_drop_ratio", 0.5)),
            spoof_small_trade_vol=float(cfg.get("spoof_small_trade_vol", 5_000.0)),
            w_imbalance=float(cfg.get("w_imbalance", 0.4)),
            w_vol_spike=float(cfg.get("w_vol_spike", 0.3)),
            w_ret_spike=float(cfg.get("w_ret_spike", 0.3)),
        )

    def _load_acct_cfg(self) -> AccountRiskConfig:
        cfg = self.system_config.get("risk_account", {}) or {}
        return AccountRiskConfig(
            max_dd_intraday=float(cfg.get("max_dd_intraday", 0.05)),
            max_dd_total=float(cfg.get("max_dd_total", 0.20)),
            max_leverage=float(cfg.get("max_leverage", 2.0)),
            concentration_limit=float(cfg.get("concentration_limit", 0.2)),
            w_dd_intraday=float(cfg.get("w_dd_intraday", 0.4)),
            w_dd_total=float(cfg.get("w_dd_total", 0.3)),
            w_leverage=float(cfg.get("w_leverage", 0.2)),
            w_concentration=float(cfg.get("w_concentration", 0.1)),
        )

    # ------------------------------------------------------------------
    # ZeroMQ 总线订阅（可选）
    # ------------------------------------------------------------------

    def attach_bus(
        self,
        bus: Optional[Any] = None,
        topics: Optional[List[str]] = None,
    ) -> None:
        """
        绑定 ZeroMQ 总线，并订阅指定 Topic。

        推荐 Topic（示例）：
        - "market.tick"     : 行情 tick，payload 为上面的 L1 字段
        - "trade.fill"      : 成交通知（订单成交）
        - "account.nav"     : 账户净值/权益更新
        - "account.position": 账户仓位结构更新（symbol -> weight）
        """
        if bus is None:
            if not HAS_ZMQ_BUS or ZmqBus is None:
                log.warning("RiskBrain.attach_bus: bus 为空且 ZmqBus 不可用，忽略总线绑定。")
                return
            bus = ZmqBus.from_system_config(self.system_config)  # type: ignore[attr-defined]

        self._bus = bus
        topics = topics or [
            "market.tick",
            "trade.fill",
            "account.nav",
            "account.position",
        ]

        for tp in topics:
            try:
                bus.subscribe(tp, self.on_event)  # 约定 subscribe(topic, callback)
                log.info("RiskBrain 订阅 Topic: %s", tp)
            except Exception as e:  # pragma: no cover
                log.warning("RiskBrain 订阅 Topic 失败: %s, err=%s", tp, e)

    # ------------------------------------------------------------------
    # 事件入口
    # ------------------------------------------------------------------

    def on_event(self, event: Dict[str, Any]) -> None:
        """
        风险大脑的统一事件入口。

        约定事件结构（示例）：
        {
          "topic": "market.tick",
          "ts": 1672531200000,
          "payload": {...}
        }

        也支持直接传入 payload 本身，只要能识别字段类型即可。
        """
        if not isinstance(event, dict):
            return

        topic = event.get("topic") or event.get("type") or ""
        payload = event.get("payload", event)

        if topic.startswith("market.tick") or "last_price" in payload:
            self._handle_market_tick(payload)
        elif topic.startswith("account.nav"):
            self._handle_account_nav(payload)
        elif topic.startswith("account.position"):
            self._handle_account_position(payload)
        elif topic.startswith("trade.fill"):
            self._handle_trade_fill(payload)
        else:
            # 其它类型先忽略，后续可以拓展
            pass

    # ------------------------------------------------------------------
    # 行情事件 → 微观结构风险
    # ------------------------------------------------------------------

    def _handle_market_tick(self, tick: Dict[str, Any]) -> None:
        symbol = str(tick.get("symbol", "")).strip()
        if not symbol:
            return
        if symbol in self.blacklist_symbols:
            # 黑名单内标的，无条件高风险
            state = self._symbol_states.get(symbol)
            if state is None:
                state = SymbolRiskState(symbol=symbol)
                self._symbol_states[symbol] = state
            state.last_ts = time.time()
            state.risk_micro = 1.0
            log.debug("RiskBrain: 黑名单标的 %s，风险评分强制为 1.0", symbol)
            return

        micro: Optional[MicrostructureRisk] = self.micro_engine.process_tick(tick)
        if micro is None:
            return

        st = self._symbol_states.get(symbol)
        if st is None:
            st = SymbolRiskState(symbol=symbol)
            self._symbol_states[symbol] = st

        st.last_ts = micro.ts
        st.risk_micro = micro.risk_score_micro
        st.last_micro = micro
        st.alerts = micro.alerts

    # ------------------------------------------------------------------
    # 账户净值 / 仓位事件
    # ------------------------------------------------------------------

    def _get_or_create_acct_state(self, account_id: str) -> AccountRiskState:
        st = self._account_states.get(account_id)
        if st is None:
            st = AccountRiskState(account_id=account_id)
            self._account_states[account_id] = st
        return st

    def _handle_account_nav(self, payload: Dict[str, Any]) -> None:
        """
        账户净值事件，推荐字段：
        {
          "account_id": "acct_ultra_main",
          "ts": 1672531200000,
          "nav": 1234567.89
        }
        """
        account_id = str(payload.get("account_id", "")).strip()
        if not account_id:
            return

        nav = float(payload.get("nav", 0.0))
        ts = self._normalize_ts(payload.get("ts"))

        st = self._get_or_create_acct_state(account_id)
        st.last_ts = ts
        st.nav = nav

        # 更新历史高点
        if st.nav_peak <= 0:
            st.nav_peak = nav
        else:
            st.nav_peak = max(st.nav_peak, nav)

        # 日内高点（简单按每天重新初始化可在上层加日切换逻辑，这里直接 max）
        if st.nav_intraday_peak <= 0:
            st.nav_intraday_peak = nav
        else:
            st.nav_intraday_peak = max(st.nav_intraday_peak, nav)

        # 回撤计算
        st.dd_total = 0.0 if st.nav_peak <= 0 else 1.0 - nav / st.nav_peak
        st.dd_intraday = 0.0 if st.nav_intraday_peak <= 0 else 1.0 - nav / st.nav_intraday_peak

        self._update_account_risk_score(st)

    def _handle_account_position(self, payload: Dict[str, Any]) -> None:
        """
        账户持仓事件，推荐字段：
        {
          "account_id": "acct_ultra_main",
          "ts": 1672531200000,
          "positions": {
            "600000.SH": 0.05,      # 持仓权重（占净值）
            "000001.SZ": 0.03
          }
        }
        """
        account_id = str(payload.get("account_id", "")).strip()
        if not account_id:
            return

        positions = payload.get("positions") or {}
        ts = self._normalize_ts(payload.get("ts"))

        st = self._get_or_create_acct_state(account_id)
        st.last_ts = ts
        st.positions = {str(sym): float(w) for sym, w in positions.items()}

        gross = sum(abs(w) for w in st.positions.values())
        st.gross_exposure = gross
        st.max_single_weight = max((abs(w) for w in st.positions.values()), default=0.0)

        self._update_account_risk_score(st)

    def _handle_trade_fill(self, payload: Dict[str, Any]) -> None:
        """
        成交事件目前主要用于可能的扩展（统计成交风控），
        这里先做占位，后续可根据需要增加逻辑。
        """
        # 例如：可以累计当日成交额、换手率，用于“过度交易”报警
        # 暂时不做强逻辑，只保留扩展点
        pass

    # ------------------------------------------------------------------
    # 账户风险评分聚合
    # ------------------------------------------------------------------

    def _update_account_risk_score(self, st: AccountRiskState) -> None:
        cfg = self.acct_cfg

        # 防御性判断
        nav = max(st.nav, 1e-8)
        gross = st.gross_exposure

        dd_total = max(0.0, st.dd_total)
        dd_intraday = max(0.0, st.dd_intraday)
        leverage = gross / nav
        concentration = st.max_single_weight

        # 转换为 [0, 1] 范围：超出阈值时逐渐趋近 1
        def score_piece(x: float, thr: float) -> float:
            if thr <= 0:
                return 0.0
            if x <= 0:
                return 0.0
            ratio = x / thr
            # 使用平滑函数，避免一下子跳顶
            return float(min(1.0, math.tanh(ratio)))

        s_dd_intraday = score_piece(dd_intraday, cfg.max_dd_intraday)
        s_dd_total = score_piece(dd_total, cfg.max_dd_total)
        s_leverage = score_piece(leverage, cfg.max_leverage)
        s_concentration = score_piece(concentration, cfg.concentration_limit)

        risk_score = (
            cfg.w_dd_intraday * s_dd_intraday
            + cfg.w_dd_total * s_dd_total
            + cfg.w_leverage * s_leverage
            + cfg.w_concentration * s_concentration
        )
        risk_score = float(max(0.0, min(risk_score, 1.0)))
        st.risk_score_acct = risk_score

        # 生成告警摘要（启发式）
        alerts: List[Dict[str, Any]] = []
        if dd_intraday >= cfg.max_dd_intraday:
            alerts.append(
                {
                    "level": "critical",
                    "type": "DD_INTRADAY_BREACH",
                    "message": f"账户 {st.account_id} 日内回撤 {dd_intraday:.2%} 超过阈值 {cfg.max_dd_intraday:.2%}",
                    "metrics": {"dd_intraday": dd_intraday},
                }
            )
        if dd_total >= cfg.max_dd_total:
            alerts.append(
                {
                    "level": "critical",
                    "type": "DD_TOTAL_BREACH",
                    "message": f"账户 {st.account_id} 总体回撤 {dd_total:.2%} 超过阈值 {cfg.max_dd_total:.2%}",
                    "metrics": {"dd_total": dd_total},
                }
            )
        if leverage >= cfg.max_leverage:
            alerts.append(
                {
                    "level": "warning",
                    "type": "LEVERAGE_HIGH",
                    "message": f"账户 {st.account_id} 杠杆 {leverage:.2f} 超过阈值 {cfg.max_leverage:.2f}",
                    "metrics": {"leverage": leverage},
                }
            )
        if concentration >= cfg.concentration_limit:
            alerts.append(
                {
                    "level": "warning",
                    "type": "CONCENTRATION_HIGH",
                    "message": f"账户 {st.account_id} 单票最大仓位 {concentration:.2%} 超过阈值 {cfg.concentration_limit:.2%}",
                    "metrics": {"max_single_weight": concentration},
                }
            )

        st.alerts = alerts

    # ------------------------------------------------------------------
    # 对外查询接口
    # ------------------------------------------------------------------

    def get_symbol_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        st = self._symbol_states.get(symbol)
        if st is None:
            return None
        d = {
            "symbol": st.symbol,
            "last_ts": st.last_ts,
            "risk_micro": st.risk_micro,
            "alerts": [asdict(a) for a in (st.alerts or [])],
        }
        if st.last_micro is not None:
            d["metrics"] = {
                "imbalance": st.last_micro.imbalance,
                "vol_z": st.last_micro.vol_z,
                "ret_z": st.last_micro.ret_z,
            }
        return d

    def get_account_risk(self, account_id: str) -> Optional[Dict[str, Any]]:
        st = self._account_states.get(account_id)
        if st is None:
            return None
        return {
            "account_id": st.account_id,
            "last_ts": st.last_ts,
            "nav": st.nav,
            "nav_peak": st.nav_peak,
            "nav_intraday_peak": st.nav_intraday_peak,
            "dd_total": st.dd_total,
            "dd_intraday": st.dd_intraday,
            "gross_exposure": st.gross_exposure,
            "max_single_weight": st.max_single_weight,
            "risk_score_acct": st.risk_score_acct,
            "alerts": st.alerts,
        }

    def snapshot(self) -> RiskSnapshot:
        ts = time.time()

        symbols = {
            sym: self.get_symbol_risk(sym) or {}
            for sym in self._symbol_states.keys()
        }
        accounts = {
            aid: self.get_account_risk(aid) or {}
            for aid in self._account_states.keys()
        }

        # 汇总所有告警
        alerts: List[Dict[str, Any]] = []
        for sym, s in symbols.items():
            for a in s.get("alerts", []):
                alerts.append({"scope": "symbol", **a})
        for aid, s in accounts.items():
            for a in s.get("alerts", []):
                alerts.append({"scope": "account", **a})

        return RiskSnapshot(
            ts=ts,
            symbols=symbols,
            accounts=accounts,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _normalize_ts(self, ts_raw: Any) -> float:
        if ts_raw is None:
            return time.time()
        if isinstance(ts_raw, (int, float)):
            if ts_raw > 1e12:
                return ts_raw / 1000.0
            return float(ts_raw)
        try:
            return float(ts_raw)  # pragma: no cover
        except Exception:
            return time.time()    # pragma: no cover


__all__ = [
    "RiskBrain",
    "RiskSnapshot",
    "AccountRiskConfig",
    "AccountRiskState",
    "SymbolRiskState",
]
