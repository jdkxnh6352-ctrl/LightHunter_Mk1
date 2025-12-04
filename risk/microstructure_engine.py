# -*- coding: utf-8 -*-
"""
risk/microstructure_engine.py

LightHunter Mk3 - MicrostructureEngine
======================================

职责
----
1. 对 L1 行情 / 成交事件进行微观结构分析：
   - 订单流不平衡 (Order Imbalance)
   - 成交量/波动率短时爆点 (Volume / Volatility Spike)
   - 简单的“疑似虚假挂单”模式 (Spoofing-like Pattern)

2. 输出标的级的微观结构风险评分与异常告警：
   - risk_score_micro: [0, 1]，越高风险越高
   - anomalies: 若干 MicrostructureAlert

3. 供 RiskBrain 调用：
   - RiskBrain 负责账户级整合和统一风险评分；
   - MicrostructureEngine 专注于**单票微观层面**。

事件格式约定
------------
传入的 tick/行情事件推荐结构（dict）：

{
  "symbol": "600000.SH",
  "ts": 1672531200000,               # 毫秒时间戳或 datetime 可序列化对象
  "last_price": 10.23,
  "last_volume": 1200,               # 本笔成交量或本tick成交增量
  "bid1": 10.22,
  "ask1": 10.23,
  "bid1_volume": 35000,
  "ask1_volume": 21000
}

如果字段缺失，部分特征会跳过计算，但不报错。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

import math
import time

try:  # pragma: no cover
    from core.logging_utils import get_logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)


log = get_logger(__name__)


# ======================================================================
# 数据结构
# ======================================================================


@dataclass
class MicrostructureConfig:
    """微观结构检测参数。"""

    window_size: int = 30           # 最近多少个 tick 做统计
    min_ticks_for_eval: int = 10    # 至少多少个 tick 才开始评估

    # 订单流不平衡阈值（如 |imbalance| > 0.6）
    imbalance_threshold: float = 0.6

    # 成交量 spike 检测（当前 vs 最近均值 的倍率）
    vol_spike_ratio: float = 3.0

    # 价格波动 spike（当前 abs(return) vs 最近均值）
    ret_spike_ratio: float = 3.0

    # spoofing 模式（大买单/卖单瞬间撤单）简单检测
    spoof_min_depth: float = 50_000.0       # 盘口挂单最少量门槛
    spoof_drop_ratio: float = 0.5           # 掉单比例，如从 10w 掉到 4w
    spoof_small_trade_vol: float = 5_000.0  # 期间成交量很少则更可疑

    # 风险评分权重
    w_imbalance: float = 0.4
    w_vol_spike: float = 0.3
    w_ret_spike: float = 0.3


@dataclass
class MicrostructureSnapshot:
    """每个 symbol 的滚动窗口状态。"""

    symbol: str
    ticks: Deque[Dict[str, Any]]
    last_update_ts: float = 0.0
    last_bid1_volume: Optional[float] = None
    last_ask1_volume: Optional[float] = None


@dataclass
class MicrostructureAlert:
    symbol: str
    ts: float
    level: str          # "info" / "warning" / "critical"
    alert_type: str     # "VOL_SPIKE" / "IMBALANCE_EXTREME" / "SPOOFING_SUSPECT" / ...
    message: str
    metrics: Dict[str, float]


@dataclass
class MicrostructureRisk:
    symbol: str
    ts: float
    risk_score_micro: float        # [0, 1]
    imbalance: float
    vol_z: float
    ret_z: float
    alerts: List[MicrostructureAlert]


# ======================================================================
# 引擎实现
# ======================================================================


class MicrostructureEngine:
    """微观结构异常分析引擎。"""

    def __init__(self, config: Optional[MicrostructureConfig] = None) -> None:
        self.cfg = config or MicrostructureConfig()
        self._states: Dict[str, MicrostructureSnapshot] = {}

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def process_tick(self, tick: Dict[str, Any]) -> Optional[MicrostructureRisk]:
        """
        处理一条 L1/成交 tick，并返回微观结构风险结果。

        若数据不足，可能返回 None。
        """
        symbol = str(tick.get("symbol", "")).strip()
        if not symbol:
            return None

        ts = self._normalize_ts(tick.get("ts"))
        state = self._states.get(symbol)
        if state is None:
            state = MicrostructureSnapshot(
                symbol=symbol,
                ticks=deque(maxlen=self.cfg.window_size),
            )
            self._states[symbol] = state

        # 1) 更新滚动窗口
        state.ticks.append(self._normalize_tick(tick))
        state.last_update_ts = ts

        if len(state.ticks) < self.cfg.min_ticks_for_eval:
            return None

        # 2) 计算核心统计：imbalance / volume spike / return spike
        imbalance = self._calc_order_imbalance(state)
        vol_z = self._calc_volume_zscore(state)
        ret_z = self._calc_return_zscore(state)

        # 3) 基于盘口行为检测简单 spoofing 模式
        spoof_alert = self._detect_spoofing(state)

        # 4) 生成风险评分
        risk_score, alerts = self._aggregate_risk(
            symbol=symbol,
            ts=ts,
            imbalance=imbalance,
            vol_z=vol_z,
            ret_z=ret_z,
            spoof_alert=spoof_alert,
        )

        return MicrostructureRisk(
            symbol=symbol,
            ts=ts,
            risk_score_micro=risk_score,
            imbalance=imbalance,
            vol_z=vol_z,
            ret_z=ret_z,
            alerts=alerts,
        )

    def get_state(self, symbol: str) -> Optional[MicrostructureSnapshot]:
        return self._states.get(symbol)

    # ------------------------------------------------------------------
    # 内部：时间戳 & Tick 规范化
    # ------------------------------------------------------------------

    def _normalize_ts(self, ts_raw: Any) -> float:
        """将各种时间格式统一为 float 秒时间戳。"""
        if ts_raw is None:
            return time.time()
        if isinstance(ts_raw, (int, float)):
            # 粗略判断是否毫秒
            if ts_raw > 1e12:  # 毫秒级别
                return ts_raw / 1000.0
            return float(ts_raw)
        # 其它情况，比如 datetime 或字符串，简单处理
        try:
            # 让 pandas / datetime 自己解析在外层做，这里只处理数字
            return float(ts_raw)  # pragma: no cover
        except Exception:
            return time.time()     # pragma: no cover

    def _normalize_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """保证 tick 至少有 last_price / last_volume 字段。"""
        t = dict(tick)
        # 字段兼容：如果没有 last_volume，但有 volume_delta / volume，则尝试映射
        if "last_volume" not in t:
            if "volume_delta" in t:
                t["last_volume"] = t["volume_delta"]
            elif "volume" in t:
                t["last_volume"] = t["volume"]
            else:
                t["last_volume"] = 0.0
        if "last_price" not in t:
            if "price" in t:
                t["last_price"] = t["price"]
        return t

    # ------------------------------------------------------------------
    # 内部：核心统计
    # ------------------------------------------------------------------

    def _calc_order_imbalance(self, state: MicrostructureSnapshot) -> float:
        """
        简化版订单不平衡：
        imbalance = (bid1_volume - ask1_volume) / (bid1_volume + ask1_volume)

        若缺少盘口信息，则返回 0。
        """
        last = state.ticks[-1]
        bid = last.get("bid1_volume")
        ask = last.get("ask1_volume")

        if bid is None or ask is None:
            return 0.0

        bid = float(bid)
        ask = float(ask)
        denom = bid + ask
        if denom <= 0:
            return 0.0
        imbalance = (bid - ask) / denom
        return float(max(-1.0, min(imbalance, 1.0)))

    def _calc_volume_zscore(self, state: MicrostructureSnapshot) -> float:
        """
        最近 window 内的成交量 z-score：
        - 当前 last_volume 与历史均值比较。
        """
        vols = [float(t.get("last_volume", 0.0)) for t in state.ticks]
        if len(vols) < 3:
            return 0.0
        cur = vols[-1]
        hist = vols[:-1]
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(len(hist) - 1, 1)
        std = math.sqrt(var)
        if std <= 1e-8:
            return 0.0
        z = (cur - mean) / std
        return float(z)

    def _calc_return_zscore(self, state: MicrostructureSnapshot) -> float:
        """
        最近 window 内的价格变化 z-score：
        - 当前 return vs 历史 return 均值。
        """
        prices = [float(t.get("last_price", 0.0)) for t in state.ticks]
        if len(prices) < 3:
            return 0.0
        rets: List[float] = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 > 0:
                rets.append((p1 / p0) - 1.0)
        if len(rets) < 3:
            return 0.0

        cur = rets[-1]
        hist = rets[:-1]
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(len(hist) - 1, 1)
        std = math.sqrt(var)
        if std <= 1e-8:
            return 0.0
        z = (cur - mean) / std
        return float(z)

    # ------------------------------------------------------------------
    # 内部：简单 spoofing 检测
    # ------------------------------------------------------------------

    def _detect_spoofing(self, state: MicrostructureSnapshot) -> Optional[MicrostructureAlert]:
        """
        极简版“疑似虚假挂单”模式：

        条件示例：
        - 上一个 tick 时，大买单数量 >= spoof_min_depth；
        - 当前 tick 时，买一挂单量骤降 ≥ spoof_drop_ratio；
        - 期间成交量很小（说明挂单并没有真正成交）。

        这里只给出一个 **启发式** 检测，供 RiskBrain 做参考。
        """
        if len(state.ticks) < 2:
            return None

        cfg = self.cfg
        last = state.ticks[-1]
        prev = state.ticks[-2]

        prev_bid = prev.get("bid1_volume")
        cur_bid = last.get("bid1_volume")
        if prev_bid is None or cur_bid is None:
            return None

        prev_bid = float(prev_bid)
        cur_bid = float(cur_bid)

        if prev_bid < cfg.spoof_min_depth:
            return None

        drop = prev_bid - cur_bid
        if drop <= 0:
            return None

        if drop < cfg.spoof_min_depth * cfg.spoof_drop_ratio:
            return None

        # 成交量变化很小
        vols = [float(t.get("last_volume", 0.0)) for t in state.ticks[-5:]]
        total_vol = sum(vols)
        if total_vol > cfg.spoof_small_trade_vol:
            return None

        # 触发疑似 spoofing 告警
        ts = self._normalize_ts(last.get("ts"))
        msg = (
            f"疑似虚假挂单: bid1_volume 从 {prev_bid:.0f} 急剧下降到 {cur_bid:.0f}, "
            f"期间成交量≈{total_vol:.0f}"
        )
        alert = MicrostructureAlert(
            symbol=state.symbol,
            ts=ts,
            level="warning",
            alert_type="SPOOFING_SUSPECT",
            message=msg,
            metrics={
                "prev_bid1_volume": prev_bid,
                "cur_bid1_volume": cur_bid,
                "drop_volume": drop,
                "recent_trade_volume": total_vol,
            },
        )
        return alert

    # ------------------------------------------------------------------
    # 内部：风险评分聚合
    # ------------------------------------------------------------------

    def _aggregate_risk(
        self,
        symbol: str,
        ts: float,
        imbalance: float,
        vol_z: float,
        ret_z: float,
        spoof_alert: Optional[MicrostructureAlert],
    ) -> Tuple[float, List[MicrostructureAlert]]:
        cfg = self.cfg
        alerts: List[MicrostructureAlert] = []

        # 订单不平衡告警
        if abs(imbalance) >= cfg.imbalance_threshold:
            level = "warning" if abs(imbalance) < 0.8 else "critical"
            msg = f"订单不平衡极端: imbalance={imbalance:.2f}"
            alerts.append(
                MicrostructureAlert(
                    symbol=symbol,
                    ts=ts,
                    level=level,
                    alert_type="IMBALANCE_EXTREME",
                    message=msg,
                    metrics={"imbalance": imbalance},
                )
            )

        # 成交量 spike 告警
        if vol_z >= cfg.vol_spike_ratio:
            level = "info" if vol_z < cfg.vol_spike_ratio * 1.5 else "warning"
            msg = f"成交量短时暴增: vol_z={vol_z:.2f}"
            alerts.append(
                MicrostructureAlert(
                    symbol=symbol,
                    ts=ts,
                    level=level,
                    alert_type="VOL_SPIKE",
                    message=msg,
                    metrics={"vol_z": vol_z},
                )
            )

        # 价格波动 spike 告警
        if abs(ret_z) >= cfg.ret_spike_ratio:
            level = "info" if abs(ret_z) < cfg.ret_spike_ratio * 1.5 else "warning"
            msg = f"价格短时剧烈波动: ret_z={ret_z:.2f}"
            alerts.append(
                MicrostructureAlert(
                    symbol=symbol,
                    ts=ts,
                    level=level,
                    alert_type="RET_SPIKE",
                    message=msg,
                    metrics={"ret_z": ret_z},
                )
            )

        if spoof_alert is not None:
            alerts.append(spoof_alert)

        # 聚合成 [0, 1] 风险评分（启发式）
        # 使用 tanh 对 z-score 做平滑，防止爆炸。
        def squash(x: float) -> float:
            return math.tanh(abs(x))

        score = (
            cfg.w_imbalance * squash(imbalance / max(cfg.imbalance_threshold, 1e-3))
            + cfg.w_vol_spike * squash(vol_z / max(cfg.vol_spike_ratio, 1e-3))
            + cfg.w_ret_spike * squash(ret_z / max(cfg.ret_spike_ratio, 1e-3))
        )

        # 如果有 critical 告警，score 提高至至少 0.8
        if any(a.level == "critical" for a in alerts):
            score = max(score, 0.8)

        score = float(max(0.0, min(score, 1.0)))
        return score, alerts


__all__ = [
    "MicrostructureConfig",
    "MicrostructureSnapshot",
    "MicrostructureAlert",
    "MicrostructureRisk",
    "MicrostructureEngine",
]
