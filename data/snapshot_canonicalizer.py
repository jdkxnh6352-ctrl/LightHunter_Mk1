from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:
    from core.logging_utils import get_logger
except Exception:  # pragma: no cover - fallback for early bootstrap
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


log = get_logger(__name__)


Number = Union[int, float, str]


@dataclass
class CanonicalSnapshot:
    """
    统一后的 L1 快照结构，用于后续订单流重构 / 特征工程。

    所有数据均使用“标准化单位”：
    - 价格: 元
    - 成交量: 股
    - 成交额: 元
    - 时间戳: 本地时间（通常为 Asia/Shanghai），由上层保证或就地生成
    """

    ts: datetime

    # 代码与市场
    code: str                # 统一格式：600000.SH / 000001.SZ / 688001.SH ...
    raw_symbol: str          # 原始代码：sh600000 / sz000001 / 600000 等
    exchange: str            # "SH" / "SZ" / "BJ" / "IDX" / ""

    # 核心价格
    last: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None

    # 成交
    volume: Optional[int] = None     # 股
    amount: Optional[float] = None   # 元

    # 五档盘口（不一定都有）
    bid_prices: List[float] = field(default_factory=list)
    bid_volumes: List[int] = field(default_factory=list)  # 股
    ask_prices: List[float] = field(default_factory=list)
    ask_volumes: List[int] = field(default_factory=list)  # 股

    # 元数据
    source: str = ""                 # tencent / eastmoney_clist / sina / generic ...
    extra: Dict[str, Any] = field(default_factory=dict)


def _to_float(value: Optional[Number], default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip()
    if not v or v in {"-", "null", "None"}:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(value: Optional[Number], default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    v = str(value).strip()
    if not v or v in {"-", "null", "None"}:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def normalize_symbol(symbol: str) -> Tuple[str, str, str]:
    """
    将各种乱七八糟的代码统一为：
        - code: 600000.SH / 000001.SZ / 688001.SH
        - raw_symbol: 保留原始形式（sh600000 / 600000 / SZ000001 等）
        - exchange: SH / SZ / BJ / IDX / ""
    """
    s = (symbol or "").strip()
    raw_symbol = s
    s_lower = s.lower().replace(" ", "")

    prefix = ""
    code = s_lower

    if s_lower.startswith(("sh", "sz", "bj", "hk")):
        prefix, code = s_lower[:2], s_lower[2:]
    else:
        code = s_lower

    if prefix == "sh":
        exchange = "SH"
    elif prefix == "sz":
        exchange = "SZ"
    elif prefix == "bj":
        exchange = "BJ"
    else:
        # 根据代码首位猜测（只用于 A 股，大概够用）
        if code.startswith(("5", "6")):
            exchange = "SH"
        elif code.startswith(("0", "1", "2", "3")):
            exchange = "SZ"
        else:
            exchange = ""

    canonical = f"{code.upper()}.{exchange}" if exchange else code.upper()
    return canonical, raw_symbol, exchange


class SnapshotCanonicalizer:
    """
    多站点 L1 快照标准化适配器。

    典型用法：
        canon = SnapshotCanonicalizer()
        snap = canon.canonicalize("tencent", raw_text, symbol_hint="sh600000")
    """

    def __init__(self) -> None:
        self._log = log

    # ---- 对外主入口 ----------------------------------------------------- #

    def canonicalize(
        self,
        source: str,
        raw: Any,
        *,
        symbol_hint: Optional[str] = None,
        ts_hint: Optional[datetime] = None,
    ) -> CanonicalSnapshot:
        src = (source or "").lower()

        if src in {"tencent", "qt", "gtimg"}:
            snap = self._from_tencent(raw, symbol_hint, ts_hint)
        elif src in {"eastmoney_clist", "eastmoney-list", "em_clist"}:
            snap = self._from_eastmoney_clist(raw, symbol_hint, ts_hint)
        elif src in {"sina", "sinajs"}:
            snap = self._from_sina(raw, symbol_hint, ts_hint)
        else:
            snap = self._from_generic(raw, symbol_hint, ts_hint)

        return snap

    # ---- 具体站点实现 --------------------------------------------------- #

    # Tencent: http://qt.gtimg.cn/q=sz000858
    # 字段定义见公开资料:
    #   0: 未知
    #   1: 名字
    #   2: 代码
    #   3: 当前价格
    #   4: 昨收
    #   5: 今开
    #   6: 成交量（手）
    #   7: 外盘
    #   8: 内盘
    #   9: 买一
    #  10: 买一量（手）
    #  11-18: 买二~买五
    #  19: 卖一
    #  20: 卖一量
    #  21-28: 卖二~卖五
    #  29: 最近逐笔成交
    #  30: 时间
    #  31: 涨跌
    #  32: 涨跌%
    #  33: 最高
    #  34: 最低
    #  35: 价格/成交量（手）/成交额
    #  36: 成交量（手）
    #  37: 成交额（万）
    #  38: 换手率
    #  39: 市盈率
    #  44: 流通市值
    #  45: 总市值
    #  47: 涨停价
    #  48: 跌停价

    def _split_tencent_line(self, text: str) -> Tuple[str, List[str]]:
        line = (text or "").strip()
        if not line:
            raise ValueError("empty tencent snapshot line")

        if line.endswith(";"):
            line = line[:-1]

        # v_sz000858="...."
        prefix, _, rest = line.partition("=")
        symbol = prefix.strip()
        if symbol.startswith("v_"):
            symbol = symbol[2:]

        rest = rest.strip()
        if rest and rest[0] in {"'", '"'} and rest[-1] in {"'", '"'}:
            rest = rest[1:-1]

        fields = rest.split("~")
        return symbol, fields

    def _from_tencent(
        self,
        raw: Any,
        symbol_hint: Optional[str],
        ts_hint: Optional[datetime],
    ) -> CanonicalSnapshot:
        if isinstance(raw, str):
            raw_symbol, fields = self._split_tencent_line(raw)
        elif isinstance(raw, (list, tuple)):
            raw_symbol = symbol_hint or ""
            fields = list(raw)
        else:
            raise TypeError(f"Unsupported Tencent raw type: {type(raw)!r}")

        symbol_for_norm = symbol_hint or raw_symbol
        code, raw_symbol_norm, exchange = normalize_symbol(symbol_for_norm)

        def f(idx: int, default: Optional[str] = None) -> Optional[str]:
            try:
                v = fields[idx]
            except IndexError:
                return default
            return v

        def ff(idx: int, default: float = 0.0) -> float:
            return _to_float(f(idx), default)

        now = datetime.now()
        t_str = (f(30) or "").strip()
        if ts_hint is not None:
            ts = ts_hint
        elif t_str and ":" in t_str:
            try:
                parts = [int(x) for x in t_str.split(":")]
                h = parts[0]
                m = parts[1] if len(parts) >= 2 else 0
                s = parts[2] if len(parts) >= 3 else 0
                ts = datetime(now.year, now.month, now.day, h, m, s)
            except Exception:
                ts = now
        else:
            ts = now

        last = ff(3)
        prev_close = ff(4) or None
        open_price = ff(5) or None

        vol_hand = ff(6)
        if vol_hand <= 0:
            vol_hand = ff(36)
        volume = int(vol_hand * 100) if vol_hand > 0 else None

        high = ff(33) or None
        low = ff(34) or None

        amount_10k = ff(37)
        amount = amount_10k * 10000 if amount_10k > 0 else None

        bid_prices: List[float] = []
        bid_volumes: List[int] = []
        ask_prices: List[float] = []
        ask_volumes: List[int] = []

        # 买一~买五
        price_idx = 9
        for _ in range(5):
            p = ff(price_idx)
            v_hand = ff(price_idx + 1)
            if p > 0 and v_hand > 0:
                bid_prices.append(p)
                bid_volumes.append(int(v_hand * 100))
            price_idx += 2

        # 卖一~卖五
        price_idx = 19
        for _ in range(5):
            p = ff(price_idx)
            v_hand = ff(price_idx + 1)
            if p > 0 and v_hand > 0:
                ask_prices.append(p)
                ask_volumes.append(int(v_hand * 100))
            price_idx += 2

        extra: Dict[str, Any] = {
            "name": f(1) or "",
            "raw_code": f(2) or "",
            "change": ff(31),
            "change_pct": ff(32),
            "turnover_ratio": ff(38),
            "pe": ff(39),
            "float_mkt_cap": ff(44),
            "total_mkt_cap": ff(45),
            "upper_limit": ff(47),
            "lower_limit": ff(48),
        }

        return CanonicalSnapshot(
            ts=ts,
            code=code,
            raw_symbol=raw_symbol_norm,
            exchange=exchange,
            last=last,
            open=open_price,
            high=high,
            low=low,
            prev_close=prev_close,
            volume=volume,
            amount=amount,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            source="tencent",
            extra=extra,
        )

    # Eastmoney: push2.eastmoney.com/api/qt/clist/get
    # 典型字段映射（clist）：
    #   f2  最新价
    #   f3  涨跌幅
    #   f4  涨跌额
    #   f5  总手
    #   f6  成交额
    #   f12 股票代码
    #   f13 市场
    #   f14 股票名称
    #   f15 最高价
    #   f16 最低价
    #   f17 开盘价
    #   f18 昨收
    #   f20 总市值
    #   f21 流通市值

    def _from_eastmoney_clist(
        self,
        raw: Any,
        symbol_hint: Optional[str],
        ts_hint: Optional[datetime],
    ) -> CanonicalSnapshot:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Eastmoney clist snapshot expects Mapping, got {type(raw)!r}"
            )

        d: Mapping[str, Any] = raw
        code_in_data = d.get("f12") or d.get("code") or symbol_hint or ""
        code, raw_symbol, exchange = normalize_symbol(str(code_in_data))

        now = datetime.now()
        ts = ts_hint or now

        last = _to_float(d.get("f2"))
        prev_close = _to_float(d.get("f18")) or None
        open_price = _to_float(d.get("f17")) or None
        high = _to_float(d.get("f15")) or None
        low = _to_float(d.get("f16")) or None

        vol_hand = _to_float(d.get("f5"))
        volume = int(vol_hand * 100) if vol_hand > 0 else None

        amount = _to_float(d.get("f6")) or None

        extra: Dict[str, Any] = {
            "name": d.get("f14") or d.get("name") or "",
            "market_raw": d.get("f13"),
            "change_pct": _to_float(d.get("f3")),
            "change": _to_float(d.get("f4")),
            "turnover_ratio": _to_float(d.get("f8")),
            "pe": _to_float(d.get("f9")),
            "volume_hand": vol_hand,
            "total_mkt_cap": _to_float(d.get("f20")),
            "float_mkt_cap": _to_float(d.get("f21")),
        }

        return CanonicalSnapshot(
            ts=ts,
            code=code,
            raw_symbol=raw_symbol,
            exchange=exchange,
            last=last,
            open=open_price,
            high=high,
            low=low,
            prev_close=prev_close,
            volume=volume,
            amount=amount,
            bid_prices=[],
            bid_volumes=[],
            ask_prices=[],
            ask_volumes=[],
            source="eastmoney_clist",
            extra=extra,
        )

    # Sina: http://hq.sinajs.cn/list=sh601006
    # 字段定义（简化版）：
    #   0: 名称
    #   1: 今日开盘价
    #   2: 昨日收盘价
    #   3: 当前价格
    #   4: 今日最高价
    #   5: 今日最低价
    #   8: 成交量（股）
    #   9: 成交额（元）
    #  10-19 买一~买五 手数 & 价格
    #  20-29 卖一~卖五 手数 & 价格
    #  30: 日期 YYYY-MM-DD
    #  31: 时间 HH:MM:SS

    def _split_sina_line(self, text: str) -> Tuple[str, List[str]]:
        line = (text or "").strip()
        if not line:
            raise ValueError("empty sina snapshot line")

        if line.endswith(";"):
            line = line[:-1]

        # var hq_str_sh601006="....";
        prefix, _, rest = line.partition("=")
        symbol = prefix.strip()
        if symbol.startswith("var"):
            # var hq_str_sh601006
            _, _, tail = symbol.partition("hq_str_")
            symbol = tail.strip()

        rest = rest.strip()
        if rest and rest[0] in {"'", '"'} and rest[-1] in {"'", '"'}:
            rest = rest[1:-1]

        fields = [x.strip() for x in rest.split(",")]
        return symbol, fields

    def _from_sina(
        self,
        raw: Any,
        symbol_hint: Optional[str],
        ts_hint: Optional[datetime],
    ) -> CanonicalSnapshot:
        if isinstance(raw, str):
            raw_symbol, fields = self._split_sina_line(raw)
        elif isinstance(raw, (list, tuple)):
            raw_symbol = symbol_hint or ""
            fields = list(raw)
        else:
            raise TypeError(f"Unsupported Sina raw type: {type(raw)!r}")

        symbol_for_norm = symbol_hint or raw_symbol
        code, raw_symbol_norm, exchange = normalize_symbol(symbol_for_norm)

        def f(idx: int, default: Optional[str] = None) -> Optional[str]:
            try:
                return fields[idx]
            except IndexError:
                return default

        def ff(idx: int, default: float = 0.0) -> float:
            return _to_float(f(idx), default)

        # 时间
        if ts_hint is not None:
            ts = ts_hint
        else:
            date_s = (f(30) or "").strip()
            time_s = (f(31) or "").strip()
            if date_s and time_s:
                try:
                    ts = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = datetime.now()
            else:
                ts = datetime.now()

        last = ff(3)
        prev_close = ff(2) or None
        open_price = ff(1) or None
        high = ff(4) or None
        low = ff(5) or None

        volume = _to_int(f(8)) or None
        amount = ff(9) or None

        bid_prices: List[float] = []
        bid_volumes: List[int] = []
        ask_prices: List[float] = []
        ask_volumes: List[int] = []

        # 买一~买五： (量, 价) 从 idx=10 开始
        idx = 10
        for _ in range(5):
            vol = _to_int(f(idx))
            price = ff(idx + 1)
            if vol > 0 and price > 0:
                bid_prices.append(price)
                bid_volumes.append(vol)
            idx += 2

        # 卖一~卖五： (量, 价) 从 idx=20 开始
        idx = 20
        for _ in range(5):
            vol = _to_int(f(idx))
            price = ff(idx + 1)
            if vol > 0 and price > 0:
                ask_prices.append(price)
                ask_volumes.append(vol)
            idx += 2

        extra: Dict[str, Any] = {
            "name": f(0) or "",
        }

        return CanonicalSnapshot(
            ts=ts,
            code=code,
            raw_symbol=raw_symbol_norm,
            exchange=exchange,
            last=last,
            open=open_price,
            high=high,
            low=low,
            prev_close=prev_close,
            volume=volume,
            amount=amount,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            source="sina",
            extra=extra,
        )

    # 通用兜底：适配已经半结构化的 dict / pydantic 对象 / dataclass
    def _from_generic(
        self,
        raw: Any,
        symbol_hint: Optional[str],
        ts_hint: Optional[datetime],
    ) -> CanonicalSnapshot:
        """
        兜底逻辑：
        - 如果是 Mapping，尝试从常见 key 中提取字段
        - 否则直接抛异常（由上层捕获）
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"Generic snapshot expects Mapping, got {type(raw)!r}")

        d: Mapping[str, Any] = raw

        code_candidate = (
            d.get("code")
            or d.get("symbol")
            or d.get("ts_code")
            or symbol_hint
            or ""
        )
        code, raw_symbol, exchange = normalize_symbol(str(code_candidate))

        ts = ts_hint or d.get("ts") or d.get("timestamp") or datetime.now()
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts))
            except Exception:
                ts = datetime.now()

        # 尝试各种常见命名
        last = _to_float(
            d.get("last")
            or d.get("price")
            or d.get("close")
            or d.get("last_price")
        )

        open_price = _to_float(d.get("open") or d.get("open_price"))
        high = _to_float(d.get("high") or d.get("high_price"))
        low = _to_float(d.get("low") or d.get("low_price"))
        prev_close = _to_float(
            d.get("prev_close")
            or d.get("pre_close")
            or d.get("last_close")
        )

        vol = d.get("volume") or d.get("vol")
        amount = d.get("amount") or d.get("turnover")

        volume = _to_int(vol) or None
        amount_val = _to_float(amount) or None

        bid_prices = list(d.get("bid_prices") or d.get("bids") or [])
        ask_prices = list(d.get("ask_prices") or d.get("asks") or [])

        bid_volumes = [int(v) for v in (d.get("bid_volumes") or [])]
        ask_volumes = [int(v) for v in (d.get("ask_volumes") or [])]

        extra_keys = set(d.keys()) - {
            "code",
            "symbol",
            "ts_code",
            "ts",
            "timestamp",
            "last",
            "price",
            "close",
            "last_price",
            "open",
            "open_price",
            "high",
            "high_price",
            "low",
            "low_price",
            "prev_close",
            "pre_close",
            "last_close",
            "volume",
            "vol",
            "amount",
            "turnover",
            "bid_prices",
            "bids",
            "ask_prices",
            "asks",
            "bid_volumes",
            "ask_volumes",
        }
        extra: Dict[str, Any] = {k: d[k] for k in extra_keys}

        return CanonicalSnapshot(
            ts=ts,
            code=code,
            raw_symbol=raw_symbol,
            exchange=exchange,
            last=last,
            open=open_price or None,
            high=high or None,
            low=low or None,
            prev_close=prev_close or None,
            volume=volume,
            amount=amount_val,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            source="generic",
            extra=extra,
        )


# 方便直接函数式调用
_default_canon = SnapshotCanonicalizer()


def canonicalize_snapshot(
    source: str,
    raw: Any,
    *,
    symbol_hint: Optional[str] = None,
    ts_hint: Optional[datetime] = None,
) -> CanonicalSnapshot:
    return _default_canon.canonicalize(source, raw, symbol_hint=symbol_hint, ts_hint=ts_hint)
