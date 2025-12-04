# -*- coding: utf-8 -*-
"""
adapters/market_adapter.py

MarketAdapter：行情适配层

职责：
- 基于 RequestEngine-S + SourceCatalog，实现对不同行情源的统一访问；
- 当前优先实现腾讯行情 (tx_quote)，后续可扩展同花顺 / 东方财富等；
- 对外输出标准化的 DataFrame（code / name / price / change / change_pct / volume / amount）。

说明：
- 这里只实现了 A 股常用的“简版行情”（适合买在盘中选股、TS 建模使用）；
- 具体字段可根据科研需要继续扩展；
- 访问频率由 SourceCatalog + RequestEngine-S 控制，保持合规、降低被封风险。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from core.logging_utils import get_logger
from request_engine import get_request_engine
from data.source_catalog import get_data_source_config, DataSourceConfig
from data_core import to_tencent_symbol

logger = get_logger(__name__)


def _safe_float(s: Any) -> Optional[float]:
    try:
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(s: Any) -> Optional[int]:
    try:
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


class MarketAdapter:
    """
    行情适配器入口。

    用法示例：

        from adapters.market_adapter import MarketAdapter

        adapter = MarketAdapter()
        df = adapter.get_realtime_quotes(["000001", "600000"], source_id="tx_quote")
        print(df.head())

    """

    def __init__(self) -> None:
        self._engine = get_request_engine()
        self._logger = get_logger(self.__class__.__name__)

    # ----------------- 对外主接口 ----------------- #

    def get_realtime_quotes(
        self,
        codes: List[str],
        source_id: str = "tx_quote",
    ) -> pd.DataFrame:
        """
        获取一组股票的实时简要行情（当前价、涨跌幅、成交量等）。

        Args:
            codes:    6 位股票代码列表（如 ["000001","600000"]，不带 .SZ/.SH 也可以）
            source_id: 数据源 ID，对应 system_config.data_sources 的键。
                       当前实现： "tx_quote"（腾讯）

        Returns:
            DataFrame，列包含：
                - code:      证券代码（6位）
                - name:      名称
                - price:     当前价
                - change:    涨跌额
                - change_pct:涨跌幅（百分数）
                - volume:    成交量（手）
                - amount:    成交额（万）
                - source:    源 ID
        """
        if not codes:
            self._logger.warning("get_realtime_quotes 被调用，但 codes 为空，返回空 DataFrame")
            return pd.DataFrame(
                columns=[
                    "code",
                    "name",
                    "price",
                    "change",
                    "change_pct",
                    "volume",
                    "amount",
                    "source",
                ]
            )

        if source_id == "tx_quote":
            return self._get_tx_quotes(codes, source_id=source_id)

        # 未来可以扩展更多源（ths_quote / em_quote 等）
        raise NotImplementedError(f"暂未实现 source_id={source_id} 的行情适配")

    # ----------------- 腾讯行情实现 ----------------- #

    def _get_tx_quotes(
        self,
        codes: List[str],
        source_id: str,
    ) -> pd.DataFrame:
        """
        调用腾讯行情接口，返回简化后的行情 DataFrame。
        """

        cfg: Optional[DataSourceConfig] = get_data_source_config(source_id)
        base_url = (cfg.base_url if cfg and cfg.base_url else "https://qt.gtimg.cn").rstrip("/")
        url = f"{base_url}/q"

        symbols: List[str] = [to_tencent_symbol(c) for c in codes]
        # 使用 s_ 前缀的简版行情
        q_param = "s_" + ",s_".join(symbols)

        self._logger.debug("Requesting Tencent quotes: url=%s q=%s", url, q_param)

        text = self._engine.get_text(
            url,
            params={"q": q_param},
            source_id=source_id,
        )

        # 返回内容类似： v_s_sz000001="51~平安银行~000001~12.34~0.24~2.00~123456~7890~...";v_s_sh600000="..."
        pattern = re.compile(r'v_[^=]+="([^"]*)"')
        matches = pattern.findall(text)

        rows: List[Dict[str, Any]] = []
        for payload in matches:
            parts = payload.split("~")
            # s_ 简版一般 >= 8 个字段，稳妥起见做长度检查
            if len(parts) < 8:
                continue

            name = parts[1]
            code = parts[2]
            price = _safe_float(parts[3])
            change = _safe_float(parts[4])
            change_pct = _safe_float(parts[5])
            volume = _safe_int(parts[6])
            amount = _safe_float(parts[7])

            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": price,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": volume,
                    "amount": amount,
                    "source": source_id,
                }
            )

        df = pd.DataFrame(rows)
        # 防御：保证列顺序
        if not df.empty:
            df = df[
                [
                    "code",
                    "name",
                    "price",
                    "change",
                    "change_pct",
                    "volume",
                    "amount",
                    "source",
                ]
            ]
        else:
            self._logger.warning(
                "从腾讯行情返回的数据无法解析或为空，codes=%s", ",".join(codes)
            )

        return df
