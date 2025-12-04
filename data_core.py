# -*- coding: utf-8 -*-
"""
data_core.py

基础数据工具模块（代码规范化、市场标识转换等）。

职责：
- 规范化股票代码（去掉 .SZ/.SH 等后缀）
- 转换为腾讯行情使用的 symbol（如 000001 -> sz000001，600000 -> sh600000）
- 简单的股票池加载函数（从文本文件读取）

后续如果有：
- 从本地数据库加载全市场代码
- 从 akshare / 官方接口拉股票列表
也可以集中放在这个模块里。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.logging_utils import get_logger

logger = get_logger(__name__)

# 以 data_core.py 所在目录作为“项目根目录”
PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_code(code: str) -> str:
    """
    规范化股票代码：
    - 去掉空格；
    - 去掉 .SZ / .SH / sz / sh 后缀；
    - 确保返回 6 位数字（如有多余前缀则保留末 6 位）。
    """
    if code is None:
        return ""

    c = str(code).strip().upper()
    c = c.replace(".SZ", "").replace(".SH", "")
    c = c.replace("SZ", "").replace("SH", "")
    # 只保留末 6 位数字
    digits = "".join(ch for ch in c if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
    return digits


def to_tencent_symbol(code: str) -> str:
    """
    将 6 位股票代码转换为腾讯行情的 symbol：
    - 以 5/6/9 开头的视为上交所：shXXXXXX
    - 否则视为深交所：szXXXXXX
    """
    digits = normalize_code(code)
    if not digits:
        return ""

    if digits.startswith(("5", "6", "9")):
        return "sh" + digits
    return "sz" + digits


def load_stock_universe(path: Optional[str] = None) -> List[str]:
    """
    从文本文件加载股票池（每行一个代码），返回代码列表（已规范化）。

    Args:
        path: 文件路径，可以是相对路径（相对项目根目录）或绝对路径。
              若为 None 或文件不存在，则返回空列表。

    文件示例：
        000001
        600000
        300750.SZ
        688686.SH
    """
    if not path:
        logger.warning("load_stock_universe: 未指定 path，返回空列表")
        return []

    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path).resolve()

    if not p.exists():
        logger.warning("load_stock_universe: 股票池文件不存在：%s", p)
        return []

    codes: List[str] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                code = normalize_code(line)
                if code:
                    codes.append(code)
    except Exception:
        logger.exception("读取股票池文件失败：%s", p)
        return []

    logger.info("加载股票池：%d 条代码，自 %s", len(codes), p)
    return codes

# ----------------------------------------------------------------------
# MarketHunter：小规模实时行情猎手（封装 MarketAdapter）
# ----------------------------------------------------------------------


class MarketHunter:
    """
    MarketHunter - 轻量版“数据猎手”。

    设计目标：
        1. 对接 adapters.market_adapter.MarketAdapter，负责从腾讯 / 新浪等站点抓实时行情；
        2. 把数据整理成后续模块喜欢的“中文列名”格式；
        3. 保持接口简单，先服务小规模测试 & 后面 TSCollector。

    目前版本侧重：
        - A 股股票（6 位代码，不带交易所前缀）；
        - 默认以腾讯行情为主（source_id = "tx_quote"）；
        - 只做一个很简单的字段映射，其它高级因子先占位为 NaN/0。
    """

    def __init__(self, *, source_id: str = "tx_quote") -> None:
        # 延迟导入，避免 data_core <-> market_adapter 循环引用
        from core.logging_utils import get_logger
        from adapters.market_adapter import MarketAdapter

        self._adapter = MarketAdapter()
        self.log = get_logger(self.__class__.__name__)
        self.source_id = source_id

        # 默认的小测试股票池（跟前面的抓数脚本保持一致）
        self._default_universe = ["000001", "600000", "300750", "002230", "605119"]

    # ------------------------------------------------------------------
    def fetch_snapshot(
        self,
        codes: list[str] | None = None,
        *,
        source_id: str | None = None,
    ):
        """
        抓一轮实时快照，返回 pandas.DataFrame。

        参数
        ----
        codes     : 股票代码列表（不带交易所前缀），为空时用内置的小股票池；
        source_id : 数据源 ID，默认 "tx_quote"（腾讯）。

        返回
        ----
        DataFrame，核心列：
            - 代码
            - 名称
            - 现价
            - 涨幅（百分比）
            - 成交量
            - 成交额
        外加若干占位列：
            - 换手率
            - 主力攻击系数
            - 板块热度
            - VWAP
            - BA_Ratio
        """
        import pandas as pd  # 局部导入，避免顶层依赖太多

        if codes is None:
            codes = list(self._default_universe)
        else:
            codes = [str(c).zfill(6) for c in codes]

        sid = source_id or self.source_id

        df_raw = self._adapter.get_realtime_quotes(codes, source_id=sid)

        if df_raw is None or df_raw.empty:
            self.log.warning("MarketHunter.fetch_snapshot: 空结果 codes=%s source_id=%s", codes, sid)
            # 返回一个结构正确但为空的 DataFrame
            return pd.DataFrame(
                columns=[
                    "代码",
                    "名称",
                    "现价",
                    "涨幅",
                    "成交量",
                    "成交额",
                    "换手率",
                    "主力攻击系数",
                    "板块热度",
                    "VWAP",
                    "BA_Ratio",
                ]
            )

        # 统一到我们后续因子 / 回测喜欢的命名
        df = pd.DataFrame()
        df["代码"] = df_raw["code"].astype(str).str.zfill(6)
        df["名称"] = df_raw.get("name", "").astype(str)

        # 价格 / 涨幅
        df["现价"] = pd.to_numeric(df_raw.get("price"), errors="coerce")
        if "change_pct" in df_raw.columns:
            df["涨幅"] = pd.to_numeric(df_raw["change_pct"], errors="coerce")
        elif "pct" in df_raw.columns:
            df["涨幅"] = pd.to_numeric(df_raw["pct"], errors="coerce")
        else:
            df["涨幅"] = pd.NA

        # 成交量 / 成交额
        df["成交量"] = pd.to_numeric(df_raw.get("volume"), errors="coerce")
        df["成交额"] = pd.to_numeric(df_raw.get("amount"), errors="coerce")

        # 先放一些占位列，后面我们会用分钟级数据 / 微观结构来补全
        df["换手率"] = pd.NA
        df["主力攻击系数"] = 0.0
        df["板块热度"] = pd.NA
        df["VWAP"] = df["现价"]
        df["BA_Ratio"] = pd.NA

        # 保持列顺序友好一点
        cols = [
            "代码",
            "名称",
            "现价",
            "涨幅",
            "成交量",
            "成交额",
            "换手率",
            "主力攻击系数",
            "板块热度",
            "VWAP",
            "BA_Ratio",
        ]
        df = df[[c for c in cols if c in df.columns]]

        return df


# 如果原来模块里已经有 __all__，把 MarketHunter 也挂进去
if "__all__" in globals():
    try:
        __all__ = list(__all__) + ["MarketHunter"]
    except Exception:
        pass



