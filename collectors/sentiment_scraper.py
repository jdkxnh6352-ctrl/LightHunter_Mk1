# -*- coding: utf-8 -*-
"""
collectors/sentiment_scraper.py

LightHunter Mk3 - 舆情采集器（情绪因子链路起点）
================================================

设计目标
--------
- 统一采集接口：支持多站点（东财股吧 / 雪球 / 其它扩展）
- 统一输出格式：symbol × ts × 文本内容，用于后续 SentimentEngine 打分
- 优先复用 RequestEngine（代理 / 限速 / UA 池），避免 IP 冻结

注意
----
1. 各网站接口经常更新，本模块提供的是**稳定骨架** + 一份可运行的初版解析逻辑。
   如需追求极致稳定性，可以针对常用站点单独调优解析逻辑。
2. 大量抓取前，请先小规模测试，确认不会触发封禁。
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from core.logging_utils import get_logger

try:
    from config.config_center import get_system_config
except ImportError:  # 方便独立测试
    def get_system_config() -> Dict[str, Any]:
        return {}

# 优先尝试使用项目内的 RequestEngine
try:
    from request_engine import RequestEngine  # type: ignore
except ImportError:  # 回退到 requests
    RequestEngine = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None

log = get_logger(__name__)


@dataclass
class SentimentScraperConfig:
    """舆情采集配置。"""

    sources: List[str]
    max_posts_per_symbol: int = 80
    timeout_sec: float = 8.0

    @classmethod
    def from_system_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "SentimentScraperConfig":
        if cfg is None:
            cfg = get_system_config()
        scfg = cfg.get("sentiment", {}) if isinstance(cfg, dict) else {}

        sources = scfg.get("sources", ["eastmoney_guba", "xueqiu"])
        max_posts = int(scfg.get("max_posts_per_symbol", 80))
        timeout = float(scfg.get("timeout_sec", 8.0))

        return cls(
            sources=sources,
            max_posts_per_symbol=max_posts,
            timeout_sec=timeout,
        )


class SentimentScraper:
    """
    舆情采集主类。

    用法示例（在 NightOps / 单独脚本里）：
        cfg = get_system_config()
        scraper = SentimentScraper(cfg)
        df_raw = scraper.scrape_for_symbols(["000001", "300750"], trade_date="2024-01-05")
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self._raw_cfg = cfg or get_system_config()
        self.cfg = SentimentScraperConfig.from_system_config(self._raw_cfg)
        self._re = self._init_request_engine(self._raw_cfg)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def scrape_for_symbols(
        self,
        symbols: Iterable[str],
        trade_date: Optional[str | dt.date | dt.datetime] = None,
    ) -> pd.DataFrame:
        """
        对一组 symbol 在指定日期附近采集舆情帖子。

        参数
        ----
        symbols    : 股票代码列表（不带交易所后缀或自定义）
        trade_date : 交易日期（可选），用于部分站点按日期过滤/排序

        返回
        ----
        DataFrame:
            columns = [
                "symbol", "ts", "source", "post_id", "title", "content",
                "author", "reply_count", "like_count", "url"
            ]
        """
        symbol_list = [str(s) for s in symbols]
        if not symbol_list:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "ts",
                    "source",
                    "post_id",
                    "title",
                    "content",
                    "author",
                    "reply_count",
                    "like_count",
                    "url",
                ]
            )

        if trade_date is None:
            date = pd.Timestamp.today().normalize()
        else:
            date = pd.to_datetime(trade_date).normalize()

        records: List[Dict[str, Any]] = []
        for sym in symbol_list:
            for source in self.cfg.sources:
                func = getattr(self, f"_fetch_from_{source}", None)
                if func is None:
                    log.warning("SentimentScraper: 未实现 source=%s，跳过。", source)
                    continue
                try:
                    posts = func(sym, date)
                    records.extend(posts)
                except Exception as e:  # pragma: no cover - 网络/解析异常
                    log.warning("SentimentScraper: symbol=%s, source=%s 抓取异常: %s", sym, source, e)

        if not records:
            log.info("SentimentScraper: symbols=%d 未抓到任何舆情记录。", len(symbol_list))
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "ts",
                    "source",
                    "post_id",
                    "title",
                    "content",
                    "author",
                    "reply_count",
                    "like_count",
                    "url",
                ]
            )

        df = pd.DataFrame.from_records(records)
        # 统一字段类型
        df["symbol"] = df["symbol"].astype(str)
        df["source"] = df["source"].astype(str)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        for col in ["reply_count", "like_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        return df

    # ------------------------------------------------------------------
    # 内部：请求工具
    # ------------------------------------------------------------------

    def _init_request_engine(self, cfg: Dict[str, Any]):
        if RequestEngine is None:
            if requests is None:
                log.warning("RequestEngine 与 requests 均不可用，SentimentScraper 将无法工作。")
                return None
            log.info("SentimentScraper: 未找到项目内 RequestEngine，将使用 requests 回退。")
            return None
        try:
            return RequestEngine(cfg)
        except Exception as e:  # pragma: no cover
            log.warning("SentimentScraper: 初始化 RequestEngine 失败，使用 requests 回退: %s", e)
            return None

    def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """统一 GET 请求入口。"""
        timeout = self.cfg.timeout_sec
        headers = headers or {}
        # 简单 UA，RequestEngine 自带 UA 池时可忽略
        headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36",
        )
        if self._re is not None:
            resp = self._re.get(url, headers=headers, timeout=timeout)
            return resp
        else:
            if requests is None:
                return None
            resp = requests.get(url, headers=headers, timeout=timeout)
            return resp

    # ------------------------------------------------------------------
    # 站点 1：东财股吧（HTML 页面，简单解析）
    # ------------------------------------------------------------------

    def _fetch_from_eastmoney_guba(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """
        抓取东财股吧列表页上的帖子概要（简单版 HTML 解析）。

        注意：
        ----
        - 实际 URL 与结构可能会调整，这里提供的是一个「可运行模板」。
        - 如需更精准控制时间范围，可以多页翻页，并在解析时按日期过滤。
        """
        if BeautifulSoup is None:
            log.warning("未安装 bs4，无法解析东财股吧 HTML。")
            return []

        # 东财股吧 list URL 约定：
        #   https://guba.eastmoney.com/list,sz000001.html
        #   https://guba.eastmoney.com/list,sh600000.html
        # 简化：仅根据代码前缀判断市场，不严谨但够用。
        code = symbol.upper()
        if code.startswith("6"):
            board = f"sh{code}"
        else:
            board = f"sz{code}"

        url = f"https://guba.eastmoney.com/list,{board}.html"
        resp = self._http_get(url)
        if resp is None:
            return []
        html = resp.text if hasattr(resp, "text") else str(resp.content)

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("div", {"id": "articlelistnew"})
        if table is None:
            return []

        rows = table.find_all("div", class_="articleh")
        records: List[Dict[str, Any]] = []

        for row in rows[: self.cfg.max_posts_per_symbol]:
            try:
                a = row.find("a", href=True)
                if not a:
                    continue
                title = a.get_text(strip=True)
                href = a["href"]
                if not href.startswith("http"):
                    href = "https://guba.eastmoney.com" + href

                spans = row.find_all("span")
                reply_count = int(spans[1].get_text(strip=True) or 0) if len(spans) > 1 else 0
                author = spans[3].get_text(strip=True) if len(spans) > 3 else ""
                # 时间列只有时分，需要补上 trade_date
                time_str = spans[4].get_text(strip=True) if len(spans) > 4 else ""
                try:
                    ts = pd.to_datetime(
                        f"{trade_date.date()} {time_str}",
                        errors="coerce",
                    )
                except Exception:
                    ts = trade_date

                records.append(
                    {
                        "symbol": symbol,
                        "ts": ts,
                        "source": "eastmoney_guba",
                        "post_id": href,
                        "title": title,
                        "content": title,  # 先用标题，若需要正文可补抓详情页
                        "author": author,
                        "reply_count": reply_count,
                        "like_count": 0,
                        "url": href,
                    }
                )
            except Exception:  # pragma: no cover - 单行解析异常
                continue

        return records

    # ------------------------------------------------------------------
    # 站点 2：雪球（JSON API，需 Cookie 才能大规模使用）
    # ------------------------------------------------------------------

    def _fetch_from_xueqiu(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """
        雪球搜索接口（简化版），真实生产环境通常需要携带 Cookie 才能稳定使用。

        URL 示例（仅供参考，可能随时间变化）：
            https://xueqiu.com/query/v1/search/status.json?q=300750&count=30&sort=time

        这里仅做一个「能跑通」的 Demo，后续可以根据你的账号 / Cookie 做增强。
        """
        # 只取前 max_posts 条结果
        url = (
            "https://xueqiu.com/query/v1/search/status.json"
            f"?q={symbol}&count={self.cfg.max_posts_per_symbol}&sort=time"
        )
        # 雪球需要 Cookie，否则会返回未登录/风控页面。
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            # 这里保留一个占位，实际使用时你可以在 config 里配置 xueqiu_cookie
        }
        xcfg = self._raw_cfg.get("sentiment", {}) if isinstance(self._raw_cfg, dict) else {}
        cookie = xcfg.get("xueqiu_cookie")
        if cookie:
            headers["Cookie"] = cookie

        resp = self._http_get(url, headers=headers)
        if resp is None:
            return []

        try:
            data = resp.json() if hasattr(resp, "json") else {}
        except Exception:
            return []

        list_obj = data.get("list") or []
        records: List[Dict[str, Any]] = []
        for item in list_obj:
            try:
                created_at = pd.to_datetime(item.get("created_at"), unit="ms", errors="coerce")
                text = str(item.get("text") or "")
                title = str(item.get("title") or "") or text[:40]
                pid = str(item.get("id") or "")
                reply_count = int(item.get("reply_count") or 0)
                like_count = int(item.get("like_count", 0) or 0)
                url_post = f"https://xueqiu.com/{item.get('user_id')}/{pid}"

                records.append(
                    {
                        "symbol": symbol,
                        "ts": created_at,
                        "source": "xueqiu",
                        "post_id": pid,
                        "title": title,
                        "content": text,
                        "author": item.get("user", {}).get("screen_name", ""),
                        "reply_count": reply_count,
                        "like_count": like_count,
                        "url": url_post,
                    }
                )
            except Exception:  # pragma: no cover
                continue

        return records


# ----------------------------------------------------------------------
# 命令行入口（可配合 Scheduler / NightOps 使用）
# ----------------------------------------------------------------------


def _load_symbol_list(path: str) -> List[str]:
    """从简单文本或 CSV 中加载 symbol 列表。"""
    try:
        df = pd.read_csv(path)
        if "symbol" in df.columns:
            return df["symbol"].astype(str).tolist()
        # 单列无表头
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).tolist()
        return []
    except Exception:
        # 尝试按行读取 txt
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:  # pragma: no cover
            log.warning("加载 symbols 文件失败: %s", e)
            return []


def main() -> None:
    parser = argparse.ArgumentParser(description="LightHunter Sentiment Scraper")
    parser.add_argument("--date", type=str, default=None, help="交易日期，如 2024-01-05")
    parser.add_argument("--symbols-file", type=str, required=True, help="包含 symbol 列的 CSV / TXT")
    parser.add_argument("--output", type=str, required=True, help="输出 CSV 路径")
    args = parser.parse_args()

    cfg = get_system_config()
    scraper = SentimentScraper(cfg)

    symbols = _load_symbol_list(args.symbols_file)
    if not symbols:
        log.error("未能从 %s 读取到任何 symbol，退出。", args.symbols_file)
        return

    df_raw = scraper.scrape_for_symbols(symbols, trade_date=args.date)
    if df_raw.empty:
        log.warning("本次未采集到任何舆情数据。")
    df_raw.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info("已保存舆情原始数据到 %s，条数=%d", args.output, len(df_raw))


if __name__ == "__main__":  # pragma: no cover
    main()
