# -*- coding: utf-8 -*-
"""
request_engine.py

LightHunter Mk4 - RequestEngine with Site-Level Rate Limiting & Circuit Breaker
==============================================================================

职责：
- 提供统一的 HTTP 请求入口（同步），支持：
  - 站点级 QPS & 并发控制
  - 站点级重试 & 指数退避
  - 站点级熔断（error_rate & 封禁信号）
  - 基于 ProxyManager / system_config 的 per-site 代理选择
  - UA 轮换（基于 network.ua_pool）

典型用法
--------

from request_engine import RequestEngine, get

engine = RequestEngine()
resp = engine.get("https://qt.gtimg.cn/q=sh600000", site="tencent_quote")
html = resp.text

# 或直接用模块级函数（默认全局实例）：
resp = get("https://qt.gtimg.cn/q=sh600000", site="tencent_quote")

配置依赖（简化说明）
------------------

system_config.json 中的 network 段大致形如：

"network": {
  "default_timeout_sec": 8,
  "max_retries": 3,
  "backoff_factor": 0.3,

  "global_limits": {
    "max_qps": 20,
    "max_concurrent": 100
  },

  "ua_pool": [ "...UA1...", "...UA2..." ],
  "ua_strategy": "random_per_session",

  "proxies": {
    "direct": {"type": "direct"},
    "v2ray_main": {
      "type": "http",
      "http": "http://127.0.0.1:10808",
      "https": "http://127.0.0.1:10808"
    }
  },

  "sites": {
    "tx_quote": {
      "max_qps": 5.0,
      "max_concurrent": 5,
      "timeout_sec": 6.0,
      "max_retries": 2,
      "backoff_factor": 0.3,
      "retry_status_codes": [500, 502, 503, 504, 429],
      "retry_on_timeout": true,
      "proxy": "v2ray_main",
      "circuit_breaker": {
        "error_rate_window": 300,
        "min_requests": 20,
        "error_rate_threshold": 0.4,
        "cooldown_sec": 900,
        "half_open_max_calls": 5
      }
    },
    ...
  }
}
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional

import requests

try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str) -> "logging.Logger":  # type: ignore
        return logging.getLogger(name)


log = get_logger(__name__)

try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        log.warning("config.config_center.get_system_config 未找到，RequestEngine 使用空配置")
        return {}


try:
    from network.proxy_manager import ProxyManager  # type: ignore
except Exception:  # pragma: no cover
    ProxyManager = None  # type: ignore


# ----------------------------------------------------------------------
# 异常定义
# ----------------------------------------------------------------------


class CircuitOpenError(RuntimeError):
    """当某站点被熔断器标记为 OPEN 时抛出。"""

    def __init__(self, site_id: str, message: str = "") -> None:
        if not message:
            message = f"Circuit breaker open for site={site_id}"
        super().__init__(message)
        self.site_id = site_id


# ----------------------------------------------------------------------
# 限流器：QPS + 并发
# ----------------------------------------------------------------------


class RateLimiter:
    """简单的每秒 QPS + 并发控制。"""

    def __init__(
        self,
        max_qps: Optional[float] = None,
        max_concurrent: Optional[int] = None,
    ) -> None:
        self.max_qps = float(max_qps) if max_qps else None
        self.max_concurrent = int(max_concurrent) if max_concurrent else None

        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()
        self._semaphore: Optional[threading.Semaphore]
        if self.max_concurrent and self.max_concurrent > 0:
            self._semaphore = threading.Semaphore(self.max_concurrent)
        else:
            self._semaphore = None

    # 内部：维护 1 秒滑动窗口
    def _prune_old(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] > 1.0:
            self._timestamps.popleft()

    def acquire(self) -> None:
        """拿一个“令牌”，必要时会 sleep。"""
        if self._semaphore is not None:
            self._semaphore.acquire()

        if self.max_qps is None:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_old(now)
                if len(self._timestamps) < self.max_qps:
                    self._timestamps.append(now)
                    return

                # 本秒额度已满，算一下要等多久
                earliest = self._timestamps[0]
                wait_time = earliest + 1.0 - now
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                time.sleep(0.01)

    def release(self) -> None:
        """释放并发令牌。"""
        if self._semaphore is not None:
            try:
                self._semaphore.release()
            except ValueError:
                # 避免重复 release 导致异常
                pass


# ----------------------------------------------------------------------
# 熔断器
# ----------------------------------------------------------------------


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    window_size_sec: int = 60
    min_requests: int = 20
    error_rate_threshold: float = 0.4
    cooldown_sec: int = 180
    half_open_max_calls: int = 5


class CircuitBreaker:
    """站点级熔断器。"""

    def __init__(self, site_id: str, cfg: CircuitBreakerConfig) -> None:
        self.site_id = site_id
        self.cfg = cfg

        self.state: str = "closed"  # closed / open / half_open
        self.window_start: float = time.monotonic()
        self.total_requests: int = 0
        self.total_failures: int = 0

        self.opened_at: Optional[float] = None
        self.half_open_trial_count: int = 0

        self._lock = threading.Lock()

    def _reset_window_if_needed(self, now: float) -> None:
        if now - self.window_start > self.cfg.window_size_sec:
            self.window_start = now
            self.total_requests = 0
            self.total_failures = 0

    def before_request(self) -> None:
        """在请求前调用，判断是否允许请求。"""
        if not self.cfg.enabled:
            return

        with self._lock:
            now = time.monotonic()

            if self.state == "open":
                if self.opened_at is not None and (now - self.opened_at) >= self.cfg.cooldown_sec:
                    # 进入 half_open 状态
                    self.state = "half_open"
                    self.half_open_trial_count = 0
                    log.info("站点 %s 熔断冷却结束，进入 half_open 状态。", self.site_id)
                else:
                    raise CircuitOpenError(self.site_id)

            self._reset_window_if_needed(now)
            self.total_requests += 1

            if self.state == "half_open":
                self.half_open_trial_count += 1

    def after_request(self, success: bool, is_ban_signal: bool = False) -> None:
        """在请求结束后调用，用于更新状态。"""
        if not self.cfg.enabled:
            return

        with self._lock:
            now = time.monotonic()
            if not success:
                self.total_failures += 1

            if self.state == "half_open":
                if not success or is_ban_signal:
                    # half_open 阶段仍然失败，重新 OPEN
                    self.state = "open"
                    self.opened_at = now
                    log.warning("站点 %s 在 half_open 阶段仍然异常，重新进入 open 状态。", self.site_id)
                elif self.half_open_trial_count >= self.cfg.half_open_max_calls:
                    # 多次成功，回到 closed
                    self.state = "closed"
                    self.opened_at = None
                    self.total_requests = 0
                    self.total_failures = 0
                    log.info("站点 %s 熔断恢复为 closed 状态。", self.site_id)
            else:
                # 正常 closed 窗口内统计 error rate
                if self.total_requests >= self.cfg.min_requests:
                    err_rate = self.total_failures / max(self.total_requests, 1)
                    if err_rate >= self.cfg.error_rate_threshold or is_ban_signal:
                        self.state = "open"
                        self.opened_at = now
                        log.warning(
                            "站点 %s 触发熔断: error_rate=%.2f, is_ban_signal=%s",
                            self.site_id,
                            err_rate,
                            is_ban_signal,
                        )


# ----------------------------------------------------------------------
# RequestEngine 本体
# ----------------------------------------------------------------------


class RequestEngine:
    """负责所有 HTTP 请求的统一入口。"""

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        self.sys_cfg = system_config or get_system_config()
        network_cfg = self.sys_cfg.get("network", {}) or {}

        # 全局默认参数
        self.default_timeout_sec: float = float(network_cfg.get("default_timeout_sec", 8.0))
        self.default_max_retries: int = int(network_cfg.get("max_retries", 2))
        self.default_backoff_factor: float = float(network_cfg.get("backoff_factor", 0.3))

        global_limits = network_cfg.get("global_limits", {}) or {}
        self.global_rate_limiter = RateLimiter(
            max_qps=global_limits.get("max_qps"),
            max_concurrent=global_limits.get("max_concurrent"),
        )

        # per-site 配置
        self.sites_cfg: Dict[str, Dict[str, Any]] = network_cfg.get("sites", {}) or {}

        # 代理配置
        self.proxies_cfg: Dict[str, Dict[str, Any]] = network_cfg.get("proxies", {}) or {}
        self.proxy_manager = ProxyManager(self.sys_cfg) if ProxyManager is not None else None

        # UA 池
        self.ua_pool: List[str] = list(network_cfg.get("ua_pool", []) or [])
        self.ua_strategy: str = str(network_cfg.get("ua_strategy", "random_per_session"))

        # 站点级组件
        self.site_rate_limiters: Dict[str, RateLimiter] = {}
        self.site_breakers: Dict[str, CircuitBreaker] = {}

        # HTTP session
        self._session = requests.Session()

        log.info(
            "RequestEngine 初始化完成: default_timeout=%.1fs, default_max_retries=%d",
            self.default_timeout_sec,
            self.default_max_retries,
        )

    # -------------------- 内部工具 --------------------

    def _resolve_site_id(self, url: str, site: Optional[str]) -> str:
        """优先使用显式 site；否则尝试由 ProxyManager/URL 推断。"""
        if site:
            return site
        # 这里可以做更复杂的 URL 解析，现在先返回 "default"
        return "default"

    def _get_site_cfg(self, site_id: str) -> Dict[str, Any]:
        return self.sites_cfg.get(site_id, {})

    def _get_site_rate_limiter(self, site_id: str, site_cfg: Dict[str, Any]) -> RateLimiter:
        if site_id not in self.site_rate_limiters:
            self.site_rate_limiters[site_id] = RateLimiter(
                max_qps=site_cfg.get("max_qps"),
                max_concurrent=site_cfg.get("max_concurrent"),
            )
        return self.site_rate_limiters[site_id]

    def _get_site_breaker(self, site_id: str, site_cfg: Dict[str, Any]) -> Optional[CircuitBreaker]:
        cb_cfg = site_cfg.get("circuit_breaker") or {}
        enabled = bool(cb_cfg.get("enabled", True))
        if not enabled:
            return None

        cfg = CircuitBreakerConfig(
            enabled=True,
            window_size_sec=int(cb_cfg.get("error_rate_window", 60)),
            min_requests=int(cb_cfg.get("min_requests", 20)),
            error_rate_threshold=float(cb_cfg.get("error_rate_threshold", 0.4)),
            cooldown_sec=int(cb_cfg.get("cooldown_sec", 180)),
            half_open_max_calls=int(cb_cfg.get("half_open_max_calls", 5)),
        )
        if site_id not in self.site_breakers:
            self.site_breakers[site_id] = CircuitBreaker(site_id, cfg)
        return self.site_breakers[site_id]

    def _select_user_agent(self) -> Optional[str]:
        if not self.ua_pool:
            return None
        if self.ua_strategy == "random_per_session":
            # 每次调用随机一个即可
            return random.choice(self.ua_pool)
        # 其它策略以后再扩展
        return random.choice(self.ua_pool)

    def _build_proxies(self, site_id: str, site_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        # 调用方显式传 proxies 时不走这里
        proxy_id = site_cfg.get("proxy") or "direct"
        cfg = self.proxies_cfg.get(proxy_id) or {}

        if not cfg:
            return None

        ptype = cfg.get("type", "direct")
        if ptype == "direct":
            return None
        if ptype in ("http", "https"):
            http_url = cfg.get("http")
            https_url = cfg.get("https", http_url)
            if not http_url and not https_url:
                return None
            proxies: Dict[str, str] = {}
            if http_url:
                proxies["http"] = http_url
            if https_url:
                proxies["https"] = https_url
            return proxies
        # 其它类型（socks 等）暂不细分，直接透传给 requests
        return {k: v for k, v in cfg.items() if k in ("http", "https")}

    # -------------------- 核心请求 --------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        site: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json: Any = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        通用请求入口。

        额外支持的自定义 keyword 参数（会被 RequestEngine 消化，不会传给 requests）：
            - site        : 站点 id，对应 system_config.network.sites 的 key
            - source_id   : 调用方自定义的“数据源/用途”标签，仅用于日志
            - max_retries : 覆盖该次请求的最大重试次数
            - backoff_factor: 覆盖该次请求的退避系数
        """
        # ---- 先从 kwargs 中取出 RequestEngine 自己用的字段，避免透传给 requests ----
        source_id = kwargs.pop("source_id", None)
        override_max_retries = kwargs.pop("max_retries", None)
        override_backoff = kwargs.pop("backoff_factor", None)

        # requests 能识别的 kwargs（如 allow_redirects / stream 等）仍然保留在 kwargs 中

        site_id = self._resolve_site_id(url, site)
        site_cfg = self._get_site_cfg(site_id)

        timeout_sec = float(
            timeout
            if timeout is not None
            else site_cfg.get("timeout_sec", self.default_timeout_sec)
        )
        max_retries = int(
            override_max_retries
            if override_max_retries is not None
            else site_cfg.get("max_retries", self.default_max_retries)
        )
        backoff_factor = float(
            override_backoff
            if override_backoff is not None
            else site_cfg.get("backoff_factor", self.default_backoff_factor)
        )

        retry_status_codes: Iterable[int] = site_cfg.get(
            "retry_status_codes", [500, 502, 503, 504, 429]
        )
        retry_on_timeout = bool(site_cfg.get("retry_on_timeout", True))

        # 限流 + 熔断组件
        site_rl = self._get_site_rate_limiter(site_id, site_cfg)
        breaker = self._get_site_breaker(site_id, site_cfg)

        # 代理选择（caller 手工指定 proxies 时优先）
        if proxies is None:
            proxies = self._build_proxies(site_id, site_cfg)

        # UA & Headers
        req_headers = dict(headers) if headers else {}
        # headers 里可能用各种大小写，这里统一判断
        header_keys_norm = {k.lower() for k in req_headers.keys()}
        if "user-agent" not in header_keys_norm:
            ua = self._select_user_agent()
            if ua:
                req_headers["User-Agent"] = ua

        last_exc: Optional[Exception] = None
        response: Optional[requests.Response] = None

        for attempt in range(max_retries + 1):
            success = False
            is_ban_signal = False

            # 熔断判定
            if breaker is not None:
                try:
                    breaker.before_request()
                except CircuitOpenError as e:
                    log.warning(
                        "请求被熔断器拒绝: site=%s, source_id=%s, url=%s, err=%s",
                        site_id,
                        source_id,
                        url,
                        e,
                    )
                    raise

            # 限流（全局 + 站点）
            self.global_rate_limiter.acquire()
            site_rl.acquire()
            try:
                try:
                    response = self._session.request(
                        method=method.upper(),
                        url=url,
                        timeout=timeout_sec,
                        headers=req_headers,
                        params=params,
                        data=data,
                        json=json,
                        proxies=proxies,
                        **kwargs,  # 注意：此处已经不含 source_id 等自定义字段
                    )
                    status = response.status_code

                    if status in retry_status_codes:
                        last_exc = RuntimeError(f"HTTP {status} for url={url}")
                        success = False
                    else:
                        success = 200 <= status < 300
                except requests.Timeout as e:
                    last_exc = e
                    success = False
                    if not retry_on_timeout:
                        raise
                except requests.RequestException as e:
                    last_exc = e
                    success = False

                # 粗略判断是否可能被封
                if response is not None:
                    if response.status_code == 403:
                        is_ban_signal = True
                    else:
                        try:
                            text_sample = response.text[:1024]
                            if any(k in text_sample for k in ("验证码", "verify", "forbidden")):
                                is_ban_signal = True
                        except Exception:
                            pass
            finally:
                site_rl.release()
                self.global_rate_limiter.release()

            # 更新熔断状态
            if breaker is not None:
                breaker.after_request(success=success, is_ban_signal=is_ban_signal)

            if success and response is not None:
                log.debug(
                    "[REQ OK] method=%s site=%s source_id=%s status=%s url=%s",
                    method,
                    site_id,
                    source_id,
                    response.status_code,
                    url,
                )
                return response

            # 不成功，准备重试
            if attempt >= max_retries:
                break

            if isinstance(last_exc, requests.Timeout) and not retry_on_timeout:
                break

            # 退避等待
            sleep_sec = backoff_factor * (2 ** attempt)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # 重试结束仍失败
        log.warning(
            "[REQ FAIL] method=%s site=%s source_id=%s url=%s last_exc=%r",
            method,
            site_id,
            source_id,
            url,
            last_exc,
        )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Request failed without explicit exception: {method} {url}")

    # -------------------- 便捷封装 --------------------

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def get_text(self, url: str, encoding: Optional[str] = None, **kwargs: Any) -> str:
        resp = self.get(url, **kwargs)
        if encoding:
            resp.encoding = encoding
        return resp.text

    def get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self.get(url, **kwargs)
        return resp.json()


# ----------------------------------------------------------------------
# 模块级默认实例 & 快捷函数
# ----------------------------------------------------------------------

_GLOBAL_ENGINE: Optional[RequestEngine] = None


def get_default_engine() -> RequestEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = RequestEngine()
    return _GLOBAL_ENGINE


def get_request_engine() -> RequestEngine:
    """向后兼容的别名。"""
    return get_default_engine()


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    return get_default_engine().request(method, url, **kwargs)


def get(url: str, **kwargs: Any) -> requests.Response:
    return get_default_engine().get(url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return get_default_engine().post(url, **kwargs)


def get_text(url: str, **kwargs: Any) -> str:
    return get_default_engine().get_text(url, **kwargs)


def get_json(url: str, **kwargs: Any) -> Any:
    return get_default_engine().get_json(url, **kwargs)
