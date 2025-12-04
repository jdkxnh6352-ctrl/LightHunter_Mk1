# -*- coding: utf-8 -*-
"""
network/proxy_manager.py

LightHunter Mk4 - ProxyManager
==============================

职责：
- 解析 config/system_config.json 中的 network.proxies 与 network.sites 段
- 根据 URL / site_id 选择合适的代理（direct / V2rayN / 代理池）
- 为 RequestEngine 提供 requests 可用的 proxies 字典

注意：
- 本模块只管“用哪个出口”，不做 QPS / 熔断（在 request_engine 里处理）
- 如果配置缺失或错误，会尽量降级为直连，并打日志警告
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


log = get_logger(__name__)

try:
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover

    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        log.warning("config.config_center.get_system_config 未找到，ProxyManager 使用空配置")
        return {}


@dataclass
class ProxyEndpoint:
    """
    代表一个可用的代理出口。
    """
    id: str
    type: str = "direct"  # direct/http/socks5
    endpoint: Optional[str] = None  # 如 "http://127.0.0.1:10809"
    weight: float = 1.0
    max_qps: Optional[float] = None  # 预留字段，目前不强制使用
    disabled: bool = False

    def to_requests_proxies(self) -> Optional[Dict[str, str]]:
        """
        转换为 requests 可接受的 proxies 字典。
        direct 类型返回 None。
        """
        if self.type == "direct" or not self.endpoint:
            return None
        # 对于 http / socks5，http 与 https 使用同一出口
        return {
            "http": self.endpoint,
            "https": self.endpoint,
        }


class ProxyManager:
    """
    代理管理器：负责根据 site_id 或 URL 选出合适的代理。

    配置约定（system_config.json）：

    network.proxies = {
      "mode": "per_site",            # disabled / global / per_site
      "default": null,               # 默认使用的 proxy_id（可选）
      "pool": [
        {
          "id": "em_primary",
          "type": "http",
          "endpoint": "http://127.0.0.1:10809",
          "weight": 1.0,
          "max_qps": 5
        },
        ...
      ],
      "v2rayn": {
        "enabled": true,
        "http_proxy": "http://127.0.0.1:10809",
        "socks_proxy": "socks5://127.0.0.1:10808"
      }
    }

    network.sites = {
      "eastmoney": {
        "domains": ["eastmoney.com", "emdata.eastmoney.com", "push2.eastmoney.com"],
        "proxy_policy": {
          "mode": "pool",            # direct / pool / default
          "pool_ids": ["em_primary", "em_backup"],
          "rotate_on_ban": true
        }
      },
      ...
    }
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        if system_config is None:
            system_config = get_system_config(refresh=False)

        self.system_config = system_config or {}
        network_cfg = self.system_config.get("network", {}) or {}

        proxies_cfg = network_cfg.get("proxies", {}) or {}
        self.mode: str = proxies_cfg.get("mode", "disabled")
        self.default_proxy_id: Optional[str] = proxies_cfg.get("default")
        self._v2rayn_cfg: Dict[str, Any] = proxies_cfg.get("v2rayn", {}) or {}

        # 加载代理池
        self._proxies: Dict[str, ProxyEndpoint] = {}
        self._load_proxy_pool(proxies_cfg)

        # 站点配置，用于根据域名匹配 site_id
        self._sites_cfg: Dict[str, Any] = network_cfg.get("sites", {}) or {}

        # 每个 site 的轮询下标
        self._rr_index: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def resolve_site_id(self, url: str, site_hint: Optional[str] = None) -> str:
        """
        根据 URL 的 hostname 匹配配置中的 site_id。
        如果提供了 site_hint，则优先使用 site_hint（当它在 sites 中存在）。
        匹配失败时，返回 "__default__"。
        """
        if site_hint and site_hint in self._sites_cfg:
            return site_hint

        host = urlparse(url).hostname or ""
        host = host.lower()

        for site_id, cfg in self._sites_cfg.items():
            domains = cfg.get("domains", []) or []
            for pattern in domains:
                pattern = (pattern or "").lower()
                if not pattern:
                    continue
                # 简单匹配：完全相等或后缀匹配
                if host == pattern or host.endswith("." + pattern) or pattern in host:
                    return site_id

        return "__default__"

    def get_proxies_for_site(self, site_id: str) -> Optional[Dict[str, str]]:
        """
        返回给 requests.request 使用的 proxies 字典。
        如果配置为 direct 或 disabled，返回 None。
        """
        if self.mode == "disabled":
            return None

        # 非 per_site 模式下，如果有 default 直接用 default
        if self.mode != "per_site":
            proxy = self._get_proxy_by_id(self.default_proxy_id)
            return proxy.to_requests_proxies() if proxy else None

        # per_site 模式下，根据 site 的 proxy_policy 决定
        site_cfg = self._sites_cfg.get(site_id) or {}
        policy = site_cfg.get("proxy_policy") or {}

        mode = policy.get("mode", "default")

        if mode == "direct":
            return None

        if mode == "default":
            proxy = self._get_proxy_by_id(self.default_proxy_id)
            return proxy.to_requests_proxies() if proxy else None

        if mode == "pool":
            pool_ids = policy.get("pool_ids") or []
            proxy = self._choose_from_pool(site_id, pool_ids)
            if proxy:
                return proxy.to_requests_proxies()
            # pool 选不到时，回退 default
            proxy = self._get_proxy_by_id(self.default_proxy_id)
            return proxy.to_requests_proxies() if proxy else None

        # 未知模式：退回 default
        log.warning("未知 proxy_policy.mode=%s for site=%s，回退 default 代理。", mode, site_id)
        proxy = self._get_proxy_by_id(self.default_proxy_id)
        return proxy.to_requests_proxies() if proxy else None

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _load_proxy_pool(self, proxies_cfg: Dict[str, Any]) -> None:
        """
        从配置中加载代理池。
        """
        pool_cfg = proxies_cfg.get("pool", []) or []
        for item in pool_cfg:
            try:
                pid = str(item.get("id"))
                if not pid:
                    continue
                p_type = item.get("type", "direct")
                endpoint = item.get("endpoint")
                weight = float(item.get("weight", 1.0) or 1.0)
                max_qps = item.get("max_qps")
                pe = ProxyEndpoint(
                    id=pid,
                    type=p_type,
                    endpoint=endpoint,
                    weight=weight,
                    max_qps=float(max_qps) if max_qps is not None else None,
                )
                self._proxies[pid] = pe
            except Exception as e:
                log.warning("加载代理配置失败，item=%s, err=%s", item, e)

        # 如果 pool 为空但 v2rayn.enabled = true，可以自动填一个入口
        if not self._proxies and self._v2rayn_cfg.get("enabled"):
            http_proxy = self._v2rayn_cfg.get("http_proxy")
            if http_proxy:
                pe = ProxyEndpoint(
                    id="v2rayn_http",
                    type="http",
                    endpoint=http_proxy,
                    weight=1.0,
                )
                self._proxies[pe.id] = pe
                if not self.default_proxy_id:
                    self.default_proxy_id = pe.id
                log.info("根据 v2rayn 自动生成代理池入口：id=%s endpoint=%s", pe.id, pe.endpoint)

    def _get_proxy_by_id(self, proxy_id: Optional[str]) -> Optional[ProxyEndpoint]:
        if not proxy_id:
            return None
        proxy = self._proxies.get(proxy_id)
        if not proxy:
            log.warning("proxy_id=%s 未在 pool 中找到，降级为直连。", proxy_id)
        return proxy

    def _choose_from_pool(self, site_id: str, pool_ids: List[str]) -> Optional[ProxyEndpoint]:
        """
        从 pool_ids 中挑选一个可用代理。
        当前实现：简单轮询（Round-Robin），跳过 disabled 的。
        若全部不可用，返回 None。
        """
        valid_ids = [pid for pid in pool_ids if pid in self._proxies and not self._proxies[pid].disabled]
        if not valid_ids:
            return None

        with self._lock:
            idx = self._rr_index.get(site_id, 0)
            proxy_id = valid_ids[idx % len(valid_ids)]
            self._rr_index[site_id] = (idx + 1) % (1 << 30)
        return self._proxies.get(proxy_id)

    # 预留接口：当某个 site + proxy 组合判定为被封时，可外部调用
    def mark_proxy_as_suspected_banned(self, site_id: str, proxy_id: str) -> None:
        """
        标记某个代理在某个站点上疑似被封，用于后续优化。
        当前实现只是打日志，不做状态调整。
        未来可以拓展为：在一定时间内降低该 proxy 在该 site 的使用优先级。
        """
        log.warning("检测到 site=%s 的 proxy=%s 疑似被封，可考虑在 ProxyManager 中降低优先级。", site_id, proxy_id)
