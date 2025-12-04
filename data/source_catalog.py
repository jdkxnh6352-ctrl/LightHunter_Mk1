# -*- coding: utf-8 -*-
"""
data/source_catalog.py

数据源目录（SourceCatalog）

职责：
- 从 system_config.data_sources 读取各站点配置（同花顺 / 东方财富 / 腾讯 等）
- 把每个数据源整理为 DataSourceConfig 对象
- 为 RequestEngine-S / CollectorMaster 提供统一的查询接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging_utils import get_logger
from config.config_center import get_system_config

logger = get_logger(__name__)


@dataclass
class DataSourceConfig:
    """
    单个数据源配置对象。

    字段说明：
    - id:        源 ID，等于 system_config.data_sources 的键，如 "tx_quote"
    - name:      显示名称，用于日志
    - role:      角色：quote / concept / funds / news / other
    - base_url:  该源的主域名，方便 Adapter 使用
    - enabled:   是否启用此源
    - max_qps:   每秒允许的最大请求数（全局）
    - max_concurrent: 允许的最大并发请求数（后续 CollectorMaster 可用）
    - timebands: 允许访问的时间段标记，比如 ["preopen", "market", "post"]
    - timeout:   单次请求超时时间（秒），为空则用 request_engine.default_timeout
    """

    id: str
    name: str
    role: str  # "quote", "concept", "funds", "news", ...
    base_url: Optional[str] = None
    enabled: bool = True
    max_qps: float = 2.0
    max_concurrent: int = 2
    timebands: List[str] = field(default_factory=lambda: ["preopen", "market", "post"])
    timeout: Optional[float] = None

    @classmethod
    def from_dict(
        cls,
        source_id: str,
        raw: Dict[str, Any],
        default_timeout: float,
    ) -> "DataSourceConfig":
        if raw is None:
            raw = {}

        name = raw.get("name", source_id)
        role = raw.get("role", "generic")
        base_url = raw.get("base_url")
        enabled = bool(raw.get("enabled", True))

        try:
            max_qps = float(raw.get("max_qps", 2.0))
        except Exception:
            max_qps = 2.0

        try:
            max_concurrent = int(raw.get("max_concurrent", 2))
        except Exception:
            max_concurrent = 2

        tb = raw.get("timebands") or raw.get("preferred_timebands") or [
            "preopen",
            "market",
            "post",
        ]
        if not isinstance(tb, list):
            tb = ["preopen", "market", "post"]

        timeout = raw.get("timeout", default_timeout)
        try:
            timeout = float(timeout) if timeout is not None else None
        except Exception:
            timeout = default_timeout

        return cls(
            id=source_id,
            name=str(name),
            role=str(role),
            base_url=base_url,
            enabled=enabled,
            max_qps=max_qps,
            max_concurrent=max_concurrent,
            timebands=[str(t) for t in tb],
            timeout=timeout,
        )


class SourceCatalog:
    """
    数据源目录单例。

    - 启动时从 system_config.data_sources 读取配置
    - 提供按 id / role 查询数据源的接口
    - 后续 RequestEngine-S / CollectorMaster 会引用这里的配置
    """

    _instance: Optional["SourceCatalog"] = None

    def __init__(self) -> None:
        self._sources: Dict[str, DataSourceConfig] = {}
        self.reload()

    @classmethod
    def instance(cls) -> "SourceCatalog":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reload(self) -> None:
        """
        从 system_config 重新加载所有数据源配置。
        """
        system_cfg = get_system_config()
        re_cfg = system_cfg.get("request_engine", {}) or {}
        default_timeout = float(re_cfg.get("default_timeout", 8.0))

        raw_sources = system_cfg.get("data_sources") or {}
        if not isinstance(raw_sources, dict):
            logger.error(
                "system_config.data_sources is not a dict, got %r",
                type(raw_sources),
            )
            raw_sources = {}

        sources: Dict[str, DataSourceConfig] = {}
        for source_id, raw_cfg in raw_sources.items():
            if not isinstance(raw_cfg, dict):
                logger.warning(
                    "data_sources.%s is not an object, skip", source_id
                )
                continue
            try:
                cfg = DataSourceConfig.from_dict(
                    source_id=source_id,
                    raw=raw_cfg,
                    default_timeout=default_timeout,
                )
                sources[source_id] = cfg
            except Exception:
                logger.exception(
                    "Failed to parse data source config for %s, skip", source_id
                )

        self._sources = sources

        logger.info(
            "SourceCatalog loaded %d data sources: %s",
            len(self._sources),
            ", ".join(sorted(self._sources.keys())),
        )

    # ------------- 查询 -------------

    def get(self, source_id: str) -> Optional[DataSourceConfig]:
        return self._sources.get(source_id)

    def list_all(self) -> List[DataSourceConfig]:
        return list(self._sources.values())

    def list_enabled(self) -> List[DataSourceConfig]:
        return [cfg for cfg in self._sources.values() if cfg.enabled]

    def list_by_role(self, role: str, enabled_only: bool = True) -> List[DataSourceConfig]:
        role = role.lower()
        result: List[DataSourceConfig] = []
        for cfg in self._sources.values():
            if cfg.role.lower() != role:
                continue
            if enabled_only and not cfg.enabled:
                continue
            result.append(cfg)
        return result


# ========== 模块级便捷函数 ==========

def get_source_catalog() -> SourceCatalog:
    return SourceCatalog.instance()


def reload_source_catalog() -> None:
    SourceCatalog.instance().reload()


def get_data_source_config(source_id: str) -> Optional[DataSourceConfig]:
    return SourceCatalog.instance().get(source_id)


def list_data_sources(
    role: Optional[str] = None,
    enabled_only: bool = True,
) -> List[DataSourceConfig]:
    catalog = SourceCatalog.instance()
    if role is None:
        sources = catalog.list_all()
        if enabled_only:
            sources = [s for s in sources if s.enabled]
        return sources
    return catalog.list_by_role(role=role, enabled_only=enabled_only)
