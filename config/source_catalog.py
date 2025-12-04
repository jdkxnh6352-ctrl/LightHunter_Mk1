# -*- coding: utf-8 -*-
"""
data/source_catalog.py

LightHunter Singularity - SourceCatalog

职责：
- 读取 `config/system_config.json` 和 `config/data_schema.json`，
  统一管理「数据源清单」与「字段 Schema」；
- 提供统一接口给采集器 / 管线 / 因子引擎使用，例如：
    - 哪个站点是某类数据的主源(primary)、备源(fallback)？
    - 某个 entity（如 stock_snapshot）的标准字段有哪些？
    - 对于某个站点，字段映射关系是什么（raw -> canonical）？

注意：
- 这一版只做“配置解析 + 查询接口”，不做实际 HTTP 请求。
- 真正的网络访问仍然由 RequestEngine / Collectors 负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

from config.config_center import get_system_config
from core.logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_SCHEMA_PATH = CONFIG_DIR / "data_schema.json"


@dataclass
class SiteConfig:
    """单个站点（eastmoney / ths / tencent 等）的网络与数据相关配置。"""

    name: str
    base_urls: List[str]
    roles: List[str]              # 支持的角色：quote / kline / fund_flow / concept / news / sentiment 等
    primary: bool = False
    timeout: float = 8.0
    qps_limit: float = 5.0        # 站点级 QPS 建议上限
    burst: int = 10               # 突发窗口内最大请求数
    enabled: bool = True
    extra: Dict[str, Any] = None  # 站点专有配置（headers 模板等）

    @classmethod
    def from_dict(cls, name: str, cfg: Dict[str, Any]) -> "SiteConfig":
        return cls(
            name=name,
            base_urls=list(cfg.get("base_urls", [])),
            roles=list(cfg.get("roles", [])),
            primary=bool(cfg.get("primary", False)),
            timeout=float(cfg.get("timeout", 8.0)),
            qps_limit=float(cfg.get("qps_limit", 5.0)),
            burst=int(cfg.get("burst", 10)),
            enabled=bool(cfg.get("enabled", True)),
            extra=dict(cfg.get("extra", {})),
        )


@dataclass
class DataSourceSpec:
    """
    一类逻辑数据源的配置。

    例如：
        realtime_snapshot: 主源 tencent, 备源 [eastmoney], entity=stock_snapshot
        daily_kline      : 主源 eastmoney, 备源 [tencent], entity=daily_bar
    """

    name: str
    primary_site: str
    fallback_sites: List[str]
    entity: str                    # 对应 data_schema.json 里的 entity 名
    notes: Optional[str] = None    # 备注


class SourceCatalog:
    """
    SourceCatalog：系统级「数据源目录」。

    典型用法：
        catalog = SourceCatalog()

        # 列出所有逻辑数据源
        catalog.list_data_sources()

        # 获取主/备数据源信息
        ds = catalog.get_data_source("realtime_snapshot")

        # 获取站点配置
        em = catalog.get_site("eastmoney")

        # 获取某个 entity 的标准字段 schema
        schema = catalog.get_entity_schema("stock_snapshot")

        # 获取某个站点下特定 entity 的字段映射
        mapping = catalog.get_field_mapping("stock_snapshot", "eastmoney")
    """

    def __init__(
        self,
        system_config: Optional[Dict[str, Any]] = None,
        data_schema_path: Optional[Path] = None,
    ) -> None:
        self._log = get_logger(self.__class__.__name__)
        self._sys_cfg = system_config or get_system_config()

        self._data_schema_path = data_schema_path or DATA_SCHEMA_PATH
        self._schema = self._load_data_schema(self._data_schema_path)

        self._sites: Dict[str, SiteConfig] = self._load_sites(self._sys_cfg)
        self._data_sources: Dict[str, DataSourceSpec] = self._load_data_sources(
            self._sys_cfg
        )

        self._log.info(
            "SourceCatalog init. sites=%d, data_sources=%d",
            len(self._sites),
            len(self._data_sources),
        )

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data_schema(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.warning(
                "data_schema.json 不存在：%s，后续字段映射相关功能将不可用。", path
            )
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
            return schema or {}
        except Exception:
            logger.exception("加载 data_schema.json 失败：%s", path)
            return {}

    @staticmethod
    def _load_sites(sys_cfg: Dict[str, Any]) -> Dict[str, SiteConfig]:
        network_cfg = sys_cfg.get("network", {}) or {}
        sites_cfg = network_cfg.get("sites", {}) or {}
        sites: Dict[str, SiteConfig] = {}
        for name, cfg in sites_cfg.items():
            try:
                sites[name] = SiteConfig.from_dict(name, cfg)
            except Exception:
                logger.exception("解析站点配置失败：name=%s cfg=%s", name, cfg)
        if not sites:
            logger.warning("system_config.network.sites 为空，后续可能无法根据站点做限流/防封。")
        return sites

    @staticmethod
    def _load_data_sources(sys_cfg: Dict[str, Any]) -> Dict[str, DataSourceSpec]:
        ds_cfg = sys_cfg.get("data_sources", {}) or {}
        result: Dict[str, DataSourceSpec] = {}
        for name, cfg in ds_cfg.items():
            primary = cfg.get("primary_site") or cfg.get("primary") or ""
            fallback = cfg.get("fallback_sites") or cfg.get("fallback") or []
            entity = cfg.get("entity") or ""
            notes = cfg.get("notes")
            if not primary or not entity:
                logger.warning(
                    "data_source[%s] 配置不完整：primary=%s entity=%s", name, primary, entity
                )
                continue
            if isinstance(fallback, str):
                fallback = [fallback]
            spec = DataSourceSpec(
                name=name,
                primary_site=primary,
                fallback_sites=list(fallback),
                entity=entity,
                notes=notes,
            )
            result[name] = spec
        if not result:
            logger.warning("system_config.data_sources 为空。")
        return result

    # ------------------------------------------------------------------
    # 站点相关接口
    # ------------------------------------------------------------------

    def list_sites(self) -> List[str]:
        return sorted(self._sites.keys())

    def get_site(self, name: str) -> Optional[SiteConfig]:
        return self._sites.get(name)

    def list_sites_by_role(self, role: str) -> List[SiteConfig]:
        role = role.lower()
        return [
            s for s in self._sites.values()
            if role in (r.lower() for r in s.roles)
        ]

    # ------------------------------------------------------------------
    # data_source 相关接口
    # ------------------------------------------------------------------

    def list_data_sources(self) -> List[str]:
        return sorted(self._data_sources.keys())

    def get_data_source(self, name: str) -> Optional[DataSourceSpec]:
        return self._data_sources.get(name)

    def get_primary_site_for(self, data_source_name: str) -> Optional[SiteConfig]:
        spec = self.get_data_source(data_source_name)
        if not spec:
            return None
        return self.get_site(spec.primary_site)

    def get_fallback_sites_for(self, data_source_name: str) -> List[SiteConfig]:
        spec = self.get_data_source(data_source_name)
        if not spec:
            return []
        return [s for s in (self.get_site(n) for n in spec.fallback_sites) if s]

    def get_entity_for(self, data_source_name: str) -> Optional[str]:
        spec = self.get_data_source(data_source_name)
        return spec.entity if spec else None

    # ------------------------------------------------------------------
    # data_schema / 字段映射相关接口
    # ------------------------------------------------------------------

    def list_entities(self) -> List[str]:
        entities = (self._schema or {}).get("entities", {}) or {}
        return sorted(entities.keys())

    def get_entity_schema(self, entity_name: str) -> Optional[Dict[str, Any]]:
        entities = (self._schema or {}).get("entities", {}) or {}
        return entities.get(entity_name)

    def get_field_mapping(self, entity_name: str, site_name: str) -> Dict[str, str]:
        """
        返回 "canonical_field -> raw_field" 的映射。

        注意：data_schema.json 中记录的是某些站点的字段名，
        这里我们根据 canonical_fields + sources[*].field_mapping 反推映射关系。
        """
        entity = self.get_entity_schema(entity_name)
        if not entity:
            self._log.warning("未知 entity：%s", entity_name)
            return {}

        sources = entity.get("sources", {}) or {}
        s_cfg = sources.get(site_name)
        if not s_cfg:
            self._log.warning(
                "entity[%s] 未配置 site[%s] 的字段映射。", entity_name, site_name
            )
            return {}

        mapping = s_cfg.get("field_mapping", {}) or {}
        # data_schema 中是 canonical->raw 还是 raw->canonical，这里约定为 canonical->raw。
        # 如果后续需要 raw->canonical，可以在调用处进行反转。
        return dict(mapping)

    def map_record_to_canonical(
        self,
        entity_name: str,
        site_name: str,
        raw_record: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        将某站点原始字段 dict 映射为标准字段 dict（仅针对配置中列出的字段）。

        Args:
            entity_name: 实体名，如 "stock_snapshot"
            site_name  : 站点名，如 "eastmoney"
            raw_record : 该站点返回的一条原始记录（已经反序列化）
            strict     : True 时如果缺少字段会告警；False 时默默跳过缺失字段。

        Returns:
            canonical_record: {canonical_field: value}
        """
        mapping = self.get_field_mapping(entity_name, site_name)
        if not mapping:
            return {}

        out: Dict[str, Any] = {}
        for canonical_field, raw_field in mapping.items():
            if raw_field in raw_record:
                out[canonical_field] = raw_record[raw_field]
            else:
                if strict:
                    self._log.warning(
                        "站点[%s] entity[%s] 缺少字段：raw_field=%s",
                        site_name,
                        entity_name,
                        raw_field,
                    )
        return out
