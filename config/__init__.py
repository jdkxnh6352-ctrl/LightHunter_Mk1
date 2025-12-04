# G:\LightHunter_Mk1\config\__init__.py
"""
配置相关统一入口

老的 xxx_config.json -> 使用 config_loader 里的函数
新的 light_hunter_config.json -> 使用 config_manager 里的函数
"""

from .config_loader import (
    load_config,
    load_system_config,
    load_risk_config,
    load_ts_pipeline_config,
    load_ai_config,
)

try:
    # Light Hunter / U2 实盘专用配置
    from .config_manager import (
        ConfigManager,
        get_config,
        reload_config,
        get_risk_config,
        get_data_config,
        get_proxy_config,
        get_ai_config,
        get_backup_config,
    )
except Exception:  # 防止缺文件时整个 import 失败
    ConfigManager = None  # type: ignore

    def get_config():
        return {}

    def reload_config():
        return {}

    def get_risk_config():
        return {}

    def get_data_config():
        return {}

    def get_proxy_config():
        return {}

    def get_ai_config():
        return {}

    def get_backup_config():
        return {}


__all__ = [
    # 旧配置加载接口
    "load_config",
    "load_system_config",
    "load_risk_config",
    "load_ts_pipeline_config",
    "load_ai_config",
    # 新配置中心接口
    "ConfigManager",
    "get_config",
    "reload_config",
    "get_risk_config",
    "get_data_config",
    "get_proxy_config",
    "get_ai_config",
    "get_backup_config",
]
