# -*- coding: utf-8 -*-
"""
config/config_center.py

LightHunter 配置中心 (Mk3+)

职责：
- 统一读取 config/system_config.json 以及按环境覆盖的 system_config.<env>.json；
- 提供一组常用工具函数：
    * get_system_config(...)     : 读取并缓存 system_config
    * reload_system_config()     : 强制重载
    * get_project_root()         : 推断项目根目录
    * get_config_path()          : 当前使用的 system_config 路径
    * get_paths_config()         : 取 system_config["paths"]
    * get_path() / resolve_path(): 从 paths 中取某个路径并解析为绝对路径
    * get_logging_config()       : 取 system_config["logging"]
    * ensure_runtime_dirs()      : 根据 paths 中的目录创建运行所需目录
    * get_env_name()             : 返回当前环境名（dev/test/prod 等）
    * get_data_schema()          : 读取 config/data_schema.json（可选）

兼容性：
- 支持旧代码里的 get_system_config(refresh=True) 调用方式；
- 支持从老入口 `import config_center` 中 re-export 的常量/函数：
  CONFIG_FILE_ENV_VAR, PROJECT_ROOT, get_path, get_paths_config, ensure_runtime_dirs 等。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -------------------- 常量 --------------------

# 可通过该环境变量指定 system_config.json 路径
CONFIG_FILE_ENV_VAR = "LIGHTHUNTER_CONFIG_PATH"
# 可通过该环境变量强制指定项目根目录
PROJECT_ROOT_ENV_VAR = "LIGHTHUNTER_PROJECT_ROOT"
# 当前运行环境名（dev / test / prod ...）
ENV_NAME_ENV_VAR = "LIGHTHUNTER_ENV"

# -------------------- 模块级缓存 --------------------
_PROJECT_ROOT: Optional[Path] = None
_SYSTEM_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_SYSTEM_CONFIG_PATH: Optional[Path] = None
_DATA_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


# ============================================================
# 项目根目录 & 基础路径
# ============================================================

def get_project_root() -> Path:
    """
    返回 LightHunter 项目根目录。

    优先级：
    1. 环境变量 LIGHTHUNTER_PROJECT_ROOT；
    2. 当前文件上两级目录（假设位于 <root>/config/config_center.py）。
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    env_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        # config/ 在项目根目录下
        root = Path(__file__).resolve().parents[1]

    _PROJECT_ROOT = root
    return root


# 向外暴露一个只读常量，兼容旧代码 `from config_center import PROJECT_ROOT`
PROJECT_ROOT: Path = get_project_root()


def _default_config_path() -> Path:
    """默认的 system_config.json 路径：<project_root>/config/system_config.json"""
    return get_project_root() / "config" / "system_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    """
    读取 JSON 文件，如果不是对象则抛错。
    """
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象(dict): {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归字典合并：override 中的字段覆盖 base。
    """
    result: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ============================================================
# system_config.json 读取与缓存
# ============================================================

def _load_system_config_from_disk() -> Dict[str, Any]:
    """
    实际从磁盘加载 system_config（不做缓存）。

    - 支持通过环境变量 LIGHTHUNTER_CONFIG_PATH 覆盖默认路径；
    - 支持按环境加载 system_config.<env>.json（env 来自环境变量 LIGHTHUNTER_ENV
      或 system_config["env"]）。
    """
    # 1) 选择配置文件路径
    env_path = os.environ.get(CONFIG_FILE_ENV_VAR)
    if env_path:
        cfg_path = Path(env_path).expanduser().resolve()
    else:
        cfg_path = _default_config_path()

    global _SYSTEM_CONFIG_PATH
    _SYSTEM_CONFIG_PATH = cfg_path

    base_cfg = _load_json(cfg_path)

    # 2) 环境 env 设置（dev/test/prod 等）
    env_name = os.environ.get(ENV_NAME_ENV_VAR) or base_cfg.get("env")
    if env_name:
        env_name = str(env_name).lower()
        # 尝试加载 system_config.<env>.json
        env_cfg_path = cfg_path.with_name(
            f"{cfg_path.stem}.{env_name}{cfg_path.suffix}"
        )
        if env_cfg_path.exists():
            try:
                env_cfg = _load_json(env_cfg_path)
                base_cfg = _deep_merge(base_cfg, env_cfg)
                logger.info("已加载环境专用配置: %s (env=%s)", env_cfg_path, env_name)
            except Exception:
                logger.exception("加载环境配置失败: %s", env_cfg_path)

    return base_cfg


def get_system_config(
    force_reload: bool = False,
    *,
    refresh: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    获取全局 system_config（带缓存）。

    参数
    ----
    force_reload : bool, default False
        True 时强制从磁盘重新加载配置。
    refresh : bool, optional
        兼容旧代码的别名，与 force_reload 含义相同。
        例如旧代码中有 get_system_config(refresh=True)，在这里也能正常工作。
    """
    global _SYSTEM_CONFIG_CACHE

    # 新旧参数统一：只要任一为 True 就强制重载
    reload_flag = bool(force_reload or (refresh or False))

    if _SYSTEM_CONFIG_CACHE is None or reload_flag:
        try:
            cfg = _load_system_config_from_disk()
        except Exception as e:
            logger.exception("加载 system_config 失败: %s", e)
            raise
        _SYSTEM_CONFIG_CACHE = cfg

    return _SYSTEM_CONFIG_CACHE or {}


def reload_system_config() -> Dict[str, Any]:
    """
    语义化别名：强制重载 system_config。
    """
    return get_system_config(force_reload=True)


def get_config_path() -> Path:
    """
    返回当前使用的 system_config.json 路径。
    """
    return _SYSTEM_CONFIG_PATH or _default_config_path()


# ============================================================
# 路径相关工具
# ============================================================

def get_paths_config(sys_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    返回 system_config['paths'] 段，缺失时返回 {}。
    """
    cfg = sys_cfg or get_system_config()
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("paths") or {}


def resolve_path(
    key: str,
    default: Optional[str] = None,
    ensure_dir: bool = False,
) -> Path:
    """
    从 system_config['paths'][key] 读取路径并解析成绝对路径。

    参数
    ----
    key : str
        paths 中的键名，例如 "data_root"、"ts_snapshots_dir"。
    default : str, optional
        当 paths 中没有该键时使用的默认相对/绝对路径。
        若为 None 且键不存在，将抛出 KeyError。
    ensure_dir : bool, default False
        若为 True，则会确保返回路径对应的目录存在：
        - 如果解析结果类似文件路径（有后缀名），则创建其父目录；
        - 否则创建该目录本身。
    """
    paths = get_paths_config()
    value = paths.get(key, default)
    if value is None:
        raise KeyError(f"system_config['paths'] 中缺少键：{key!r}")

    p = Path(str(value))
    if not p.is_absolute():
        p = get_project_root() / p

    p = p.resolve()

    if ensure_dir:
        target = p if p.suffix == "" else p.parent
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("创建目录失败: %s", target)

    return p


def get_path(key: str, default: Optional[str] = None) -> str:
    """
    兼容层：返回 paths[key] 对应的绝对路径字符串。

    等价于：str(resolve_path(key, default))
    供旧代码 `from config_center import get_path` 使用。
    """
    return str(resolve_path(key, default=default, ensure_dir=False))


def get_logging_config(sys_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    返回 system_config['logging'] 段，缺失时返回 {}。
    """
    cfg = sys_cfg or get_system_config()
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("logging") or {}


def ensure_runtime_dirs(sys_cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    根据 system_config['paths'] 中的配置，尽量创建运行过程中会用到的目录。

    约定：只对 key 名中包含 'dir'/'root'/'folder'/'home'/'base' 的路径进行创建，
    避免误把诸如 duckdb_path 这类文件路径当成目录。
    """
    paths = get_paths_config(sys_cfg)
    if not paths:
        return

    candidates = []
    for k, v in paths.items():
        if not isinstance(v, str) or not v:
            continue
        kl = k.lower()
        if any(tok in kl for tok in ("dir", "root", "folder", "home", "base")):
            candidates.append(v)

    for raw in candidates:
        p = Path(raw)
        if not p.is_absolute():
            p = get_project_root() / p
        p = p.resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("创建运行目录失败: %s", p)


# ============================================================
# 其它辅助配置
# ============================================================

def get_env_name(default: str = "dev") -> str:
    """
    返回当前环境名（dev/test/prod 等）。

    优先级：
    1. 环境变量 LIGHTHUNTER_ENV；
    2. system_config['env']；
    3. 函数参数 default。
    """
    cfg = get_system_config()
    return str(os.environ.get(ENV_NAME_ENV_VAR) or cfg.get("env") or default)


def get_data_schema(force_reload: bool = False) -> Dict[str, Any]:
    """
    读取 config/data_schema.json（如果存在）。

    data_source_quality_checker 等工具会优先调用本函数；
    若文件不存在或解析失败，返回 {}。
    """
    global _DATA_SCHEMA_CACHE

    if _DATA_SCHEMA_CACHE is not None and not force_reload:
        return _DATA_SCHEMA_CACHE

    path = get_project_root() / "config" / "data_schema.json"
    if not path.exists():
        logger.warning("data_schema.json 不存在：%s", path)
        _DATA_SCHEMA_CACHE = {}
        return _DATA_SCHEMA_CACHE

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("data_schema.json 必须是 JSON 对象(dict)")
    except Exception:
        logger.exception("加载 data_schema.json 失败: %s", path)
        data = {}

    _DATA_SCHEMA_CACHE = data
    return _DATA_SCHEMA_CACHE


__all__ = [
    # 常量
    "CONFIG_FILE_ENV_VAR",
    "PROJECT_ROOT",
    # 根路径
    "get_project_root",
    # system_config 相关
    "get_system_config",
    "reload_system_config",
    "get_config_path",
    # 路径相关
    "get_paths_config",
    "get_path",
    "resolve_path",
    "get_logging_config",
    "ensure_runtime_dirs",
    # 其它
    "get_env_name",
    "get_data_schema",
]
