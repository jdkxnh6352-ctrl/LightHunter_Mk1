# -*- coding: utf-8 -*-
"""
模块名称：ConfigLoader Mk-Core
路径: G:/LightHunter_Mk1/config_loader.py

功能：
- 统一从 config/*.json 读取配置；
- 支持“默认参数 + 用户覆盖”的深度合并；
- 任何配置异常时，自动回退到代码内置默认值，不拖垮核心流程。
"""

import os
import json
from typing import Any, Dict

BASE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(BASE_DIR, "config")


def ensure_config_dir() -> str:
    """确保 config 目录存在。"""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)
    return CONFIG_DIR


def get_config_path(name: str) -> str:
    """返回 config/<name> 的绝对路径。"""
    ensure_config_dir()
    return os.path.join(CONFIG_DIR, name)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并 dict，后者覆盖前者；用于“默认配置 + 用户覆盖”。

    注意：会就地修改 base，并返回 base。
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)  # type: ignore[assignment]
        else:
            base[k] = v
    return base


def load_config(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 config/<name> 读取 JSON，与 default 深度合并。

    - 文件不存在 / JSON 解析失败时，直接返回 default；
    - 所有异常打印一行提示，但不抛出，保证交易主流程稳定。
    """
    # 用 json round-trip 做一个深拷贝，避免 default 被外部修改
    cfg: Dict[str, Any] = json.loads(json.dumps(default))
    path = get_config_path(name)

    if not os.path.exists(path):
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if isinstance(user_cfg, dict):
            cfg = _deep_update(cfg, user_cfg)
    except Exception as e:
        print(f"[CONFIG] Failed to load {name}: {e}. Using defaults.")

    return cfg
