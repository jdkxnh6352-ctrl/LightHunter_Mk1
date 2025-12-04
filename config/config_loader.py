# -*- coding: utf-8 -*-
"""
模块名称：Config Loader
路径: G:/LightHunter_Mk1/config/config_loader.py

功能：
- 统一读取 config/*.json；
- 支持提供 default dict，并在读取时做递归合并；
- 读取失败或缺失时自动回退到默认值，保证系统不崩。
"""

import os
import json
import copy
from typing import Any, Dict, Optional


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个 dict，override 覆盖 base；
    非 dict 节点直接覆盖。
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    name: str,
    default: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    读取 config/{name}.json 并与 default 合并。

    :param name: 不带后缀的文件名，例如 "risk_config"
    :param default: 默认配置（可为 None）
    :param base_dir: 可选，指定配置目录；默认为当前文件所在目录
    :return: 合并后的配置 dict
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    cfg_path = os.path.join(base_dir, f"{name}.json")
    cfg = copy.deepcopy(default) if isinstance(default, dict) else {}

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if cfg:
                    cfg = _deep_update(cfg, loaded)
                else:
                    cfg = loaded
        except Exception as e:
            print(f"[CONFIG] 读取 {cfg_path} 失败，使用默认配置。错误: {e}")
    else:
        # 找不到文件时，仅返回 default，不强行创建
        if not cfg:
            cfg = {}

    return cfg


def load_system_config(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取系统级配置：system_config.json"""
    return load_config("system_config", default=default)


def load_risk_config(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取风控配置：risk_config.json"""
    return load_config("risk_config", default=default)


def load_ts_pipeline_config(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 TS 管线配置：ts_pipeline.json"""
    return load_config("ts_pipeline", default=default)


def load_ai_config(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 AI / 模型配置：ai_config.json"""
    return load_config("ai_config", default=default)
