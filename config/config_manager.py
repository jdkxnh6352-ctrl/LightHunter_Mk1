# -*- coding: utf-8 -*-
"""
ConfigManager Mk-Hub 统一配置中心

- 默认配置文件：config/light_hunter_config.json
- 首次运行自动写入默认模板；以后你只改 JSON 就能影响整个项目。
"""

import os
import json
import copy
from typing import Any, Dict, Optional


class ConfigManager:
    """
    轻量级配置中心（单例）：

    - 默认配置 + 用户配置 深度合并
    - 缺失字段自动补全，不会因为你删错一个 key 整个系统就崩
    """

    _instance: Optional["ConfigManager"] = None

    # ------------------------------------------------------------------
    # 默认配置（你日后想扩展，也可以在这里加字段）
    # ------------------------------------------------------------------
    DEFAULT_CONFIG: Dict[str, Any] = {
        "meta": {
            "version": "Mk-Config R10",
            "description": "LightHunter central config",
        },
        # 路径相关
        "paths": {
            # 项目根目录（一般不用改）
            "project_root": ".",
            # 配置文件目录
            "config_dir": "config",
            # 报告目录
            "reports_dir": "reports",
            # 日志目录
            "log_dir": "logs",
            # 监控 / 体检相关输出
            "monitor_dir": "monitor",
        },
        # 风控配置
        "risk": {
            "account": {
                # 总体最大回撤（相对初始资金）
                "max_total_drawdown_pct": 30.0,
                # 单日最大亏损（%）
                "max_daily_loss_pct": -4.0,
                # 单日最大回撤（%）
                "max_daily_drawdown_pct": 6.0,
                # 最低权益比例
                "min_equity_ratio": 0.5,
            },
        },
        # U2 日常打分过滤参数（默认兜底）
        "u2_daily_filter": {
            "min_price": 3.0,
            "max_price": 60.0,
            "min_amount_today": 20000000,
            "min_amount_20d": 30000000,
            "max_abs_ret_1": 0.09,
            "min_ret_5": -0.25,
            "max_ret_5": 0.25,
            "max_ret_20": 0.60,
            "min_prob": 0.55,
            "top_k": 30,
        },
        # U2 过滤方案集合：按 profile / tag 区分
        # 这里只给一个兜底 default，具体方案由你在 json 里覆盖
        "u2_daily_filter_profiles": {
            "default": {}
        },
    }

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config_filename: str = "light_hunter_config.json",
    ) -> None:
        if root_dir is None:
            # config/ 的上一级就是项目根
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.root_dir = root_dir
        self.config_dir = os.path.join(self.root_dir, "config")
        os.makedirs(self.config_dir, exist_ok=True)

        self.config_path = os.path.join(self.config_dir, config_filename)
        self.config: Dict[str, Any] = {}

        self._load_or_init()
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # 单例入口
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 内部：加载 / 初始化
    # ------------------------------------------------------------------
    def _load_or_init(self) -> None:
        # 第一次没有文件：直接写一份默认模板出来
        if not os.path.exists(self.config_path):
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save()
            return

        # 有文件就尝试读取
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 读失败就兜底用默认
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save()
            return

        # 默认 + 用户配置 深度合并
        base = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config = self._deep_merge(base, data)

        # 把当前版本号写回去
        self.config.setdefault("meta", {})
        self.config["meta"]["version"] = self.DEFAULT_CONFIG["meta"]["version"]

        # 顺手把合并后的结果写回 json，方便你直接看最终生效配置
        self._save()

    def _save(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception:
            # 写失败不至于把主流程干崩，静默跳过
            pass

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归深度合并：
        - override 中的值优先
        - dict 就一层一层往下合并
        """
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    def _ensure_dirs(self) -> None:
        """根据 paths 里定义的目录，自动创建一下。"""
        paths = self.config.get("paths", {}) or {}

        # 把相对路径转换成基于项目根目录的绝对路径
        for key in ["log_dir", "reports_dir", "monitor_dir"]:
            d = paths.get(key)
            if not d:
                continue
            if not os.path.isabs(d):
                d = os.path.join(self.root_dir, d)
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        return self.config

    def reload(self) -> Dict[str, Any]:
        self._load_or_init()
        self._ensure_dirs()
        return self.config


# ======================================================================
# 便捷函数（外部模块一般只用这些）
# ======================================================================

def get_config(refresh: bool = False) -> Dict[str, Any]:
    """
    读取总配置。

    refresh=True 时会重新从 json 读一遍（用于你改了配置想立刻生效）。
    """
    mgr = ConfigManager.instance()
    if refresh:
        return mgr.reload()
    return mgr.get_config()


def get_paths_config() -> Dict[str, Any]:
    return get_config().get("paths", {}) or {}


def get_risk_config() -> Dict[str, Any]:
    return get_config().get("risk", {}) or {}


def get_u2_filter_config(profile: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 U2 日常过滤配置。

    profile:
        - None           : 只用 u2_daily_filter 这一份；
        - 某个字符串值     : 在 u2_daily_filter 基础上，再叠加
                            u2_daily_filter_profiles[profile] 里的覆盖项。
    """
    cfg = get_config()
    base = (cfg.get("u2_daily_filter", {}) or {}).copy()
    profiles = cfg.get("u2_daily_filter_profiles", {}) or {}

    if profile:
        overrides = profiles.get(profile)
        if isinstance(overrides, dict):
            base.update(overrides)

    return base
