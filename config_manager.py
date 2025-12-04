# -*- coding: utf-8 -*-
"""
模块名称：ConfigManager Mk-Hub
版本：Mk-Config R10 (Central Control)
路径: G:/LightHunter_Mk1/config/config_manager.py

核心功能：
- 统一管理 LightHunter 所有可调参数（风控 / 抓数节奏 / 代理 / AI 超参 / 阈值 / 备份）。
- 默认配置文件：config/light_hunter_config.json
- 首次运行自动写入默认模板；后续人为修改即可全局生效。
"""

import os
import json
import copy
from typing import Any, Dict, Optional


class ConfigManager:
    """
    轻量级配置中心：
    - 单例模式（全局共享一份配置）
    - 默认配置 + 用户配置 深度合并（新增字段自动补全，不破坏已有 json）
    """

    _instance: Optional["ConfigManager"] = None

    # --------------------------
    # 默认配置（可按需扩展）
    # --------------------------
    DEFAULT_CONFIG: Dict[str, Any] = {
        "meta": {
            "version": "Mk-Config R10",
            "description": "LightHunter central config",
        },
        # 路径相关（便于后面统一调整）
        "paths": {
            "log_dir": "log",
            "blackbox_file": "market_blackbox.csv",
            "memory_file": "memory_core.json",
            "account_file": "Hunter_Account.json",
            "trade_history_file": "trade_history.csv",
            "ts_db_file": "ts_data.db",
            "market_ts_db_file": "market_ts.db",
            "config_file": "config/light_hunter_config.json",
        },
        # 风控参数（5.x / 4.x 系列会逐步迁移过来）
        "risk": {
            "account": {
                # 总资产最大回撤（相对初始资金）
                "max_total_drawdown_pct": 12.0,
                # 单日最大亏损（%），如 -4 表示亏4% 熔断
                "max_daily_loss_pct": -4.0,
                # 单日最大回撤（%）
                "max_daily_drawdown_pct": 6.0,
                # 最低权益比例（相对初始资金），低于则拉闸
                "min_equity_ratio": 0.5,
            },
            "intraday": {
                # 日内 PnL 风险闸门（可后续接 Commander / HUD）
                "pnl_stop_pct": -3.0,
                "pnl_warn_pct": -1.5,
                "risk_on_default": True,
            },
            "kelly": {
                # 凯利系数基准缩放（0~1），0.5 表示半 Kelly
                "base_fraction": 0.5,
                # 单笔最大资金占比上限
                "max_fraction_per_trade": 0.4,
                # 每笔最小开仓金额
                "min_trade_cash": 5000.0,
                # 赔率（平均盈亏比）
                "odds": 2.0,
                # 最低胜率门槛
                "win_prob_floor": 0.51,
            },
        },
        # 抓数节奏 / HTTP 行为
        "data": {
            # 快照模式：tencent / em / multi
            "snapshot_mode": "tencent",
            # Commander 主循环的扫描周期（秒）
            "commander": {
                "scan_interval_sec": 5,
            },
            # TSCollector / TSRecorder 等采样周期（秒）
            "ts_collector": {
                "interval_sec": 20,
            },
            # RequestEngine 相关 HTTP 参数
            "http": {
                "base_timeout": 4.0,
                "max_retries": 3,
                "max_total_retries": 3,
            },
        },
        # 代理参数（V2RayN / Clash / SSR…）
        "proxy": {
            # 是否启用代理（False = 始终直连）
            "enabled": True,
            # 是否优先扫描本地端口
            "prefer_local": True,
            # 本地代理常见端口（可以按自己 V2rayN/Clash 配置改）
            "local_scan_ports": [10808, 7890, 10809, 8080, 4780, 5700],
            # 探测代理是否可用的测试 URL
            "test_url": "http://www.baidu.com",
            # 探测超时（秒）
            "probe_timeout_sec": 1.5,
            # 外部代理列表（如有固定出口，可在 JSON 里写死）
            # 例如：["http://user:pass@ip:port", "http://127.0.0.1:7890"]
            "fixed_proxies": [],
            # 代理池列表文件（可选）
            "proxy_pool_file": "proxy_pool.txt",
        },
        # AI 超参数（后续会接更多：SeqBrain / RiskBrain / TS 模型等）
        "ai": {
            "combat_brain": {
                "hidden_layer_sizes": [64, 32],
                "max_iter": 500,
                "random_state": 42,
            },
            "risk_brain": {
                "horizon_min": 10,
                "dd_threshold": -3.0,
                "min_amount": 2e7,
                "min_turnover": 1.0,
            },
        },
        # 一些全局阈值（后面可渐进迁移）
        "thresholds": {
            "attack_zforce_base": 1.5,
            "risk_prob_block": 0.75,
            "risk_prob_warn": 0.6,
        },
        # 备份中心配置
        "backup": {
            "output_dir": "backups",
            "compress": True,
            "keep_last_n": 10,
            # 是否自动包含这些类型的文件
            "include_db": True,
            "include_csv": True,
            "include_json": True,
            "include_log": True,
            # 额外强制打包的关键文件
            "extra_files": [
                "gene_config.json",
                "hunter_brain.pkl",
                "hunter_scaler.pkl",
                "risk_brain.pkl",
                "data_quality_report.json",
            ],
            # 不需要递归进入的目录
            "exclude_dirs": ["__pycache__", ".git", ".idea", ".vscode", "venv"],
        },
    }

    def __init__(self, root_dir: Optional[str] = None, config_filename: str = "light_hunter_config.json"):
        if root_dir is None:
            # config/ 目录的上一级就是项目根目录
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.root_dir = root_dir
        self.config_dir = os.path.join(self.root_dir, "config")
        os.makedirs(self.config_dir, exist_ok=True)

        self.config_path = os.path.join(self.config_dir, config_filename)
        self.config: Dict[str, Any] = {}

        self._load_or_init()
        self._ensure_dirs()

    # --------------------------
    # 单例获取
    # --------------------------
    @classmethod
    def instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --------------------------
    # 内部：加载 / 初始化
    # --------------------------
    def _load_or_init(self) -> None:
        if not os.path.exists(self.config_path):
            # 首次运行：写入默认配置
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 读失败直接回落默认
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save()
            return

        # 深度合并：用户配置优先，缺失字段自动补全
        self.config = self._deep_merge(copy.deepcopy(self.DEFAULT_CONFIG), data)
        # 把 version 写回，便于后续升级
        self.config.setdefault("meta", {})
        self.config["meta"]["version"] = self.DEFAULT_CONFIG["meta"]["version"]
        self._save()  # 顺手写回一份整理后的 JSON

    def _save(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception:
            # 配置写失败不致命，不抛异常影响主流程
            pass

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归深度合并：
        - override 中的值优先；
        - 对于 dict，逐层合并；
        """
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    def _ensure_dirs(self) -> None:
        """
        确保日志目录 / 备份目录存在。
        """
        paths = self.config.get("paths", {})
        log_dir = paths.get("log_dir")
        if log_dir:
            os.makedirs(os.path.join(self.root_dir, log_dir), exist_ok=True)

        backup_cfg = self.config.get("backup", {})
        backup_dir = backup_cfg.get("output_dir")
        if backup_dir:
            os.makedirs(os.path.join(self.root_dir, backup_dir), exist_ok=True)

    # --------------------------
    # 对外接口
    # --------------------------
    def get_config(self) -> Dict[str, Any]:
        return self.config

    def reload(self) -> Dict[str, Any]:
        self._load_or_init()
        self._ensure_dirs()
        return self.config


# 便捷函数：供各模块直接 import 使用
def get_config() -> Dict[str, Any]:
    return ConfigManager.instance().get_config()


def reload_config() -> Dict[str, Any]:
    return ConfigManager.instance().reload()


def get_risk_config() -> Dict[str, Any]:
    return get_config().get("risk", {})


def get_data_config() -> Dict[str, Any]:
    return get_config().get("data", {})


def get_proxy_config() -> Dict[str, Any]:
    return get_config().get("proxy", {})


def get_ai_config() -> Dict[str, Any]:
    return get_config().get("ai", {})


def get_backup_config() -> Dict[str, Any]:
    return get_config().get("backup", {})
