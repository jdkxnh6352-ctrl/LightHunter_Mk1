# -*- coding: utf-8 -*-
"""
模块名称：LogUtils Mk-Unity
版本：Mk-Log R10 (Central Logging)
路径: G:/LightHunter_Mk1/log_utils.py

功能：
- 初始化统一日志目录 log/；
- 创建按日期切分的 lighthunter_YYYYMMDD.log；
- 提供 get_logger() 供各子系统使用 (RequestEngine / MarketHunter / Commander 等)；
- 避免重复初始化，多模块多次调用也安全。
"""

import logging
import os
from datetime import datetime
from typing import Optional

_LOG_INITIALIZED = False


def init_logging(
    log_dir: str = "log",
    level: int = int(logging.DEBUG),
    filename_prefix: str = "lighthunter",
) -> None:
    """
    初始化全局日志系统：
    - log_dir: 日志存放目录
    - level: 记录级别（默认 DEBUG，方便抓细节）
    - filename_prefix: 日志文件前缀
    """
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:
        return

    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{date_str}.log")

    root = logging.getLogger()
    root.setLevel(level)

    # 防止重复添加 handler
    def _has_handler(tag: str) -> bool:
        for h in root.handlers:
            if getattr(h, "_lh_tag", None) == tag:
                return True
        return False

    # 文件日志：全量 DEBUG
    if not _has_handler("LH_MAIN_FILE"):
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh._lh_tag = "LH_MAIN_FILE"  # type: ignore[attr-defined]
        root.addHandler(fh)

    # 控制台：保持干净一点，默认 WARNING+
    if not _has_handler("LH_CONSOLE"):
        fmt_console = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt_console)
        ch._lh_tag = "LH_CONSOLE"  # type: ignore[attr-defined]
        root.addHandler(ch)

    _LOG_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    对外统一入口：
    - 自动保证 init_logging() 已执行；
    - 返回指定 name 的 logger。
    """
    if not _LOG_INITIALIZED:
        init_logging()
    return logging.getLogger(name)
