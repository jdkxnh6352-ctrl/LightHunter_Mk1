# -*- coding: utf-8 -*-
"""
模块：core.logging_utils
作用：为整个 LightHunter 项目提供统一的日志入口。

设计原则：
- 优先复用根目录下的 log_utils.get_logger / init_logging（如果存在）；
- 若 log_utils 不存在，则自动回退到一个简单的 logging.basicConfig 配置；
- 其他模块一律通过 core.logging_utils.get_logger 来拿 logger。

用法示例：
    from core.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("系统启动")

这样后续即便我们内部调整日志实现，只要保持这个接口不变，
所有业务代码都不用改。
"""

from typing import Optional
import logging

# 尝试复用项目现有的 log_utils（如果存在的话）
try:
    from log_utils import get_logger as _legacy_get_logger, init_logging as _legacy_init_logging  # type: ignore
except Exception:  # log_utils 不存在或者导入失败时，走降级路径
    _legacy_get_logger = None
    _legacy_init_logging = None


_LOGGING_INITIALIZED = False


def _init_fallback_logging() -> None:
    """
    当项目中没有 log_utils，或者 log_utils 导入失败时，
    使用一个简单的 basicConfig 作为兜底方案。
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _LOGGING_INITIALIZED = True


def init_logging() -> None:
    """
    统一的日志初始化入口：
    - 如果有 log_utils，则调用其中的 init_logging；
    - 否则使用 fallback 配置。
    """
    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return

    if _legacy_init_logging is not None:
        # 使用项目里已经实现好的日志系统
        _legacy_init_logging()
        _LOGGING_INITIALIZED = True
    else:
        # 否则就用兜底方案
        _init_fallback_logging()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取指定名称的 logger。
    - 会确保日志系统已经初始化；
    - 如果有 log_utils，则复用其中的 get_logger；
    - 否则使用 logging.getLogger。
    """
    # 先初始化日志
    init_logging()

    if _legacy_get_logger is not None:
        return _legacy_get_logger(name)

    # 兜底：直接返回标准 logging 的 logger
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)
