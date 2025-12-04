# -*- coding: utf-8 -*-
"""
根目录 config_center.py

兼容层：为了兼容老代码中直接 `import config_center` 的写法，
这里简单地从 `config.config_center` 转 re-export 所有关键接口。

新代码请优先使用：

    from config.config_center import get_system_config, get_path
"""

from config.config_center import (  # noqa: F401
    CONFIG_FILE_ENV_VAR,
    PROJECT_ROOT,
    ensure_runtime_dirs,
    get_config_path,
    get_logging_config,
    get_path,
    get_paths_config,
    get_system_config,
    reload_system_config,
)

__all__ = [
    "CONFIG_FILE_ENV_VAR",
    "PROJECT_ROOT",
    "get_system_config",
    "reload_system_config",
    "get_config_path",
    "get_logging_config",
    "get_paths_config",
    "get_path",
    "ensure_runtime_dirs",
]
