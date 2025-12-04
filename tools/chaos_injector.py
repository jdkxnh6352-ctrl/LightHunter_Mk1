# -*- coding: utf-8 -*-
"""
tools/chaos_injector.py

LightHunter Mk4 - Chaos Injector
================================

用于在本地环境主动注入各类“可控故障”，配合：

- MonitorDaemon（ops/monitor_daemon.py）
- Web HUD（hud/web_server.py）
- NightOps / DayOps / Scheduler
- DataGuardian / 数据质量报告
- 交易链路（PaperBroker + TradeCore）

来验证整套系统在极端异常情况下的表现。

当前内置的 Chaos 场景示例：

1. N1_EM_BLOCK
   - 模拟东财域名被封 / 访问失败：
   - 将 eastmoney.com 相关域名路由到一个不存在的代理（chaos_blackhole）。

2. N2_THS_BLOCK
   - 模拟同花顺域名被封：
   - 将 10jqka.com.cn 等域名路由到 chaos_blackhole。

3. NET_TIMEOUT
   - 全局网络超时收紧：
   - 将 network.default_timeout_sec 调低（例如 0.5s），制造大量超时。

4. D1_DUCKDB_LOCK
   - 模拟 DuckDB 被长事务锁住：
   - 打开 DuckDB，执行 BEGIN EXCLUSIVE 并保持一段时间。

5. D3_DELETE_PARQUET_SLICE
   - 模拟某交易日 / 部分标的 Parquet 缺失：
   - 根据日期字符串匹配 Parquet 文件名，选择性删除（需要 -y 确认）。

注意：
- 所有“修改 system_config.json”类场景，都会先生成一个
  system_config.json.chaos_bak 备份文件，用于一键恢复。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# ----------------------------------------------------------------------
# 日志统一
# ----------------------------------------------------------------------

try:  # 优先使用工程内的 logging_utils
    from core.logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str):
        return logging.getLogger(name)


log = get_logger(__name__)

# 可选 DuckDB 支持
try:  # pragma: no cover
    import duckdb  # type: ignore

    HAS_DUCKDB = True
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore
    HAS_DUCKDB = False

# 尝试读取系统配置（用于推断路径）
try:  # pragma: no cover
    from config.config_center import get_system_config  # type: ignore
except Exception:  # pragma: no cover
    def get_system_config(refresh: bool = False) -> Dict[str, Any]:
        return {}


# ----------------------------------------------------------------------
# 场景定义
# ----------------------------------------------------------------------


@dataclass
class ScenarioDef:
    """
    Chaos 场景定义。
    """
    id: str
    group: str
    description: str
    touches_config: bool
    handler: Callable[[Dict[str, Any], argparse.Namespace, str], None]


# ----------------------------------------------------------------------
# 工具函数：配置文件读写
# ----------------------------------------------------------------------


def _resolve_config_path(sys_cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    尝试根据 system_config 推断 system_config.json 所在路径。
    优先顺序：
    1. 环境变量 LIGHTHUNTER_CONFIG_PATH
    2. sys_cfg.paths.project_root + sys_cfg.paths.config_dir + system_config.json
    3. ./config/system_config.json
    4. ./system_config.json
    """
    env_path = os.environ.get("LIGHTHUNTER_CONFIG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    sys_cfg = sys_cfg or get_system_config()
    if isinstance(sys_cfg, dict):
        paths = sys_cfg.get("paths", {}) or {}
        project_root = paths.get("project_root", ".")
        config_dir = paths.get("config_dir", "config")
        candidate = os.path.join(project_root, config_dir, "system_config.json")
        if os.path.isfile(candidate):
            return candidate

    # fallback
    for candidate in (
        os.path.join("config", "system_config.json"),
        "system_config.json",
    ):
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("未找到 system_config.json，请确认当前工作目录是否是项目根目录。")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Dict[str, Any], path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


def _ensure_backup(config_path: str) -> str:
    """
    确保存在一个原始的备份文件：
    config_path + ".chaos_bak"
    """
    backup_path = config_path + ".chaos_bak"
    if not os.path.isfile(backup_path):
        shutil.copy2(config_path, backup_path)
        log.info("ChaosInjector: 已创建配置备份 %s", backup_path)
    else:
        log.info("ChaosInjector: 已存在配置备份 %s", backup_path)
    return backup_path


def _restore_backup(config_path: str) -> None:
    backup_path = config_path + ".chaos_bak"
    if not os.path.isfile(backup_path):
        log.warning("ChaosInjector: 未找到备份文件 %s，无需恢复。", backup_path)
        return
    shutil.copy2(backup_path, config_path)
    log.info("ChaosInjector: 已从备份恢复配置 %s <- %s", config_path, backup_path)


# ----------------------------------------------------------------------
# 工具函数：推断 DuckDB / Parquet 路径
# ----------------------------------------------------------------------


def _resolve_duckdb_path(sys_cfg: Dict[str, Any]) -> str:
    duckdb_cfg = sys_cfg.get("duckdb", {}) or {}
    db_path = duckdb_cfg.get("db_path")
    if db_path:
        return db_path

    storage_ts = sys_cfg.get("storage", {}).get("ts", {}) or {}
    db_path = storage_ts.get("duckdb_path")
    if db_path:
        return db_path

    paths = sys_cfg.get("paths", {}) or {}
    db_path = paths.get("duckdb_path", "data/lighthunter.duckdb")
    return db_path


def _resolve_parquet_dir(sys_cfg: Dict[str, Any]) -> str:
    storage_ts = sys_cfg.get("storage", {}).get("ts", {}) or {}
    pq_dir = storage_ts.get("parquet_dir")
    if pq_dir:
        return pq_dir

    paths = sys_cfg.get("paths", {}) or {}
    pq_dir = paths.get("parquet_dir", "data/parquet")
    return pq_dir


# ----------------------------------------------------------------------
# 场景实现：网络相关
# ----------------------------------------------------------------------


def _scenario_em_block(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    """
    N1_EM_BLOCK：模拟东财被封锁，将 eastmoney 相关域名路由到 chaos_blackhole。
    """
    cfg = _load_json(config_path)
    network = cfg.setdefault("network", {})
    proxies = network.setdefault("proxies", {})
    pool = proxies.setdefault("pool", [])
    routes = network.setdefault("routes", {})

    blackhole_id = "chaos_blackhole"

    # 确保存在 blackhole 代理（指向一个必然失败的端口）
    if not any(p.get("id") == blackhole_id for p in pool):
        pool.append(
            {
                "id": blackhole_id,
                "type": "http",
                "http_proxy": "http://127.0.0.1:65535",
                "socks_proxy": None,
                "weight": 1.0,
            }
        )

    targets = [
        "eastmoney.com",
        "emdata.eastmoney.com",
        "push2.eastmoney.com",
    ]
    for host in targets:
        routes[host] = blackhole_id

    _save_json(cfg, config_path)
    log.info("已应用 N1_EM_BLOCK：eastmoney 相关域名将通过 chaos_blackhole 代理（请求将全部失败）。")


def _scenario_ths_block(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    """
    N2_THS_BLOCK：模拟同花顺被封锁，将 10jqka 域名路由到 chaos_blackhole。
    """
    cfg = _load_json(config_path)
    network = cfg.setdefault("network", {})
    proxies = network.setdefault("proxies", {})
    pool = proxies.setdefault("pool", [])
    routes = network.setdefault("routes", {})

    blackhole_id = "chaos_blackhole"

    if not any(p.get("id") == blackhole_id for p in pool):
        pool.append(
            {
                "id": blackhole_id,
                "type": "http",
                "http_proxy": "http://127.0.0.1:65535",
                "socks_proxy": None,
                "weight": 1.0,
            }
        )

    targets = [
        "10jqka.com.cn",
        "basic.10jqka.com.cn",
    ]
    for host in targets:
        routes[host] = blackhole_id

    _save_json(cfg, config_path)
    log.info("已应用 N2_THS_BLOCK：10jqka 域名将通过 chaos_blackhole 代理（请求将全部失败）。")


def _scenario_net_timeout(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    """
    NET_TIMEOUT：整体网络超时调得很紧，模拟网络抖动/慢响应。
    """
    cfg = _load_json(config_path)
    network = cfg.setdefault("network", {})

    orig_timeout = float(network.get("default_timeout_sec", 8.0))
    new_timeout = float(getattr(args, "timeout", 0.5) or 0.5)
    network["default_timeout_sec"] = new_timeout

    log.info(
        "NET_TIMEOUT：default_timeout_sec 从 %.3f 调整为 %.3f（请求更容易超时）。",
        orig_timeout,
        new_timeout,
    )
    _save_json(cfg, config_path)


# ----------------------------------------------------------------------
# 场景实现：DuckDB / 存储相关
# ----------------------------------------------------------------------


def _scenario_duckdb_lock(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    """
    D1_DUCKDB_LOCK：打开 DuckDB 并持有 EXCLUSIVE 锁一段时间，
    模拟其它写入线程遇到“数据库忙 / locked”。
    """
    if not HAS_DUCKDB:
        log.error("D1_DUCKDB_LOCK：未安装 duckdb Python 包，无法执行该场景。请先 pip install duckdb。")
        return

    db_path = _resolve_duckdb_path(sys_cfg)
    duration = float(getattr(args, "duration", 60.0) or 60.0)

    log.info("D1_DUCKDB_LOCK：准备锁定 DuckDB 文件：%s，持续约 %.1f 秒。", db_path, duration)
    log.info("提示：在锁定期间，使用 TSStorage / DuckDB 写入的任务应该报错或重试。")

    conn = None
    try:
        conn = duckdb.connect(db_path)
        conn.execute("BEGIN EXCLUSIVE;")
        log.info("D1_DUCKDB_LOCK：已获取 EXCLUSIVE 锁，开始睡眠。按 Ctrl+C 可提前结束。")
        time.sleep(duration)
        log.info("D1_DUCKDB_LOCK：锁定结束，准备释放。")
    except KeyboardInterrupt:
        log.info("D1_DUCKDB_LOCK：收到中断信号，提前释放锁。")
    except Exception as e:
        log.error("D1_DUCKDB_LOCK：执行过程中发生错误：%s", e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        log.info("D1_DUCKDB_LOCK：连接已关闭。")


def _scenario_delete_parquet_slice(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    """
    D3_DELETE_PARQUET_SLICE：模拟部分 Parquet 文件缺失（某天 / 部分标的）。

    根据 --date 参数（例如 20240105），在 parquet 目录中查找包含该字符串的文件名，
    并删除其中一部分（或者全部）。该操作不可逆，请务必小心使用。
    """
    trade_date = getattr(args, "date", None)
    if not trade_date:
        log.error("D3_DELETE_PARQUET_SLICE：需要 --date 参数，例如 --date 20240105。")
        return

    parquet_dir = _resolve_parquet_dir(sys_cfg)
    if not os.path.isdir(parquet_dir):
        log.error("D3_DELETE_PARQUET_SLICE：Parquet 目录不存在：%s", parquet_dir)
        return

    # 收集候选文件
    candidates = []
    for root, _dirs, files in os.walk(parquet_dir):
        for fname in files:
            if trade_date in fname and fname.lower().endswith((".parquet", ".pq")):
                full_path = os.path.join(root, fname)
                candidates.append(full_path)

    if not candidates:
        log.warning("D3_DELETE_PARQUET_SLICE：未在 %s 下找到包含 '%s' 的 Parquet 文件。", parquet_dir, trade_date)
        return

    log.info("D3_DELETE_PARQUET_SLICE：找到 %d 个候选 Parquet 文件（日期包含 %s）：", len(candidates), trade_date)
    for p in candidates:
        log.info("  %s", p)

    # 控制删除比例
    ratio = float(getattr(args, "ratio", 0.5) or 0.5)
    ratio = max(0.0, min(1.0, ratio))
    delete_count = max(1, int(len(candidates) * ratio))
    to_delete = candidates[:delete_count]

    if not getattr(args, "yes", False):
        log.warning(
            "准备删除 %d / %d 个文件（比例=%.2f）。此操作不可逆，如确认请添加 -y/--yes。当前为 dry-run，仅打印不删除。",
            delete_count,
            len(candidates),
            ratio,
        )
        return

    log.warning("即将删除以下 %d 个 Parquet 文件（不可恢复）：", delete_count)
    for p in to_delete:
        log.warning("  DELETE %s", p)

    for p in to_delete:
        try:
            os.remove(p)
        except Exception as e:
            log.error("删除文件失败 %s: %s", p, e)

    log.info("D3_DELETE_PARQUET_SLICE：删除完成。后续可通过 replay_and_healthcheck / DataGuardian 观察缺失影响。")


# ----------------------------------------------------------------------
# 场景注册表
# ----------------------------------------------------------------------


def _build_scenarios() -> Dict[str, ScenarioDef]:
    return {
        "N1_EM_BLOCK": ScenarioDef(
            id="N1_EM_BLOCK",
            group="network",
            description="模拟东方财富（eastmoney）接口被封，所有 eastmoney 域名路由到不可达代理。",
            touches_config=True,
            handler=_scenario_em_block,
        ),
        "N2_THS_BLOCK": ScenarioDef(
            id="N2_THS_BLOCK",
            group="network",
            description="模拟同花顺（10jqka）接口被封，10jqka 域名路由到不可达代理。",
            touches_config=True,
            handler=_scenario_ths_block,
        ),
        "NET_TIMEOUT": ScenarioDef(
            id="NET_TIMEOUT",
            group="network",
            description="整体网络超时参数调低，制造更多超时错误。",
            touches_config=True,
            handler=_scenario_net_timeout,
        ),
        "D1_DUCKDB_LOCK": ScenarioDef(
            id="D1_DUCKDB_LOCK",
            group="storage",
            description="锁住 DuckDB 数据库一段时间，模拟数据库忙 / locked。",
            touches_config=False,
            handler=_scenario_duckdb_lock,
        ),
        "D3_DELETE_PARQUET_SLICE": ScenarioDef(
            id="D3_DELETE_PARQUET_SLICE",
            group="storage",
            description="删除指定日期的一部分 Parquet 文件，模拟数据缺失。",
            touches_config=False,
            handler=_scenario_delete_parquet_slice,
        ),
    }


SCENARIOS = _build_scenarios()


# ----------------------------------------------------------------------
# CLI 主逻辑
# ----------------------------------------------------------------------


def _cmd_list(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    print("可用 Chaos 场景：")
    for sid, s in sorted(SCENARIOS.items()):
        print(f"- {sid:<22} [{s.group}] {'(修改配置)' if s.touches_config else '           '}  {s.description}")


def _cmd_apply(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    scenario_id = (args.scenario or "").upper()
    if scenario_id not in SCENARIOS:
        print(f"未知场景：{scenario_id}")
        print("使用 `python -m tools.chaos_injector list` 查看可用场景。")
        sys.exit(1)

    scenario = SCENARIOS[scenario_id]
    log.info("准备应用 Chaos 场景：%s - %s", scenario.id, scenario.description)

    if scenario.touches_config:
        _ensure_backup(config_path)

    scenario.handler(sys_cfg, args, config_path)
    log.info("Chaos 场景 %s 已执行完成。", scenario.id)
    if scenario.touches_config:
        log.info("如需恢复原始配置，可运行：python -m tools.chaos_injector revert")


def _cmd_revert(sys_cfg: Dict[str, Any], args: argparse.Namespace, config_path: str) -> None:
    _restore_backup(config_path)


def main(argv: Optional[list[str]] = None) -> None:
    sys_cfg = get_system_config()
    try:
        config_path = _resolve_config_path(sys_cfg)
    except FileNotFoundError as e:
        log.error("无法解析 system_config.json 路径：%s", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="python -m tools.chaos_injector",
        description="LightHunter Chaos Injection 工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="列出可用的 Chaos 场景")

    # apply
    p_apply = subparsers.add_parser("apply", help="应用某个 Chaos 场景")
    p_apply.add_argument("scenario", help="场景 ID，例如 N2_THS_BLOCK")
    p_apply.add_argument("--timeout", type=float, default=0.5, help="NET_TIMEOUT 场景使用的新超时时间（秒）")
    p_apply.add_argument("--duration", type=float, default=60.0, help="D1_DUCKDB_LOCK 场景锁定时长（秒）")
    p_apply.add_argument("--date", type=str, help="D3_DELETE_PARQUET_SLICE 场景的日期字符串（例如 20240105）")
    p_apply.add_argument("--ratio", type=float, default=0.5, help="D3_DELETE_PARQUET_SLICE 删除比例（0~1）")
    p_apply.add_argument("-y", "--yes", action="store_true", help="对不可逆操作直接执行，无需再次确认")

    # revert
    p_revert = subparsers.add_parser(
        "revert",
        help="从 system_config.json.chaos_bak 恢复配置（撤销所有配置类 Chaos 场景）",
    )

    args = parser.parse_args(argv)

    if args.command == "list":
        _cmd_list(sys_cfg, args, config_path)
    elif args.command == "apply":
        _cmd_apply(sys_cfg, args, config_path)
    elif args.command == "revert":
        _cmd_revert(sys_cfg, args, config_path)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
