# -*- coding: utf-8 -*-
"""
模块名称：ProcessWatcher Mk-Keeper
版本：Mk-Keeper R10 (Auto-Restart Supervisor)
路径: G:/LightHunter_Mk1/process_watcher.py

功能：
- 作为统一的“守护进程”入口，负责拉起并监控核心模块；
- 当前支持：
    * collector : ts_collector.py  （分时采集 / TS 写库）
    * commander : commander.py     （实盘指挥官 / 交易引擎）
- 子进程异常退出（退出码 != 0）时，自动等待一小段时间后重启；
- 限制单位时间内的重启次数，避免“疯狂重启”。

用法示例：
    python process_watcher.py collector
    python process_watcher.py commander
"""

import argparse
import subprocess
import sys
import time
import os
import datetime
from typing import List

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    HAS_COLOR = True
except Exception:
    HAS_COLOR = False


def _c(text: str, color) -> str:
    if not HAS_COLOR:
        return text
    return color + text + Style.RESET_ALL


def build_command(role: str) -> List[str]:
    """
    根据 role 构造子进程启动命令，统一使用当前 Python 解释器。
    """
    python_exe = sys.executable or "python"

    if role == "collector":
        return [python_exe, "-u", "ts_collector.py"]
    elif role == "commander":
        return [python_exe, "-u", "commander.py"]
    else:
        raise ValueError(f"未知角色：{role}")


def supervise(role: str, delay: int = 5, max_restarts_per_hour: int = 20) -> None:
    """
    守护主循环：
    - 运行目标子进程；
    - 正常退出（exit code == 0）则直接结束；
    - 异常退出则记录一次“失败”，若 1 小时内失败次数未超过上限，则延时后重启。
    """
    cmd = build_command(role)
    workdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workdir)

    tag = role.upper()
    restart_times = []  # 记录每次“异常退出”的时间戳

    print(_c(f"[WATCH][{tag}] 工作目录：{workdir}", Fore.CYAN if HAS_COLOR else ""))
    print(_c(f"[WATCH][{tag}] 启动命令：{' '.join(cmd)}", Fore.CYAN if HAS_COLOR else ""))
    print(
        _c(
            f"[WATCH][{tag}] 异常退出后自动重启，间隔 {delay}s，每小时最多 {max_restarts_per_hour} 次。\n",
            Fore.CYAN if HAS_COLOR else "",
        )
    )

    while True:
        start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(_c(f"[WATCH][{tag}] 子进程启动 @ {start_ts}", Fore.GREEN if HAS_COLOR else ""))

        try:
            # 继承当前控制台的输入/输出，HUD 直接显示在终端里
            exit_code = subprocess.call(cmd)
        except KeyboardInterrupt:
            # 用户 Ctrl+C 直接退出 watcher
            print(_c(f"\n[WATCH][{tag}] 收到键盘中断，守护进程退出。", Fore.YELLOW if HAS_COLOR else ""))
            break
        except Exception as e:
            print(_c(f"[WATCH][{tag}] 启动子进程异常：{e}", Fore.RED if HAS_COLOR else ""))
            exit_code = -999

        end_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(_c(f"[WATCH][{tag}] 子进程退出 @ {end_ts}，退出码 = {exit_code}", Fore.YELLOW if HAS_COLOR else ""))

        # 正常退出：不再重启
        if exit_code == 0:
            print(_c(f"[WATCH][{tag}] 子进程正常结束，守护进程不再重启。", Fore.GREEN if HAS_COLOR else ""))
            break

        # 异常退出：记录一次失败
        now = time.time()
        restart_times.append(now)
        # 只保留最近 1 小时的记录
        one_hour_ago = now - 3600
        restart_times = [t for t in restart_times if t >= one_hour_ago]

        if len(restart_times) > max_restarts_per_hour:
            print(
                _c(
                    f"[WATCH][{tag}] 1 小时内异常退出次数超过 {max_restarts_per_hour} 次，"
                    f"停止自动重启，请检查日志与网络 / 代码问题。",
                    Fore.RED if HAS_COLOR else "",
                )
            )
            break

        print(
            _c(
                f"[WATCH][{tag}] 子进程异常退出，{delay} 秒后尝试第 {len(restart_times)} 次重启...",
                Fore.MAGENTA if HAS_COLOR else "",
            )
        )
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            print(_c(f"\n[WATCH][{tag}] 等待期间被中断，守护进程退出。", Fore.YELLOW if HAS_COLOR else ""))
            break


def main():
    parser = argparse.ArgumentParser(description="LightHunter Process Watcher (Mk-Keeper R10)")
    parser.add_argument(
        "role",
        choices=["collector", "commander"],
        help="要守护的角色：collector=ts_collector.py，commander=commander.py",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="子进程异常退出后重启的等待秒数（默认：5）",
    )
    parser.add_argument(
        "--max-restarts-per-hour",
        type=int,
        default=20,
        help="1 小时内最多允许的重启次数（默认：20；过多说明系统有严重问题）。",
    )

    args = parser.parse_args()
    supervise(
        role=args.role,
        delay=max(1, int(args.delay)),
        max_restarts_per_hour=max(1, int(args.max_restarts_per_hour)),
    )


if __name__ == "__main__":
    main()
