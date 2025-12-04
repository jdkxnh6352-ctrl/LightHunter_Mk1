# -*- coding: utf-8 -*-
"""
模块名称：SystemOptimizer Mk-Overclock
版本：Mk-Opt R20 (Affinity + TCP Hints + Strategy Gene HUD)
路径: G:/LightHunter_Mk1/system_optimizer.py

功能：
- 探测本机 CPU / 内存 / 操作系统信息；
- 为 LightHunter 进程提供 CPU 绑定 & 进程优先级提升的统一入口；
- 输出 Windows / Linux 的 TCP 调优建议（打印，不自动修改系统）；
- 读取/生成 system_optimizer_config.json，方便你手工微调核心分配。
- 新增：如存在 gene_config.json，则展示当前策略基因权重 + 决策缺口指标（FN / FP）。
"""

import os
import json
import platform
import argparse
from typing import Dict, List, Any, Optional

try:
    import psutil  # type: ignore

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False


class SystemOptimizer:
    def __init__(self, config_path: str = "system_optimizer_config.json"):
        self.config_path = config_path
        self.sys_info = self._detect_system()
        self.config = self._load_or_init_config()
        self.gene_info: Optional[Dict[str, Any]] = self._load_gene_info()

    # --------------------------------------------------
    # 基础探测
    # --------------------------------------------------
    def _detect_system(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "cpu_count_logical": os.cpu_count() or 1,
            "cpu_count_physical": None,
            "memory_total_gb": None,
        }
        if HAS_PSUTIL:
            try:
                info["cpu_count_physical"] = psutil.cpu_count(logical=False)
            except Exception:
                info["cpu_count_physical"] = None
            try:
                mem = psutil.virtual_memory()
                info["memory_total_gb"] = round(mem.total / (1024**3), 1)
            except Exception:
                info["memory_total_gb"] = None
        return info

    # --------------------------------------------------
    # 配置：从文件加载 / 默认生成
    # --------------------------------------------------
    def _default_config(self) -> Dict[str, Any]:
        """
        按 CPU 核心大致分三类角色：
        - fetch  : 采集（高实时）
        - compute: 计算（AI / 回测）
        - io     : 后勤（日志 / QA）
        默认策略：前 40% 核给采集，中间 30% 给计算，最后 30% 给后勤。
        你可以手动修改 system_optimizer_config.json 进行微调。
        """
        total = self.sys_info.get("cpu_count_logical") or 1
        cores = list(range(total))

        if total <= 4:
            fetch_cores = cores
            compute_cores: List[int] = cores
            io_cores: List[int] = cores
        else:
            n_fetch = max(1, int(total * 0.4))
            n_compute = max(1, int(total * 0.3))
            # 剩余给 IO
            fetch_cores = cores[:n_fetch]
            compute_cores = cores[n_fetch : n_fetch + n_compute]
            io_cores = cores[n_fetch + n_compute :]

            if not io_cores:
                io_cores = compute_cores

        cfg = {
            "version": "Mk-Opt R20",
            "cpu_map": {
                "fetch": fetch_cores,
                "compute": compute_cores,
                "io": io_cores,
            },
            # 是否在 apply_affinity_for_current_process 时同时提升优先级
            "boost_priority": True,
            # TCP 调优模式（当前仅用于打印建议）
            "tcp_profile": "safe_default",
        }
        return cfg

    def _load_or_init_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cfg
            except Exception:
                pass

        cfg = self._default_config()
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            print(
                f"[SYS] 默认 system_optimizer_config.json 已生成 -> {self.config_path}"
            )
        except Exception:
            print("[SYS] 无法写入 system_optimizer_config.json，但不影响运行。")
        return cfg

    # --------------------------------------------------
    # 读取基因信息（可选）
    # --------------------------------------------------
    def _load_gene_info(self) -> Optional[Dict[str, Any]]:
        """
        可选：尝试读取 gene_config.json，用于在 Summary 中展示策略基因 & 决策缺口信息。
        """
        path = "gene_config.json"
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return None

    # --------------------------------------------------
    # 打印系统 & 配置摘要
    # --------------------------------------------------
    def print_summary(self) -> None:
        print("\n===== SystemOptimizer Summary =====")
        print(
            f" OS      : {self.sys_info.get('os')} "
            f"{self.sys_info.get('os_release')}"
        )
        print(f" Version : {self.sys_info.get('os_version')}")
        print(
            f" CPU     : logical={self.sys_info.get('cpu_count_logical')}, "
            f"physical={self.sys_info.get('cpu_count_physical')}"
        )
        print(f" Memory  : {self.sys_info.get('memory_total_gb')} GB")
        cpu_map = self.config.get("cpu_map", {})
        print(" CPU Map :")
        for role in ("fetch", "compute", "io"):
            cores = cpu_map.get(role, [])
            print(f"   - {role:7s}: {cores}")
        print(" Priority: auto-boost =", self.config.get("boost_priority", True))
        print(" TCP     : profile    =", self.config.get("tcp_profile", "safe_default"))

        # 策略基因 / 决策缺口 HUD
        gene = self.gene_info
        if gene:
            print("\n--- Strategy Gene / Decision Gap ---")
            updated_at = gene.get("updated_at", "N/A")
            src = gene.get("source_file", "")
            src_name = os.path.basename(src) if src else ""
            uses_ts = gene.get("uses_ts_label", False)
            fit = gene.get("fitness_score", None)
            weights = gene.get("weights", {})

            print(f" Gene updated : {updated_at}")
            if src_name:
                print(f" Source file  : {src_name}")
            print(f" TS labels    : {uses_ts}")
            if weights:
                print(f" Weights      : {weights}")
            if fit is not None:
                print(f" Fitness      : {fit:.3f}")

            gap = gene.get("decision_gap", {})
            if gap:
                fn_rate = float(gap.get("fn_rate", 0.0))
                fp_rate = float(gap.get("fp_rate", 0.0))
                print(
                    f" DecisionGap  : FN={fn_rate*100:.1f}% (高分没买), "
                    f"FP={fp_rate*100:.1f}% (低分乱买)"
                )

        print("===================================\n")

    # --------------------------------------------------
    # CPU 绑定 & 优先级
    # --------------------------------------------------
    def apply_affinity_for_current_process(self, role: str = "fetch") -> None:
        """
        将当前 Python 进程绑定到配置文件中的指定核心。
        role: 'fetch' / 'compute' / 'io'
        """
        if not HAS_PSUTIL:
            print(
                "[SYS] 未安装 psutil，无法设置 CPU 亲和性。请先执行：pip install psutil"
            )
            return

        role = role.lower()
        cpu_map: Dict[str, List[int]] = self.config.get("cpu_map", {})
        cores: Optional[List[int]] = cpu_map.get(role)

        if not cores:
            print(f"[SYS] 配置中没有找到角色 {role} 对应的核心，跳过 affinity。")
            return

        cores = [c for c in cores if isinstance(c, int) and c >= 0]

        if not cores:
            print(f"[SYS] 角色 {role} 的核心列表为空，跳过 affinity。")
            return

        try:
            p = psutil.Process(os.getpid())
            p.cpu_affinity(cores)  # type: ignore[attr-defined]
            print(f"[SYS] 当前进程已绑定到角色 {role} 的核心：{cores}")
        except Exception as e:
            print(f"[SYS] 设置 CPU affinity 失败：{e}")

    def boost_priority(self) -> None:
        """
        提升当前进程优先级：
        - Windows: HIGH_PRIORITY_CLASS
        - Linux/Unix: nice 值 -5（需要足够权限，否则自动忽略）
        """
        if not HAS_PSUTIL:
            print(
                "[SYS] 未安装 psutil，无法调整进程优先级。可选：pip install psutil"
            )
            return

        os_name = self.sys_info.get("os", "")
        try:
            p = psutil.Process(os.getpid())
            if os_name == "Windows":
                try:
                    p.nice(psutil.HIGH_PRIORITY_CLASS)  # type: ignore[attr-defined]
                    print("[SYS] 已将当前进程优先级提升到 HIGH_PRIORITY_CLASS (Windows)。")
                except Exception as e:
                    print(f"[SYS] 设置 Windows 进程优先级失败：{e}")
            else:
                # Linux / macOS：nice 值越低优先级越高，-5 已经算比较客气
                try:
                    p.nice(-5)
                    print("[SYS] 已尝试将当前进程 nice 调整为 -5（需要权限，不成功会被忽略）。")
                except Exception as e:
                    print(f"[SYS] 设置 nice 失败：{e}")
        except Exception as e:
            print(f"[SYS] 获取进程信息失败：{e}")

    # --------------------------------------------------
    # TCP / 网络调优建议（只打印，不自动执行）
    # --------------------------------------------------
    def print_tcp_tuning_hints(self) -> None:
        """
        输出一组可选的 TCP 调优命令（仅供参考，需要你手动在管理员终端执行）。
        目的：提高高并发 HTTP 请求的稳定性，减少 TIME_WAIT 堆积。
        """
        os_name = self.sys_info.get("os", "").lower()
        print("\n===== TCP / Network Tuning Hints =====")
        if os_name == "windows":
            print("【Windows 建议 - 请在“以管理员身份运行”的 PowerShell 或 CMD 中手动执行】")
            print("1) 查看当前 TCP 全局设置：")
            print("   netsh int tcp show global\n")
            print("2) 开启/确认推荐的全局设置（相对保守，不会太激进）：")
            print("   netsh int tcp set global autotuninglevel=normal")
            print("   netsh int tcp set global rss=enabled")
            print("   netsh int tcp set global ecncapability=disabled")
            print("   netsh int tcp set global timestamps=disabled\n")
            print("3) 如果你经常跑超高并发（>= 2000 连接），可以考虑在注册表中调大临时端口范围，")
            print("   但这有一定风险，建议先创建系统还原点，再酌情操作。")
        elif os_name in ("linux", "darwin"):
            print("【Linux/macOS 建议 - 请在 root/sudo 下手动执行】")
            print("1) 查看当前参数：")
            print("   sysctl net.ipv4.ip_local_port_range")
            print("   sysctl net.ipv4.tcp_tw_reuse")
            print("   sysctl net.core.somaxconn\n")
            print("2) 相对保守的优化示例（请按需修改）：")
            print("   sysctl -w net.ipv4.ip_local_port_range=\"10000 65000\"")
            print("   sysctl -w net.ipv4.tcp_tw_reuse=1")
            print("   sysctl -w net.core.somaxconn=4096\n")
            print("   # 若要永久生效，请写入 /etc/sysctl.conf 后执行 sysctl -p")
        else:
            print("未识别到常见系统类型，仅建议：")
            print("- 保持系统更新；")
            print("- 避免同时开启大量无意义连接；")
            print("- 如遇连接上限，优先从应用侧降低并发或引入更多代理出口。")

        print("======================================\n")


# ------------------------------------------------------
# 命令行入口
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LightHunter System Optimizer (Mk-Opt R20)"
    )
    parser.add_argument(
        "--role",
        type=str,
        default="fetch",
        choices=["fetch", "compute", "io", "none"],
        help="为当前进程设置哪种角色的 CPU 绑定（默认：fetch）。",
    )
    parser.add_argument(
        "--no-tcp",
        action="store_true",
        help="仅做 CPU 绑定和优先级调整，不打印 TCP 调优建议。",
    )
    args = parser.parse_args()

    opt = SystemOptimizer()
    opt.print_summary()

    if args.role != "none":
        opt.apply_affinity_for_current_process(role=args.role)
        if opt.config.get("boost_priority", True):
            opt.boost_priority()

    if not args.no_tcp:
        opt.print_tcp_tuning_hints()


if __name__ == "__main__":
    main()
