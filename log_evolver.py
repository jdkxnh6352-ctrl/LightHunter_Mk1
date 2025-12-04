# -*- coding: utf-8 -*-
"""
模块名称：LogEvolver Mk-Doctor
版本：Mk-Doctor R10 (Log-Driven Defense)
路径: G:/LightHunter_Mk1/log_evolver.py

功能：
- 扫描 logs 目录下的日志文件，抽取高频 / 高危错误场景；
- 按 站点(THS / EM / TX / OTHER) + 错误类型(HTTP_403 / TIMEOUT 等) 统计“痛点画像”；
- 根据统计结果，反推一组建议的防御参数（限速 / 并发 / 重试次数）；
- 输出 log_evolution_report.json，供你手工审阅，或后续接入 RequestEngine / MarketHunter。

说明：
- 这是“研究工具”，不会自动改配置，只是给出“防御策略建议”；
- 你可以每天盘后跑一次，结合 DataGuardian 一起看数据健康 & 错误画像。
"""

import os
import re
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional

from colorama import Fore, Style, init

init(autoreset=True)


class LogEvolver:
    """
    日志驱动自我进化实验室：
    - 目前假设你的统一日志都落在 ./logs 目录（可以通过 log_dir 参数修改）；
    - 日志格式不做强假设，只要行里带有 'ERROR' / 'CRITICAL' /
      'Traceback' / 'failed' / '失败' 之类关键词，就会尝试纳入统计。
    """

    def __init__(
        self,
        log_dir: str = "logs",
        pattern: str = ".log",
        output_file: str = "log_evolution_report.json",
    ):
        self.log_dir = log_dir
        self.pattern = pattern
        self.output_file = output_file

    # ------------------------ 主入口 ------------------------ #
    def run(self) -> Dict[str, Any]:
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "[LOG] LogEvolver Mk-Doctor Online, scanning logs..."
            + Style.RESET_ALL
        )
        records = self._collect_records()
        if not records:
            print(
                Fore.YELLOW
                + "[LOG] 没有在日志里发现明显的错误行，或者 logs 目录为空。"
                + Style.RESET_ALL
            )
            summary = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_error_lines": 0,
                "stats": {},
                "site_defense": {},
                "top_error_examples": {},
                "notes": "No error-like lines found. Check log_dir or logging config.",
            }
            self._save_report(summary)
            return summary

        summary = self._aggregate(records)
        self._print_summary(summary)
        self._save_report(summary)
        return summary

    # ------------------------ 日志收集 ------------------------ #
    def _iter_log_files(self) -> List[str]:
        if not os.path.isdir(self.log_dir):
            return []
        files: List[str] = []
        for name in os.listdir(self.log_dir):
            if not name:
                continue
            if self.pattern and not name.endswith(self.pattern):
                continue
            full = os.path.join(self.log_dir, name)
            if os.path.isfile(full):
                files.append(full)
        return sorted(files)

    def _collect_records(self) -> List[Dict[str, Any]]:
        """
        从所有日志文件里抽取“疑似错误”的行，结构化为记录：
        {time, module, site, etype, message, file}
        """
        files = self._iter_log_files()
        if not files:
            print(
                Fore.YELLOW
                + f"[LOG] log_dir={self.log_dir} 下没有匹配 *{self.pattern} 的日志文件。"
                + Style.RESET_ALL
            )
            return []

        records: List[Dict[str, Any]] = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        rec = self._parse_line(line, os.path.basename(path))
                        if rec is not None:
                            records.append(rec)
            except Exception as e:
                print(
                    Fore.RED
                    + f"[LOG] 读取日志 {path} 失败: {e}"
                    + Style.RESET_ALL
                )
        print(
            Fore.CYAN
            + f"[LOG] 已从 {len(files)} 个日志文件抽取 {len(records)} 条错误相关记录。"
            + Style.RESET_ALL
        )
        return records

    # ------------------------ 单行解析 ------------------------ #
    _TS_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    def _parse_line(self, line: str, filename: str) -> Optional[Dict[str, Any]]:
        raw = line.strip()
        if not raw:
            return None

        # 初步过滤：只抓明显跟错误有关的行
        lowered = raw.lower()
        hit = any(
            key in lowered
            for key in [
                "error",
                "critical",
                "exception",
                "traceback",
                "failed",
                "失败",
                "crash",
            ]
        )
        if not hit:
            return None

        # 时间戳（可选）
        ts_match = self._TS_PATTERN.search(raw)
        t_obj: Optional[datetime] = None
        if ts_match:
            ts_str = ts_match.group(1)
            try:
                t_obj = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                t_obj = None

        module = self._guess_module(raw, filename)
        site = self._guess_site(raw)
        etype = self._guess_error_type(raw)

        return {
            "time": t_obj.isoformat() if t_obj else None,
            "module": module,
            "site": site,
            "etype": etype,
            "message": raw,
            "file": filename,
        }

    @staticmethod
    def _guess_module(msg: str, filename: str) -> str:
        msg_lower = msg.lower()
        if "requestengine" in msg_lower or "[req]" in msg_lower:
            return "RequestEngine"
        if "commander" in msg_lower or "[cmd]" in msg_lower:
            return "Commander"
        if "markethunter" in msg_lower or "data_core" in msg_lower:
            return "MarketHunter"
        if "nightops" in msg_lower or "[night]" in msg_lower:
            return "NightOps"
        if "ts_recorder" in msg_lower or "ts_data" in msg_lower:
            return "TSRecorder"
        if "riskbrain" in msg_lower:
            return "RiskBrain"
        if "sequencebrain" in msg_lower or "seqbrain" in msg_lower:
            return "SequenceBrain"
        if "dataguardian" in msg_lower:
            return "DataGuardian"
        # fallback：根据文件名大概判断
        base = filename.lower()
        if "request" in base:
            return "RequestEngine"
        if "commander" in base:
            return "Commander"
        if "night" in base:
            return "NightOps"
        if "risk" in base:
            return "RiskBrain"
        return "Unknown"

    @staticmethod
    def _guess_site(msg: str) -> str:
        m = msg.lower()
        if "10jqka.com.cn" in m or "ths" in m:
            return "THS"
        if "eastmoney.com" in m or "push2.eastmoney" in m or "em" in m:
            return "EM"
        if "gtimg.cn" in m or "qt.gtimg" in m or "tencent" in m:
            return "TX"
        if "akshare" in m:
            return "AKSHARE"
        if "sina.com" in m:
            return "SINA"
        return "OTHER"

    @staticmethod
    def _guess_error_type(msg: str) -> str:
        m = msg.lower()
        if " 403" in m or "http 403" in m or "status_code=403" in m:
            return "HTTP_403_FORBIDDEN"
        if " 429" in m or "http 429" in m or "too many requests" in m:
            return "HTTP_429_TOO_MANY"
        if "timeout" in m or "timed out" in m:
            return "TIMEOUT"
        if "proxyerror" in m or "proxy" in m:
            return "PROXY_ERROR"
        if "connection" in m and ("reset" in m or "aborted" in m or "refused" in m):
            return "CONNECTION_ERROR"
        if "ssl" in m or "tls" in m:
            return "TLS_SSL_ERROR"
        if "parse" in m or "解析" in m or "结构变化" in m:
            return "PARSER_ERROR"
        if "keyboardinterrupt" in m:
            return "USER_INTERRUPT"
        if "memoryerror" in m or "out of memory" in m:
            return "MEMORY_ERROR"
        if "crash" in m:
            return "CRASH"
        return "OTHER_ERROR"

    # ------------------------ 统计与策略推演 ------------------------ #
    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        by_type: Counter = Counter()
        by_site: Counter = Counter()
        by_module: Counter = Counter()
        examples_by_type: Dict[str, List[str]] = defaultdict(list)
        type_site: Dict[str, Counter] = defaultdict(Counter)

        for rec in records:
            etype = rec["etype"]
            site = rec["site"]
            module = rec["module"]
            by_type[etype] += 1
            by_site[site] += 1
            by_module[module] += 1
            type_site[site][etype] += 1
            if len(examples_by_type[etype]) < 5:
                examples_by_type[etype].append(rec["message"])

        site_defense = self._derive_defense(type_site)

        stats = {
            "total_error_lines": total,
            "by_type": dict(by_type),
            "by_site": dict(by_site),
            "by_module": dict(by_module),
            "type_site_matrix": {
                site: dict(c) for site, c in type_site.items()
            },
        }

        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_error_lines": total,
            "stats": stats,
            "site_defense": site_defense,
            "top_error_examples": examples_by_type,
            "notes": (
                "This report is heuristic. Review site_defense and adjust "
                "RequestEngine / MarketHunter 配置时请结合实际情况。"
            ),
        }
        return summary

    def _derive_defense(self, type_site: Dict[str, Counter]) -> Dict[str, Any]:
        """
        根据 每站点 × 错误类型 的统计，给出推荐的防御策略参数。
        返回结构：
        {
          "THS": {
             "risk_level": "high",
             "error_profile": {...},
             "suggested": {
                "max_concurrency": 2,
                "max_retries": 1,
                "cooldown_sec": 2.5,
             }
          },
          ...
        }
        """
        site_defense: Dict[str, Any] = {}
        for site, counter in type_site.items():
            total = sum(counter.values())
            if total == 0:
                continue
            share = {
                etype: count / total for etype, count in counter.items()
            }

            # 粗暴估个风险等级
            high_signals = 0.0
            high_signals += share.get("HTTP_403_FORBIDDEN", 0.0) * 3.0
            high_signals += share.get("HTTP_429_TOO_MANY", 0.0) * 3.0
            high_signals += share.get("PARSER_ERROR", 0.0) * 2.0
            high_signals += share.get("CONNECTION_ERROR", 0.0) * 1.5

            # 默认值
            risk_level = "low"
            max_concurrency = 8
            max_retries = 3
            cooldown_sec = 0.3

            # 按错误结构调整参数
            if high_signals > 1.5 or total > 200:
                risk_level = "high"
                max_concurrency = 2
                max_retries = 1
                cooldown_sec = 2.5
            elif high_signals > 0.7 or total > 80:
                risk_level = "medium"
                max_concurrency = 4
                max_retries = 2
                cooldown_sec = 1.0

            # 如果 TIMEOUT 非常多，但 403/429 不多，可以略微提高重试但降低并发
            timeout_share = share.get("TIMEOUT", 0.0)
            if timeout_share > 0.4 and share.get("HTTP_403_FORBIDDEN", 0.0) < 0.1:
                max_retries = max(max_retries, 3)
                max_concurrency = min(max_concurrency, 4)
                cooldown_sec = max(cooldown_sec, 0.8)

            site_defense[site] = {
                "risk_level": risk_level,
                "total_errors": int(total),
                "error_profile": {k: float(round(v * 100.0, 2)) for k, v in share.items()},
                "suggested": {
                    "max_concurrency": int(max_concurrency),
                    "max_retries": int(max_retries),
                    "cooldown_sec": float(round(cooldown_sec, 2)),
                },
            }

        return site_defense

    # ------------------------ 输出 ------------------------ #
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        total = summary.get("total_error_lines", 0)
        stats = summary.get("stats", {})
        by_type: Dict[str, int] = stats.get("by_type", {})
        by_site: Dict[str, int] = stats.get("by_site", {})
        by_module: Dict[str, int] = stats.get("by_module", {})
        site_defense: Dict[str, Any] = summary.get("site_defense", {})

        print(
            Fore.YELLOW
            + f"\n[LOG] 总错误相关行数: {total}"
            + Style.RESET_ALL
        )

        if by_type:
            print(
                Fore.CYAN
                + "[LOG] 按错误类型统计 (Top 8):"
                + Style.RESET_ALL
            )
            for etype, cnt in sorted(
                by_type.items(), key=lambda x: x[1], reverse=True
            )[:8]:
                print(f"  - {etype:<24} : {cnt:>5d}")

        if by_site:
            print(
                Fore.CYAN
                + "\n[LOG] 按站点统计:"
                + Style.RESET_ALL
            )
            for site, cnt in sorted(
                by_site.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {site:<8} : {cnt:>5d}")

        if by_module:
            print(
                Fore.CYAN
                + "\n[LOG] 按模块统计:"
                + Style.RESET_ALL
            )
            for mod, cnt in sorted(
                by_module.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {mod:<12} : {cnt:>5d}")

        if site_defense:
            print(
                Fore.MAGENTA
                + "\n[LOG] 防御策略建议（按站点）："
                + Style.RESET_ALL
            )
            for site, cfg in site_defense.items():
                risk = cfg.get("risk_level", "low")
                prof = cfg.get("error_profile", {})
                sug = cfg.get("suggested", {})
                risk_color = (
                    Fore.RED
                    if risk == "high"
                    else (Fore.YELLOW if risk == "medium" else Fore.GREEN)
                )
                print(
                    f"  [{site}] risk={risk_color}{risk}{Style.RESET_ALL}, "
                    f"total_err={cfg.get('total_errors', 0)}, "
                    f"403={prof.get('HTTP_403_FORBIDDEN', 0.0):.1f}%, "
                    f"429={prof.get('HTTP_429_TOO_MANY', 0.0):.1f}%, "
                    f"TIMEOUT={prof.get('TIMEOUT', 0.0):.1f}%"
                )
                print(
                    f"       建议: max_concurrency={sug.get('max_concurrency')}, "
                    f"max_retries={sug.get('max_retries')}, "
                    f"cooldown_sec={sug.get('cooldown_sec')}"
                )

        print(
            Fore.GREEN
            + f"\n[LOG] 详细报告已生成 -> {self.output_file}"
            + Style.RESET_ALL
        )

    def _save_report(self, summary: Dict[str, Any]) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(
                Fore.RED
                + f"[LOG] 写入 {self.output_file} 失败: {e}"
                + Style.RESET_ALL
            )


def main():
    lab = LogEvolver()
    lab.run()


if __name__ == "__main__":
    main()
