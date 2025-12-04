# -*- coding: utf-8 -*-
"""
hud/web_server.py

LightHunter Mk4 - Web HUD 实时看板（战役版）
=========================================

在 Mk3 Web HUD 基础上做了两类增强：

1）继续保持：
    - 通过 ZeroMQ 订阅：
        - "account.nav"      : 账户净值
        - "account.position" : 账户持仓
        - "risk.alert"       : 风险告警（来自 RiskBrain）
    - 从 ExperimentLab 或 experiments.jsonl 中读取最近实验记录；
    - /api/dashboard 返回完整 dashboard JSON。

2）新增两周战役监控支持：
    - 读取 MonitorDaemon 输出的 summary JSON（默认：paths.monitor_dir/state.json）；
    - 在首页增加“战役监控总览”卡片，展示：
        - 关键 Topic 延迟（EWMA）
        - 队列堆积情况
        - 各组件错误计数
    - /api/monitor_summary 返回该 summary JSON。

运行方式：
----------
python -m hud.web_server
或
python hud/web_server.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template

from core.logging_utils import get_logger
from config.config_center import get_system_config

log = get_logger(__name__)

# 可选：ZeroMQ 总线
try:  # pragma: no cover
    from bus.zmq_bus import ZmqBus  # type: ignore

    HAS_ZMQ_BUS = True
except Exception:  # pragma: no cover
    ZmqBus = None  # type: ignore
    HAS_ZMQ_BUS = False

# 可选：ExperimentLab
try:  # pragma: no cover
    from lab.experiment_lab import ExperimentLab  # type: ignore

    HAS_EXP_LAB = True
except Exception:  # pragma: no cover
    ExperimentLab = None  # type: ignore
    HAS_EXP_LAB = False


# ======================================================================
# 内部数据结构
# ======================================================================


@dataclass
class AccountView:
    account_id: str
    last_ts: float = 0.0
    nav: float = 0.0
    nav_peak: float = 0.0
    dd_total: float = 0.0
    dd_intraday: float = 0.0
    gross_exposure: float = 0.0
    max_single_weight: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "normal"
    positions: Dict[str, float] = field(default_factory=dict)


@dataclass
class SymbolRiskView:
    account_id: str
    symbol: str
    last_ts: float = 0.0
    score: float = 0.0
    level: str = "normal"
    alert_count: int = 0


@dataclass
class RiskAlertView:
    ts: float
    scope: str  # "account" / "symbol" / "general"
    level: str
    account_id: Optional[str] = None
    symbol: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentView:
    experiment_id: str
    model_id: str = ""
    task_type: str = ""
    status: str = ""
    ts: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# DashboardState：核心内存态
# ======================================================================


class DashboardState:
    """
    用于 Web HUD 的内存态，线程安全：

    - 通过 EventBus 消息更新 accounts / risk / alerts；
    - 周期性从 ExperimentLab 或 JSONL 中读取最近实验；
    - 提供 get_dashboard() 给 Flask 视图渲染。
    """

    def __init__(self, system_config: Optional[Dict[str, Any]] = None) -> None:
        self.system_config = system_config or get_system_config()
        hud_cfg = self.system_config.get("hud", {}) or {}
        self.max_alerts = int(hud_cfg.get("max_alerts", 100))
        self.max_experiments = int(hud_cfg.get("max_experiments", 20))

        paths_cfg = self.system_config.get("paths", {}) or {}
        self.experiments_dir = paths_cfg.get(
            "experiments_dir", os.path.join("data", "experiments")
        )

        # 状态容器
        self._accounts: Dict[str, AccountView] = {}
        self._symbols: Dict[str, SymbolRiskView] = {}
        self._alerts: deque[RiskAlertView] = deque(maxlen=self.max_alerts)
        self._experiments: List[ExperimentView] = []

        self._lock = threading.Lock()

        # ExperimentLab（如果有的话）
        self.exp_lab = None
        if HAS_EXP_LAB and ExperimentLab is not None:
            try:
                if hasattr(ExperimentLab, "from_system_config"):
                    self.exp_lab = ExperimentLab.from_system_config(
                        self.system_config
                    )  # type: ignore[attr-defined]
                else:
                    self.exp_lab = ExperimentLab()  # type: ignore[call-arg]
                log.info("DashboardState: 已初始化 ExperimentLab。")
            except Exception as e:
                log.warning("DashboardState: 初始化 ExperimentLab 失败: %s", e)
                self.exp_lab = None

        log.info(
            "DashboardState 初始化完成: experiments_dir=%s, max_alerts=%d, max_experiments=%d",
            self.experiments_dir,
            self.max_alerts,
            self.max_experiments,
        )

    # ------------------------------------------------------------------
    # EventBus 事件入口（由 web_server 调用）
    # ------------------------------------------------------------------

    def handle_bus_message(self, msg: Dict[str, Any]) -> None:
        """
        统一处理 EventBus 推来的消息。

        约定消息结构：
        {
          "topic": "account.nav",
          "ts": 1672531200000,
          "payload": {...}
        }
        也支持 payload 本身为消息。
        """
        topic = str(msg.get("topic") or msg.get("type") or "")
        payload = msg.get("payload", msg)

        if topic.startswith("account.nav"):
            self._on_account_nav(payload)
        elif topic.startswith("account.position"):
            self._on_account_position(payload)
        elif topic == "risk.alert" or str(payload.get("type", "")).startswith("risk."):
            self._on_risk_alert(payload)

    # ------------------------------------------------------------------
    # 账户净值事件
    # ------------------------------------------------------------------

    def _on_account_nav(self, payload: Dict[str, Any]) -> None:
        """
        预期结构：
        {
          "account_id": "acct_ultra_main",
          "ts": 1672531200000,
          "nav": 1234567.89,
          "nav_peak": ... (可选)
        }
        """
        account_id = str(payload.get("account_id") or "unknown")
        nav = float(payload.get("nav", 0.0))
        nav_peak = float(payload.get("nav_peak", nav))
        dd_total = float(payload.get("dd_total", 0.0))
        dd_intraday = float(payload.get("dd_intraday", 0.0))
        gross_exposure = float(payload.get("gross_exposure", 0.0))
        max_single_weight = float(payload.get("max_single_weight", 0.0))
        ts = self._normalize_ts(payload.get("ts"))

        with self._lock:
            acct = self._accounts.get(account_id)
            if acct is None:
                acct = AccountView(account_id=account_id)
                self._accounts[account_id] = acct

            acct.last_ts = ts
            acct.nav = nav
            acct.nav_peak = max(acct.nav_peak, nav_peak)
            acct.dd_total = dd_total
            acct.dd_intraday = dd_intraday
            acct.gross_exposure = gross_exposure
            acct.max_single_weight = max_single_weight

    # ------------------------------------------------------------------
    # 账户持仓事件
    # ------------------------------------------------------------------

    def _on_account_position(self, payload: Dict[str, Any]) -> None:
        """
        预期结构：
        {
          "account_id": "acct_ultra_main",
          "ts": 1672531200000,
          "positions": {
            "600000.SH": 0.05,
            "000001.SZ": 0.03
          }
        }
        """
        account_id = str(payload.get("account_id") or "unknown")
        ts = self._normalize_ts(payload.get("ts"))
        positions_raw = payload.get("positions") or {}
        positions = {str(sym): float(w) for sym, w in positions_raw.items()}

        with self._lock:
            acct = self._accounts.get(account_id)
            if acct is None:
                acct = AccountView(account_id=account_id)
                self._accounts[account_id] = acct
            acct.last_ts = ts
            acct.positions = positions

    # ------------------------------------------------------------------
    # 风险告警事件（来自 RiskBrain）
    # ------------------------------------------------------------------

    def _on_risk_alert(self, payload: Dict[str, Any]) -> None:
        """
        预期结构（来自 RiskBrain.publish）：

        账户告警：
        {
          "type": "risk.account_alert",
          "account_id": "acct_ultra_main",
          "level": "warning"/"critical",
          "score": 0.82,
          "message": "...",
          ...
        }

        标的告警：
        {
          "type": "risk.symbol_alert",
          "account_id": "acct_ultra_main",
          "symbol": "600000.SH",
          "level": "warning"/"critical",
          "score": 0.91,
          "message": "...",
          ...
        }
        """
        tp = str(payload.get("type", ""))
        level = str(payload.get("level", "info"))
        message = str(payload.get("message", ""))
        score = float(payload.get("score", 0.0))
        ts = self._normalize_ts(payload.get("ts"))

        with self._lock:
            if tp == "risk.account_alert":
                account_id = str(payload.get("account_id") or "unknown")
                # 更新账户视图
                acct = self._accounts.get(account_id)
                if acct is None:
                    acct = AccountView(account_id=account_id)
                    self._accounts[account_id] = acct
                acct.last_ts = ts
                acct.risk_score = score
                acct.risk_level = level

                alert = RiskAlertView(
                    ts=ts,
                    scope="account",
                    level=level,
                    account_id=account_id,
                    symbol=None,
                    message=message,
                    data={k: v for k, v in payload.items() if k not in {"message"}},
                )
                self._alerts.append(alert)

            elif tp == "risk.symbol_alert":
                account_id = str(payload.get("account_id") or "unknown")
                symbol = str(payload.get("symbol") or "unknown")

                key = f"{account_id}/{symbol}"
                sym_view = self._symbols.get(key)
                if sym_view is None:
                    sym_view = SymbolRiskView(account_id=account_id, symbol=symbol)
                    self._symbols[key] = sym_view
                sym_view.last_ts = ts
                sym_view.score = score
                sym_view.level = level
                sym_view.alert_count += 1

                alert = RiskAlertView(
                    ts=ts,
                    scope="symbol",
                    level=level,
                    account_id=account_id,
                    symbol=symbol,
                    message=message,
                    data={k: v for k, v in payload.items() if k not in {"message"}},
                )
                self._alerts.append(alert)

            else:
                # 其它风险事件类型，简单存成 info
                alert = RiskAlertView(
                    ts=ts,
                    scope="general",
                    level=level,
                    account_id=None,
                    symbol=None,
                    message=message or tp,
                    data=payload,
                )
                self._alerts.append(alert)

    # ------------------------------------------------------------------
    # 实验结果读取（周期拉取）
    # ------------------------------------------------------------------

    def refresh_experiments(self) -> None:
        """
        从 ExperimentLab 或 JSONL 文件中加载最近若干实验。
        """
        try:
            if self.exp_lab is not None and hasattr(
                self.exp_lab, "list_recent_experiments"
            ):
                recs = self.exp_lab.list_recent_experiments(
                    limit=self.max_experiments
                )  # type: ignore[attr-defined]
                exps = []
                for r in recs:
                    exps.append(
                        ExperimentView(
                            experiment_id=str(
                                r.get("experiment_id") or r.get("id") or ""
                            ),
                            model_id=str(r.get("model_id", "")),
                            task_type=str(r.get("task_type", "")),
                            status=str(r.get("status", "")),
                            ts=float(r.get("ts", 0.0)),
                            metrics=r.get("metrics", {}),
                        )
                    )
            else:
                exps = self._load_experiments_from_jsonl(self.max_experiments)
        except Exception as e:
            log.warning("DashboardState.refresh_experiments: 读取实验记录失败: %s", e)
            exps = []

        with self._lock:
            self._experiments = exps

    def _load_experiments_from_jsonl(self, limit: int) -> List[ExperimentView]:
        paths_cfg = self.system_config.get("paths", {}) or {}
        default_path = os.path.join(
            paths_cfg.get("experiments_dir", "experiments"), "experiments.jsonl"
        )
        # 向后兼容：如果 experiments_dir 未配置，则尝试默认路径
        path = default_path
        if not os.path.exists(path):
            # 保持旧逻辑：直接 experiments_dir/experiments.jsonl
            path = os.path.join(self.experiments_dir, "experiments.jsonl")

        if not os.path.exists(path):
            return []

        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            log.warning("DashboardState._load_experiments_from_jsonl 读取失败: %s", e)
            return []

        # 按 ts 排序，取最近的 limit 条
        rows.sort(key=lambda r: float(r.get("ts", 0.0)), reverse=True)
        rows = rows[:limit]

        exps: List[ExperimentView] = []
        for r in rows:
            exps.append(
                ExperimentView(
                    experiment_id=str(r.get("experiment_id") or r.get("id") or ""),
                    model_id=str(r.get("model_id", "")),
                    task_type=str(r.get("task_type", "")),
                    status=str(r.get("status", "")),
                    ts=float(r.get("ts", 0.0)),
                    metrics=r.get("metrics", {}),
                )
            )
        return exps

    # ------------------------------------------------------------------
    # 提供给 Flask 的 Dashboard 汇总
    # ------------------------------------------------------------------

    def get_dashboard(self) -> Dict[str, Any]:
        with self._lock:
            accounts = [asdict(a) for a in self._accounts.values()]
            # 按风险评分排序（高→低）
            accounts.sort(key=lambda x: x.get("risk_score", 0.0), reverse=True)

            # 风险最高的若干标的
            sym_views = list(self._symbols.values())
            sym_views.sort(key=lambda s: s.score, reverse=True)
            symbols = [asdict(s) for s in sym_views[:20]]

            # 最近告警（按时间倒序）
            alerts = sorted(list(self._alerts), key=lambda a: a.ts, reverse=True)
            alerts_dict = [asdict(a) for a in alerts]

            # 实验结果
            exps = [asdict(e) for e in self._experiments]

        return {
            "accounts": accounts,
            "symbols": symbols,
            "alerts": alerts_dict,
            "experiments": exps,
        }

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ts(ts_raw: Any) -> float:
        if ts_raw is None:
            return time.time()
        if isinstance(ts_raw, (int, float)):
            if float(ts_raw) > 1e12:
                return float(ts_raw) / 1000.0
            return float(ts_raw)
        try:
            import pandas as pd

            return pd.to_datetime(ts_raw).timestamp()
        except Exception:
            return time.time()


# ======================================================================
# Flask 应用 & EventBus 集成
# ======================================================================


def create_app() -> Flask:
    system_config = get_system_config()
    paths_cfg = system_config.get("paths", {}) or {}
    monitor_dir = paths_cfg.get("monitor_dir", os.path.join("logs", "monitor"))

    app = Flask(
        __name__, template_folder=os.path.join(os.path.dirname(__file__), "templates")
    )

    state = DashboardState(system_config=system_config)

    # 背景线程：周期刷新实验列表
    def experiments_loop() -> None:
        while True:
            try:
                state.refresh_experiments()
            except Exception as e:
                log.warning("experiments_loop 异常: %s", e)
            time.sleep(30)  # 每 30 秒刷新一次实验列表

    t = threading.Thread(
        target=experiments_loop, name="ExperimentRefresh", daemon=True
    )
    t.start()

    # 可选：ZeroMQ 总线订阅
    if HAS_ZMQ_BUS and ZmqBus is not None:
        try:
            bus = ZmqBus.from_system_config(system_config)  # type: ignore[attr-defined]
            # 订阅相关 Topic
            for topic in ["account.nav", "account.position", "risk.alert"]:
                try:
                    bus.subscribe(topic, state.handle_bus_message)  # type: ignore[attr-defined]
                    log.info("Web HUD: 已订阅 Topic=%s", topic)
                except Exception as e:
                    log.warning("Web HUD: 订阅 Topic=%s 失败: %s", topic, e)
        except Exception as e:
            log.warning("Web HUD: 初始化 ZmqBus 失败: %s", e)
    else:
        log.warning(
            "Web HUD: 未发现 ZmqBus，实时数据需要自行注入 DashboardState.handle_bus_message。"
        )

    # ------------------- Monitor summary 读取工具 ------------------- #

    def load_monitor_summary() -> Dict[str, Any]:
        """
        从 MonitorDaemon 写入的 state.json 中读取最近一次汇总。
        """
        path = os.path.join(monitor_dir, "state.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("HUD 读取 monitor summary 失败: %s", e)
            return {}

    # ------------------------------------------------------------------
    # HTTP 路由
    # ------------------------------------------------------------------

    @app.route("/")
    def index() -> str:
        dash = state.get_dashboard()
        monitor_summary = load_monitor_summary()
        return render_template(
            "dashboard.html",
            dashboard=dash,
            monitor_summary=monitor_summary,
        )

    @app.route("/api/dashboard")
    def api_dashboard():
        dash = state.get_dashboard()
        return jsonify(dash)

    @app.route("/api/monitor_summary")
    def api_monitor_summary():
        return jsonify(load_monitor_summary())

    return app


# CLI 入口
def main() -> None:  # pragma: no cover
    app = create_app()
    cfg = get_system_config()
    hud_cfg = (cfg.get("hud") or {}) if isinstance(cfg, dict) else {}
    host = hud_cfg.get("host", "0.0.0.0")
    port = int(hud_cfg.get("port", 5000))
    debug = bool(hud_cfg.get("debug", False))

    log.info("Web HUD 启动中: http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":  # pragma: no cover
    main()
