# -*- coding: utf-8 -*-
"""
模块名称：TSBacktest Mk-TimeLab
版本：Mk-TimeLab R90 (Intraday + Risk Core)
路径: G:/LightHunter_Mk1/ts_backtest.py

功能：
- 从 ts_data.db 的 snapshots 表中读取分时数据；
- 按“触发条件（如：涨幅≥X%、成交额≥Y）→ 持有Z分钟”模拟盘中超短交易；
- 可选接入 RiskBrain，对每个候选买点做“未来10分钟大回撤概率”过滤；
- 输出 ts_backtest_trades.csv + ts_backtest_report.json，用于科研 & 策略评估。

假设 ts_data.db.snapshots 结构：
  ts TEXT 'YYYY-MM-DD HH:MM:SS'
  code TEXT
  price REAL
  pct REAL
  amount REAL
  turnover_rate REAL
"""

import os
import sqlite3
import json
import datetime

import pandas as pd
import numpy as np
from colorama import Fore, Style

# --------------------------------------------------
# 尝试加载 RiskBrain（若不可用则自动降级）
# --------------------------------------------------
try:
    from risk_brain import RiskBrain  # 本地模块
    RISK_AVAILABLE = True
except ImportError:
    RiskBrain = None  # type: ignore
    RISK_AVAILABLE = False


class TSBacktester:
    def __init__(self, db_path: str = "ts_data.db"):
        self.db_path = db_path
        self.trades_file = "ts_backtest_trades.csv"
        self.report_file = "ts_backtest_report.json"

        # 交易成本（粗略模拟，和 TradeCore 接近）
        self.commission_rate = 0.00025
        self.stamp_duty = 0.001
        # 统一滑点：买入+卖出合计大约 0.4%
        self.total_slippage_pct = 0.4

        # 风险大脑（可选）
        self.risk_brain = None
        if RISK_AVAILABLE:
            try:
                self.risk_brain = RiskBrain()
                if not getattr(self.risk_brain, "is_trained", False):
                    # 模型文件还没训练好，仅做提醒
                    print(
                        Fore.YELLOW
                        + "[TSBT] RiskBrain 已加载但尚未训练，风险过滤将默认关闭。"
                        + Style.RESET_ALL
                    )
            except Exception as e:
                print(
                    Fore.YELLOW
                    + f"[TSBT] RiskBrain 初始化失败，将不启用风险过滤: {e}"
                    + Style.RESET_ALL
                )
                self.risk_brain = None

    # --------------------------------------------------
    # 工具函数
    # --------------------------------------------------
    def _connect(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"ts_data.db not found: {self.db_path}")
        return sqlite3.connect(self.db_path)

    def _get_available_dates(self):
        """
        返回 snapshots 中存在数据的交易日列表：['2025-01-02', '2025-01-03', ...]
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT substr(ts,1,10) AS d FROM snapshots ORDER BY d;"
        )
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _load_day_df(self, date_str: str) -> pd.DataFrame:
        conn = self._connect()
        query = """
        SELECT ts, code, price, pct, amount, turnover_rate
        FROM snapshots
        WHERE substr(ts,1,10) = ?
        ORDER BY code, ts;
        """
        df = pd.read_sql_query(query, conn, params=(date_str,))
        conn.close()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df

    @staticmethod
    def _in_trading_session(t: datetime.time) -> bool:
        """
        只在连续竞价时段做交易：
        - 09:35 - 11:29
        - 13:00 - 14:55
        """
        if datetime.time(9, 35) <= t <= datetime.time(11, 29):
            return True
        if datetime.time(13, 0) <= t <= datetime.time(14, 55):
            return True
        return False

    # --------------------------------------------------
    # 内部：单点风险评分封装
    # --------------------------------------------------
    def _score_risk_for_row(
        self,
        code: str,
        row: pd.Series,
        risk_threshold: float,
        risk_on: bool,
    ):
        """
        对当前候选买点调用 RiskBrain，返回 (risk_prob, should_block)

        - risk_on = False 时，直接返回 (np.nan, False)
        - RiskBrain 不可用时也自动返回 (np.nan, False)
        """
        if (not risk_on) or (self.risk_brain is None):
            return np.nan, False

        try:
            cur_pct = float(row["pct"])
            cur_amt = float(row["amount"])
            cur_turn = float(row["turnover_rate"])

            # RiskBrain 训练时基于 market_ts.db.snapshot_1m，
            # 这里用简化特征对齐：涨幅 / 换手率 / 成交额，主力攻击系数先置 0
            df_risk = pd.DataFrame(
                {
                    "代码": [str(code)],
                    "涨幅": [cur_pct],
                    "换手率": [cur_turn],
                    "主力攻击系数": [0.0],
                    "成交额": [cur_amt],
                }
            )
            prob_map = self.risk_brain.predict_risk(df_risk)
            risk_prob = float(prob_map.get(str(code), 0.0))

            should_block = risk_prob >= risk_threshold
            return risk_prob, should_block
        except Exception as e:
            # 风控出错时不阻断，只打个 Yellow 日志
            print(
                Fore.YELLOW
                + f"[TSBT][RISK] 评估 {code} 风险时异常：{e}"
                + Style.RESET_ALL
            )
            return np.nan, False

    # --------------------------------------------------
    # 回测核心：按“触发条件 + 持有时间”模拟交易
    # --------------------------------------------------
    def run_backtest(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        entry_pct: float = 5.0,
        min_amount: float = 2e7,
        horizon_min: int = 10,
        cooldown_min: int = 15,
        max_trades_per_day: int = 50,
        use_risk_brain: bool = False,
        risk_threshold: float = 0.6,
    ):
        """
        :param date_from: 起始日期 'YYYY-MM-DD'（None = 用数据库中最早日期）
        :param date_to: 结束日期 'YYYY-MM-DD'（None = 用数据库中最晚日期）
        :param entry_pct: 触发买入的当刻涨幅阈值 (%)
        :param min_amount: 触发买入的当刻成交额阈值 (元)
        :param horizon_min: 持有时间（分钟），到达后在该股票下一条快照价卖出
        :param cooldown_min: 同一只股票冷却时间（分钟），避免频繁进出同一票
        :param max_trades_per_day: 控制每天最多开多少笔（避免样本爆炸）
        :param use_risk_brain: 是否启用 RiskBrain 风险过滤
        :param risk_threshold: 视为“高风险”的概率阈值 (0~1)，高于该值则阻断开仓
        """
        dates = self._get_available_dates()
        if not dates:
            print(
                Fore.RED
                + "[TSBT] NO data in ts_data.db.snapshots."
                + Style.RESET_ALL
            )
            return pd.DataFrame(), {}

        if date_from is None:
            date_from = dates[0]
        if date_to is None:
            date_to = dates[-1]

        selected_dates = [d for d in dates if date_from <= d <= date_to]
        if not selected_dates:
            print(
                Fore.RED
                + f"[TSBT] No dates between {date_from} ~ {date_to}."
                + Style.RESET_ALL
            )
            return pd.DataFrame(), {}

        # 判断 RiskBrain 是否真正可用
        risk_on = (
            use_risk_brain
            and self.risk_brain is not None
            and getattr(self.risk_brain, "is_trained", False)
        )
        if use_risk_brain and not risk_on:
            print(
                Fore.YELLOW
                + "[TSBT] 你请求启用 RiskBrain，但模型尚未就绪，自动降级为 OFF。"
                + Style.RESET_ALL
            )

        risk_tag = (
            f" | RiskBrain=ON(th={risk_threshold:.2f})"
            if risk_on
            else " | RiskBrain=OFF"
        )

        print(
            Fore.CYAN
            + f"[TSBT] Backtest {selected_dates[0]} ~ {selected_dates[-1]} "
            f"| entry_pct≥{entry_pct:.1f}% | min_amount≥{min_amount/1e7:.1f}千万 "
            f"| hold {horizon_min}m{risk_tag}"
            + Style.RESET_ALL
        )

        trades = []

        for d in selected_dates:
            day_df = self._load_day_df(d)
            if day_df.empty:
                continue

            print(
                Fore.YELLOW
                + f"[TSBT] Replaying {d} . rows={len(day_df)}"
                + Style.RESET_ALL
            )

            # 每天统计开仓数量限制
            trades_today = 0

            # 按股票分组，逐票模拟更直观
            grouped = day_df.groupby("code", sort=False)

            for code, g in grouped:
                if trades_today >= max_trades_per_day:
                    break

                g = g.sort_values("ts").reset_index(drop=True)
                if len(g) < 5:
                    continue

                last_entry_time = None

                for i in range(len(g)):
                    row = g.iloc[i]
                    ts = row["ts"]
                    t = ts.time()

                    if not self._in_trading_session(t):
                        continue

                    # 冷却时间：避免同一票频繁开仓
                    if last_entry_time is not None:
                        dt_min = (ts - last_entry_time).total_seconds() / 60.0
                        if dt_min < cooldown_min:
                            continue

                    cur_pct = float(row["pct"])
                    cur_amt = float(row["amount"])

                    # 触发条件：涨幅 + 成交额门槛
                    if cur_pct < entry_pct:
                        continue
                    if cur_amt < min_amount:
                        continue

                    # -------- RiskBrain 风险过滤 --------
                    risk_prob, should_block = self._score_risk_for_row(
                        code=str(code),
                        row=row,
                        risk_threshold=risk_threshold,
                        risk_on=risk_on,
                    )
                    if should_block:
                        # 高风险：这笔不做
                        continue

                    # 找到 horizon_min 之后的第一条记录作为卖出点
                    exit_deadline = ts + datetime.timedelta(minutes=horizon_min)
                    future = g[g["ts"] >= exit_deadline]
                    if future.empty:
                        # 当天后面没数据了，无法形成完整交易
                        break
                    exit_row = future.iloc[0]

                    entry_price = float(row["price"])
                    exit_price = float(exit_row["price"])
                    if entry_price <= 0 or exit_price <= 0:
                        continue

                    raw_ret = (exit_price - entry_price) / entry_price * 100.0
                    # 扣掉滑点 + 手续费 + 税的近似影响
                    net_ret = raw_ret - self.total_slippage_pct

                    trades.append(
                        {
                            "date": d,
                            "code": str(code),
                            "entry_ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "exit_ts": exit_row["ts"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "entry_pct": cur_pct,
                            "entry_amount": cur_amt,
                            "entry_turnover": float(row["turnover_rate"]),
                            "raw_return_pct": raw_ret,
                            "net_return_pct": net_ret,
                            # 风险大脑输出（若未启用则为 NaN）
                            "risk_prob": float(risk_prob)
                            if not np.isnan(risk_prob)
                            else np.nan,
                        }
                    )

                    last_entry_time = ts
                    trades_today += 1

                    if trades_today >= max_trades_per_day:
                        break

        if not trades:
            print(
                Fore.YELLOW
                + "[TSBT] No trades generated with current parameters."
                + Style.RESET_ALL
            )
            return pd.DataFrame(), {}

        trades_df = pd.DataFrame(trades)
        self._save_trades(trades_df)
        report = self._build_report(
            trades_df,
            entry_pct,
            min_amount,
            horizon_min,
            use_risk_brain=risk_on,
            risk_threshold=risk_threshold if risk_on else None,
        )
        self._save_report(report)

        self._print_summary(report)
        return trades_df, report

    # --------------------------------------------------
    # 输出 & 报告
    # --------------------------------------------------
    def _save_trades(self, trades_df: pd.DataFrame):
        try:
            trades_df.to_csv(
                self.trades_file,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                Fore.GREEN
                + f"[TSBT] Trades saved -> {self.trades_file} (N={len(trades_df)})"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[TSBT][ERROR] failed to save trades: {e}"
                + Style.RESET_ALL
            )

    def _save_report(self, report: dict):
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(
                Fore.GREEN
                + f"[TSBT] Report saved -> {self.report_file}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[TSBT][ERROR] failed to save report: {e}"
                + Style.RESET_ALL
            )

    def _build_report(
        self,
        trades_df: pd.DataFrame,
        entry_pct: float,
        min_amount: float,
        horizon_min: int,
        use_risk_brain: bool = False,
        risk_threshold: float | None = None,
    ) -> dict:
        n = len(trades_df)
        wins = trades_df[trades_df["net_return_pct"] > 0]
        win_rate = len(wins) / n * 100.0 if n > 0 else 0.0

        avg_ret = trades_df["net_return_pct"].mean()
        med_ret = trades_df["net_return_pct"].median()
        best = trades_df["net_return_pct"].max()
        worst = trades_df["net_return_pct"].min()

        by_date = (
            trades_df.groupby("date")["net_return_pct"]
            .agg(["count", "mean", "sum"])
            .reset_index()
        )

        report: dict = {
            "generated_at": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "params": {
                "entry_pct": entry_pct,
                "min_amount": min_amount,
                "horizon_min": horizon_min,
                "use_risk_brain": bool(use_risk_brain),
                "risk_threshold": (
                    float(risk_threshold)
                    if use_risk_brain and risk_threshold is not None
                    else None
                ),
            },
            "global": {
                "trades": int(n),
                "win_rate_pct": float(win_rate),
                "avg_return_pct": float(avg_ret),
                "median_return_pct": float(med_ret),
                "best_return_pct": float(best),
                "worst_return_pct": float(worst),
            },
            "by_date": by_date.to_dict(orient="records"),
        }

        # 若有 risk_prob，则附加简单风险画像
        if "risk_prob" in trades_df.columns:
            sub = trades_df.dropna(subset=["risk_prob"])
            if not sub.empty:
                report["risk_stats"] = {
                    "avg_risk_all": float(sub["risk_prob"].mean()),
                    "avg_risk_win": float(
                        sub[sub["net_return_pct"] > 0]["risk_prob"].mean()
                    )
                    if (sub["net_return_pct"] > 0).any()
                    else None,
                    "avg_risk_loss": float(
                        sub[sub["net_return_pct"] <= 0]["risk_prob"].mean()
                    )
                    if (sub["net_return_pct"] <= 0).any()
                    else None,
                }

        return report

    def _print_summary(self, report: dict):
        g = report.get("global", {})
        p = report.get("params", {})
        r = report.get("risk_stats", {})

        print(
            Fore.YELLOW
            + Style.BRIGHT
            + "\n[TSBT] === Time-Series Backtest Summary ==="
            + Style.RESET_ALL
        )
        print(f"  Trades    : {g.get('trades', 0)}")
        print(f"  WinRate   : {g.get('win_rate_pct', 0.0):.2f}%")
        print(f"  AvgRet    : {g.get('avg_return_pct', 0.0):+.2f}%")
        print(f"  Median    : {g.get('median_return_pct', 0.0):+.2f}%")
        print(
            f"  Best/Worst: {g.get('best_return_pct', 0.0):+.2f}% / "
            f"{g.get('worst_return_pct', 0.0):+.2f}%"
        )
        print(
            f"  RiskBrain : {'ON' if p.get('use_risk_brain') else 'OFF'} "
            f"(th={p.get('risk_threshold')})"
        )

        if r:
            print("  RiskStats : ", end="")
            print(
                f"avg_all={r.get('avg_risk_all'):.3f}"
                if r.get("avg_risk_all") is not None
                else "avg_all=None",
                end="; ",
            )
            if r.get("avg_risk_win") is not None:
                print(f"avg_win={r.get('avg_risk_win'):.3f}", end="; ")
            if r.get("avg_risk_loss") is not None:
                print(f"avg_loss={r.get('avg_risk_loss'):.3f}", end="; ")
            print()

    # --------------------------------------------------
    # 交互入口：快速回测最近一个交易日
    # --------------------------------------------------
    def run_interactive(self):
        from colorama import init as color_init

        color_init(autoreset=True)
        print(
            Fore.CYAN
            + Style.BRIGHT
            + """
        ################################################
        #   LIGHT HUNTER - Time-Series Backtest Lab    #
        #            Mk-TimeLab R90                    #
        ################################################
        """
            + Style.RESET_ALL
        )

        dates = self._get_available_dates()
        if not dates:
            print(
                Fore.RED
                + "[TSBT] No data in ts_data.db, please run ts_collector.py first."
                + Style.RESET_ALL
            )
            return

        last_date = dates[-1]
        print(
            Fore.YELLOW
            + f"[TSBT] Available dates: {', '.join(dates[-5:])}"
            + Style.RESET_ALL
        )
        print(
            Fore.YELLOW
            + f"[TSBT] Default = last date {last_date}"
            + Style.RESET_ALL
        )

        date_from = (
            input(
                Fore.YELLOW
                + f"  Start date [YYYY-MM-DD, Enter={last_date}]: "
                + Style.RESET_ALL
            ).strip()
            or last_date
        )
        date_to = (
            input(
                Fore.YELLOW
                + f"  End date   [YYYY-MM-DD, Enter={date_from}]: "
                + Style.RESET_ALL
            ).strip()
            or date_from
        )

        entry_str = input(
            Fore.YELLOW
            + "  Entry pct threshold (default 5.0): "
            + Style.RESET_ALL
        ).strip()
        entry_pct = float(entry_str) if entry_str else 5.0

        amt_str = input(
            Fore.YELLOW
            + "  Min amount (万, default 2000 万): "
            + Style.RESET_ALL
        ).strip()
        min_amount = (float(amt_str) * 1e4) if amt_str else 2e7

        horizon_str = input(
            Fore.YELLOW
            + "  Holding horizon (minutes, default 10): "
            + Style.RESET_ALL
        ).strip()
        horizon_min = int(horizon_str) if horizon_str else 10

        # RiskBrain 开关（若模型就绪）
        use_risk = False
        risk_th = 0.6
        if (
            self.risk_brain is not None
            and getattr(self.risk_brain, "is_trained", False)
        ):
            ans = input(
                Fore.YELLOW
                + "  Use RiskBrain filter? [y/N]: "
                + Style.RESET_ALL
            ).strip().lower()
            if ans == "y":
                use_risk = True
                r_str = input(
                    Fore.YELLOW
                    + "  Risk threshold (0~1, default 0.6): "
                    + Style.RESET_ALL
                ).strip()
                if r_str:
                    try:
                        risk_th = float(r_str)
                    except Exception:
                        risk_th = 0.6
        else:
            print(
                Fore.YELLOW
                + "[TSBT] RiskBrain not ready (no model), skip risk filter."
                + Style.RESET_ALL
            )

        self.run_backtest(
            date_from=date_from,
            date_to=date_to,
            entry_pct=entry_pct,
            min_amount=min_amount,
            horizon_min=horizon_min,
            use_risk_brain=use_risk,
            risk_threshold=risk_th,
        )


if __name__ == "__main__":
    bt = TSBacktester()
    bt.run_interactive()
