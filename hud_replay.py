# -*- coding: utf-8 -*-
"""
模块名称：HUDReplay Mk-Chronicle
版本：Mk-Replay R10
路径: G:/LightHunter_Mk1/hud_replay.py

功能：
- 从 market_blackbox.csv 读取某个交易日的 HUD 快照；
- 可选加载 ts_data.db，检查该日分时覆盖度，并按时间顺序回放；
- 使用 Commander 的 CombatBrain + 本地 trade_history.csv 重建账户画像，
  在终端重放当天作战画面（只读回放，不会下单）。
"""

import os
import sqlite3
import time
import datetime
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
from colorama import init, Fore, Style

from commander import Commander

init(autoreset=True)


class ReplayTrader:
    """
    简化版 Trader：
    - 从 trade_history.csv 读取指定交易日的成交记录；
    - 按时间推进，重建账户现金 + 持仓；
    - 提供 get_risk_metrics() 与 Commander 现有 HUD 接口兼容。
    """

    def __init__(
        self,
        trade_history_file: str = "trade_history.csv",
        trade_date: Optional[str] = None,
        initial_capital: float = 200000.0,
    ):
        self.initial_capital = float(initial_capital)
        self.account = {
            "initial_capital": self.initial_capital,
            "total_assets": self.initial_capital,
            "cash": self.initial_capital,
            "position_value": 0.0,
            "positions": {},  # code -> {"name": str, "volume": int, "cost": float}
        }

        self._equity_return_pct = 0.0
        self.max_dd_pct = 0.0
        self._day_pnl_pct = 0.0
        self.day_max_dd_pct = 0.0
        self._circuit_tripped = False  # 回放模式永远不触发熔断

        self.trades: List[Dict] = []
        self._cursor = 0
        self._last_time: Optional[datetime.datetime] = None

        self._load_trades(trade_history_file, trade_date)

    def _load_trades(self, trade_history_file: str, trade_date: Optional[str]):
        if not os.path.exists(trade_history_file):
            print(
                Fore.YELLOW
                + f"[REPLAY] trade_history 文件不存在：{trade_history_file}，将以空账户回放。"
                + Style.RESET_ALL
            )
            return

        try:
            df = pd.read_csv(trade_history_file)
        except Exception as e:
            print(
                Fore.RED
                + f"[REPLAY] 读取 trade_history 失败：{e}，将以空账户回放。"
                + Style.RESET_ALL
            )
            return

        if df.empty or "Time" not in df.columns:
            print(
                Fore.YELLOW
                + "[REPLAY] trade_history 为空或缺少 Time 列，将以空账户回放。"
                + Style.RESET_ALL
            )
            return

        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        if trade_date:
            try:
                target_date = datetime.datetime.strptime(trade_date, "%Y-%m-%d").date()
                df = df[df["Time"].dt.date == target_date]
            except Exception:
                pass

        if df.empty:
            print(
                Fore.YELLOW
                + f"[REPLAY] 交易日 {trade_date} 没有成交记录，将以空账户回放。"
                + Style.RESET_ALL
            )
            return

        df = df.sort_values("Time")
        self.trades = df.to_dict("records")
        print(
            Fore.CYAN
            + f"[REPLAY] 已载入 {len(self.trades)} 条成交记录用于账户回放。"
            + Style.RESET_ALL
        )

    def _apply_trade(self, tr: Dict):
        action = str(tr.get("Action", "")).upper()
        code = str(tr.get("Code", "")).strip()
        if not code or action not in ("BUY", "SELL"):
            return

        name = str(tr.get("Name", code))
        try:
            price = float(tr.get("Price", 0.0))
            vol = int(tr.get("Vol", 0))
        except Exception:
            return
        if vol <= 0 or price <= 0:
            return

        amount = float(tr.get("Amount", price * vol))
        fee = float(tr.get("Fee", 0.0))

        positions: Dict[str, Dict] = self.account["positions"]

        if action == "BUY":
            cash_cost = amount + fee
            self.account["cash"] -= cash_cost

            pos = positions.get(code)
            if pos:
                old_vol = int(pos.get("volume", 0))
                old_cost = float(pos.get("cost", price))
                new_vol = old_vol + vol
                if new_vol > 0:
                    new_cost = (old_cost * old_vol + price * vol) / new_vol
                else:
                    new_cost = price
                pos["volume"] = new_vol
                pos["cost"] = new_cost
                pos["name"] = name
            else:
                positions[code] = {"name": name, "volume": vol, "cost": price}

        elif action == "SELL":
            pos = positions.get(code)
            if not pos:
                # 没有找到对应持仓，可能是历史残留，忽略
                return
            old_vol = int(pos.get("volume", 0))
            if old_vol <= 0:
                return

            sell_vol = min(old_vol, vol)
            cash_in = amount - fee
            self.account["cash"] += cash_in

            pos["volume"] = old_vol - sell_vol
            if pos["volume"] <= 0:
                positions.pop(code, None)

    def _mark_to_market(self, price_map: Dict[str, float]):
        positions: Dict[str, Dict] = self.account["positions"]
        pos_val = 0.0
        for code, pos in positions.items():
            vol = int(pos.get("volume", 0))
            if vol <= 0:
                continue
            cost = float(pos.get("cost", 0.0))
            price = float(price_map.get(code, cost if cost > 0 else 0.0))
            pos_val += price * vol

        self.account["position_value"] = pos_val
        equity = float(self.account["cash"]) + pos_val
        self.account["total_assets"] = equity

        if self.initial_capital > 0:
            self._equity_return_pct = (equity - self.initial_capital) / self.initial_capital * 100.0
            self._day_pnl_pct = self._equity_return_pct
        else:
            self._equity_return_pct = 0.0
            self._day_pnl_pct = 0.0

        # 简单最大回撤统计
        if not hasattr(self, "_equity_peak"):
            self._equity_peak = self.initial_capital
        if equity > self._equity_peak:
            self._equity_peak = equity
        if self._equity_peak > 0:
            dd = (self._equity_peak - equity) / self._equity_peak * 100.0
            if dd > self.max_dd_pct:
                self.max_dd_pct = dd
            if dd > self.day_max_dd_pct:
                self.day_max_dd_pct = dd

    def update_to_time(self, current_time: datetime.datetime, price_map: Dict[str, float]):
        """
        推进到 current_time：执行当前时间之前的所有成交，然后按最新价格做市值。
        """
        while self._cursor < len(self.trades):
            tr = self.trades[self._cursor]
            tr_time = tr.get("Time")
            if isinstance(tr_time, str):
                try:
                    tr_time = datetime.datetime.fromisoformat(tr_time)
                except Exception:
                    tr_time = None
            if tr_time is None or tr_time > current_time:
                break
            self._apply_trade(tr)
            self._cursor += 1

        self._last_time = current_time
        self._mark_to_market(price_map)

    def get_risk_metrics(self) -> Dict[str, float]:
        return {
            "equity_return_pct": float(self._equity_return_pct),
            "max_dd_pct": float(self.max_dd_pct),
            "day_pnl_pct": float(self._day_pnl_pct),
            "day_max_dd_pct": float(self.day_max_dd_pct),
            "circuit_tripped": bool(self._circuit_tripped),
        }


def load_blackbox_frames(
    blackbox_file: str,
    trade_date: str,
) -> List[Tuple[datetime.datetime, pd.DataFrame]]:
    """
    按 Time 分组，返回某个交易日的所有 HUD 帧列表。
    """
    if not os.path.exists(blackbox_file):
        print(
            Fore.RED
            + f"[REPLAY] 找不到 blackbox 文件：{blackbox_file}"
            + Style.RESET_ALL
        )
        return []

    try:
        df = pd.read_csv(blackbox_file)
    except Exception as e:
        print(
            Fore.RED
            + f"[REPLAY] 读取 blackbox 失败：{e}"
            + Style.RESET_ALL
        )
        return []

    if df.empty or "Time" not in df.columns:
        print(
            Fore.YELLOW
            + "[REPLAY] blackbox 为空或缺少 Time 列。"
            + Style.RESET_ALL
        )
        return []

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    try:
        target_date = datetime.datetime.strptime(trade_date, "%Y-%m-%d").date()
        df = df[df["Time"].dt.date == target_date]
    except Exception:
        pass

    if df.empty:
        print(
            Fore.YELLOW
            + f"[REPLAY] blackbox 中没有 {trade_date} 的数据。"
            + Style.RESET_ALL
        )
        return []

    # 按时间 + Final_Score 排序，保证和当时 HUD 接近
    if "Final_Score" in df.columns:
        df = df.sort_values(["Time", "Final_Score"], ascending=[True, False])
    else:
        df = df.sort_values("Time")

    frames: List[Tuple[datetime.datetime, pd.DataFrame]] = []
    for ts, g in df.groupby("Time"):
        frames.append((ts, g.copy()))

    print(
        Fore.CYAN
        + f"[REPLAY] 已装载 {len(frames)} 帧 HUD 快照用于回放。"
        + Style.RESET_ALL
    )
    return frames


def build_ts_index(
    ts_db_path: str,
    trade_date: str,
) -> Dict[datetime.datetime, pd.DataFrame]:
    """
    读取 ts_data.db.snapshots 中某个交易日的全部分时数据，
    并按 ts 分组，构建 {ts -> DataFrame} 索引。
    """
    ts_map: Dict[datetime.datetime, pd.DataFrame] = {}
    if not os.path.exists(ts_db_path):
        print(
            Fore.YELLOW
            + f"[REPLAY] ts_data.db 不存在：{ts_db_path}，本次回放将不使用分时数据。"
            + Style.RESET_ALL
        )
        return ts_map

    try:
        conn = sqlite3.connect(ts_db_path)
        df = pd.read_sql_query(
            """
            SELECT ts, code, price, pct, amount, turnover_rate
            FROM snapshots
            WHERE substr(ts,1,10) = ?
            ORDER BY ts, code
            """,
            conn,
            params=(trade_date,),
        )
        conn.close()
    except Exception as e:
        print(
            Fore.RED
            + f"[REPLAY] 读取 ts_data.db 失败：{e}，本次回放将不使用分时数据。"
            + Style.RESET_ALL
        )
        return ts_map

    if df.empty:
        print(
            Fore.YELLOW
            + f"[REPLAY] ts_data.db 中 {trade_date} 无分时记录。"
            + Style.RESET_ALL
        )
        return ts_map

    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts_dt"])
    for ts, g in df.groupby("ts_dt"):
        ts_map[ts] = g.copy()

    print(
        Fore.CYAN
        + f"[REPLAY] 已装载 {len(ts_map)} 个时间点的分时快照。"
        + Style.RESET_ALL
    )
    return ts_map


def run_replay(
    trade_date: str,
    blackbox_file: str = "market_blackbox.csv",
    ts_db_path: str = "ts_data.db",
    trade_history_file: str = "trade_history.csv",
    speed: float = 2.0,
    step_mode: bool = False,
):
    """
    核心回放入口：
    - 初始化 Commander，但不启动实盘；
    - 用 ReplayTrader 替换其中的 trader；
    - 按时间顺序回放 HUD。
    """
    frames = load_blackbox_frames(blackbox_file, trade_date)
    if not frames:
        return

    ts_map = build_ts_index(ts_db_path, trade_date)

    # 实例化 Commander，并切换到 REPLAY 模式（不启动 start_monitor）
    cmd = Commander()
    cmd.current_mode = "REPLAY"
    cmd.is_running = False
    cmd.latest_news = []  # 回放时不拉取实时新闻
    cmd.hot_concepts = []
    cmd.north_flow = 0.0
    cmd.prev_north_flow = 0.0
    cmd.north_delta = 0.0

    # 用 ReplayTrader 替换原始 PaperTrader
    init_cap = getattr(getattr(cmd, "trader", None), "account", {}).get(
        "initial_capital", 200000.0
    )
    cmd.trader = ReplayTrader(
        trade_history_file=trade_history_file,
        trade_date=trade_date,
        initial_capital=init_cap,
    )

    total_frames = len(frames)
    master_price_map: Dict[str, float] = {}

    print(
        Fore.MAGENTA
        + Style.BRIGHT
        + f"[REPLAY] 开始回放 {trade_date} （共 {total_frames} 帧）..."
        + Style.RESET_ALL
    )

    try:
        for idx, (ts, frame_df) in enumerate(frames, start=1):
            # 1) 准备价格映射（用于账户市值计算）
            frame_df = frame_df.copy()
            frame_df["代码"] = frame_df["代码"].astype(str)
            frame_df["现价"] = pd.to_numeric(frame_df["现价"], errors="coerce").fillna(0.0)
            price_map = {row["代码"]: float(row["现价"]) for _, row in frame_df.iterrows()}
            master_price_map.update(price_map)

            # 2) 分时数据：用于 Mood 估计 & TS 覆盖统计
            ts_rows = 0
            mood_df = None
            if ts_map:
                # ts 精度到秒，与 blackbox Time 对齐
                ts_key = ts.replace(microsecond=0)
                if ts_key in ts_map:
                    ts_frame = ts_map[ts_key]
                    ts_rows = len(ts_frame)
                    # 转成 CombatBrain 需要的结构：代码/涨幅/成交额
                    mood_df = pd.DataFrame(
                        {
                            "代码": ts_frame["code"].astype(str),
                            "涨幅": pd.to_numeric(ts_frame["pct"], errors="coerce").fillna(0.0),
                            "成交额": pd.to_numeric(ts_frame["amount"], errors="coerce").fillna(0.0),
                        }
                    )

            if mood_df is None:
                # 若该时间点没有 TS 数据，就用 blackbox 的 Top 列表估计情绪
                mood_df = frame_df[["代码", "涨幅"]].copy()
                mood_df["成交额"] = pd.to_numeric(
                    frame_df.get("成交额", 0.0), errors="coerce"
                ).fillna(0.0)

            # 3) 市场情绪估计（复用 CombatBrain 的 assess_market_mood）
            mood_score, mood_desc, _ = cmd.brain.assess_market_mood(
                mood_df, 0.0, 0.0
            )
            cmd.mood_history.append(mood_score)
            cmd.risk_on = "PANIC" not in mood_desc and "Freezing" not in mood_desc

            # 4) 推进账户到当前时间点
            cmd.trader.update_to_time(ts, master_price_map)

            # 5) 调用原有 HUD 渲染当前帧
            cmd._print_hud(
                frame_df,
                mood_desc,
                kinetic_map={},
                sector_map={},
                ladder_dict={},
                flow_map={},
                vector_map={},
                mood_score=mood_score,
            )

            # 在 HUD 下面补一行回放信息（不会被 _print_hud 清屏覆盖）
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            print(
                Fore.MAGENTA
                + f"[REPLAY] {ts_str} | Frame {idx}/{total_frames} | TS rows: {ts_rows}"
                + Style.RESET_ALL
            )

            # 6) 回放节奏控制
            if step_mode:
                user_in = input(
                    Fore.YELLOW
                    + "按回车播放下一帧，输入 q 后回车可提前结束回放："
                    + Style.RESET_ALL
                ).strip()
                if user_in.lower() == "q":
                    break
            else:
                # 默认每帧约 1 秒，可用 speed 调快/调慢
                base_sleep = 1.0
                sleep_time = base_sleep / max(speed, 0.1)
                time.sleep(sleep_time)

        print(
            Fore.GREEN
            + Style.BRIGHT
            + "\n[REPLAY] 回放结束。"
            + Style.RESET_ALL
        )
    except KeyboardInterrupt:
        print(
            Fore.RED
            + "\n[REPLAY] 用户中断回放。"
            + Style.RESET_ALL
        )


def main():
    parser = argparse.ArgumentParser(
        description="LightHunter HUD Replay (Mk-Replay R10)"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="回放的交易日，例如 2024-01-15（YYYY-MM-DD）。",
    )
    parser.add_argument(
        "--blackbox",
        type=str,
        default="market_blackbox.csv",
        help="HUD 记录文件路径（默认：market_blackbox.csv）。",
    )
    parser.add_argument(
        "--ts-db",
        type=str,
        default="ts_data.db",
        help="分时库 ts_data.db 路径（默认：ts_data.db）。",
    )
    parser.add_argument(
        "--trade-history",
        type=str,
        default="trade_history.csv",
        help="成交记录 trade_history.csv 路径（默认：trade_history.csv）。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="回放速度倍数（>1 加速，<1 减速，默认 2.0）。",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="逐帧模式：每帧按回车继续，便于细致研判。默认关闭。",
    )

    args = parser.parse_args()
    run_replay(
        trade_date=args.date,
        blackbox_file=args.blackbox,
        ts_db_path=args.ts_db,
        trade_history_file=args.trade_history,
        speed=args.speed,
        step_mode=args.step,
    )


if __name__ == "__main__":
    main()
