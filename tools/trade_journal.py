# -*- coding: utf-8 -*-
"""
tools/trade_journal.py

实盘记录 + 复盘统计 小工具（独立于 U2 回测，可配合 trade_history.csv 使用）

主要功能
--------
1）实盘记录（交互式）
    - 运行命令：python -m tools.trade_journal --mode add
    - 终端里一笔一笔输入：日期、代码、名称、方向、价格、数量、手续费、备注
    - 自动追加到 trade_history.csv

2）复盘统计
    - 运行命令：python -m tools.trade_journal --mode stats [可选过滤条件]
    - 从 trade_history.csv 读取记录，按 Time 排序
    - 假设只有做多（BUY 开仓，SELL 平仓）
    - 计算每笔已平仓交易的盈亏、胜率、最大回撤等
    - 可以按日期区间 / 策略过滤

说明
----
- 默认使用 ConfigManager 里的 paths.trade_history_file 路径；
  如果取不到，就退回当前目录下的 trade_history.csv。
- trade_history.csv 的最少字段：
    Time, Code, Name, Action, Price, Vol, Amount, Fee
  其中：
    - Time: "YYYY-MM-DD HH:MM" 或 "YYYY-MM-DD"
    - Action: BUY / SELL
    - Amount 如果缺失，会用 Price * Vol 自动补
    - Fee 缺失则按 0 处理
- 本脚本在不破坏现有字段的前提下，可以额外写入：
    Strategy, Note
  这些字段 Commander / hud_replay 不会用到，但对你复盘有帮助。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


# ------------------------------------------------------------
# 配置 & 数据模型
# ------------------------------------------------------------

TRADE_COLUMNS = [
    "Time",      # 交易时间
    "Code",      # 代码
    "Name",      # 名称（可空）
    "Action",    # BUY / SELL
    "Price",     # 成交价
    "Vol",       # 成交量（股数）
    "Amount",    # 成交金额（Price * Vol）
    "Fee",       # 手续费（元）
    "Strategy",  # 策略标签，例如 U2
    "Note",      # 备注
]


def resolve_trade_history_path() -> str:
    """
    优先从 ConfigManager 里读取 paths.trade_history_file，
    读不到就退回 trade_history.csv。
    """
    try:
        from config.config_manager import ConfigManager  # type: ignore

        cfg = ConfigManager.get_instance().get_config()
        paths = cfg.get("paths", {}) or {}
        path = paths.get("trade_history_file", "trade_history.csv")
        return os.path.abspath(path)
    except Exception:
        return os.path.abspath("trade_history.csv")


def load_trade_history(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[JOURNAL] 读取 {path} 失败，将从空表开始。错误：{e}")
            df = pd.DataFrame(columns=TRADE_COLUMNS)
    else:
        df = pd.DataFrame(columns=TRADE_COLUMNS)

    # 保证所有需要的列都存在
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_trade_history(df: pd.DataFrame, path: str) -> None:
    df_sorted = df.sort_values("Time")
    df_sorted.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[JOURNAL] 成交记录已保存：{path}")


# ------------------------------------------------------------
# 交互式录入
# ------------------------------------------------------------

def _input_with_default(prompt: str, default: Optional[str] = None) -> str:
    if default:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    s = input(full).strip()
    if not s and default is not None:
        return default
    return s


def interactive_add(trade_file: str, default_strategy: str = "U2") -> None:
    """
    交互式录入一批成交。
    """
    print("\n==== 实盘记录模式（交互式）====")
    print(f"[JOURNAL] 当前使用的成交记录文件：{trade_file}")
    print("提示：回车留空表示使用默认值；代码留空直接结束录入。\n")

    df = load_trade_history(trade_file)

    today = dt.date.today().strftime("%Y-%m-%d")
    default_time = today

    while True:
        code = _input_with_default("代码 (回车结束本次录入)", "")
        if not code:
            break

        name = _input_with_default("名称（可选）", "")
        action_raw = _input_with_default("方向 (B=买入, S=卖出)", "B").upper()
        if action_raw in {"B", "BUY", "1"}:
            action = "BUY"
        elif action_raw in {"S", "SELL", "-1"}:
            action = "SELL"
        else:
            print("  无法识别方向，默认视为 BUY。")
            action = "BUY"

        time_str = _input_with_default("成交时间 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM)", default_time)
        try:
            # 只输入日期的话，补成 15:00
            if len(time_str) == 10:
                time_dt = dt.datetime.strptime(time_str, "%Y-%m-%d")
                time_dt = time_dt.replace(hour=15, minute=0)
            else:
                time_dt = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except Exception:
            print("  时间格式错误，使用今天 15:00。")
            time_dt = dt.datetime.combine(dt.date.today(), dt.time(15, 0))

        price_str = _input_with_default("成交价", "")
        vol_str = _input_with_default("成交量（股）", "")
        fee_str = _input_with_default("手续费（元，可空，默认 0）", "0")
        strategy = _input_with_default("策略标签", default_strategy)
        note = _input_with_default("备注（可空）", "")

        try:
            price = float(price_str)
            vol = int(float(vol_str))
            fee = float(fee_str) if fee_str else 0.0
        except Exception:
            print("  价格 / 数量 / 手续费解析失败，忽略本笔记录。")
            continue

        amount = price * vol

        row = {
            "Time": time_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Code": code,
            "Name": name,
            "Action": action,
            "Price": price,
            "Vol": vol,
            "Amount": amount,
            "Fee": fee,
            "Strategy": strategy,
            "Note": note,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        print(f"  已记录：{time_dt.date()} {code} {name} {action} {vol}@{price}  手续费={fee:.2f}")

    save_trade_history(df, trade_file)
    print("\n[JOURNAL] 本次录入完成。\n")


# ------------------------------------------------------------
# 复盘统计
# ------------------------------------------------------------

@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_ratio: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    max_drawdown_pct: float
    max_consecutive_win: int
    max_consecutive_loss: int


def _simulate_pnl(df: pd.DataFrame, initial_equity: float = 200_000.0) -> TradeStats:
    """
    用最简单的方式，根据 BUY / SELL 序列推演已平仓盈亏 + 权益曲线。

    假设：
    - 只有多头交易（BUY 建仓 / 加仓，SELL 减仓 / 平仓）
    - 手续费 Fee 全部算在成本里
    """
    if df.empty:
        return TradeStats(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_ratio=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown_pct=0.0,
            max_consecutive_win=0,
            max_consecutive_loss=0,
        )

    df = df.sort_values("Time").copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    # 规范 Action
    df["Action"] = df["Action"].astype(str).str.upper()
    df["Side"] = df["Action"].str[0]  # B or S

    # 补金额 / 手续费
    df["Price"] = df["Price"].astype(float)
    df["Vol"] = df["Vol"].astype(float)
    if "Amount" not in df.columns:
        df["Amount"] = df["Price"] * df["Vol"]
    else:
        df["Amount"] = df["Amount"].fillna(df["Price"] * df["Vol"])
    if "Fee" not in df.columns:
        df["Fee"] = 0.0
    else:
        df["Fee"] = df["Fee"].fillna(0.0).astype(float)

    # 仓位：code -> (vol, avg_cost)
    positions: Dict[str, Dict[str, float]] = {}
    realized_pnls: List[float] = []
    cum_pnl_list: List[float] = []
    cum_pnl = 0.0

    for _, row in df.iterrows():
        code = str(row["Code"])
        side = row["Side"]
        price = float(row["Price"])
        vol = float(row["Vol"])
        fee = float(row["Fee"])

        if vol <= 0:
            realized_pnls.append(0.0)
            cum_pnl_list.append(cum_pnl)
            continue

        pos = positions.get(code, {"vol": 0.0, "cost": 0.0})

        if side == "B":  # 买入：加仓，成本 = 旧成本 + 新成交 + 手续费
            total_cost_old = pos["cost"] * pos["vol"]
            total_cost_new = total_cost_old + price * vol + fee
            pos["vol"] += vol
            if pos["vol"] > 0:
                pos["cost"] = total_cost_new / pos["vol"]
            else:
                pos["cost"] = 0.0
            positions[code] = pos
            pnl = 0.0

        elif side == "S":  # 卖出：减少仓位，实现盈亏
            if pos["vol"] <= 0:
                # 没有持仓就卖，视作无成本，这种一般是录入错误，提示一下
                cost_basis = 0.0
                print(f"[JOURNAL][WARN] 卖出时 {code} 无持仓，视作零成本处理。")
            else:
                sell_vol = min(vol, pos["vol"])
                cost_basis = pos["cost"] * sell_vol
                pos["vol"] -= sell_vol
                if pos["vol"] <= 0:
                    pos["vol"] = 0.0
                    pos["cost"] = 0.0
                positions[code] = pos

            revenue = price * vol - fee
            pnl = revenue - cost_basis

        else:
            pnl = 0.0

        cum_pnl += pnl
        realized_pnls.append(pnl)
        cum_pnl_list.append(cum_pnl)

    df["realized_pnl"] = realized_pnls
    df["cum_pnl"] = cum_pnl_list

    closed = df[df["realized_pnl"] != 0].copy()
    total_trades = len(closed)
    if total_trades == 0:
        return TradeStats(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_ratio=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown_pct=0.0,
            max_consecutive_win=0,
            max_consecutive_loss=0,
        )

    wins = closed[closed["realized_pnl"] > 0]
    losses = closed[closed["realized_pnl"] < 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    win_ratio = winning_trades / total_trades * 100.0

    gross_profit = float(wins["realized_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["realized_pnl"].sum()) if not losses.empty else 0.0
    net_profit = gross_profit + gross_loss

    avg_win = float(wins["realized_pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["realized_pnl"].mean()) if not losses.empty else 0.0

    # 权益曲线 + 最大回撤（基于已实现盈亏）
    equity = initial_equity + df["cum_pnl"]
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd_pct = float(dd.min() * 100.0)

    # 连赢 / 连亏
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for v in closed["realized_pnl"]:
        if v > 0:
            streak = streak + 1 if streak >= 0 else 1
        elif v < 0:
            streak = streak - 1 if streak <= 0 else -1
        else:
            streak = 0

        max_win_streak = max(max_win_streak, streak) if streak > 0 else max_win_streak
        max_loss_streak = min(max_loss_streak, streak) if streak < 0 else max_loss_streak

    return TradeStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_ratio=win_ratio,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown_pct=max_dd_pct,
        max_consecutive_win=max_win_streak,
        max_consecutive_loss=abs(max_loss_streak),
    )


def run_stats(
    trade_file: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy: Optional[str] = None,
    initial_equity: float = 200_000.0,
) -> None:
    print("\n==== 复盘统计模式 ====")
    print(f"[JOURNAL] 成交记录文件：{trade_file}")

    df = load_trade_history(trade_file)
    if df.empty:
        print("[JOURNAL] trade_history 为空，目前没有任何记录。")
        return

    # 时间 / 策略过滤
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df["Date"] = df["Time"].dt.date

    if start_date:
        try:
            d0 = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            df = df[df["Date"] >= d0]
        except Exception:
            print(f"[JOURNAL] start_date 格式错误：{start_date}，忽略。")

    if end_date:
        try:
            d1 = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            df = df[df["Date"] <= d1]
        except Exception:
            print(f"[JOURNAL] end_date 格式错误：{end_date}，忽略。")

    if strategy:
        df = df[(df["Strategy"].fillna("").astype(str) == strategy)]

    if df.empty:
        print("[JOURNAL] 经过过滤后没有任何记录。")
        return

    stats = _simulate_pnl(df, initial_equity=initial_equity)

    print("\n==== 整体统计（基于已平仓盈亏）====")
    print(f"总交易笔数   : {stats.total_trades}")
    print(f"盈利笔数     : {stats.winning_trades}")
    print(f"亏损笔数     : {stats.losing_trades}")
    print(f"胜率         : {stats.win_ratio:6.2f}%")
    print(f"总盈利       : {stats.gross_profit:10.2f}")
    print(f"总亏损       : {stats.gross_loss:10.2f}")
    print(f"净盈利       : {stats.net_profit:10.2f}")
    print(f"单笔平均盈利 : {stats.avg_win:10.2f}")
    print(f"单笔平均亏损 : {stats.avg_loss:10.2f}")
    print(f"最大回撤     : {stats.max_drawdown_pct:6.2f}%")
    print(f"最大连赢笔数 : {stats.max_consecutive_win}")
    print(f"最大连亏笔数 : {stats.max_consecutive_loss}")

    # 按年 / 按月简单拆一下（基于已实现盈亏）
    closed = df.copy()
    # 先用同样逻辑算出 realized_pnl
    _ = _simulate_pnl(closed, initial_equity=initial_equity)  # 给 closed 增加列用
    if "realized_pnl" in closed.columns:
        print("\n==== 按年份统计（已平仓盈亏）====")
        closed["year"] = closed["Time"].dt.year
        yearly = closed.groupby("year")["realized_pnl"].sum().reset_index()
        for _, r in yearly.iterrows():
            print(f"{int(r['year'])}: {r['realized_pnl']:10.2f}")

        print("\n==== 按月份统计（已平仓盈亏）====")
        closed["ym"] = closed["Time"].dt.strftime("%Y-%m")
        monthly = closed.groupby("ym")["realized_pnl"].sum().reset_index()
        for _, r in monthly.iterrows():
            print(f"{r['ym']}: {r['realized_pnl']:10.2f}")

    print("\n[JOURNAL] 复盘统计完成。\n")


# ------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter 实盘记录 + 复盘统计 小工具",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["add", "stats"],
        required=True,
        help="运行模式：add=交互式录入成交，stats=做复盘统计。",
    )
    parser.add_argument(
        "--trade-file",
        type=str,
        default=None,
        help="可选，自定义成交记录 csv 路径；默认用 ConfigManager.paths.trade_history_file。",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="复盘统计起始日期（含），YYYY-MM-DD，仅在 --mode stats 时生效。",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="复盘统计结束日期（含），YYYY-MM-DD，仅在 --mode stats 时生效。",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="按策略标签过滤，例如 U2；为空则不过滤。",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=200_000.0,
        help="初始权益（用于构造权益曲线 & 最大回撤，默认 200000）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trade_file = args.trade_file or resolve_trade_history_path()

    if args.mode == "add":
        interactive_add(trade_file=trade_file, default_strategy=args.strategy or "U2")
    elif args.mode == "stats":
        run_stats(
            trade_file=trade_file,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
            initial_equity=args.initial_equity,
        )
    else:
        raise SystemExit(f"未知 mode: {args.mode}")


if __name__ == "__main__":
    main()
