#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
u1_paper_trade_helper.py

功能：
1）add 模式：交互式记录一笔实盘/纸上交易，追加到 data/u1_paper_trades.csv
2）summary 模式：读取 data/u1_paper_trades.csv，输出简单的绩效统计

用法示例（在项目根目录 LightHunter_Mk1 下）：
    python tools/u1_paper_trade_helper.py add
    python tools/u1_paper_trade_helper.py summary
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ---------- 基础路径配置 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = DATA_DIR / "u1_paper_trades.csv"


# ---------- 一些小工具函数 ----------

def _parse_date(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    print(f"[WARN] 日期格式不识别：{s}，请用 2025-12-03 或 20251203 这种格式。")
    return None


def _parse_float(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        print(f"[WARN] 不是数字：{s}，先按空值处理。")
        return None


def _parse_int(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        print(f"[WARN] 不是整数：{s}，先按空值处理。")
        return None


def _normalize_side(s: str):
    s = (s or "").strip().upper()
    if s in ("B", "BUY", "多", "1"):
        return "BUY"
    if s in ("S", "SELL", "空", "-1"):
        return "SELL"
    print(f"[WARN] 无法识别的方向：{s}，先按 BUY 处理。")
    return "BUY"


# ---------- add：交互式新增一笔交易 ----------

def cmd_add(args):
    print("=" * 60)
    print("【U1 纸上实盘记录】新增一笔交易（按回车用默认值或留空）")
    print("=" * 60)

    today_str = datetime.today().strftime("%Y-%m-%d")

    open_date = _parse_date(input(f"开仓日期 [默认 {today_str}]: ") or today_str)
    symbol = input("股票代码（如 000001.SZ 或 600000.SH）: ").strip()
    if not symbol:
        print("[ERROR] 股票代码不能为空。")
        return

    side = _normalize_side(input("方向 [B=买入, S=卖出]（默认 B）: ") or "B")
    open_price = _parse_float(input("开仓价格（元）: "))
    shares = _parse_int(input("数量（股）: "))

    close_date = _parse_date(input("平仓日期（还没卖就留空）: "))
    close_price = _parse_float(input("平仓价格（还没卖就留空）: "))

    notes = input("备注（可空，比如 '按模型信号买入'）: ").strip()

    # 如果没给 trade_id，就自动生成一个
    auto_trade_id = f"{open_date}_{symbol}_{side}"
    trade_id = input(f"交易 ID（可空，默认 {auto_trade_id}）: ").strip() or auto_trade_id

    # 组装成一行
    row = {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "open_date": open_date.strftime("%Y-%m-%d") if open_date else "",
        "open_price": open_price,
        "shares": shares,
        "close_date": close_date.strftime("%Y-%m-%d") if close_date else "",
        "close_price": close_price,
        "notes": notes,
    }

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=list(row.keys()))

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # 重新计算 PnL / 收益率
    df = _recompute_pnl(df)

    df.to_csv(LOG_PATH, index=False)
    print(f"\n[OK] 已保存到 {LOG_PATH}")
    print(f"当前总记录数：{len(df)} 笔。")


def _recompute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """根据 open/close 价格和方向，计算 pnl / return_pct / holding_days"""
    # 确保这些列存在
    for col in ["open_price", "close_price", "shares"]:
        if col not in df.columns:
            df[col] = np.nan

    # 方向：BUY=1, SELL=-1
    side_sign = df["side"].map(lambda x: 1 if str(x).upper() == "BUY" else -1)

    open_price = pd.to_numeric(df["open_price"], errors="coerce")
    close_price = pd.to_numeric(df["close_price"], errors="coerce")
    shares = pd.to_numeric(df["shares"], errors="coerce")

    not_na_mask = (~open_price.isna()) & (~close_price.isna()) & (~shares.isna())

    pnl = np.where(
        not_na_mask,
        (close_price - open_price) * shares * side_sign,
        np.nan,
    )

    cost = np.where(
        (~open_price.isna()) & (~shares.isna()),
        open_price * shares,
        np.nan,
    )
    return_pct = np.where(
        (not_na_mask & (cost != 0)),
        pnl / cost,
        np.nan,
    )

    df["pnl"] = pnl
    df["return_pct"] = return_pct

    # 持仓天数
    def _days_delta(row):
        try:
            if not row.get("open_date") or not row.get("close_date"):
                return np.nan
            d1 = datetime.strptime(str(row["open_date"]), "%Y-%m-%d").date()
            d2 = datetime.strptime(str(row["close_date"]), "%Y-%m-%d").date()
            return (d2 - d1).days
        except Exception:
            return np.nan

    df["holding_days"] = df.apply(_days_delta, axis=1)

    return df


# ---------- summary：汇总绩效 ----------

def cmd_summary(args):
    if not LOG_PATH.exists():
        print(f"[ERROR] 找不到交易记录文件：{LOG_PATH}")
        print("先运行：python tools/u1_paper_trade_helper.py add 来记录至少一笔交易。")
        return

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        print("[WARN] 当前没有任何记录。")
        return

    df = _recompute_pnl(df)

    # 完结的交易：有 close_price
    closed = df[df["close_price"].notna()].copy()

    print("=" * 70)
    print(f"【U1 纸上实盘汇总】文件：{LOG_PATH.name}")
    print(f"总记录数：{len(df)} 笔，其中已平仓 {len(closed)} 笔。")
    print("=" * 70)

    if len(closed) == 0:
        print("暂时没有已平仓的交易，等有卖出的再看统计。")
        return

    total_pnl = closed["pnl"].sum()
    avg_pnl = closed["pnl"].mean()
    max_win = closed["pnl"].max()
    max_loss = closed["pnl"].min()

    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]

    win_rate = len(wins) / len(closed) if len(closed) > 0 else np.nan
    avg_win = wins["pnl"].mean() if not wins.empty else np.nan
    avg_loss = losses["pnl"].mean() if not losses.empty else np.nan

    avg_holding = closed["holding_days"].mean()

    print("【整体表现】")
    print(f"- 已平仓笔数：{len(closed)}")
    print(f"- 总收益（元）：{total_pnl:,.2f}")
    print(f"- 平均每笔收益（元）：{avg_pnl:,.2f}")
    print(f"- 胜率：{win_rate * 100:5.2f}%")
    print(f"- 平均盈利单（元）：{avg_win:,.2f}")
    print(f"- 平均亏损单（元）：{avg_loss:,.2f}")
    print(f"- 最大盈利单（元）：{max_win:,.2f}")
    print(f"- 最大亏损单（元）：{max_loss:,.2f}")
    print(f"- 平均持仓天数：{avg_holding:.2f} 天")
    print()

    # 按月份简单看看
    try:
        closed["close_month"] = pd.to_datetime(closed["close_date"]).dt.to_period("M").astype(str)
        by_month = closed.groupby("close_month")["pnl"].sum().reset_index()
        print("【按月份汇总】")
        for _, row in by_month.iterrows():
            print(f"- {row['close_month']}: {row['pnl']:,.2f} 元")
        print()
    except Exception:
        pass

    # 最近几笔交易
    print("【最近 5 笔已平仓】")
    cols_show = ["trade_id", "symbol", "side", "open_date", "close_date", "open_price", "close_price", "shares", "pnl", "return_pct"]
    for col in cols_show:
        if col not in closed.columns:
            closed[col] = np.nan

    tail = closed.sort_values("close_date").tail(5)
    # 简单打印一行一行
    for _, row in tail.iterrows():
        print("-" * 70)
        print(f"ID: {row['trade_id']}")
        print(f"标的: {row['symbol']}  方向: {row['side']}")
        print(f"开仓: {row['open_date']} @ {row['open_price']}  数量: {row['shares']}")
        print(f"平仓: {row['close_date']} @ {row['close_price']}")
        print(f"结果: PnL={row['pnl']:.2f} 元, 收益率={row['return_pct'] * 100 if pd.notna(row['return_pct']) else float('nan'):5.2f}%")
    print("-" * 70)
    print("\n[提示] 这个脚本只是第一版，后面我们可以再加：按策略分组、按持仓天数分层等更高级分析。")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="U1 纸上实盘记录 + 汇总脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_add = subparsers.add_parser("add", help="新增一笔纸上/实盘交易记录")
    p_add.set_defaults(func=cmd_add)

    p_summary = subparsers.add_parser("summary", help="汇总当前所有交易表现")
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
