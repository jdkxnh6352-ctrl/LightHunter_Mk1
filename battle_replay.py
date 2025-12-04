# -*- coding: utf-8 -*-
"""
模块名称：BattleReplay Mk-Chronicle
版本：Mk-Chronicle R90 (Multi-Horizon Label Edition)
路径: G:/LightHunter_Mk1/battle_replay.py

功能：
- 从 trade_history.csv 中还原每一笔完整交易（BUY->SELL）；
- 利用 ts_data.db 中的 snapshots 还原持仓期间的价格路径：
  * 计算每笔交易的 MFE/MAE（最大顺/逆风浮动）
  * 计算从进场起未来多时间尺度的收益画像：
        - 10 分钟窗口：Future_10M_Max / Future_10M_Min
        - 20 分钟窗口：Future_20M_Max / Future_20M_Min
        - 30 分钟窗口：Future_30M_Max / Future_30M_Min（兼容旧版本）
        - 当日收盘：EOD_Return_pct（从进场到当天收盘的涨跌幅）
- 结合 market_blackbox.csv，对每笔交易附上进场时的因子状态：
  * Entry_Z_Force / Entry_Resilience / Entry_NN_Prob / Entry_Final_Score / Entry_Info 等
- 生成 battle_report.csv，供后续 FactorLab / RiskLabelLab / GA / AI / RiskBrain 使用。

多标签说明（Entry_* 系列）：
- Entry_future_gain_10M / 20M / 30M   : 对应各窗口内最大涨幅（%）
- Entry_label_10M / 20M / 30M         : 是否在对应窗口内最大涨幅 >= 3%（1=好样本）
- Entry_future_gain                    : 兼容字段 = Entry_future_gain_30M
- Entry_label                          : 兼容字段 = Entry_label_30M
- EOD_Return_pct                       : 从进场到当日收盘的收益率（%）
- Entry_label_EOD                      : 收盘收益率 >= 3% 视为 1，否则 0

注意：
- 保留旧字段，FactorLab / RiskLabelLab 现有逻辑无需改动即可继续使用；
- 新增多时间尺度标签，为后续「时间尺度剖面」和多任务模型提供原始弹药。
"""

import os
import sqlite3
import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from colorama import init, Fore, Style

init(autoreset=True)


class BattleReplay:
    def __init__(
        self,
        trade_file: str = "trade_history.csv",
        ts_db: str = "ts_data.db",
        blackbox_file: str = "market_blackbox.csv",
        output_file: str = "battle_report.csv",
    ):
        self.trade_file = trade_file
        self.ts_db = ts_db
        self.blackbox_file = blackbox_file
        self.output_file = output_file

        self.conn = self._open_ts_db()
        self.bb_df = self._load_blackbox()

    # --------------------------------------------------
    # 初始化：时序库 & 黑匣子
    # --------------------------------------------------
    def _open_ts_db(self):
        if not os.path.exists(self.ts_db):
            print(
                Fore.YELLOW
                + f"[BR] 未找到 {self.ts_db}，将只做交易复盘，不计算 MFE/MAE 和 future gain。"
                + Style.RESET_ALL
            )
            return None
        try:
            conn = sqlite3.connect(self.ts_db)
            print(
                Fore.CYAN
                + f"[BR] 已连接时序库: {self.ts_db}"
                + Style.RESET_ALL
            )
            return conn
        except Exception as e:
            print(
                Fore.RED
                + f"[BR] 打开 {self.ts_db} 失败: {e}"
                + Style.RESET_ALL
            )
            return None

    def _load_blackbox(self) -> pd.DataFrame:
        if not os.path.exists(self.blackbox_file):
            print(
                Fore.YELLOW
                + f"[BR] 未找到 {self.blackbox_file}，将无法附加进场因子。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.blackbox_file)
            if "Time" not in df.columns:
                print(
                    Fore.YELLOW
                    + f"[BR] {self.blackbox_file} 中缺少 Time 列。"
                    + Style.RESET_ALL
                )
                return pd.DataFrame()
            df["Time"] = pd.to_datetime(
                df["Time"], errors="coerce"
            )
            df = df.dropna(subset=["Time"])
            print(
                Fore.CYAN
                + f"[BR] 已加载黑匣子: {self.blackbox_file} (rows={len(df)})"
                + Style.RESET_ALL
            )
            return df
        except Exception as e:
            print(
                Fore.RED
                + f"[BR] 读取 {self.blackbox_file} 失败: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

    # --------------------------------------------------
    # 读取 & 配对交易
    # --------------------------------------------------
    def _load_trades(self) -> pd.DataFrame:
        if not os.path.exists(self.trade_file):
            print(
                Fore.RED
                + f"[BR] 未找到 {self.trade_file}，请先用 Commander 跑出交易记录。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.trade_file)
            df["Time"] = pd.to_datetime(
                df["Time"], errors="coerce"
            )
            df = df.dropna(subset=["Time"])
            df = df.sort_values("Time")
            print(
                Fore.CYAN
                + f"[BR] 已加载交易记录: {self.trade_file} (rows={len(df)})"
                + Style.RESET_ALL
            )
            return df
        except Exception as e:
            print(
                Fore.RED
                + f"[BR] 读取 {self.trade_file} 失败: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

    def _pair_trades(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        根据 trade_history.csv 中的 BUY/SELL 记录，配对完整交易。
        假设 PaperTrader 每次 BUY 建仓，SELL 全仓卖出。
        """
        open_pos: Dict[str, Dict[str, Any]] = {}
        trades: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            action = str(row["Action"]).upper()
            code = str(row["Code"])
            name = str(row["Name"])
            t = row["Time"]

            price = float(row["Price"])
            vol = int(row["Vol"])
            amount = float(row["Amount"])
            fee = float(row["Fee"])

            if action == "BUY":
                open_pos[code] = {
                    "code": code,
                    "name": name,
                    "time": t,
                    "price": price,
                    "vol": vol,
                    "amount": amount,
                    "fee": fee,
                }
            elif action == "SELL":
                if code not in open_pos:
                    # 没有对应 BUY，可能是旧数据，跳过
                    continue
                buy = open_pos.pop(code)
                trade = self._build_trade(buy, row)
                trades.append(trade)

        if open_pos:
            print(
                Fore.YELLOW
                + f"[BR] 注意：有 {len(open_pos)} 笔持仓尚未卖出，未计入战斗报告。"
                + Style.RESET_ALL
            )

        return trades

    # --------------------------------------------------
    # 单笔交易构建 + TS 指标（多时间尺度）
    # --------------------------------------------------
    def _build_trade(
        self, buy_row: Dict[str, Any], sell_row: pd.Series
    ) -> Dict[str, Any]:
        code = buy_row["code"]
        name = buy_row["name"]

        entry_time: datetime.datetime = buy_row["time"]
        exit_time: datetime.datetime = sell_row["Time"]

        entry_price = float(buy_row["price"])
        exit_price = float(sell_row["Price"])
        vol = int(buy_row["vol"])

        entry_amount = float(buy_row["Amount"])
        entry_fee = float(buy_row["Fee"])
        exit_amount = float(sell_row["Amount"])
        exit_fee = float(sell_row["Fee"])

        gross_pnl = exit_amount - entry_amount
        net_pnl = (exit_amount - exit_fee) - (
            entry_amount + entry_fee
        )
        cost_basis = entry_amount + entry_fee
        ret_pct = (
            net_pnl / cost_basis * 100 if cost_basis > 0 else 0.0
        )

        hold_minutes = (
            exit_time - entry_time
        ).total_seconds() / 60.0

        # TS：MFE/MAE + 多 horizon future gain + 当日收盘收益
        (
            mfe_pct,
            mae_pct,
            future_max_10,
            future_min_10,
            future_max_20,
            future_min_20,
            future_max_30,
            future_min_30,
            eod_ret,
        ) = self._calc_ts_metrics(
            code, entry_time, exit_time, entry_price
        )

        trade: Dict[str, Any] = {
            "Code": code,
            "Name": name,
            "Entry_Time": entry_time,
            "Exit_Time": exit_time,
            "Entry_Price": round(entry_price, 4),
            "Exit_Price": round(exit_price, 4),
            "Volume": vol,
            "Entry_Amount": round(entry_amount, 2),
            "Entry_Fee": round(entry_fee, 2),
            "Exit_Amount": round(exit_amount, 2),
            "Exit_Fee": round(exit_fee, 2),
            "Gross_PnL": round(gross_pnl, 2),
            "Net_PnL": round(net_pnl, 2),
            "Return_pct": round(ret_pct, 3),
            "Hold_Minutes": round(hold_minutes, 1),
            "MFE_pct": (
                round(mfe_pct, 3)
                if mfe_pct is not None
                else np.nan
            ),
            "MAE_pct": (
                round(mae_pct, 3)
                if mae_pct is not None
                else np.nan
            ),
            # 30m horizon（保持旧字段名称兼容）
            "Future_30M_Max": (
                round(future_max_30, 3)
                if future_max_30 is not None
                else np.nan
            ),
            "Future_30M_Min": (
                round(future_min_30, 3)
                if future_min_30 is not None
                else np.nan
            ),
        }

        # -------- 多时间尺度 Future_XM_* --------
        trade["Future_10M_Max"] = (
            round(future_max_10, 3) if future_max_10 is not None else np.nan
        )
        trade["Future_10M_Min"] = (
            round(future_min_10, 3) if future_min_10 is not None else np.nan
        )

        trade["Future_20M_Max"] = (
            round(future_max_20, 3) if future_max_20 is not None else np.nan
        )
        trade["Future_20M_Min"] = (
            round(future_min_20, 3) if future_min_20 is not None else np.nan
        )

        # EOD 收盘收益
        trade["EOD_Return_pct"] = (
            round(eod_ret, 3) if eod_ret is not None else np.nan
        )

        # -------- 多标签构造（10m / 20m / 30m / EOD）--------
        # 统一门槛：未来窗口内最大涨幅 >= 3% 视为好样本
        thr = 3.0

        # 10m
        if future_max_10 is not None:
            trade["Entry_future_gain_10M"] = round(future_max_10, 3)
            trade["Entry_label_10M"] = int(
                1 if future_max_10 >= thr else 0
            )
        else:
            trade["Entry_future_gain_10M"] = np.nan
            trade["Entry_label_10M"] = np.nan

        # 20m
        if future_max_20 is not None:
            trade["Entry_future_gain_20M"] = round(future_max_20, 3)
            trade["Entry_label_20M"] = int(
                1 if future_max_20 >= thr else 0
            )
        else:
            trade["Entry_future_gain_20M"] = np.nan
            trade["Entry_label_20M"] = np.nan

        # 30m（兼容旧字段）
        if future_max_30 is not None:
            trade["Entry_future_gain_30M"] = round(future_max_30, 3)
            trade["Entry_label_30M"] = int(
                1 if future_max_30 >= thr else 0
            )
            # 兼容字段（旧版使用）
            trade["Entry_future_gain"] = trade["Entry_future_gain_30M"]
            trade["Entry_label"] = trade["Entry_label_30M"]
        else:
            trade["Entry_future_gain_30M"] = np.nan
            trade["Entry_label_30M"] = np.nan
            trade["Entry_future_gain"] = np.nan
            trade["Entry_label"] = np.nan

        # EOD label
        if eod_ret is not None:
            trade["Entry_label_EOD"] = int(1 if eod_ret >= thr else 0)
        else:
            trade["Entry_label_EOD"] = np.nan

        # 从黑匣子附加进场因子
        self._attach_entry_factors(trade)

        return trade

    def _calc_ts_metrics(
        self,
        code: str,
        entry_time: datetime.datetime,
        exit_time: datetime.datetime,
        entry_price: float,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """
        从 ts_data.snapshots 中提取：
        - [entry, exit] 区间的 MFE / MAE
        - [entry, entry+10m] 区间的 future max/min
        - [entry, entry+20m] 区间的 future max/min
        - [entry, entry+30m] 区间的 future max/min
        - 当日收盘 EOD_Return_pct（从进场到当日最后一笔价）

        返回：
        (mfe_pct, mae_pct,
         future_max_10, future_min_10,
         future_max_20, future_min_20,
         future_max_30, future_min_30,
         eod_ret)
        """
        if self.conn is None or entry_price <= 0:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        try:
            start_str = entry_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            exit_str = exit_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # 三个 horizon
            h10_end = entry_time + datetime.timedelta(minutes=10)
            h20_end = entry_time + datetime.timedelta(minutes=20)
            h30_end = entry_time + datetime.timedelta(minutes=30)

            h10_end_str = h10_end.strftime("%Y-%m-%d %H:%M:%S")
            h30_end_str = h30_end.strftime("%Y-%m-%d %H:%M:%S")

            # 1) 持仓期内的路径 → MFE / MAE
            trade_df = pd.read_sql_query(
                """
                SELECT ts, price FROM snapshots
                WHERE code = ? AND ts >= ? AND ts <= ?
                ORDER BY ts
                """,
                self.conn,
                params=(code, start_str, exit_str),
            )
            mfe_pct = mae_pct = None
            if not trade_df.empty:
                prices = trade_df["price"].astype(
                    float
                ).values
                ret = (prices - entry_price) / entry_price * 100.0
                mfe_pct = float(ret.max())
                mae_pct = float(ret.min())

            # 2) 未来 30 分钟的空间（一次查询，内部切 10/20/30）
            future_df = pd.read_sql_query(
                """
                SELECT ts, price FROM snapshots
                WHERE code = ? AND ts >= ? AND ts <= ?
                ORDER BY ts
                """,
                self.conn,
                params=(code, start_str, h30_end_str),
            )

            future_max_10 = future_min_10 = None
            future_max_20 = future_min_20 = None
            future_max_30 = future_min_30 = None

            if not future_df.empty:
                future_df = future_df.copy()
                future_df["ts"] = pd.to_datetime(future_df["ts"])
                prices_f = future_df["price"].astype(
                    float
                ).values
                ret_f = (
                    prices_f - entry_price
                ) / entry_price * 100.0

                dt_min = (
                    (future_df["ts"] - entry_time)
                    .dt.total_seconds()
                    / 60.0
                )

                # 10m 窗口
                mask10 = dt_min <= 10.0 + 1e-6
                if mask10.any():
                    sub10 = ret_f[mask10.values]
                    future_max_10 = float(sub10.max())
                    future_min_10 = float(sub10.min())

                # 20m 窗口
                mask20 = dt_min <= 20.0 + 1e-6
                if mask20.any():
                    sub20 = ret_f[mask20.values]
                    future_max_20 = float(sub20.max())
                    future_min_20 = float(sub20.min())

                # 30m 窗口（保持旧逻辑）
                mask30 = dt_min <= 30.0 + 1e-6
                if mask30.any():
                    sub30 = ret_f[mask30.values]
                    future_max_30 = float(sub30.max())
                    future_min_30 = float(sub30.min())

            # 3) EOD 收盘收益（同一交易日最后一笔价）
            eod_ret = None
            trade_day_str = entry_time.strftime("%Y-%m-%d")
            day_start_str = trade_day_str + " 09:15:00"
            day_end_str = trade_day_str + " 15:05:00"

            eod_df = pd.read_sql_query(
                """
                SELECT ts, price FROM snapshots
                WHERE code = ? AND ts >= ? AND ts <= ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                self.conn,
                params=(code, day_start_str, day_end_str),
            )
            if not eod_df.empty:
                eod_price = float(
                    eod_df["price"].astype(float).iloc[0]
                )
                if eod_price > 0:
                    eod_ret = (
                        (eod_price - entry_price)
                        / entry_price
                        * 100.0
                    )

            return (
                mfe_pct,
                mae_pct,
                future_max_10,
                future_min_10,
                future_max_20,
                future_min_20,
                future_max_30,
                future_min_30,
                eod_ret,
            )

        except Exception as e:
            print(
                Fore.RED
                + f"[BR][TS] 计算 {code} TS 指标失败: {e}"
                + Style.RESET_ALL
            )
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    # --------------------------------------------------
    # 黑匣子：进场因子对齐
    # --------------------------------------------------
    def _attach_entry_factors(self, trade: Dict[str, Any]):
        """
        根据进场时间 & 代码，在 market_blackbox.csv 中找最近的一条记录，
        把当时 CombatBrain 的因子状态贴到交易上。
        """
        if self.bb_df is None or self.bb_df.empty:
            return

        code = trade["Code"]
        entry_time = trade["Entry_Time"]

        sub = self.bb_df[self.bb_df["代码"] == code]
        if sub.empty:
            return

        sub = sub.copy()
        sub["dt_diff"] = (
            sub["Time"] - entry_time
        ).abs()
        sub = sub.sort_values("dt_diff")
        row = sub.iloc[0]

        # 若时间差过大（> 60 秒），认为不是同一次扫描，放弃
        if row["dt_diff"].total_seconds() > 60:
            return

        # 映射字段
        mapping = {
            "涨幅": "Entry_pct",
            "换手率": "Entry_turnover",
            "主力攻击系数": "Entry_force",
            "Z_Force": "Entry_Z_Force",
            "Resilience": "Entry_Resilience",
            "NN_Prob": "Entry_NN_Prob",
            "Final_Score": "Entry_Final_Score",
            "Info": "Entry_Info",
            "形态": "Entry_Pattern",
            "指令": "Entry_Cmd",
        }

        for col, new_name in mapping.items():
            if col in row.index:
                trade[new_name] = row[col]

    # --------------------------------------------------
    # 公开入口：run()
    # --------------------------------------------------
    def run(self) -> pd.DataFrame:
        print(
            Fore.CYAN
            + "[BR] Battle Replay 初始化..."
            + Style.RESET_ALL
        )

        df_trades = self._load_trades()
        if df_trades.empty:
            return pd.DataFrame()

        trades = self._pair_trades(df_trades)
        if not trades:
            print(
                Fore.YELLOW
                + "[BR] 尚无完整买卖配对交易。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        report_df = pd.DataFrame(trades)
        # 保存到 CSV
        try:
            report_df.to_csv(
                self.output_file,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                Fore.GREEN
                + f"\n[BR] 战斗报告已生成 -> {self.output_file} (Trades: {len(report_df)})"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[BR] 保存 {self.output_file} 失败: {e}"
                + Style.RESET_ALL
            )

        self._print_stats(report_df)
        return report_df

    # --------------------------------------------------
    # 简单统计（扩展展示多标签信息）
    # --------------------------------------------------
    def _print_stats(self, df: pd.DataFrame):
        total = len(df)
        wins = df[df["Net_PnL"] > 0]
        win_rate = (
            len(wins) / total * 100 if total > 0 else 0.0
        )

        avg_ret = float(df["Return_pct"].mean())
        med_ret = float(df["Return_pct"].median())
        avg_hold = float(df["Hold_Minutes"].mean())

        mfe_mean = float(df["MFE_pct"].mean(skipna=True))
        mae_mean = float(df["MAE_pct"].mean(skipna=True))

        pnl_total = float(df["Net_PnL"].sum())
        pnl_color = (
            Fore.RED if pnl_total >= 0 else Fore.GREEN
        )

        print(
            Fore.YELLOW
            + "\n================= BATTLE SUMMARY ================="
            + Style.RESET_ALL
        )
        print(f"  总交易笔数 : {total}")
        print(f"  胜率       : {win_rate:.2f}%")
        print(
            f"  平均收益率 : {avg_ret:.2f}% (中位数: {med_ret:.2f}%)"
        )
        print(
            f"  平均持仓   : {avg_hold:.1f} 分钟"
        )
        if not np.isnan(mfe_mean):
            print(f"  平均 MFE   : {mfe_mean:.2f}%")
        if not np.isnan(mae_mean):
            print(f"  平均 MAE   : {mae_mean:.2f}%")
        print(
            f"  总净盈亏   : {pnl_color}{pnl_total:.2f}{Style.RESET_ALL}"
        )

        # 30m 旧版本统计（兼容）
        if "Entry_future_gain" in df.columns:
            fg_mean = float(
                df["Entry_future_gain"].mean(
                    skipna=True
                )
            )
            print(
                f"  平均未来 30 分钟最大涨幅: {fg_mean:.2f}%"
            )

        if "Entry_label" in df.columns:
            good = df[df["Entry_label"] == 1]
            bad = df[df["Entry_label"] == 0]
            print(
                f"  Label(30m)=1 样本数: {len(good)}, Label(30m)=0 样本数: {len(bad)}"
            )

        # —— 多时间尺度标签简要统计 ——
        for horizon in ["10M", "20M", "30M", "EOD"]:
            col_gain = (
                f"Entry_future_gain_{horizon}"
                if horizon != "EOD"
                else "EOD_Return_pct"
            )
            col_label = (
                f"Entry_label_{horizon}"
                if horizon != "EOD"
                else "Entry_label_EOD"
            )

            if col_gain not in df.columns or col_label not in df.columns:
                continue

            sub = df[~df[col_gain].isna()].copy()
            if sub.empty:
                continue

            mean_gain = float(sub[col_gain].mean())
            pos = sub[sub[col_label] == 1]
            neg = sub[sub[col_label] == 0]
            print(
                f"  [{horizon}] 平均收益: {mean_gain:.2f}% | "
                f"Label=1: {len(pos)}, Label=0: {len(neg)}"
            )


if __name__ == "__main__":
    br = BattleReplay()
    br.run()
