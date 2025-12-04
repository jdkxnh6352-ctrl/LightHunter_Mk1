# -*- coding: utf-8 -*-
"""
模块名称：DecisionGapLab Mk-Decision
版本：Mk-Decision R10 (HighScore Miss & LowScore Misfire)
路径: G:/LightHunter_Mk1/decision_gap_lab.py

功能：
- 从 market_blackbox_labeled.csv / market_blackbox.csv 中读取当日候选股快照；
- 从 battle_report.csv 中读取真实成交记录（由 BattleReplay 生成）；
- 定义两类“决策落差”样本：
    1) 高分没买（HighScore_NoTrade）：
        * 当天在黑匣子里评分靠前 (Final_Score >= 高分分位数)，
        * 但当日没有任何成交记录的股票；
    2) 低分乱买（LowScore_Trade）：
        * 真实成交时的 Entry_Final_Score 处于较低分位 (<= 低分分位数) 的交易。
- 统一抽取因子画像（Score / Z_Force / NN_Prob / 涨幅 / 换手 / 主力攻击系数 /
  TS 标签 / 未来收益 / 真实收益），
  生成：
    * decision_gap_samples.csv：逐样本明细；
    * decision_gap_profile.csv：按 Group 汇总的因子画像。

说明：
- 若存在 market_blackbox_labeled.csv，则优先使用其中的 future_gain / label 信息；
- 若不存在，则退化为使用 market_blackbox.csv，仅做因子分布分析；
- 低分乱买组的真实收益来自 battle_report.csv 中的 Return_pct；
- 高分没买若存在 future_gain，则可近似视为“如果当时买入，未来 10~30 分钟能否赚钱”的 proxy。
"""

import os
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)


class DecisionGapLab:
    def __init__(
        self,
        blackbox_file: str = "market_blackbox_labeled.csv",
        fallback_blackbox_file: str = "market_blackbox.csv",
        battle_report_file: str = "battle_report.csv",
        samples_file: str = "decision_gap_samples.csv",
        profile_file: str = "decision_gap_profile.csv",
        high_score_quantile: float = 0.9,
        low_score_quantile: float = 0.3,
    ):
        """
        :param blackbox_file: 优先使用的黑匣子文件（带 TS 标签更佳）
        :param fallback_blackbox_file: 若上述文件不存在，则退回到该文件
        :param battle_report_file: BattleReplay 输出的完整交易报告
        :param samples_file: 决策落差样本明细输出
        :param profile_file: 因子画像统计输出
        :param high_score_quantile: 高分没买阈值的分位数 (0~1)
        :param low_score_quantile: 低分乱买阈值的分位数 (0~1)
        """
        self.blackbox_file = (
            blackbox_file if os.path.exists(blackbox_file) else fallback_blackbox_file
        )
        self.battle_report_file = battle_report_file
        self.samples_file = samples_file
        self.profile_file = profile_file
        self.high_q = float(high_score_quantile)
        self.low_q = float(low_score_quantile)

    # --------------------------------------------------
    # 公共入口
    # --------------------------------------------------
    def run(self) -> pd.DataFrame:
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "[DG-DECISION] DecisionGapLab 初始化..."
            + Style.RESET_ALL
        )

        bb = self._load_blackbox()
        br = self._load_battle_report()

        if bb.empty:
            print(
                Fore.RED
                + "[DG-DECISION] 黑匣子数据为空，无法分析决策落差。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        if br.empty:
            print(
                Fore.RED
                + "[DG-DECISION] battle_report.csv 为空，尚无真实交易可对比。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        # 分位阈值
        high_thr, low_thr = self._calc_score_thresholds(bb, br)

        print(
            Fore.CYAN
            + f"[DG-DECISION] 高分没买阈值 Final_Score >= {high_thr:.2f} "
            f"(q={self.high_q:.2f})；低分乱买阈值 Entry_Final_Score <= {low_thr:.2f} "
            f"(q={self.low_q:.2f})."
            + Style.RESET_ALL
        )

        high_no_trade = self._build_high_score_no_trade(bb, br, high_thr)
        low_trade = self._build_low_score_trade(br, low_thr)

        if high_no_trade.empty and low_trade.empty:
            print(
                Fore.YELLOW
                + "[DG-DECISION] 没有命中的高分没买 / 低分乱买样本。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        samples = pd.concat([high_no_trade, low_trade], ignore_index=True)
        self._save_samples(samples)

        profile = self._build_profile(samples)
        self._save_profile(profile)
        self._print_summary(profile)

        return profile

    # --------------------------------------------------
    # 数据加载
    # --------------------------------------------------
    def _load_blackbox(self) -> pd.DataFrame:
        if not os.path.exists(self.blackbox_file):
            print(
                Fore.RED
                + f"[DG-DECISION] 找不到黑匣子文件: {self.blackbox_file}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.blackbox_file)
        except Exception as e:
            print(
                Fore.RED
                + f"[DG-DECISION] 读取 {self.blackbox_file} 失败: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        if df.empty:
            print(
                Fore.YELLOW
                + f"[DG-DECISION] {self.blackbox_file} 为空。"
                + Style.RESET_ALL
            )
            return df

        # 时间列 & 代码标准化
        time_col = None
        for c in ["Time", "time", "时间"]:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            print(
                Fore.YELLOW
                + "[DG-DECISION] 黑匣子中缺少 Time 列，将无法按日期对齐，只能做全局统计。"
                + Style.RESET_ALL
            )
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df["date"] = df[time_col].dt.date

        code_col = "代码" if "代码" in df.columns else None
        if code_col is None:
            for c in ["code", "Code", "证券代码"]:
                if c in df.columns:
                    code_col = c
                    break
        if code_col is None:
            print(
                Fore.RED
                + "[DG-DECISION] 黑匣子中缺少 代码 列。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        df["Code"] = df[code_col].astype(str).str[-6:].str.zfill(6)

        # 评分字段
        if "Final_Score" not in df.columns:
            print(
                Fore.RED
                + "[DG-DECISION] 黑匣子中缺少 Final_Score 列，无法判断高分。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        print(
            Fore.GREEN
            + f"[DG-DECISION] 已加载黑匣子 {self.blackbox_file}，rows={len(df)}。"
            + Style.RESET_ALL
        )
        return df

    def _load_battle_report(self) -> pd.DataFrame:
        if not os.path.exists(self.battle_report_file):
            print(
                Fore.RED
                + f"[DG-DECISION] 找不到 battle_report 文件: {self.battle_report_file}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(
                self.battle_report_file,
                parse_dates=["Entry_Time", "Exit_Time"],
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[DG-DECISION] 读取 {self.battle_report_file} 失败: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        if df.empty:
            print(
                Fore.YELLOW
                + "[DG-DECISION] battle_report.csv 为空。"
                + Style.RESET_ALL
            )
            return df

        # 标准化代码 & 日期
        df["Code"] = df["Code"].astype(str).str[-6:].str.zfill(6)
        if "Entry_Time" in df.columns:
            df["Entry_Date"] = df["Entry_Time"].dt.date

        print(
            Fore.GREEN
            + f"[DG-DECISION] 已加载 battle_report.csv，trades={len(df)}。"
            + Style.RESET_ALL
        )
        return df

    # --------------------------------------------------
    # 阈值计算
    # --------------------------------------------------
    def _calc_score_thresholds(
        self, bb: pd.DataFrame, br: pd.DataFrame
    ) -> (float, float):
        # 高分阈值：黑匣子 Final_Score 全局分位
        high_thr = float(
            bb["Final_Score"].dropna().quantile(self.high_q)
        )

        # 低分阈值：battle_report 中 Entry_Final_Score 分位；若不存在则退化为 Entry_Z_Force
        if "Entry_Final_Score" in br.columns:
            base_col = "Entry_Final_Score"
        elif "Entry_Z_Force" in br.columns:
            base_col = "Entry_Z_Force"
        else:
            base_col = None

        if base_col is not None:
            low_thr = float(
                pd.to_numeric(br[base_col], errors="coerce")
                .dropna()
                .quantile(self.low_q)
            )
        else:
            # 没有任何打分列，则无法定义低分乱买；设为 0 作为占位
            low_thr = 0.0

        return high_thr, low_thr

    # --------------------------------------------------
    # 构建：高分没买样本
    # --------------------------------------------------
    def _build_high_score_no_trade(
        self, bb: pd.DataFrame, br: pd.DataFrame, high_thr: float
    ) -> pd.DataFrame:
        if "date" not in bb.columns:
            # 没有日期信息，则只在全局维度上看“曾经高分又从未成交”的股票
            print(
                Fore.YELLOW
                + "[DG-DECISION] 黑匣子缺少日期信息，高分没买只做全局层面。"
                + Style.RESET_ALL
            )
            # 取每个 Code 的最高分
            agg = (
                bb.groupby("Code", as_index=False)["Final_Score"]
                .max()
                .rename(columns={"Final_Score": "Best_Score"})
            )
            high = agg[agg["Best_Score"] >= high_thr].copy()
            traded_codes = set(br["Code"].unique())
            high["Traded_Flag"] = high["Code"].isin(traded_codes)
            high = high[~high["Traded_Flag"]].copy()
            if high.empty:
                return pd.DataFrame()

            # 从原黑匣子中抽取对应的最高分样本一条
            merged = bb.merge(
                high[["Code", "Best_Score"]],
                on=["Code"],
                how="inner",
            )
            merged = merged[merged["Final_Score"] == merged["Best_Score"]]
            # 每个 Code 取一条代表样本
            merged = (
                merged.sort_values(["Code", "Final_Score"], ascending=[True, False])
                .drop_duplicates(subset=["Code"])
            )

            samples = self._to_canonical_samples(
                merged,
                group_name="HighScore_NoTrade",
                is_trade=False,
            )
            return samples

        # 正常情况：按日期 + 代码维度判断
        # 1) 每日每股取最高分
        bb_day = (
            bb.groupby(["date", "Code"], as_index=False)["Final_Score"]
            .max()
            .rename(columns={"Final_Score": "Best_Score"})
        )

        # 2) 标记当日是否有成交
        if "Entry_Date" in br.columns:
            trade_keys = (
                br[["Entry_Date", "Code"]]
                .dropna()
                .drop_duplicates()
                .assign(
                    key=lambda x: x["Entry_Date"].astype(str)
                    + "_"
                    + x["Code"].astype(str)
                )
            )
            trade_key_set = set(trade_keys["key"].tolist())
            bb_day = bb_day.assign(
                key=lambda x: x["date"].astype(str) + "_" + x["Code"].astype(str)
            )
            bb_day["Traded_Flag"] = bb_day["key"].isin(trade_key_set)
        else:
            traded_codes = set(br["Code"].unique())
            bb_day["Traded_Flag"] = bb_day["Code"].isin(traded_codes)

        # 3) 高分且当日未成交
        mask = (bb_day["Best_Score"] >= high_thr) & (~bb_day["Traded_Flag"])
        high = bb_day[mask].copy()
        if high.empty:
            print(
                Fore.YELLOW
                + "[DG-DECISION] 没有命中的高分没买样本。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        # 4) 回到原黑匣子中，取对应日期+代码的最高分一条记录
        bb["date"] = bb["date"].astype(str)
        high["date"] = high["date"].astype(str)

        merged = bb.merge(
            high[["date", "Code", "Best_Score"]],
            on=["date", "Code"],
            how="inner",
        )
        merged = merged[merged["Final_Score"] == merged["Best_Score"]]

        # 每个 (date, Code) 取一条代表样本
        merged = (
            merged.sort_values(["date", "Code", "Final_Score"], ascending=[True, True, False])
            .drop_duplicates(subset=["date", "Code"])
        )

        samples = self._to_canonical_samples(
            merged,
            group_name="HighScore_NoTrade",
            is_trade=False,
        )
        return samples

    # --------------------------------------------------
    # 构建：低分乱买样本
    # --------------------------------------------------
    def _build_low_score_trade(
        self, br: pd.DataFrame, low_thr: float
    ) -> pd.DataFrame:
        if low_thr == 0.0 and "Entry_Final_Score" not in br.columns:
            print(
                Fore.YELLOW
                + "[DG-DECISION] battle_report 中没有 Entry_Final_Score / Entry_Z_Force，"
                "无法定义低分乱买。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        if "Entry_Final_Score" in br.columns:
            base_col = "Entry_Final_Score"
        else:
            base_col = "Entry_Z_Force"

        br = br.copy()
        br[base_col] = pd.to_numeric(br[base_col], errors="coerce")

        low_trades = br[br[base_col] <= low_thr].copy()
        if low_trades.empty:
            print(
                Fore.YELLOW
                + "[DG-DECISION] 没有命中的低分乱买样本。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        samples = self._to_canonical_samples(
            low_trades,
            group_name="LowScore_Trade",
            is_trade=True,
        )
        return samples

    # --------------------------------------------------
    # 样本标准化：统一字段
    # --------------------------------------------------
    def _to_canonical_samples(
        self,
        df: pd.DataFrame,
        group_name: str,
        is_trade: bool,
    ) -> pd.DataFrame:
        """
        将黑匣子 / battle_report 的原始字段映射为统一的因子列：
        - code / name / date / time
        - score / z_force / nn_prob / pct / turnover / force / resilience
        - future_gain / label / real_return
        """
        out = pd.DataFrame()
        out["Group"] = group_name

        # 基本信息
        if "Code" in df.columns:
            out["Code"] = df["Code"].astype(str).str[-6:].str.zfill(6)
        elif "代码" in df.columns:
            out["Code"] = df["代码"].astype(str).str[-6:].str.zfill(6)
        else:
            out["Code"] = ""

        name_col = None
        for c in ["Name", "名称"]:
            if c in df.columns:
                name_col = c
                break
        out["Name"] = df[name_col] if name_col else out["Code"]

        # 日期 & 时间
        if is_trade:
            # 来自 battle_report
            if "Entry_Time" in df.columns:
                dt = pd.to_datetime(df["Entry_Time"], errors="coerce")
                out["Date"] = dt.dt.date.astype(str)
                out["Time"] = dt.dt.strftime("%H:%M:%S")
            else:
                out["Date"] = ""
                out["Time"] = ""
        else:
            # 来自黑匣子
            time_col = None
            for c in ["Time", "time", "时间"]:
                if c in df.columns:
                    time_col = c
                    break
            if time_col is not None:
                dt = pd.to_datetime(df[time_col], errors="coerce")
                out["Date"] = dt.dt.date.astype(str)
                out["Time"] = dt.dt.strftime("%H:%M:%S")
            elif "date" in df.columns:
                out["Date"] = df["date"].astype(str)
                out["Time"] = ""
            else:
                out["Date"] = ""
                out["Time"] = ""

        # 统一因子列
        def pick(col_trade, col_bb, default=np.nan):
            if is_trade:
                if col_trade in df.columns:
                    return pd.to_numeric(df[col_trade], errors="coerce")
            else:
                if col_bb in df.columns:
                    return pd.to_numeric(df[col_bb], errors="coerce")
            return pd.Series(default, index=df.index)

        out["score"] = pick("Entry_Final_Score", "Final_Score")
        out["z_force"] = pick("Entry_Z_Force", "Z_Force")
        out["nn_prob"] = pick("Entry_NN_Prob", "NN_Prob")
        out["pct"] = pick("Entry_pct", "涨幅")
        out["turnover"] = pick("Entry_turnover", "换手率")
        out["force"] = pick("Entry_force", "主力攻击系数")
        out["resilience"] = pick("Entry_Resilience", "Resilience")

        # TS 标签 & 未来收益（如果有）
        # 黑匣子带标签：future_gain / label
        # battle_report：Entry_future_gain / Entry_label
        if is_trade:
            out["future_gain"] = pick("Entry_future_gain", None)
            out["label"] = pick("Entry_label", None)
        else:
            # 黑匣子
            if "future_gain" in df.columns:
                out["future_gain"] = pd.to_numeric(df["future_gain"], errors="coerce")
            elif "Entry_future_gain" in df.columns:
                out["future_gain"] = pd.to_numeric(df["Entry_future_gain"], errors="coerce")
            else:
                out["future_gain"] = np.nan

            if "label" in df.columns:
                out["label"] = pd.to_numeric(df["label"], errors="coerce")
            elif "Entry_label" in df.columns:
                out["label"] = pd.to_numeric(df["Entry_label"], errors="coerce")
            else:
                out["label"] = np.nan

        # 真实收益只对成交样本有意义
        if is_trade and "Return_pct" in df.columns:
            out["real_return"] = pd.to_numeric(df["Return_pct"], errors="coerce")
        else:
            out["real_return"] = np.nan

        # 附带 Info 字段方便肉眼排查
        info_col = None
        for c in ["Entry_Info", "Info"]:
            if c in df.columns:
                info_col = c
                break
        out["Info"] = df[info_col].astype(str) if info_col else ""

        return out

    # --------------------------------------------------
    # 样本 & 因子画像输出
    # --------------------------------------------------
    def _save_samples(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        try:
            df.to_csv(self.samples_file, index=False, encoding="utf-8-sig")
            print(
                Fore.GREEN
                + f"[DG-DECISION] 样本明细已保存 -> {self.samples_file} "
                f"(rows={len(df)})"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[DG-DECISION] 保存 {self.samples_file} 失败: {e}"
                + Style.RESET_ALL
            )

    def _build_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        groups = []
        for g_name, sub in df.groupby("Group"):
            sub = sub.copy()
            n = len(sub)
            win_future = (
                float((sub["future_gain"] > 0).mean() * 100.0)
                if "future_gain" in sub.columns
                else np.nan
            )
            win_real = (
                float((sub["real_return"] > 0).mean() * 100.0)
                if "real_return" in sub.columns
                else np.nan
            )

            def safe_mean(col):
                return float(pd.to_numeric(sub[col], errors="coerce").mean()) if col in sub.columns else np.nan

            groups.append(
                {
                    "Group": g_name,
                    "N_Samples": n,
                    "Avg_score": round(safe_mean("score"), 3),
                    "Avg_z_force": round(safe_mean("z_force"), 3),
                    "Avg_nn_prob": round(safe_mean("nn_prob"), 3),
                    "Avg_pct": round(safe_mean("pct"), 3),
                    "Avg_turnover": round(safe_mean("turnover"), 3),
                    "Avg_force": round(safe_mean("force"), 3),
                    "Avg_resilience": round(safe_mean("resilience"), 3),
                    "Avg_future_gain": round(safe_mean("future_gain"), 3),
                    "Avg_real_return": round(safe_mean("real_return"), 3),
                    "WinRate_future_pct": round(win_future, 2)
                    if not np.isnan(win_future)
                    else np.nan,
                    "WinRate_real_pct": round(win_real, 2)
                    if not np.isnan(win_real)
                    else np.nan,
                }
            )

        profile = pd.DataFrame(groups)
        return profile

    def _save_profile(self, df: pd.DataFrame):
        if df is None or df.empty:
            print(
                Fore.YELLOW
                + "[DG-DECISION] 无因子画像可保存。"
                + Style.RESET_ALL
            )
            return
        try:
            df.to_csv(self.profile_file, index=False, encoding="utf-8-sig")
            print(
                Fore.GREEN
                + f"[DG-DECISION] 因子画像已保存 -> {self.profile_file}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[DG-DECISION] 保存 {self.profile_file} 失败: {e}"
                + Style.RESET_ALL
            )

    def _print_summary(self, df: pd.DataFrame):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[DG-DECISION] === 决策落差因子画像 Summary ==="
            + Style.RESET_ALL
        )
        if df is None or df.empty:
            print(Fore.YELLOW + "  (空)" + Style.RESET_ALL)
            return

        for _, row in df.iterrows():
            print(
                Fore.WHITE
                + f"  {row['Group']:<16} | N={int(row['N_Samples']):4d} | "
                f"AvgScore={row['Avg_score']:7.2f} | "
                f"AvgFuture={row['Avg_future_gain']:7.2f}% | "
                f"AvgReal={row['Avg_real_return']:7.2f}% | "
                f"WinF={row['WinRate_future_pct'] if not np.isnan(row['WinRate_future_pct']) else np.nan:6.2f}% | "
                f"WinR={row['WinRate_real_pct'] if not np.isnan(row['WinRate_real_pct']) else np.nan:6.2f}%"
                + Style.RESET_ALL
            )


if __name__ == "__main__":
    lab = DecisionGapLab()
    lab.run()
