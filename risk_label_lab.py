# -*- coding: utf-8 -*-
"""
模块名称：RiskLabelLab Mk-Confluence
版本：Mk-Confluence R20 (Label × Risk × Regime)
路径: G:/LightHunter_Mk1/risk_label_lab.py

功能：
- 读取 battle_report.csv（由 battle_replay.py 生成）；
- 使用 Entry_label + Future_30M_Min 构造“标签 × 风险”四象限；
- 对每个象限统计收益 / 胜率 / 回撤画像；
- 扫描不同风险阈值（-1.5 / -2 / -2.5 / -3 / -4 / -5），
  评估“只保留安全 Alpha”的收益-样本数权衡；
- 新增：按行情 Regime（牛 / 震荡 / 熊）拆分四象限与阈值扫描：
  * 优先从 market_regimes.csv 读取日期→regime 映射；
  * 无配置文件则自动按日度收益中位数划分 Bull / Range / Bear；
- 输出 risk_label_profile.csv，控制台打印总结。

说明：
- 完全基于真实交易结果，不依赖 RiskBrain 模型文件；
- 后续可以把 RiskBrain 预测概率也接进来，做“三维联动”。
"""

import os
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)


class RiskLabelLab:
    def __init__(
        self,
        battle_report_file: str = "battle_report.csv",
        output_profile_file: str = "risk_label_profile.csv",
        default_risk_threshold: float = -3.0,
        regime_config_file: str = "market_regimes.csv",
    ):
        """
        :param battle_report_file: battle_replay.py 生成的战斗复盘文件
        :param output_profile_file: 联动分析输出文件
        :param default_risk_threshold: 默认风险阈值 (Future_30M_Min <= 阈值 即视为高风险)
        :param regime_config_file: 行情 Regime 配置文件（可选），列: trade_date/date, regime
        """
        self.battle_report_file = battle_report_file
        self.output_profile_file = output_profile_file
        self.default_risk_threshold = default_risk_threshold
        self.regime_config_file = regime_config_file

    # --------------------------------------------------
    # 主入口
    # --------------------------------------------------
    def run(self):
        df = self._load_battle_report()
        if df.empty:
            return

        df = self._prepare_df(df)
        if df.empty:
            return

        df = self._assign_regime(df)

        print(
            Fore.CYAN
            + Style.BRIGHT
            + f"[RL-LAB] Using risk threshold = {self.default_risk_threshold:.2f}%."
            + Style.RESET_ALL
        )
        df = self._build_risk_flag(df, self.default_risk_threshold)

        # 1) 四象限分析（支持 Regime）
        quad_profiles = self._analyze_quadrants(df)

        # 2) 风险阈值扫描（支持 Regime）
        sweep_profiles = self._risk_threshold_sweep(df)

        # 3) 合并保存
        profile_df = pd.concat(
            [quad_profiles, sweep_profiles], ignore_index=True
        )
        self._save_profile(profile_df)
        self._print_summary(profile_df)

        return profile_df

    # --------------------------------------------------
    # 数据加载 & 清洗 & Regime 赋值
    # --------------------------------------------------
    def _load_battle_report(self) -> pd.DataFrame:
        if not os.path.exists(self.battle_report_file):
            print(
                Fore.RED
                + f"[RL-LAB] battle_report.csv not found: {self.battle_report_file}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.battle_report_file)
            print(
                Fore.GREEN
                + f"[RL-LAB] Loaded battle_report: {len(df)} rows."
                + Style.RESET_ALL
            )
            return df
        except Exception as e:
            print(
                Fore.RED
                + f"[RL-LAB] Failed to read battle_report: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 标准化需要的字段
        num_cols = [
            "Net_PnL",
            "Return_pct",
            "Hold_Minutes",
            "MFE_pct",
            "MAE_pct",
            "Future_30M_Max",
            "Future_30M_Min",
            "Entry_future_gain",
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 时间列兜底解析
        if "Entry_Time" in df.columns and not np.issubdtype(
            df["Entry_Time"].dtype, np.datetime64
        ):
            df["Entry_Time"] = pd.to_datetime(df["Entry_Time"], errors="coerce")

        # 标签列
        if "Entry_label" not in df.columns:
            print(
                Fore.RED
                + "[RL-LAB] battle_report 缺少 Entry_label 列，无法做标签 × 风控联动。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        df["Entry_label"] = pd.to_numeric(df["Entry_label"], errors="coerce")

        # 去掉无效行
        df = df.dropna(
            subset=["Entry_label", "Return_pct", "Future_30M_Min"]
        ).copy()

        if df.empty:
            print(
                Fore.RED
                + "[RL-LAB] battle_report 有效行为空，检查是否已正确运行 battle_replay.py。"
                + Style.RESET_ALL
            )
            return df

        print(
            Fore.CYAN
            + f"[RL-LAB] Cleaned battle_report, remaining rows = {len(df)}."
            + Style.RESET_ALL
        )

        return df

    def _assign_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        给每笔交易打上 Regime 标签（与 FactorLab 一致）：
        - 优先从 self.regime_config_file 读取；
        - 否则用日度收益中位数（Return_pct / Entry_pct / Entry_future_gain）三等分。
        """
        if "Entry_Time" not in df.columns:
            print(
                Fore.YELLOW
                + "[RL-LAB] 缺少 Entry_Time 列，无法按日期划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        if not np.issubdtype(df["Entry_Time"].dtype, np.datetime64):
            df["Entry_Time"] = pd.to_datetime(df["Entry_Time"], errors="coerce")

        if df["Entry_Time"].isna().all():
            print(
                Fore.YELLOW
                + "[RL-LAB] Entry_Time 全部无效，无法按日期划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        df["Trade_Date"] = df["Entry_Time"].dt.date

        # 1) 尝试外部 Regime 配置
        if self.regime_config_file and os.path.exists(self.regime_config_file):
            try:
                reg = pd.read_csv(self.regime_config_file)
                date_col = None
                for c in ["trade_date", "date", "Date"]:
                    if c in reg.columns:
                        date_col = c
                        break
                regime_col = None
                for c in ["regime", "Regime", "market_regime"]:
                    if c in reg.columns:
                        regime_col = c
                        break
                if date_col and regime_col:
                    reg["Trade_Date"] = pd.to_datetime(
                        reg[date_col], errors="coerce"
                    ).dt.date
                    reg = reg.dropna(subset=["Trade_Date"])
                    regime_map = dict(
                        zip(reg["Trade_Date"], reg[regime_col].astype(str))
                    )
                    df["Regime"] = df["Trade_Date"].map(regime_map).fillna("Unknown")
                    print(
                        Fore.CYAN
                        + f"[RL-LAB] Regime 映射已从 {self.regime_config_file} 加载。"
                        + Style.RESET_ALL
                    )
                    return df
                else:
                    print(
                        Fore.YELLOW
                        + f"[RL-LAB] {self.regime_config_file} 缺少 date/regime 列，回退到自动 Regime。"
                        + Style.RESET_ALL
                    )
            except Exception as e:
                print(
                    Fore.YELLOW
                    + f"[RL-LAB] 读取 {self.regime_config_file} 失败，回退到自动 Regime: {e}"
                    + Style.RESET_ALL
                )

        # 2) 自动 Regime
        metric_col = None
        for c in ["Return_pct", "Entry_pct", "Entry_future_gain"]:
            if c in df.columns:
                metric_col = c
                break

        if metric_col is None:
            print(
                Fore.YELLOW
                + "[RL-LAB] 无合适列用于自动划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        day_stats = df.groupby("Trade_Date")[metric_col].median().dropna()
        if day_stats.empty:
            print(
                Fore.YELLOW
                + "[RL-LAB] 自动划分行情 Regime 失败（无有效日度统计），全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        q1 = day_stats.quantile(0.33)
        q2 = day_stats.quantile(0.66)

        def _regime_from_val(v: float) -> str:
            if v >= q2:
                return "Bull"
            elif v <= q1:
                return "Bear"
            else:
                return "Range"

        regime_by_date = {d: _regime_from_val(v) for d, v in day_stats.items()}
        df["Regime"] = df["Trade_Date"].map(regime_by_date).fillna("Unknown")

        reg_counts = df["Regime"].value_counts()
        dist_str = ", ".join(f"{k}:{v}" for k, v in reg_counts.items())
        print(
            Fore.CYAN
            + f"[RL-LAB] 已按日度 {metric_col} 自动划分 Regime(Bull/Range/Bear)。样本分布: {dist_str}"
            + Style.RESET_ALL
        )
        return df

    # --------------------------------------------------
    # 构造风险标签
    # --------------------------------------------------
    def _build_risk_flag(
        self, df: pd.DataFrame, threshold: float
    ) -> pd.DataFrame:
        """
        在 df 上新增：
        - Risk_Flag: 1 = Future_30M_Min <= threshold (高风险)
                     0 = 否
        """
        df = df.copy()
        df["Risk_Flag"] = np.where(
            df["Future_30M_Min"] <= threshold, 1, 0
        )
        return df

    # --------------------------------------------------
    # 四象限分析（支持 Regime）
    # --------------------------------------------------
    def _profile_group(self, name: str, sub: pd.DataFrame, regime_label: str) -> dict:
        if sub is None or sub.empty:
            return {
                "Group": name,
                "Regime": regime_label,
                "N": 0,
                "WinRate_pct": 0.0,
                "Avg_Return_pct": 0.0,
                "Med_Return_pct": 0.0,
                "Total_Net_PnL": 0.0,
                "Avg_MFE_pct": 0.0,
                "Avg_MAE_pct": 0.0,
                "Worst_Return_pct": 0.0,
                "Worst_FutureMin_pct": 0.0,
                "Avg_FutureMax_pct": 0.0,
                "Avg_FutureMin_pct": 0.0,
                "Avg_FutureGain_pct": 0.0,
            }

        n = len(sub)
        win_mask = sub["Return_pct"] > 0
        win_rate = float(win_mask.sum()) / n * 100.0

        avg_ret = float(sub["Return_pct"].mean())
        med_ret = float(sub["Return_pct"].median())
        total_pnl = float(sub["Net_PnL"].sum()) if "Net_PnL" in sub.columns else 0.0

        avg_mfe = float(sub["MFE_pct"].mean()) if "MFE_pct" in sub.columns else 0.0
        avg_mae = float(sub["MAE_pct"].mean()) if "MAE_pct" in sub.columns else 0.0

        worst_ret = float(sub["Return_pct"].min())
        worst_fmin = float(sub["Future_30M_Min"].min())

        avg_fmax = float(sub["Future_30M_Max"].mean()) if "Future_30M_Max" in sub.columns else 0.0
        avg_fmin = float(sub["Future_30M_Min"].mean())
        avg_fg = float(sub["Entry_future_gain"].mean()) if "Entry_future_gain" in sub.columns else 0.0

        return {
            "Group": name,
            "Regime": regime_label,
            "N": n,
            "WinRate_pct": round(win_rate, 2),
            "Avg_Return_pct": round(avg_ret, 3),
            "Med_Return_pct": round(med_ret, 3),
            "Total_Net_PnL": round(total_pnl, 2),
            "Avg_MFE_pct": round(avg_mfe, 3),
            "Avg_MAE_pct": round(avg_mae, 3),
            "Worst_Return_pct": round(worst_ret, 3),
            "Worst_FutureMin_pct": round(worst_fmin, 3),
            "Avg_FutureMax_pct": round(avg_fmax, 3),
            "Avg_FutureMin_pct": round(avg_fmin, 3),
            "Avg_FutureGain_pct": round(avg_fg, 3),
        }

    def _analyze_quadrants(self, df: pd.DataFrame) -> pd.DataFrame:
        print(
            Fore.CYAN
            + "\n[RL-LAB] === Quadrant Analysis (Entry_label × Risk_Flag × Regime) ==="
            + Style.RESET_ALL
        )

        regimes = ["ALL"]
        if "Regime" in df.columns:
            regimes += sorted(df["Regime"].dropna().unique().tolist())

        profiles = []

        for regime_label in regimes:
            if regime_label == "ALL":
                base_df = df
            else:
                base_df = df[df["Regime"] == regime_label].copy()
                if base_df.empty:
                    continue

            print(
                Fore.CYAN
                + f"\n[RL-LAB] --- Regime = {regime_label} ---"
                + Style.RESET_ALL
            )

            quad_defs = [
                ("Q1_Alpha_Safe", (base_df["Entry_label"] == 1) & (base_df["Risk_Flag"] == 0)),
                ("Q2_Alpha_Danger", (base_df["Entry_label"] == 1) & (base_df["Risk_Flag"] == 1)),
                ("Q3_NonAlpha_Safe", (base_df["Entry_label"] == 0) & (base_df["Risk_Flag"] == 0)),
                ("Q4_NonAlpha_Danger", (base_df["Entry_label"] == 0) & (base_df["Risk_Flag"] == 1)),
            ]

            for name, mask in quad_defs:
                sub = base_df[mask].copy()
                prof = self._profile_group(name, sub, regime_label)
                profiles.append(prof)

                print(
                    Fore.YELLOW
                    + f"  [{regime_label}] {name}: N={prof['N']}, WinRate={prof['WinRate_pct']:.2f}%, "
                    f"AvgRet={prof['Avg_Return_pct']:.3f}%, "
                    f"WorstRet={prof['Worst_Return_pct']:.3f}%"
                    + Style.RESET_ALL
                )

        return pd.DataFrame(profiles)

    # --------------------------------------------------
    # 风险阈值扫描（支持 Regime）
    # --------------------------------------------------
    def _risk_threshold_sweep(self, df: pd.DataFrame) -> pd.DataFrame:
        print(
            Fore.CYAN
            + "\n[RL-LAB] === Risk Threshold Sweep (for Entry_label=1, by Regime) ==="
            + Style.RESET_ALL
        )

        # 只看有 Alpha 标签的样本
        df_alpha = df[df["Entry_label"] == 1].copy()
        if df_alpha.empty:
            print(
                Fore.RED
                + "[RL-LAB] No Entry_label=1 samples found, skip sweep."
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        thresholds = [-1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
        records = []

        regimes = ["ALL"]
        if "Regime" in df_alpha.columns:
            regimes += sorted(df_alpha["Regime"].dropna().unique().tolist())

        for regime_label in regimes:
            if regime_label == "ALL":
                base_df = df_alpha
            else:
                base_df = df_alpha[df_alpha["Regime"] == regime_label].copy()
                if base_df.empty:
                    continue

            print(
                Fore.MAGENTA
                + f"\n[RL-LAB] [Regime={regime_label}] Risk Threshold Sweep:"
                + Style.RESET_ALL
            )

            for thr in thresholds:
                safe_mask = base_df["Future_30M_Min"] > thr
                safe_sub = base_df[safe_mask].copy()

                if safe_sub.empty:
                    n = 0
                    win_rate = 0.0
                    avg_ret = 0.0
                else:
                    n = len(safe_sub)
                    win_rate = float((safe_sub["Return_pct"] > 0).sum()) / n * 100.0
                    avg_ret = float(safe_sub["Return_pct"].mean())

                records.append(
                    {
                        "Group": f"Sweep_Thr>{thr:.1f}",
                        "Regime": regime_label,
                        "N": n,
                        "WinRate_pct": round(win_rate, 2),
                        "Avg_Return_pct": round(avg_ret, 3),
                        "Med_Return_pct": np.nan,
                        "Total_Net_PnL": np.nan,
                        "Avg_MFE_pct": np.nan,
                        "Avg_MAE_pct": np.nan,
                        "Worst_Return_pct": np.nan,
                        "Worst_FutureMin_pct": np.nan,
                        "Avg_FutureMax_pct": np.nan,
                        "Avg_FutureMin_pct": np.nan,
                        "Avg_FutureGain_pct": np.nan,
                    }
                )

                print(
                    Fore.MAGENTA
                    + f"  [R={regime_label}] Thr>{thr:.1f}% | N={n}, WinRate={win_rate:.2f}%, AvgRet={avg_ret:.3f}%"
                    + Style.RESET_ALL
                )

        return pd.DataFrame(records)

    # --------------------------------------------------
    # 输出 & 控制台总结
    # --------------------------------------------------
    def _save_profile(self, df: pd.DataFrame):
        if df is None or df.empty:
            print(
                Fore.RED
                + "[RL-LAB] No profile to save."
                + Style.RESET_ALL
            )
            return

        try:
            df.to_csv(
                self.output_profile_file,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                Fore.GREEN
                + f"[RL-LAB] Profile saved -> {self.output_profile_file}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[RL-LAB] Failed to save profile: {e}"
                + Style.RESET_ALL
            )

    def _print_summary(self, df: pd.DataFrame):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[RL-LAB] === Summary (by Regime) ==="
            + Style.RESET_ALL
        )
        if df is None or df.empty:
            print(Fore.YELLOW + "  Empty profile." + Style.RESET_ALL)
            return

        core = df[df["Group"].str.startswith("Q")].copy()
        if core.empty:
            print(Fore.YELLOW + "  No quadrant data." + Style.RESET_ALL)
            return

        for regime_label, g in core.groupby("Regime"):
            print(
                Fore.CYAN
                + f"\n  --- Regime = {regime_label} ---"
                + Style.RESET_ALL
            )
            for _, row in g.iterrows():
                print(
                    Fore.WHITE
                    + f"  {row['Group']:<18} | N={int(row['N']):4d} | "
                    f"Win={row['WinRate_pct']:6.2f}% | AvgRet={row['Avg_Return_pct']:7.3f}% | "
                    f"WorstRet={row['Worst_Return_pct']:7.3f}% | WorstFMin={row['Worst_FutureMin_pct']:7.3f}%"
                    + Style.RESET_ALL
                )


if __name__ == "__main__":
    lab = RiskLabelLab()
    lab.run()
