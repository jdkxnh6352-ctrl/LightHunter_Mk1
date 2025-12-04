# -*- coding: utf-8 -*-
"""
模块名称：FactorLab Mk-Research
版本：Mk-Research R20 (Factor Diagnostics + Regime Profile)
路径: G:/LightHunter_Mk1/factor_lab.py

功能：
- 读取 battle_report.csv（BattleReplay 输出的每笔完整交易）；
- 按多个核心因子分桶统计：
  * Entry_Z_Force（进场 Z 力度）
  * Entry_future_gain（TS 未来收益标签）
  * Entry_NN_Prob（神经网络置信度）
  * Entry_Final_Score（总评分）
  * Entry_label（TS 好/坏样本标签）
- 新增：按行情 Regime（牛市 / 震荡 / 市况差）做剖面：
  * 优先从 market_regimes.csv 读取日期→regime 映射；
  * 如无配置文件，则自动按“日度收益中位数”三等分为 Bull / Range / Bear；
- 输出：
  * 控制台汇总（总盈亏、胜率、Top/Bottom 因子桶）；
  * factor_profile.csv（多一个 Regime 列，可按行情 regime 过滤分析）。
"""

import os
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)


class FactorLab:
    def __init__(
        self,
        battle_file: str = "battle_report.csv",
        output_file: str = "factor_profile.csv",
        regime_config_file: str = "market_regimes.csv",
    ):
        self.battle_file = battle_file
        self.output_file = output_file
        # 行情 Regime 配置文件（可选）：
        # 期望列: trade_date/date, regime
        self.regime_config_file = regime_config_file

    # --------------------------------------------------
    # 主入口
    # --------------------------------------------------
    def run(self) -> pd.DataFrame:
        print(
            Fore.CYAN
            + "[FL] Factor Research Lab Initializing..."
            + Style.RESET_ALL
        )

        if not os.path.exists(self.battle_file):
            print(
                Fore.RED
                + f"[FL] {self.battle_file} 不存在，请先运行 battle_replay.py 生成战斗报告。"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(
                self.battle_file,
                parse_dates=["Entry_Time", "Exit_Time"],
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[FL] 读取 {self.battle_file} 失败: {e}"
                + Style.RESET_ALL
            )
            return pd.DataFrame()

        if df.empty:
            print(
                Fore.RED
                + "[FL] battle_report.csv 为空，暂无可分析交易。"
                + Style.RESET_ALL
            )
            return df

        df = self._clean_df(df)
        df = self._assign_regime(df)
        self._print_global_stats(df)

        profiles = []

        # 准备 Regime 列表：ALL + 各 regime
        regimes = ["ALL"]
        if "Regime" in df.columns:
            regimes += sorted(df["Regime"].dropna().unique().tolist())

        # 逐 Regime 进行因子剖面
        for regime_label in regimes:
            if regime_label == "ALL":
                sub_df = df
            else:
                sub_df = df[df["Regime"] == regime_label].copy()
                if sub_df.empty:
                    continue
                print(
                    Fore.CYAN
                    + f"\n[FL] === Regime = {regime_label} 因子剖面 ==="
                    + Style.RESET_ALL
                )

            # 1) Z_Force 进场力度分桶
            if "Entry_Z_Force" in sub_df.columns:
                profiles += self._analyze_factor(
                    sub_df,
                    col="Entry_Z_Force",
                    bins=[-999, 0, 1, 2, 3, 999],
                    labels=[
                        "Z<=0",
                        "0<Z<=1",
                        "1<Z<=2",
                        "2<Z<=3",
                        "Z>3",
                    ],
                    regime_label=regime_label,
                )

            # 2) 未来收益标签（TS_Label / future_gain）
            if "Entry_future_gain" in sub_df.columns:
                profiles += self._analyze_factor(
                    sub_df,
                    col="Entry_future_gain",
                    bins=[-999, 0, 3, 6, 10, 999],
                    labels=[
                        "FG<=0",
                        "0<FG<=3",
                        "3<FG<=6",
                        "6<FG<=10",
                        "FG>10",
                    ],
                    regime_label=regime_label,
                )

            # 3) 神经网络置信度
            if "Entry_NN_Prob" in sub_df.columns:
                profiles += self._analyze_factor(
                    sub_df,
                    col="Entry_NN_Prob",
                    bins=[-0.01, 0.4, 0.7, 0.85, 0.93, 1.01],
                    labels=[
                        "P<=0.4",
                        "0.4<P<=0.7",
                        "0.7<P<=0.85",
                        "0.85<P<=0.93",
                        "P>0.93",
                    ],
                    regime_label=regime_label,
                )

            # 4) 综合评分 Final_Score
            if "Entry_Final_Score" in sub_df.columns:
                q = sub_df["Entry_Final_Score"].quantile
                cuts = sorted(
                    list(
                        set(
                            [
                                sub_df["Entry_Final_Score"].min() - 1,
                                float(q(0.2)),
                                float(q(0.4)),
                                float(q(0.6)),
                                float(q(0.8)),
                                sub_df["Entry_Final_Score"].max() + 1,
                            ]
                        )
                    )
                )
                labels = [
                    "Score Q1",
                    "Score Q2",
                    "Score Q3",
                    "Score Q4",
                    "Score Q5",
                ]
                if len(cuts) == 6:
                    profiles += self._analyze_factor(
                        sub_df,
                        col="Entry_Final_Score",
                        bins=cuts,
                        labels=labels,
                        regime_label=regime_label,
                    )

            # 5) TS label（0/1）
            if "Entry_label" in sub_df.columns:
                profiles += self._analyze_label_factor(
                    sub_df,
                    "Entry_label",
                    regime_label=regime_label,
                )

        profile_df = pd.DataFrame(profiles)
        if not profile_df.empty:
            self._save_profile(profile_df)
            self._print_best_buckets(profile_df)

        # 相关性矩阵简单打印（全样本维度）
        self._print_correlations(df)

        return profile_df

    # --------------------------------------------------
    # 清洗 & Regime 赋值 & 全局统计
    # --------------------------------------------------
    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 基本字段缺失处理
        numeric_cols = [
            "Net_PnL",
            "Return_pct",
            "Hold_Minutes",
            "MFE_pct",
            "MAE_pct",
            "Entry_Z_Force",
            "Entry_future_gain",
            "Entry_NN_Prob",
            "Entry_Final_Score",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["Hold_Minutes"] = df.get("Hold_Minutes", 0).fillna(0).astype(int)

        # Entry_Time 若未解析，在这里做一次兜底
        if "Entry_Time" in df.columns and not np.issubdtype(
            df["Entry_Time"].dtype, np.datetime64
        ):
            df["Entry_Time"] = pd.to_datetime(df["Entry_Time"], errors="coerce")

        return df

    def _assign_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        给每笔交易打上 Regime 标签：
        - 优先从 self.regime_config_file 读取；
        - 否则用日度收益中位数自动三等分为 Bull / Range / Bear；
        - 无法判断时，统一标记为 'ALL'。
        """
        if "Entry_Time" not in df.columns:
            print(
                Fore.YELLOW
                + "[FL] battle_report 缺少 Entry_Time 列，无法按日期划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        if not np.issubdtype(df["Entry_Time"].dtype, np.datetime64):
            df["Entry_Time"] = pd.to_datetime(df["Entry_Time"], errors="coerce")

        if df["Entry_Time"].isna().all():
            print(
                Fore.YELLOW
                + "[FL] Entry_Time 全部无效，无法按日期划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        df["Trade_Date"] = df["Entry_Time"].dt.date

        # 1) 尝试外部配置文件：market_regimes.csv
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
                        + f"[FL] Regime 映射已从 {self.regime_config_file} 加载。"
                        + Style.RESET_ALL
                    )
                    return df
                else:
                    print(
                        Fore.YELLOW
                        + f"[FL] {self.regime_config_file} 缺少 date/regime 列，回退到自动 Regime 划分。"
                        + Style.RESET_ALL
                    )
            except Exception as e:
                print(
                    Fore.YELLOW
                    + f"[FL] 读取 {self.regime_config_file} 失败，回退到自动 Regime 划分: {e}"
                    + Style.RESET_ALL
                )

        # 2) 自动 Regime：按日度中位 Return_pct / Entry_pct / Entry_future_gain 划分
        metric_col = None
        for c in ["Return_pct", "Entry_pct", "Entry_future_gain"]:
            if c in df.columns:
                metric_col = c
                break

        if metric_col is None:
            print(
                Fore.YELLOW
                + "[FL] 无合适列用于自动划分行情 Regime，全部视为 'ALL'。"
                + Style.RESET_ALL
            )
            df["Regime"] = "ALL"
            return df

        day_stats = df.groupby("Trade_Date")[metric_col].median().dropna()
        if day_stats.empty:
            print(
                Fore.YELLOW
                + "[FL] 自动划分行情 Regime 失败（无有效日度统计），全部视为 'ALL'。"
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
            + f"[FL] 已按日度 {metric_col} 自动划分 Regime(Bull/Range/Bear)。样本分布: {dist_str}"
            + Style.RESET_ALL
        )
        return df

    def _print_global_stats(self, df: pd.DataFrame):
        total_trades = len(df)
        total_pnl = float(df["Net_PnL"].sum()) if "Net_PnL" in df.columns else 0.0
        wins = df[df["Net_PnL"] > 0] if "Net_PnL" in df.columns else df.iloc[0:0]
        win_rate = (
            len(wins) / total_trades * 100 if total_trades > 0 else 0.0
        )

        avg_ret = float(df["Return_pct"].mean()) if "Return_pct" in df.columns else 0.0
        med_ret = float(df["Return_pct"].median()) if "Return_pct" in df.columns else 0.0
        avg_hold = float(df["Hold_Minutes"].mean()) if "Hold_Minutes" in df.columns else 0.0

        mfe_mean = (
            float(df["MFE_pct"].mean()) if "MFE_pct" in df.columns else np.nan
        )
        mae_mean = (
            float(df["MAE_pct"].mean()) if "MAE_pct" in df.columns else np.nan
        )

        pnl_color = Fore.RED if total_pnl >= 0 else Fore.GREEN

        print(
            Fore.YELLOW
            + "\n================= GLOBAL TRADE STATS ================="
            + Style.RESET_ALL
        )
        print(f"  总交易笔数 : {total_trades}")
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
            f"  总净盈亏   : {pnl_color}{total_pnl:.2f}{Style.RESET_ALL}"
        )

    # --------------------------------------------------
    # 单因子分桶分析（支持 Regime）
    # --------------------------------------------------
    def _analyze_factor(
        self,
        df: pd.DataFrame,
        col: str,
        bins,
        labels,
        regime_label: str = "ALL",
    ):
        sub = df[~df[col].isna()].copy()
        if sub.empty:
            return []

        sub["bucket"] = pd.cut(
            sub[col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        profiles = []
        for bucket, g in sub.groupby("bucket"):
            if g.empty:
                continue
            n = len(g)
            win_rate = (
                (g["Net_PnL"] > 0).mean() * 100 if "Net_PnL" in g.columns else np.nan
            )
            avg_ret = (
                g["Return_pct"].mean() if "Return_pct" in g.columns else np.nan
            )
            med_ret = (
                g["Return_pct"].median() if "Return_pct" in g.columns else np.nan
            )
            mfe = (
                g["MFE_pct"].mean() if "MFE_pct" in g.columns else np.nan
            )
            mae = (
                g["MAE_pct"].mean() if "MAE_pct" in g.columns else np.nan
            )

            profiles.append(
                {
                    "Factor": col,
                    "Bucket": str(bucket),
                    "Regime": regime_label,
                    "Trades": int(n),
                    "WinRate_pct": round(win_rate, 2)
                    if not np.isnan(win_rate)
                    else np.nan,
                    "Avg_Return_pct": round(avg_ret, 3)
                    if not np.isnan(avg_ret)
                    else np.nan,
                    "Med_Return_pct": round(med_ret, 3)
                    if not np.isnan(med_ret)
                    else np.nan,
                    "Avg_MFE_pct": round(mfe, 3)
                    if not np.isnan(mfe)
                    else np.nan,
                    "Avg_MAE_pct": round(mae, 3)
                    if not np.isnan(mae)
                    else np.nan,
                }
            )
        print(
            Fore.CYAN
            + f"[FL] 因子 {col} ({regime_label}) 已完成分桶分析。"
            + Style.RESET_ALL
        )
        return profiles

    def _analyze_label_factor(
        self, df: pd.DataFrame, col: str, regime_label: str = "ALL"
    ):
        sub = df[~df[col].isna()].copy()
        if sub.empty:
            return []
        profiles = []

        for label_val, g in sub.groupby(col):
            n = len(g)
            win_rate = (
                (g["Net_PnL"] > 0).mean() * 100 if "Net_PnL" in g.columns else np.nan
            )
            avg_ret = (
                g["Return_pct"].mean() if "Return_pct" in g.columns else np.nan
            )
            med_ret = (
                g["Return_pct"].median() if "Return_pct" in g.columns else np.nan
            )

            profiles.append(
                {
                    "Factor": col,
                    "Bucket": f"Label={int(label_val)}",
                    "Regime": regime_label,
                    "Trades": int(n),
                    "WinRate_pct": round(win_rate, 2)
                    if not np.isnan(win_rate)
                    else np.nan,
                    "Avg_Return_pct": round(avg_ret, 3)
                    if not np.isnan(avg_ret)
                    else np.nan,
                    "Med_Return_pct": round(med_ret, 3)
                    if not np.isnan(med_ret)
                    else np.nan,
                    "Avg_MFE_pct": np.nan,
                    "Avg_MAE_pct": np.nan,
                }
            )

        print(
            Fore.CYAN
            + f"[FL] Label 因子 {col} ({regime_label}) 已完成分档分析。"
            + Style.RESET_ALL
        )
        return profiles

    # --------------------------------------------------
    # 保存 & 输出结果
    # --------------------------------------------------
    def _save_profile(self, df: pd.DataFrame):
        try:
            df.to_csv(
                self.output_file, index=False, encoding="utf-8-sig"
            )
            print(
                Fore.GREEN
                + f"\n[FL] 因子档位统计已保存 -> {self.output_file}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED
                + f"[FL] 保存 {self.output_file} 失败: {e}"
                + Style.RESET_ALL
            )

    def _print_best_buckets(self, df: pd.DataFrame):
        print(
            Fore.YELLOW
            + "\n================= TOP FACTOR BUCKETS ================="
            + Style.RESET_ALL
        )
        df2 = df.dropna(subset=["Avg_Return_pct", "Trades"]).copy()
        df2 = df2[df2["Trades"] >= 3]  # 至少 3 笔交易才有统计意义

        if df2.empty:
            print(
                Fore.YELLOW
                + "  暂无足够样本的因子档位。"
                + Style.RESET_ALL
            )
            return

        top = df2.sort_values(
            "Avg_Return_pct", ascending=False
        ).head(15)

        for _, r in top.iterrows():
            regime = r.get("Regime", "ALL")
            print(
                f"  [{regime:<6}] {r['Factor']:<18} {str(r['Bucket']):<12} | "
                f"N={int(r['Trades']):<3} | "
                f"Win={r['WinRate_pct']:.1f}% | "
                f"AvgRet={r['Avg_Return_pct']:.2f}%"
            )

    def _print_correlations(self, df: pd.DataFrame):
        factors = []
        for c in [
            "Entry_Z_Force",
            "Entry_future_gain",
            "Entry_NN_Prob",
            "Entry_Final_Score",
        ]:
            if c in df.columns:
                factors.append(c)

        if "Return_pct" not in df.columns or not factors:
            return

        sub = df[["Return_pct"] + factors].dropna()
        if len(sub) < 10:
            return

        corr = sub.corr()["Return_pct"].drop("Return_pct")
        print(
            Fore.YELLOW
            + "\n================= CORRELATIONS (Global) ================="
            + Style.RESET_ALL
        )
        for name, val in corr.items():
            color = (
                Fore.RED
                if val > 0.15
                else (Fore.GREEN if val < -0.15 else Fore.WHITE)
            )
            print(
                f"  Corr(Return_pct, {name:<16}) = "
                f"{color}{val:+.3f}{Style.RESET_ALL}"
            )


if __name__ == "__main__":
    lab = FactorLab()
    lab.run()
