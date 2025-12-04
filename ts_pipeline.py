# -*- coding: utf-8 -*-
"""
模块名称：TSPipeline Mk-BrainForge
版本：Mk-BrainForge R20 (TS Label + GA + Repair)
路径: G:/LightHunter_Mk1/ts_pipeline.py

功能：
- 一键跑通整条科研链路：
  1) DataGuardian 体检数据资产 + TSDataRepairer 自动修复缺数据；
  2) TSLabeler 基于 ts_data.db.snapshots 为 market_blackbox 打未来20分钟标签；
  3) CombatBrain 训练神经网络大脑 (hunter_brain.pkl)；
  4) GeneticOptimizer 遗传进化出 gene_config.json，Commander 实盘自动加载。
"""

import traceback
from colorama import init, Fore, Style

init(autoreset=True)


class TSPipeline:
    def __init__(
        self,
        ts_db_path: str = "ts_data.db",
        blackbox_file: str = "market_blackbox.csv",
        labeled_file: str = "market_blackbox_labeled.csv",
    ):
        self.ts_db_path = ts_db_path
        self.blackbox_file = blackbox_file
        self.labeled_file = labeled_file

    # --------------------------------------------------
    # 总控入口
    # --------------------------------------------------
    def run_full_pipeline(self):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + """
################################################
#       LIGHT HUNTER - TS Brain Forge          #
#     Mk-BrainForge R20 (TS + GA Core)         #
################################################
"""
            + Style.RESET_ALL
        )

        # 1) 数据体检（黑匣子 & ts_data & 交易记录）+ 缺数据修复
        self._stage_data_guardian()

        # 2) TS 打标签：给黑匣子每一行打“未来20分钟最高涨幅”标签
        self._stage_ts_label()

        # 3) 训练 AI 大脑：CombatBrain，优先吃 labeled 文件
        self._stage_train_brain()

        # 4) 遗传进化：用带标签黑匣子做 GA，进化出 gene_config.json
        self._stage_genetic_evolution()

        print(
            Fore.GREEN
            + Style.BRIGHT
            + "\n[PIPE] 全流程完成：TS 标签 + AI 训练 + GA 基因档案 已就绪。"
            + Style.RESET_ALL
        )
        print(
            Fore.YELLOW
            + "       下次启动实盘 (菜单 1) 时，Commander 会自动加载最新 gene_config 权重。\n"
            + Style.RESET_ALL
        )

    # --------------------------------------------------
    # 阶段 1：数据体检 + 缺数据自动修复
    # --------------------------------------------------
    def _stage_data_guardian(self):
        try:
            from data_guardian import DataGuardian
        except ImportError:
            print(
                Fore.YELLOW
                + "[PIPE] 未找到 data_guardian.py，跳过数据体检阶段。"
                + Style.RESET_ALL
            )
            return

        # 尝试导入 TSDataRepairer（可选）
        try:
            from ts_data_repair import TSDataRepairer  # type: ignore
            has_repairer = True
        except ImportError:
            TSDataRepairer = None  # type: ignore
            has_repairer = False

        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[STAGE 1] DataGuardian - 数据资产体检 & TS 缺数据修复 ."
            + Style.RESET_ALL
        )

        summary = None
        try:
            guardian = DataGuardian(
                blackbox_file=self.blackbox_file,
                ts_db_path=self.ts_db_path,
                trade_history_file="trade_history.csv",
                account_file="Hunter_Account.json",
                report_file="data_quality_report.json",
                market_ts_db="market_ts.db",
            )
            summary = guardian.run_full_check()
        except Exception:
            print(
                Fore.RED
                + "[PIPE][STAGE1] DataGuardian 执行异常："
                + Style.RESET_ALL
            )
            print(traceback.format_exc())

        # 若没有 Repairer 模块，体检做到这里就结束
        if not has_repairer:
            print(
                Fore.YELLOW
                + "[PIPE][STAGE1] 未找到 ts_data_repair.py，仅生成体检报告，不做自动修复。"
                + Style.RESET_ALL
            )
            return

        # 触发 TSDataRepairer
        try:
            repairer = TSDataRepairer(
                ts_db=self.ts_db_path,
                market_ts_db="market_ts.db",
                ts_dataset_dir="ts_datasets",
                report_file="ts_repair_report.json",
            )
            if summary is not None:
                repairer.attach_guardian_summary(summary)
                repairer.repair_all(auto_guardian=False)
            else:
                # 体检失败时，修复器内部再尝试一次 DataGuardian
                repairer.repair_all(auto_guardian=True)
        except Exception:
            print(
                Fore.RED
                + "[PIPE][STAGE1] TSDataRepairer 执行异常："
                + Style.RESET_ALL
            )
            print(traceback.format_exc())

    # --------------------------------------------------
    # 阶段 2：TS 打标签
    # --------------------------------------------------
    def _stage_ts_label(self):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[STAGE 2] TSLabeler - 基于 ts_data.db 打未来20分钟标签 ."
            + Style.RESET_ALL
        )
        try:
            from ts_labeler import TSLabeler
        except ImportError:
            print(
                Fore.RED
                + "[PIPE][STAGE2] 未找到 ts_labeler.py，无法进行 TS 打标签。"
                + Style.RESET_ALL
            )
            return

        try:
            labeler = TSLabeler(
                ts_db_path=self.ts_db_path,
                blackbox_file=self.blackbox_file,
                output_file=self.labeled_file,
                horizon_minutes=20,
                gain_threshold=3.0,
            )
            df_labeled = labeler.run()
            if df_labeled is None or df_labeled.empty:
                print(
                    Fore.YELLOW
                    + "[PIPE][STAGE2] 标注结果为空，后续 AI 训练将退回使用原始黑匣子。"
                    + Style.RESET_ALL
                )
            else:
                print(
                    Fore.GREEN
                    + f"[PIPE][STAGE2] 完成 TS 打标签，样本数: {len(df_labeled)}."
                    + Style.RESET_ALL
                )
        except Exception:
            print(
                Fore.RED
                + "[PIPE][STAGE2] TSLabeler 执行异常："
                + Style.RESET_ALL
            )
            print(traceback.format_exc())

    # --------------------------------------------------
    # 阶段 3：训练 AI 大脑
    # --------------------------------------------------
    def _stage_train_brain(self):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[STAGE 3] CombatBrain - AI 大脑训练 ."
            + Style.RESET_ALL
        )
        try:
            from alpha_strategy import CombatBrain
        except ImportError:
            print(
                Fore.RED
                + "[PIPE][STAGE3] 未找到 alpha_strategy.py，无法训练 CombatBrain。"
                + Style.RESET_ALL
            )
            return

        try:
            brain = CombatBrain()
            # CombatBrain 内部会优先使用 market_blackbox_labeled.csv
            brain.train_brain(csv_path=self.blackbox_file)
        except Exception:
            print(
                Fore.RED
                + "[PIPE][STAGE3] CombatBrain 训练异常："
                + Style.RESET_ALL
            )
            print(traceback.format_exc())

    # --------------------------------------------------
    # 阶段 4：遗传进化 GA
    # --------------------------------------------------
    def _stage_genetic_evolution(self):
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "\n[STAGE 4] GeneticOptimizer - GA 基因进化 ."
            + Style.RESET_ALL
        )
        try:
            from backtest_core import GeneticOptimizer
        except ImportError:
            print(
                Fore.RED
                + "[PIPE][STAGE4] 未找到 backtest_core.py，无法运行 GA 遗传进化。"
                + Style.RESET_ALL
            )
            return

        try:
            optimizer = GeneticOptimizer(
                csv_file=self.blackbox_file,
                labeled_file=self.labeled_file,
            )
            optimizer.run_evolution()
        except Exception:
            print(
                Fore.RED
                + "[PIPE][STAGE4] GeneticOptimizer 执行异常："
                + Style.RESET_ALL
            )
            print(traceback.format_exc())


def main():
    pipe = TSPipeline()
    pipe.run_full_pipeline()


if __name__ == "__main__":
    main()
