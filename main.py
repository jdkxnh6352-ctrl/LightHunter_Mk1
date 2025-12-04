# -*- coding: utf-8 -*-
"""
Light Hunter Apex - 终极启动入口
版本: Mk-AI Neural-Link R130 (Battle Forge)
路径: G:\\LightHunter_Mk1\\main.py
"""
import sys
from colorama import init, Fore, Style
from backtest_core import GeneticOptimizer  # 遗传回测

init(autoreset=True)


def main():
    print(
        Fore.CYAN
        + Style.BRIGHT
        + """
    ########################################################
    #           LIGHT HUNTER AI (猎光·奇点协议)            #
    #        R130 Singularity - Battle Forge Pipeline      #
    ########################################################
    """
        + Style.RESET_ALL
    )

    print("  [1] 启动实盘作战 (Live Trading)")
    print("  [2] 训练 AI 大脑 (Train Neural Net)")
    print("  [3] 遗传算法回测 (Evolution Backtest)")
    print("  [4] 战斗复盘实验室 (Battle Replay Lab)")
    print("  [5] 一键 TS 标注 + 训练 + GA (TS Label + Train + GA)")
    print("  [6] 退出 (Exit)")
    print("  [7] 夜间科研流水线 (NightOps 全链路盘后科研)")

    choice = input(
        Fore.YELLOW + "\n  >>> Select Protocol [1-7]: " + Style.RESET_ALL
    ).strip()

    if choice == "1":
        try:
            from commander import Commander  # 实盘指挥官

            cm = Commander()
            cm.start_monitor()
        except Exception as e:
            print(f"[ERROR] {e}")
            input()
            main()

    elif choice == "2":
        print(Fore.BLUE + "[*] Initializing Neural Core." + Style.RESET_ALL)
        try:
            from alpha_strategy import CombatBrain  # 神经网络内核

            brain = CombatBrain()
            brain.train_brain()
        except Exception as e:
            print(
                Fore.RED + f"[AI][ERROR] Neural training failed: {e}" + Style.RESET_ALL
            )
        input("\nPress Enter to return.")
        main()

    elif choice == "3":
        print(Fore.BLUE + "[*] Initializing Genetic Engine." + Style.RESET_ALL)
        try:
            optimizer = GeneticOptimizer()
            best_gene = optimizer.run_evolution()
            print(
                Fore.GREEN
                + f"\n[RESULT] Optimal Genome: {best_gene}"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(
                Fore.RED + f"[GA][ERROR] Genetic backtest failed: {e}" + Style.RESET_ALL
            )
        input("\nPress Enter to return.")
        main()

    elif choice == "4":
        print(Fore.BLUE + "[*] Launching Battle Replay Lab." + Style.RESET_ALL)
        try:
            from battle_replay import BattleReplay  # 战斗复盘

            br = BattleReplay()
            df = br.run()
            if df is not None and not df.empty:
                print(
                    Fore.GREEN
                    + f"\n[BR] 战斗报告已生成，交易笔数: {len(df)}"
                    + Style.RESET_ALL
                )
        except Exception as e:
            print(Fore.RED + f"[BR][ERROR] {e}" + Style.RESET_ALL)
        input("\nPress Enter to return.")
        main()

    elif choice == "5":
        print(
            Fore.BLUE
            + "[*] Launching TS Label + Neural Train + GA Pipeline."
            + Style.RESET_ALL
        )
        try:
            from train_ga_pipeline import TrainingOrchestrator

            orchestrator = TrainingOrchestrator()
            orchestrator.run_all()
        except Exception as e:
            print(Fore.RED + f"[PIPE][ERROR] {e}" + Style.RESET_ALL)
        input("\nPress Enter to return.")
        main()

    elif choice == "7":
        print(
            Fore.BLUE
            + "[*] Launching NightOps Full Research Pipeline."
            + Style.RESET_ALL
        )
        try:
            from night_ops import NightOps  # 盘后科研流水线

            ops = NightOps()
            ops.run_all()
        except Exception as e:
            print(
                Fore.RED + f"[NIGHT][ERROR] {e}" + Style.RESET_ALL
            )
        input("\nPress Enter to return.")
        main()

    elif choice == "6":
        sys.exit()

    else:
        print(
            Fore.RED
            + "  [!] Invalid choice. Please select 1-7."
            + Style.RESET_ALL
        )
        input("\nPress Enter to retry.")
        main()


if __name__ == "__main__":
    main()
