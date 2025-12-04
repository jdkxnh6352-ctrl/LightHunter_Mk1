# -*- coding: utf-8 -*-
"""
U2 候选股过滤脚本
-----------------

用法示例（在 venv 已激活、当前目录为项目根 G:\LightHunter_Mk1 时）：

    python tools\u2_filter_candidates.py ^
        --input reports\u2_daily_raw_20251128.csv ^
        --output reports\u2_daily_filtered_20251128.csv

你只需要把 --input 换成 u2_daily_scoring 脚本实际生成的 CSV 路径即可。
"""

import argparse
import datetime as dt
import os
from typing import List, Optional

import pandas as pd

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    HAS_COLOR = True
except Exception:
    HAS_COLOR = False


def _c(msg: str, color: str) -> str:
    if HAS_COLOR:
        return color + msg + Style.RESET_ALL
    return msg


# -----------------------------
# 一些工具函数
# -----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    在 df 里按候选名列表找第一列存在的列名，没有就返回 None。
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# 过滤主逻辑
# -----------------------------
def apply_filters(
    df: pd.DataFrame,
    max_n: int = 80,
) -> pd.DataFrame:
    """
    按预设规则过滤 U2 候选股。

    规则：
      1) 剔除 ST / *ST / 退市
      2) 价格区间 8 ~ 80 元（有价格列才生效）
      3) 流通市值区间 50e8 ~ 800e8（有市值列才生效）
      4) 成交额 / 换手等流动性条件（有对应列才生效）
      5) 按 U2 得分排序，最多保留前 max_n 只
    """

    df = df.copy()

    # 1) 剔除 ST / 退市
    name_col = pick_col(df, ["name", "名称", "sec_name"])
    if name_col is not None:
        mask_st = (
            df[name_col].astype(str).str.contains("ST")
            | df[name_col].astype(str).str.contains("退")
        )
        before = len(df)
        df = df[~mask_st].copy()
        print(
            _c(
                f"[U2-FILTER] 剔除 ST / 退市：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )
    else:
        print(_c("[U2-FILTER] 找不到名称列，跳过 ST 过滤。", Fore.YELLOW if HAS_COLOR else ""))

    # 2) 价格区间
    price_col = pick_col(df, ["close", "收盘价", "现价", "price"])
    if price_col is not None:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        before = len(df)
        df = df[(df[price_col] >= 8.0) & (df[price_col] <= 80.0)].copy()
        print(
            _c(
                f"[U2-FILTER] 价格 8~80 元：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )
    else:
        print(_c("[U2-FILTER] 找不到价格列，跳过价格过滤。", Fore.YELLOW if HAS_COLOR else ""))

    # 3) 流通市值区间
    mcap_col = pick_col(df, ["float_mktcap", "circulating_mv", "流通市值", "mktcap"])
    if mcap_col is not None:
        df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
        before = len(df)
        df = df[(df[mcap_col] >= 50e8) & (df[mcap_col] <= 800e8)].copy()
        print(
            _c(
                f"[U2-FILTER] 流通市值 50~800 亿：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )
    else:
        print(_c("[U2-FILTER] 找不到流通市值列，跳过市值过滤。", Fore.YELLOW if HAS_COLOR else ""))

    # 4) 成交额 / 换手 之类的流动性过滤（有就用，没有就算了）
    amt_col = pick_col(df, ["amount", "成交额", "amt"])
    if amt_col is not None:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
        before = len(df)
        df = df[df[amt_col] >= 3e8].copy()
        print(
            _c(
                f"[U2-FILTER] 当日成交额 ≥ 3 亿：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )

    amt20_col = pick_col(df, ["amt_mean_20", "amt20", "avg_amount_20"])
    if amt20_col is not None:
        df[amt20_col] = pd.to_numeric(df[amt20_col], errors="coerce")
        before = len(df)
        df = df[df[amt20_col] >= 2e8].copy()
        print(
            _c(
                f"[U2-FILTER] 20 日平均成交额 ≥ 2 亿：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )

    turn20_col = pick_col(df, ["turn_20", "vol_20", "turnover_20"])
    if turn20_col is not None:
        df[turn20_col] = pd.to_numeric(df[turn20_col], errors="coerce")
        before = len(df)
        df = df[df[turn20_col] >= 1.5].copy()
        print(
            _c(
                f"[U2-FILTER] 20 日平均换手率 ≥ 1.5%：{before} -> {len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )

    # 5) 按 U2 得分排序，只取前 max_n
    score_col = pick_col(df, ["u2_score", "score", "prob", "pred_prob"])
    if score_col is not None:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.sort_values(score_col, ascending=False)
    else:
        print(_c("[U2-FILTER] 找不到得分列，只按原顺序输出。", Fore.YELLOW if HAS_COLOR else ""))

    if max_n is not None and max_n > 0 and len(df) > max_n:
        df = df.head(max_n).copy()
        print(
            _c(
                f"[U2-FILTER] 只保留前 {max_n} 只：{len(df)}",
                Fore.CYAN if HAS_COLOR else "",
            )
        )

    df.reset_index(drop=True, inplace=True)
    return df


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U2 日常候选股过滤器")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="u2_daily_scoring 生成的原始 CSV 路径",
    )
    p.add_argument(
        "--output",
        type=str,
        required=False,
        help="过滤后的 CSV 输出路径（默认在同目录加 _filtered 后缀）",
    )
    p.add_argument(
        "--max-n",
        type=int,
        default=80,
        help="最多保留多少只股票（默认 80）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = args.input
    if not os.path.exists(in_path):
        print(_c(f"[U2-FILTER][ERROR] 找不到输入文件：{in_path}", Fore.RED if HAS_COLOR else ""))
        return

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_filtered" + (ext or ".csv")

    print(_c(f"[U2-FILTER] 读取原始候选股：{in_path}", Fore.GREEN if HAS_COLOR else ""))

    df = pd.read_csv(in_path)
    if df.empty:
        print(_c("[U2-FILTER][WARN] 输入 CSV 为空，直接退出。", Fore.YELLOW if HAS_COLOR else ""))
        return

    print(_c(f"[U2-FILTER] 原始样本数：{len(df)}", Fore.GREEN if HAS_COLOR else ""))

    df_filtered = apply_filters(df, max_n=int(args.max_n))

    print(
        _c(
            f"[U2-FILTER] 过滤后剩余：{len(df_filtered)} 条，保存到：{out_path}",
            Fore.GREEN if HAS_COLOR else "",
        )
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_filtered.to_csv(out_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
