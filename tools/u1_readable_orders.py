# -*- coding: utf-8 -*-
"""
U1 可读版下单建议脚本（T+1 卖出版）

用法示例：
    python -m tools.u1_readable_orders --job-id ultrashort_main --tag u1_v1_base_rf --as-of 20251031

- 不写 --as-of：会自动在 reports/ 下面找到最新日期的 u1_orders_*.csv 来生成建议
- 默认买入时间：as_of 的下一个交易日 09:31:00
- 默认卖出时间：买入日的下一个交易日 14:55:00  （T+1 卖出）
"""

import argparse
from datetime import datetime, date, time, timedelta
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 可读版下单建议（T+1 卖出版）"
    )
    parser.add_argument(
        "--job-id", required=True, help="任务 ID，例如 ultrashort_main"
    )
    parser.add_argument(
        "--tag", required=True, help="模型 / 策略 tag，例如 u1_v1_base_rf"
    )
    parser.add_argument(
        "--as-of",
        help="下单基准日期，格式 YYYYMMDD；不填则自动寻找 reports 下最新一份 u1_orders_*.csv",
    )
    return parser.parse_args()


def find_orders_file(job_id: str, tag: str, as_of: str | None):
    """根据 job_id/tag/as_of 找到对应的 u1_orders_*.csv，并返回 (as_of_str, Path)."""

    reports_dir = Path("reports")

    if as_of:
        as_of_str = as_of
        fname = f"u1_orders_{job_id}_{tag}_{as_of_str}.csv"
        path = reports_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"找不到下单列表文件: {path}")
    else:
        # 自动找最新一份
        pattern = f"u1_orders_{job_id}_{tag}_*.csv"
        files = sorted(reports_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"在 {reports_dir} 下找不到匹配文件: {pattern}"
            )
        path = files[-1]
        # 文件名形如 u1_orders_ultrashort_main_u1_v1_base_rf_20251031.csv
        as_of_str = path.stem.split("_")[-1]

    return as_of_str, path


def next_biz_day(d: date) -> date:
    """简单的“下一个工作日”（周一 ~ 周五），暂不处理法定节假日。"""
    cur = d
    while True:
        cur += timedelta(days=1)
        # weekday: Mon=0 ... Sun=6
        if cur.weekday() < 5:
            return cur


def format_money(x: float) -> str:
    """金额格式化，输出形如 123,456.7"""
    return f"{x:,.1f}"


def main() -> None:
    args = parse_args()

    # 1. 找到下单文件
    as_of_str, orders_path = find_orders_file(args.job_id, args.tag, args.as_of)
    print(f"[U1Readable] 读取下单建议文件: {orders_path}")

    df = pd.read_csv(orders_path)
    if df.empty:
        print("[U1Readable] 下单列表为空，没有可推荐的股票。")
        return

    # 2. 计算买入 / 卖出时间（T+1）
    as_of_date = datetime.strptime(as_of_str, "%Y%m%d").date()
    buy_date = next_biz_day(as_of_date)       # as_of 的下一个交易日
    sell_date = next_biz_day(buy_date)        # 再往后一个交易日 → T+1 卖出

    buy_dt = datetime.combine(buy_date, time(9, 31))
    sell_dt = datetime.combine(sell_date, time(14, 55))

    # 3. 终端展示
    print()
    print(
        f"===== {buy_date.strftime('%Y-%m-%d')} U1 模型（日内超短）下单建议（可读版） ====="
    )
    print(f"任务: job_id={args.job_id}, tag={args.tag}")
    print(f"候选股票数量: {len(df)} （只展示前 50 条）")
    print()

    max_rows = 50
    show_df = df.head(max_rows).copy()

    for idx, row in show_df.iterrows():
        code = str(row.get("code", "")).zfill(6)

        side = str(row.get("side", "BUY")).upper()
        side_label = "[买入]" if side.startswith("B") else "[卖出]"

        weight = row.get("target_weight", row.get("weight", None))
        amount = row.get("target_amount", row.get("amount", row.get("amt", None)))

        # 仓位、金额显示
        if weight is not None and not pd.isna(weight):
            weight_str = f"{float(weight) * 100:.1f}%"
        else:
            weight_str = "N/A"

        if amount is not None and not pd.isna(amount):
            # 这里按“万”为单位展示
            amount_str = format_money(float(amount) / 1e4) + " 万"
        else:
            amount_str = "N/A"

        print(
            f"{idx + 1:02d}. {side_label} {code} | "
            f"建议买入时间: {buy_dt:%Y-%m-%d %H:%M:%S} | "
            f"建议卖出时间: {sell_dt:%Y-%m-%d %H:%M:%S} | "
            f"目标仓位: {weight_str} | 目标金额: {amount_str}"
        )

    print()
    print("（以上为可读版下单建议，仅供科研/模拟使用，真实交易请结合实际行情谨慎操作）")

    # 4. 保存 markdown 版本，方便手机 / 记事本查看
    md_path = Path("reports") / f"u1_readable_orders_{args.job_id}_{args.tag}_{as_of_str}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            f"# {buy_date.strftime('%Y-%m-%d')} U1 模型（日内超短）下单建议（T+1 卖出，可读版）\n\n"
        )
        f.write(f"- 任务: job_id=`{args.job_id}`, tag=`{args.tag}`\n")
        f.write(f"- as_of（信号生成日）: {as_of_date}\n")
        f.write(f"- 建议买入时间: {buy_dt:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- 建议卖出时间: {sell_dt:%Y-%m-%d %H:%M:%S}  （T+1 卖出）\n")
        f.write("\n---\n\n")

        for idx, row in show_df.iterrows():
            code = str(row.get("code", "")).zfill(6)
            side = str(row.get("side", "BUY")).upper()
            side_label = "买入" if side.startswith("B") else "卖出"

            weight = row.get("target_weight", row.get("weight", None))
            amount = row.get("target_amount", row.get("amount", row.get("amt", None)))

            if weight is not None and not pd.isna(weight):
                weight_str = f"{float(weight) * 100:.1f}%"
            else:
                weight_str = "N/A"

            if amount is not None and not pd.isna(amount):
                amount_str = format_money(float(amount) / 1e4) + " 万"
            else:
                amount_str = "N/A"

            f.write(
                f"{idx + 1:02d}. **{side_label} {code}**  | "
                f"仓位: {weight_str} | 金额: {amount_str}\n"
            )

        f.write(
            "\n> 以上为模型信号，不构成任何投资建议，仅供科研 / 模拟回测使用。\n"
        )

    print(f"[U1Readable] 可读版下单建议已保存到 markdown: {md_path}")

    if args.as_of is None:
        print(
            "\n[U1Readable] 提示：如果不写 --as-of，会自动找到最新日期那一份 u1_orders_*.csv"
        )


if __name__ == "__main__":
    main()
