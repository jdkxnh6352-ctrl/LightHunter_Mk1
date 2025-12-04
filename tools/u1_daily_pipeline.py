# tools/u1_daily_pipeline.py
"""
U1 日内一键流水线：
1）调用日打分引擎 tools.u1_daily_scoring_ml
2）根据 topK 打分 + 当前持仓，生成下单列表

使用示例（在项目根目录）：
    python -m tools.u1_daily_pipeline ^
        --job-id ultrashort_main ^
        --as-of 2025-10-31 ^
        --tag u1_v1_base_rf ^
        --top-k 3 ^
        --min-price 3 ^
        --max-price 80 ^
        --min-amount 20000000 ^
        --positions-csv data/live/positions_20251030.csv ^
        --output-orders reports/u1_orders_ultrashort_main_u1_v1_base_rf_20251031.csv
"""

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"


# ---------- 工具函数 ----------

def _date_to_str(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def build_scores_paths(job_id: str, tag: str, as_of: dt.date):
    """根据之前约定的命名规则，推导 full/top3 路径。"""
    ds = _date_to_str(as_of)
    base = f"u1_scores_{job_id}_{tag}_{ds}"
    full = REPORT_DIR / f"{base}_full.csv"
    top3 = REPORT_DIR / f"{base}_top3.csv"
    return full, top3


# ---------- 第 1 步：调用日打分引擎 ----------

def run_daily_scoring_ml(
    job_id: str,
    as_of: dt.date,
    tag: str,
    top_k: int,
    min_price: float,
    max_price: float,
    min_amount: float,
) -> None:
    """
    通过 subprocess 调用已有的 tools.u1_daily_scoring_ml 模块。
    这样不用改你原来的打分脚本。
    """
    cmd = [
        sys.executable,
        "-m",
        "tools.u1_daily_scoring_ml",
        "--job-id",
        job_id,
        "--as-of",
        as_of.strftime("%Y-%m-%d"),
        "--top-k",
        str(top_k),
        "--min-price",
        str(min_price),
        "--max-price",
        str(max_price),
        "--min-amount",
        str(int(min_amount)),
        "--tag",
        tag,
    ]

    print("[PIPELINE] 调用日打分引擎：", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError("tools.u1_daily_scoring_ml 运行失败，请检查上面的报错。")

    print("[PIPELINE] 日打分完成。")


# ---------- 第 2 步：根据打分 + 持仓生成下单列表 ----------

def load_positions(path: Path) -> Dict[str, float]:
    """
    简单读取当前持仓文件：
    要求至少有两列：code, position
    你可以按自己实际格式调整这里的解析逻辑。
    """
    positions: Dict[str, float] = {}
    if not path.exists():
        print(f"[PIPELINE] 未找到持仓文件 {path} ，默认当前空仓。")
        return positions

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row["code"]).strip()
            pos = float(row.get("position", 0.0))
            positions[code] = pos

    print(f"[PIPELINE] 已读取持仓 {len(positions)} 只股票。")
    return positions


def load_topk_scores(path: Path) -> List[Dict[str, str]]:
    """
    读取 topK 打分结果：
    默认假设有列：code, close, amount, u1_score（根据你现有 CSV 调整）
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到 topK 打分文件：{path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[PIPELINE] 已读取 topK 打分记录 {len(rows)} 条。")
    return rows


def build_order_list(
    scores: List[Dict[str, str]],
    positions: Dict[str, float],
    target_weight: float,
) -> List[Dict[str, str]]:
    """
    非常简单的下单逻辑示例：

    - 对 topK 标的，统一目标权重 = target_weight（如 0.03 = 3%）
    - 其他持仓，目标权重 = 0（全部卖出）
    - 暂时只输出「目标权重」，成交股数在券商侧用现金和价格再算。

    你可以以后再迭代成更复杂的仓位控制逻辑。
    """
    codes_top = [str(r["code"]).strip() for r in scores]

    orders: List[Dict[str, str]] = []

    # 1）先给 topK 标的生成“买入 / 加仓”目标
    for row in scores:
        code = str(row["code"]).strip()
        cur_pos = positions.get(code, 0.0)
        tgt_weight = target_weight
        orders.append(
            {
                "code": code,
                "current_pos": f"{cur_pos:.4f}",
                "target_weight": f"{tgt_weight:.4f}",
                "action": "BUY" if tgt_weight > cur_pos else "HOLD",
            }
        )

    # 2）对在持仓里但不在 topK 里的，目标权重 = 0（卖出）
    for code, cur_pos in positions.items():
        if code in codes_top:
            continue
        if cur_pos <= 0:
            continue
        orders.append(
            {
                "code": code,
                "current_pos": f"{cur_pos:.4f}",
                "target_weight": "0.0000",
                "action": "SELL",
            }
        )

    print(f"[PIPELINE] 已生成下单记录 {len(orders)} 条。")
    return orders


def save_orders(orders: List[Dict[str, str]], path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["code", "current_pos", "target_weight", "action"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in orders:
            writer.writerow(row)

    print(f"[PIPELINE] 下单列表已保存：{path}")


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(description="U1 日内一键流水线（打分 + 下单列表）")
    parser.add_argument("--job-id", required=True, help="如 ultrashort_main")
    parser.add_argument("--as-of", required=True, help="打分交易日，格式 YYYY-MM-DD")
    parser.add_argument("--tag", required=True, help="策略/模型标签，如 u1_v1_base_rf")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--max-price", type=float, default=80.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000)

    parser.add_argument(
        "--positions-csv",
        type=str,
        required=False,
        default=str(DATA_DIR / "live" / "positions_latest.csv"),
        help="当前持仓文件（code, position）",
    )
    parser.add_argument(
        "--target-weight",
        type=float,
        default=0.03,
        help="topK 标的目标权重（比如 0.03 = 3%）",
    )
    parser.add_argument(
        "--output-orders",
        type=str,
        required=False,
        help="下单列表输出路径；默认根据日期自动生成",
    )

    args = parser.parse_args()

    as_of_date = _parse_date(args.as_of)
    job_id = args.job_id
    tag = args.tag

    # 1）跑日打分
    run_daily_scoring_ml(
        job_id=job_id,
        as_of=as_of_date,
        tag=tag,
        top_k=args.top_k,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
    )

    # 2）推导打分文件路径
    full_scores_path, topk_scores_path = build_scores_paths(job_id, tag, as_of_date)
    print("[PIPELINE] 预期 full 打分文件：", full_scores_path)
    print("[PIPELINE] 预期 topK 打分文件：", topk_scores_path)

    # 3）读取持仓 + topK 打分
    positions = load_positions(Path(args.positions_csv))
    topk_scores = load_topk_scores(topk_scores_path)

    # 4）生成下单列表
    orders = build_order_list(
        scores=topk_scores,
        positions=positions,
        target_weight=args.target_weight,
    )

    # 5）保存下单列表
    if args.output_orders:
        orders_path = Path(args.output_orders)
    else:
        ds = _date_to_str(as_of_date)
        orders_path = REPORT_DIR / f"u1_orders_{job_id}_{tag}_{ds}.csv"

    save_orders(orders, orders_path)

    print("[PIPELINE] 全流程完成。")


if __name__ == "__main__":
    main()
