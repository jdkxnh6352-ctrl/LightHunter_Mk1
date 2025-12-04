# -*- coding: utf-8 -*-
"""
简单版：从东方财富拉一批 A 股日 K，生成 data/ultrashort_raw.csv

- 使用你项目里的 request_engine，自动走当前代理/重试配置
- 默认拉 5 只股票、近 3 年数据，你可以按需修改 CODES / START_DATE / END_DATE
"""

from __future__ import annotations

import csv
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from request_engine import get  # 用你现有的请求引擎


# ---------------- 配置区域：可以按需改 ----------------

# 采集的股票池（随便先来几只，之后你可以换成自己想要的一揽子）
CODES: List[str] = [
    "000001",  # 平安银行
    "600000",  # 浦发银行
    "300750",  # 宁德时代
    "002230",  # 科大讯飞
    "605119",  # 贵州三力
]

# 起止日期（注意不要太长，接口一次返回会比较大）
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 输出文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "ultrashort_raw.csv"


# ---------------- 工具函数 ----------------


def code_to_secid(code: str) -> str:
    """
    东财 secid 规则（常见写法）：
      - 上海：1.600000
      - 深圳：0.000001 / 0.300750 / 0.002230

    这里只做一个简单映射：
      - 以 '6' 开头 -> 上海，前缀 '1.'
      - 其他 -> 深圳，前缀 '0.'
    """
    code = str(code)
    if code.startswith("6"):
        mkt = "1"  # SH
    else:
        mkt = "0"  # SZ
    return f"{mkt}.{code}"


def fetch_kline_one(code: str) -> List[Dict[str, Any]]:
    """
    拉单只股票的日 K，返回若干行字典：
      {trade_date, code, name, open, high, low, close, volume, amount}
    """
    secid = code_to_secid(code)

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    # 这些参数是东财 k 线接口常见参数，必要时你可以微调
    params = {
        "secid": secid,
        "klt": "101",  # 101=日线
        "fqt": "1",  # 前复权
        "beg": START_DATE.replace("-", ""),
        "end": END_DATE.replace("-", ""),
        "fields1": "f1,f2,f3,f4,f5",
        # f51=日期, f52=开盘, f53=收盘, f54=最高, f55=最低, f56=成交量, f57=成交额, ...
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }

    print(f"[FETCH] {code}  -> {url}")
    r = get(url, site="eastmoney_quote", params=params, timeout=8)
    r.raise_for_status()
    data = r.json()

    if not data or "data" not in data or not data["data"]:
        print(f"[WARN] {code} 返回 data 为空，跳过。")
        return []

    name = data["data"].get("name", "") or ""
    klines = data["data"].get("klines") or []
    rows: List[Dict[str, Any]] = []

    for line in klines:
        # 一条示例：2020-01-02,10.01,10.20,10.30,9.90,123456,123456789,2.34,1.9,0.19,0.50
        parts = str(line).split(",")
        if len(parts) < 7:
            continue

        trade_date = parts[0]
        try:
            open_ = float(parts[1])
            close = float(parts[2])
            high = float(parts[3])
            low = float(parts[4])
            volume = float(parts[5])
            amount = float(parts[6])
        except Exception:
            continue

        rows.append(
            {
                "trade_date": trade_date,
                "code": code,
                "name": name,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        )

    print(f"[FETCH] {code} 共拿到 {len(rows)} 行。")
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for idx, code in enumerate(CODES, start=1):
        try:
            rows = fetch_kline_one(code)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] 抓取 {code} 出错: {e}")
        # 简单控频，避免对目标站点压力过大
        time.sleep(1.5 + random.random())

    if not all_rows:
        print("[WARN] 没有采集到任何数据，退出。")
        return

    # 按日期+代码排序
    all_rows.sort(key=lambda r: (r["trade_date"], r["code"]))

    # 写 CSV
    fieldnames = [
        "trade_date",
        "code",
        "name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[DONE] 共写入 {len(all_rows)} 行到 {OUT_CSV}")


if __name__ == "__main__":
    main()
