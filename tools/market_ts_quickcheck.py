# -*- coding: utf-8 -*-
"""
简单的 market_ts.db 体检脚本

功能：
- 找到项目根目录下的 market_ts.db
- 列出所有表
- 检查 snapshot_1m 表：
  - 字段列表
  - 总行数
  - 每个交易日的行数（最近几天）
  - 打印最近 5 条记录
"""

import os
import sqlite3
from collections import Counter


def get_project_root() -> str:
    """根据当前文件位置，推断项目根目录 G:/LightHunter_Mk1"""
    here = os.path.abspath(__file__)
    tools_dir = os.path.dirname(here)
    project_root = os.path.dirname(tools_dir)
    return project_root


def main() -> None:
    project_root = get_project_root()
    db_path = os.path.join(project_root, "market_ts.db")

    print(f"[CHECK] 项目根目录: {project_root}")
    print(f"[CHECK] 准备检查 DB: {db_path}")

    if not os.path.exists(db_path):
        print("[ERROR] 找不到 market_ts.db ，确认采集脚本是在同一项目下跑的。")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1) 列出所有表
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r["name"] for r in cur.fetchall()]
    print(f"[CHECK] 数据库中共有 {len(tables)} 个表: {tables}")

    if "snapshot_1m" not in tables:
        print("[WARN] 没有找到 snapshot_1m 表，确认采集脚本是否正常创建表。")
        conn.close()
        return

    print("\n====== snapshot_1m 表检查 ======")

    # 2) 列出字段
    cur.execute("PRAGMA table_info(snapshot_1m)")
    cols_info = cur.fetchall()
    cols = [r["name"] for r in cols_info]
    print(f"[CHECK] 字段列表 ({len(cols)}): {cols}")

    # 3) 统计总行数
    cur.execute("SELECT COUNT(*) AS cnt FROM snapshot_1m")
    total_rows = cur.fetchone()["cnt"]
    print(f"[CHECK] 总行数: {total_rows}")

    # 自动探测日期列名
    date_col = None
    for cand in ("trade_date", "trading_date", "date"):
        if cand in cols:
            date_col = cand
            break

    if date_col:
        # 4) 最近几个交易日的行数分布
        sql = f"""
            SELECT {date_col} AS d, COUNT(*) AS c
            FROM snapshot_1m
            GROUP BY {date_col}
            ORDER BY {date_col} DESC
            LIMIT 10
        """
        cur.execute(sql)
        rows = cur.fetchall()
        print("\n[CHECK] 最近 10 个交易日的行数：")
        for r in rows:
            print(f"  - {r['d']}: {r['c']} 行")
    else:
        print("[WARN] 没找到 trade_date / trading_date 字段，暂时无法按交易日统计。")

    # 5) 打印最近 5 条记录
    order_col = None
    for cand in ("ts", "datetime", "time", "rowid"):
        if cand in cols:
            order_col = cand
            break
    if order_col is None:
        order_col = "rowid"

    print(f"\n[CHECK] 按 {order_col} 倒序打印最近 5 条记录：")
    cur.execute(f"SELECT * FROM snapshot_1m ORDER BY {order_col} DESC LIMIT 5")
    last_rows = cur.fetchall()
    for i, r in enumerate(last_rows, 1):
        print(f"---- Row #{i} ----")
        for k in cols:
            print(f"{k}: {r[k]}")
        print("--------------")

    conn.close()
    print("\n[CHECK] market_ts.db 快速体检完成。")


if __name__ == "__main__":
    main()
