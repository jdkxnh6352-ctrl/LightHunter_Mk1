# -*- coding: utf-8 -*-
"""
test_market_ts_capture_once.py

强制用 MarketTSCollector 抓一帧 1 分钟快照，写入 market_ts.db 的 snapshot_1m。
不走 while 循环，也不管交易时间，用来做管道自检。
"""

from colorama import init
from market_ts_collector import MarketTSCollector, DB_FILE


def main() -> None:
    # 让彩色输出正常
    init(autoreset=True)

    # 为了安全起见，只写前 100 只成交额最大的股票
    collector = MarketTSCollector(
        interval=60,
        db_path=DB_FILE,
        max_rows_per_frame=100,
        enable_bus=False,   # 先关掉 ZMQ 总线，避免多余依赖
    )

    print("[TEST] 准备强制抓取一帧 1m 快照并写入 snapshot_1m ...")

    # 直接调用内部的一次性抓帧函数（不走 run() 里的死循环）
    rows = collector._capture_once()  # noqa: SLF001  （测试脚本里用私有方法没关系）

    print(f"[TEST] 本次强制抓帧写入条数： {rows}")

    # 手动关一下连接（避免 Windows 下文件句柄占用）
    if collector.conn is not None:
        collector.conn.close()


if __name__ == "__main__":
    main()
