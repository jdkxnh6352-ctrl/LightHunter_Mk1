# -*- coding: utf-8 -*-
"""
test_market_hunter.py

小规模“选股猎手”自检脚本：
- 通过 MarketHunter 调用腾讯行情；
- 看看一小篮子股票的实时数据；
- 顺便再确认一次：代理 + 防封配置是否都正常工作。
"""

import time

from data_core import MarketHunter


def main() -> None:
    hunter = MarketHunter()

    sample_codes = ["000001", "600000", "300750", "002230", "605119"]
    print("== MarketHunter 小规模选股测试 ==")
    print("样本股票池：", sample_codes)

    for i in range(1, 4):
        print("\n" + "=" * 30)
        print(f"第 {i} 轮采样 @ {time.strftime('%H:%M:%S')}")

        df = hunter.fetch_snapshot(sample_codes)

        # 只打印关键几列，避免一屏太挤
        cols = [c for c in ["代码", "名称", "现价", "涨幅", "成交额"] if c in df.columns]
        print(df[cols].to_string(index=False))

        if i < 3:
            print("等待 10 秒后继续下一轮...\n")
            time.sleep(10)

    print("\n== MarketHunter 小规模测试结束 ==")


if __name__ == "__main__":
    main()
