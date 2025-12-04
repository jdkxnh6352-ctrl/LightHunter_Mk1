# test_ts_collect_once.py
# 作用：绕过交易时间判断，只用 TSCollector 抓一次并写入 ts_data.db

from ts_collector import TSCollector

def main():
    collector = TSCollector(interval=20, enable_bus=False)

    # 注意：_capture_once 是内部方法，但我们只是做本地测试，可以直接调用
    n = collector._capture_once()  # type: ignore[attr-defined]

    print(f"\n本次采集写入 ts_data.db 的快照行数：{n}")

if __name__ == "__main__":
    main()
