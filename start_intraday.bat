@echo off
cd /d G:\LightHunter_Mk1

call .venv\Scripts\activate.bat

rem 1) 启动 1 分钟行情采集
start "MTS Collector" python market_ts_collector.py

rem 2) 简单提示
echo ===============================
echo 已启动: 1m tick 采集器 (market_ts_collector.py)
echo 记得中午和收盘前看一眼窗口有無紅色報錯。
echo ===============================
pause
