@echo off
cd /d G:\LightHunter_Mk1
call .venv\Scripts\activate.bat

rem 1) 从 market_ts.db 汇总到 snapshot_1d（这个脚本我们后面已经写好/会写）
python -m tools.build_snapshot_1d_from_ts

rem 2) U2 日常打分（自动找最新交易日）
python -m tools.u2_daily_scoring

rem 3) 生成候选股体检 / 回测（可选）
python -m tools.u2_daily_backtest_stats ^
  --input reports/u2_daily_backtest_demo.csv ^
  --output-equity reports/u2_backtest_equity_live.csv ^
  --output-yearly reports/u2_backtest_yearly_live.csv

echo ===============================
echo 收盤流程腳本已跑完，請打開 reports/ 看結果。
echo ===============================
pause
