@echo off
chcp 65001 >nul
title LightHunter - Commander Watcher
cd /d %~dp0

echo [START] LightHunter Commander 守护进程（commander）...
echo 当前目录：%CD%
echo.

REM 如果没有在 PATH 中配置 Python，可以把下面这一行改成具体路径，例如：
REM   "G:\Python39\python.exe" -u process_watcher.py commander
python -u process_watcher.py commander

echo.
echo ProcessWatcher 已退出。按任意键关闭窗口...
pause >nul
