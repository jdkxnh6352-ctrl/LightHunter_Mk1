#!/usr/bin/env bash
# 路径: G:/LightHunter_Mk1/start_collector.sh  （Linux / WSL 环境下使用）

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[START] LightHunter TSCollector 守护进程 (collector)..."
echo "[WORKDIR] $DIR"

# 如需指定 python 解释器，可以把 python3 改成绝对路径
python3 -u process_watcher.py collector "$@"
