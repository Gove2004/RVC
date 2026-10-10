#!/usr/bin/env sh
# RVC 训练命令行向导（Linux / macOS）
# 用法: ./start-train.sh [数据集目录]
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
    exec "$PY" app.py --train "$@"
elif [ -x ".venv/Scripts/python.exe" ]; then
    exec ".venv/Scripts/python.exe" app.py --train "$@"
else
    exec python3 app.py --train "$@"
fi
