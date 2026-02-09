#!/bin/bash
# API 重启脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "正在停止现有 API 服务..."
pkill -9 -f gunicorn 2>/dev/null
sleep 3

echo "正在启动 API 服务..."
cd "$SCRIPT_DIR"

# 加载 .env（如果存在）
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
elif [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# 激活 conda 环境（自动检测）
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lora}"
if [ -n "$CONDA_EXE" ]; then
    CONDA_SH="$(dirname $(dirname "$CONDA_EXE"))/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
else
    CONDA_SH=""
fi
if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
fi

# 设置环境变量
export STORAGE_ROOT="${STORAGE_ROOT:-$PROJECT_ROOT/data}"
export MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/pretrained_models}"
export TRAINING_OUTPUT_ROOT="${TRAINING_OUTPUT_ROOT:-$STORAGE_ROOT/outputs}"
export DATASET_PATH="${DATASET_PATH:-$STORAGE_ROOT/datasets}"
export RAW_PATH="${RAW_PATH:-$STORAGE_ROOT/raw}"
export GALLERY_ROOT="${GALLERY_ROOT:-$STORAGE_ROOT/gallery}"
export INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-$STORAGE_ROOT/outputs/inference}"
export LORA_ROOT="${LORA_ROOT:-$STORAGE_ROOT/outputs}"
export QWEN_VL_API_KEY="${QWEN_VL_API_KEY:-}"

# 后台启动（使用 app:create_app() 工厂模式，gthread worker）
nohup gunicorn --config gunicorn.conf.py "app:create_app()" > /tmp/api_output.log 2>&1 &

sleep 8

# 检查是否启动成功
if pgrep -f gunicorn > /dev/null; then
    echo "API 服务启动成功！"
    echo "日志文件: /tmp/api_output.log"
    ps aux | grep gunicorn | grep -v grep
    echo ""
    echo "测试 API 响应..."
    curl -s -o /dev/null -w "HTTP %{http_code}" --connect-timeout 5 http://localhost:8080/health
    echo ""
else
    echo "API 服务启动失败，查看日志:"
    tail -50 /tmp/api_output.log
fi
