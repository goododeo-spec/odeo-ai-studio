#!/bin/bash

# ODEO AI Studio - API 服务启动脚本
# =====================================
# 用法: bash start_api.sh
# 可通过环境变量或 .env 文件覆盖所有配置

echo "=========================================="
echo "  ODEO AI Studio - 正在启动 API 服务"
echo "=========================================="

# ---------- 自动检测路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 加载 .env（如果存在） ----------
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "加载配置: $SCRIPT_DIR/.env"
    set -a; source "$SCRIPT_DIR/.env"; set +a
elif [ -f "$PROJECT_ROOT/.env" ]; then
    echo "加载配置: $PROJECT_ROOT/.env"
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# ---------- 环境变量（可通过 .env 或 export 覆盖） ----------
export STORAGE_ROOT="${STORAGE_ROOT:-$PROJECT_ROOT/data}"
export MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/pretrained_models}"
export TRAINING_OUTPUT_ROOT="${TRAINING_OUTPUT_ROOT:-$STORAGE_ROOT/outputs}"
export DATASET_PATH="${DATASET_PATH:-$STORAGE_ROOT/datasets}"
export RAW_PATH="${RAW_PATH:-$STORAGE_ROOT/raw}"
export GALLERY_ROOT="${GALLERY_ROOT:-$STORAGE_ROOT/gallery}"
export INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-$STORAGE_ROOT/outputs/inference}"
export LORA_ROOT="${LORA_ROOT:-$STORAGE_ROOT/outputs}"

# Qwen VL API Key (用于视频描述生成，请在 .env 中配置)
export QWEN_VL_API_KEY="${QWEN_VL_API_KEY:-}"

echo ""
echo "环境配置:"
echo "  STORAGE_ROOT: $STORAGE_ROOT"
echo "  MODELS_ROOT: $MODELS_ROOT"
echo "  TRAINING_OUTPUT_ROOT: $TRAINING_OUTPUT_ROOT"
echo "  GALLERY_ROOT: $GALLERY_ROOT"
echo ""

# ---------- 切换到 API 目录 ----------
cd "$SCRIPT_DIR"

# ---------- 激活 conda 环境 ----------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lora}"
# 自动检测 conda 路径
if [ -n "$CONDA_EXE" ]; then
    CONDA_SH="$(dirname $(dirname "$CONDA_EXE"))/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "警告: 未找到 conda，请确保 Python 环境已正确配置"
    CONDA_SH=""
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
    echo "已激活 conda 环境: $CONDA_ENV_NAME"
fi

# ---------- 检查 gunicorn ----------
if ! command -v gunicorn &> /dev/null; then
    echo "错误: gunicorn 未安装，正在安装..."
    pip install gunicorn
fi

# ---------- 停止已有的服务 ----------
pkill -f "gunicorn.*app:create_app" 2>/dev/null
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
sleep 2

# ---------- 启动 Gunicorn 服务 ----------
echo "正在启动 Gunicorn 服务器 (gthread worker)..."
echo "访问地址: http://0.0.0.0:8080"
echo ""

# 使用 gunicorn 配置文件启动
# 注意: 使用 app:create_app() 工厂模式入口，不走 wsgi.py
exec gunicorn --config gunicorn.conf.py "app:create_app()"
