#!/bin/bash
# 测试并重启 API 服务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "===== 步骤 1: 验证 Python 语法 ====="
cd "$SCRIPT_DIR"

# 加载 .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
elif [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

# 激活 conda
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

python3 -c "
from models.training import TrainingTaskRequest
print('TrainingTaskRequest OK')
from services.training_service import training_service
print('training_service OK')
from routes.training import training_bp
print('training_bp OK')
print('所有模块加载成功！')
"

if [ $? -ne 0 ]; then
    echo "Python 语法错误，请检查代码"
    exit 1
fi

echo ""
echo "===== 步骤 2: 停止现有服务 ====="
pkill -f "gunicorn.*app:create_app" 2>/dev/null
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
sleep 3
echo "已停止现有服务"

echo ""
echo "===== 步骤 3: 启动 API 服务 ====="
export STORAGE_ROOT="${STORAGE_ROOT:-$PROJECT_ROOT/data}"
export MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/pretrained_models}"
export TRAINING_OUTPUT_ROOT="${TRAINING_OUTPUT_ROOT:-$STORAGE_ROOT/outputs}"
export DATASET_PATH="${DATASET_PATH:-$STORAGE_ROOT/datasets}"
export RAW_PATH="${RAW_PATH:-$STORAGE_ROOT/raw}"
export GALLERY_ROOT="${GALLERY_ROOT:-$STORAGE_ROOT/gallery}"
export INFERENCE_OUTPUT_ROOT="${INFERENCE_OUTPUT_ROOT:-$STORAGE_ROOT/outputs/inference}"
export LORA_ROOT="${LORA_ROOT:-$STORAGE_ROOT/outputs}"
export QWEN_VL_API_KEY="${QWEN_VL_API_KEY:-}"

nohup gunicorn --config gunicorn.conf.py "app:create_app()" > /tmp/api_output.log 2>&1 &
sleep 5

echo ""
echo "===== 步骤 4: 检查服务状态 ====="
if pgrep -f gunicorn > /dev/null; then
    echo "API 服务启动成功！"
    ps aux | grep gunicorn | grep -v grep
else
    echo "API 服务启动失败"
    echo "查看日志:"
    tail -50 /tmp/api_output.log
    exit 1
fi

echo ""
echo "===== 步骤 5: 测试 API 响应 ====="
sleep 2
response=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/v1/training/list 2>/dev/null)
http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    echo "API 响应正常 (HTTP $http_code)"
    echo "响应数据: $body" | head -c 200
    echo "..."
else
    echo "API 响应异常 (HTTP $http_code)"
    echo "响应: $body"
fi

echo ""
echo "===== 完成 ====="
echo "现在可以刷新浏览器测试提交训练功能"
echo "日志文件: /tmp/api_output.log"
