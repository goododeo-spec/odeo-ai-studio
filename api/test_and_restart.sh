#!/bin/bash
# 测试并重启 API 服务

echo "===== 步骤 1: 验证 Python 语法 ====="
cd /home/disk2/diffusion-pipe/api
source /opt/conda/etc/profile.d/conda.sh
conda activate lora

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
pkill -f "gunicorn.*odeo-ai-studio" 2>/dev/null
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
sleep 3
echo "已停止现有服务"

echo ""
echo "===== 步骤 3: 启动 API 服务 ====="
export STORAGE_ROOT=/home/disk2/lora_training
export MODELS_ROOT=/home/disk1/pretrained_models
export TRAINING_OUTPUT_ROOT=/home/disk2/lora_training/outputs
export DATASET_PATH=/home/disk2/lora_training/datasets
export RAW_PATH=/home/disk2/lora_training/raw
export GALLERY_ROOT=/home/disk2/lora_training/gallery
export INFERENCE_OUTPUT_ROOT=/home/disk2/lora_training/outputs/inference
export LORA_ROOT=/home/disk2/lora_training/outputs
export QWEN_VL_API_KEY="sk-3fee7787593f4a3e95f338e8303033c8"

nohup gunicorn --config gunicorn.conf.py wsgi:app > /tmp/api_output.log 2>&1 &
sleep 5

echo ""
echo "===== 步骤 4: 检查服务状态 ====="
if pgrep -f "gunicorn.*odeo-ai-studio" > /dev/null; then
    echo "✓ API 服务启动成功！"
    ps aux | grep gunicorn | grep -v grep
else
    echo "✗ API 服务启动失败"
    echo "查看日志:"
    cat /tmp/api_output.log
    exit 1
fi

echo ""
echo "===== 步骤 5: 测试 API 响应 ====="
sleep 2
response=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/v1/training/list 2>/dev/null)
http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" = "200" ]; then
    echo "✓ API 响应正常 (HTTP $http_code)"
    echo "响应数据: $body" | head -c 200
    echo "..."
else
    echo "✗ API 响应异常 (HTTP $http_code)"
    echo "响应: $body"
fi

echo ""
echo "===== 完成 ====="
echo "现在可以刷新浏览器测试提交训练功能"
echo "日志文件: /tmp/api_output.log"
