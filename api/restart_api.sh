#!/bin/bash
# API 重启脚本

echo "正在停止现有 API 服务..."
pkill -9 -f gunicorn 2>/dev/null
sleep 3

echo "正在启动 API 服务..."
cd /home/disk2/diffusion-pipe/api

# 激活 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate lora

# 设置环境变量
export STORAGE_ROOT=/home/disk2/lora_training
export MODELS_ROOT=/home/disk1/pretrained_models
export TRAINING_OUTPUT_ROOT=/home/disk2/lora_training/outputs
export DATASET_PATH=/home/disk2/lora_training/datasets
export RAW_PATH=/home/disk2/lora_training/raw
export GALLERY_ROOT=/home/disk2/lora_training/gallery
export INFERENCE_OUTPUT_ROOT=/home/disk2/lora_training/outputs/inference
export LORA_ROOT=/home/disk2/lora_training/outputs
export QWEN_VL_API_KEY="sk-3fee7787593f4a3e95f338e8303033c8"

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
