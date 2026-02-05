#!/bin/bash
# API 重启脚本

echo "正在停止现有 API 服务..."
pkill -f "gunicorn.*odeo-ai-studio" 2>/dev/null
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
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

# 后台启动
nohup gunicorn --config gunicorn.conf.py wsgi:app > /tmp/api_output.log 2>&1 &

sleep 3

# 检查是否启动成功
if pgrep -f "gunicorn.*odeo-ai-studio" > /dev/null; then
    echo "API 服务启动成功！"
    echo "日志文件: /tmp/api_output.log"
    ps aux | grep gunicorn | grep -v grep
else
    echo "API 服务启动失败，查看日志:"
    cat /tmp/api_output.log
fi
