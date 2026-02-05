#!/bin/bash
# 快速重启 API 服务

pkill -9 -f gunicorn 2>/dev/null
sleep 2

cd /home/disk2/diffusion-pipe/api
source /opt/conda/etc/profile.d/conda.sh
conda activate lora

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
sleep 4

echo "=== 检查服务状态 ==="
ps aux | grep gunicorn | grep -v grep
echo ""
echo "=== 测试 API ==="
curl -s http://localhost:8080/api/v1/training/list | head -c 200
echo ""
echo "=== 完成 ==="
