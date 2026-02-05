#!/bin/bash

# ODEO AI Studio - API 服务启动脚本
# =====================================

echo "=========================================="
echo "  ODEO AI Studio - 正在启动 API 服务"
echo "=========================================="

# 训练 GPU 分配:
# - 训练: GPU 0-3
# - 推理: GPU 4-7

# 环境变量配置
export STORAGE_ROOT=/home/disk2/lora_training
export MODELS_ROOT=/home/disk1/pretrained_models
export TRAINING_OUTPUT_ROOT=/home/disk2/lora_training/outputs
export DATASET_PATH=/home/disk2/lora_training/datasets
export RAW_PATH=/home/disk2/lora_training/raw
export GALLERY_ROOT=/home/disk2/lora_training/gallery
export INFERENCE_OUTPUT_ROOT=/home/disk2/lora_training/outputs/inference
export LORA_ROOT=/home/disk2/lora_training/outputs

# Qwen VL API Key (用于视频描述生成)
export QWEN_VL_API_KEY="sk-3fee7787593f4a3e95f338e8303033c8"

echo ""
echo "环境配置:"
echo "  STORAGE_ROOT: $STORAGE_ROOT"
echo "  TRAINING_OUTPUT_ROOT: $TRAINING_OUTPUT_ROOT"
echo "  GALLERY_ROOT: $GALLERY_ROOT"
echo ""
echo "GPU 分配:"
echo "  训练 GPU: 0-3"
echo "  推理 GPU: 4-7"
echo ""
echo "ComfyUI 节点映射:"
echo "  Node 71: LoRA 选择"
echo "  Node 81: 触发词"
echo "  Node 58: 输入图片"
echo "  Node 30: 输出结果"
echo ""

# 切换到 API 目录
cd /home/disk2/diffusion-pipe/api

# 激活 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate lora

# 检查 gunicorn 是否安装
if ! command -v gunicorn &> /dev/null; then
    echo "错误: gunicorn 未安装，正在安装..."
    pip install gunicorn gevent -i https://mirrors.aliyun.com/pypi/simple/
fi

# 停止已有的服务
pkill -f "gunicorn.*wsgi:app" 2>/dev/null
sleep 2

# 启动 Gunicorn 服务
echo "正在启动 Gunicorn 服务器..."
echo "访问地址: http://0.0.0.0:8080"
echo ""

# 使用 gunicorn 配置文件启动
exec gunicorn --config gunicorn.conf.py wsgi:app
