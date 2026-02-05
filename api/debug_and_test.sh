#!/bin/bash
# 调试和测试 API

echo "===== 1. 检查 API 服务状态 ====="
ps aux | grep gunicorn | grep -v grep
echo ""

echo "===== 2. 测试 API 健康检查 ====="
curl -s http://localhost:8080/api/v1/training/list | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'状态: code={d.get(\"code\")}, 任务数={d.get(\"data\",{}).get(\"pagination\",{}).get(\"total\",0)}')"
echo ""

echo "===== 3. 查看最近 API 日志 ====="
tail -30 /tmp/api_output.log 2>/dev/null || echo "日志文件不存在"
echo ""

echo "===== 4. 测试创建训练任务 ====="
TASK_ID="test_$(date +%Y%m%d_%H%M%S)"
echo "创建任务: $TASK_ID"

RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "'$TASK_ID'",
    "gpu_id": -1,
    "model_type": "wan",
    "description": "API测试任务_'$TASK_ID'",
    "dataset": {
      "resolutions": [480],
      "enable_ar_bucket": true,
      "ar_buckets": [0.5, 0.75, 1.0],
      "frame_buckets": [1, 29, 49],
      "directory": [{"path": "/home/disk2/lora_training/datasets", "num_repeats": 5}]
    },
    "config": {
      "epochs": 60,
      "save_every_n_epochs": 5,
      "micro_batch_size_per_gpu": 1,
      "gradient_accumulation_steps": 1,
      "gradient_clipping": 1.0,
      "warmup_steps": 20,
      "model": {
        "ckpt_path": "/home/disk1/pretrained_models/Wan2.1-I2V-14B-480P",
        "dtype": "bfloat16",
        "transformer_dtype": "float8"
      },
      "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16"
      },
      "optimizer": {
        "type": "adamw_optimi",
        "lr": 5e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01
      }
    },
    "raw_videos": [],
    "processed_videos": []
  }')

echo "API 响应:"
echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))"
echo ""

# 检查响应状态
CODE=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('code', 0))")
if [ "$CODE" = "201" ] || [ "$CODE" = "200" ]; then
    echo "✓ 任务创建成功！"
else
    echo "✗ 任务创建失败，code=$CODE"
    echo ""
    echo "===== 查看完整日志 ====="
    tail -50 /tmp/api_output.log 2>/dev/null
fi

echo ""
echo "===== 5. 验证任务是否存在 ====="
curl -s http://localhost:8080/api/v1/training/$TASK_ID | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'任务状态: {d.get(\"data\",{}).get(\"status\", \"未找到\")}')" 2>/dev/null || echo "获取任务失败"
echo ""

echo "===== 调试完成 ====="
