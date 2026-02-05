#!/bin/bash
# 测试创建训练任务 API

echo "===== 测试创建训练任务 API ====="

# 发送测试请求
curl -X POST http://localhost:8080/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test_task_'$(date +%s)'",
    "gpu_id": -1,
    "model_type": "wan",
    "description": "API测试任务",
    "dataset": {
      "resolutions": [480],
      "enable_ar_bucket": true,
      "ar_buckets": [0.5, 0.75, 1.0],
      "frame_buckets": [1, 29, 49],
      "directory": [{"path": "/home/disk2/lora_training/datasets", "num_repeats": 5}]
    },
    "config": {
      "epochs": 60,
      "save_every_n_epochs": 5
    },
    "raw_videos": [],
    "processed_videos": []
  }' 2>&1

echo ""
echo "===== 完成 ====="
