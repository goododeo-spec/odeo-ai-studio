# Diffusion-Pipe 训练后端 API 文档

## 概述

本文档描述了基于 diffusion-pipe 训练框架的 Flask 后端 API 接口，支持数据预处理、GPU 管理、训练任务调度等功能。

### 基本信息

- **基础 URL**: `http://localhost:5000/api/v1`
- **Content-Type**: `application/json` (除文件上传接口)
- **认证方式**: API Key (在请求头中传递 `Authorization: Bearer <API_KEY>`)
- **响应格式**: JSON

### 通用响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": {},
  "timestamp": 1704067200
}
```

### 错误码说明

| 错误码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权或 API Key 无效 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 409 | 资源冲突 (如 GPU 已被占用) |
| 422 | 数据验证失败 |
| 500 | 服务器内部错误 |

---

## 1. 数据预处理 API

### 1.1 上传并预处理视频

**接口路径**: `/preprocess/upload`

**请求方法**: `POST`

**Content-Type**: `multipart/form-data`

**描述**: 上传训练视频和提示词，执行预处理步骤，生成处理后的视频和提示词

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| video | file | 是 | 视频文件 (支持 mp4, avi, mov, mkv) |
| prompt | string | 是 | 文本提示词 |
| model_type | string | 是 | 模型类型 (flux, hunyuan-video, wan, etc.) |
| fps | integer | 否 | 输出帧率 (默认: 模型默认值) |
| resolution | string | 否 | 分辨率 (如 "512x512", "1024x1024") |

**请求示例**:

```bash
curl -X POST http://localhost:5000/api/v1/preprocess/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "video=@/path/to/video.mp4" \
  -F "prompt=A beautiful landscape with mountains" \
  -F "model_type=hunyuan-video" \
  -F "fps=16" \
  -F "resolution=512x512"
```

**响应示例** (成功):

```json
{
  "code": 200,
  "message": "预处理完成",
  "data": {
    "task_id": "preprocess_20250212_07123456",
    "status": "completed",
    "output": {
      "video_path": "/data/preprocessed/video_20250212_07123456.mp4",
      "prompt": "A beautiful landscape with mountains",
      "metadata": {
        "fps": 16,
        "resolution": "512x512",
        "duration": 10.5,
        "frame_count": 168,
        "original_video": "/tmp/upload_abc123.mp4"
      }
    },
    "processing_time": 12.34,
    "created_at": "2025-02-12T07:12:34Z"
  },
  "timestamp": 1704067200
}
```

**响应示例** (处理中):

```json
{
  "code": 202,
  "message": "预处理进行中",
  "data": {
    "task_id": "preprocess_20250212_07123456",
    "status": "processing",
    "progress": 65,
    "current_step": "视频裁剪和缩放",
    "eta_seconds": 5
  },
  "timestamp": 1704067200
}
```

**响应示例** (错误):

```json
{
  "code": 422,
  "message": "视频文件格式不支持",
  "data": null,
  "timestamp": 1704067200
}
```

### 1.2 获取预处理任务状态

**接口路径**: `/preprocess/status/{task_id}`

**请求方法**: `GET`

**描述**: 查询预处理任务的执行状态

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "preprocess_20250212_07123456",
    "status": "completed", // completed, processing, failed, queued
    "progress": 100,
    "current_step": "完成",
    "result": {
      "video_path": "/data/preprocessed/video_20250212_07123456.mp4",
      "prompt": "A beautiful landscape with mountains",
      "metadata": {
        "fps": 16,
        "resolution": "512x512",
        "duration": 10.5
      }
    },
    "created_at": "2025-02-12T07:12:34Z",
    "updated_at": "2025-02-12T07:12:46Z"
  },
  "timestamp": 1704067200
}
```

---

## 2. GPU 状态查询 API

### 2.1 获取所有 GPU 状态

**接口路径**: `/gpu/status`

**请求方法**: `GET`

**描述**: 获取所有 GPU 的实时状态信息

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "gpus": [
      {
        "gpu_id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total": 24576,
        "memory_used": 8192,
        "memory_free": 16384,
        "memory_utilization": 33,
        "utilization_gpu": 75,
        "temperature": 65,
        "power_usage": 250,
        "power_limit": 450,
        "status": "available", // available, training, preprocessing, unknown
        "current_task": null,
        "task_type": null,
        "task_progress": null,
        "estimated_remaining_time": null
      },
      {
        "gpu_id": 1,
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total": 24576,
        "memory_used": 20480,
        "memory_free": 4096,
        "memory_utilization": 83,
        "utilization_gpu": 98,
        "temperature": 78,
        "power_usage": 420,
        "power_limit": 450,
        "status": "training",
        "current_task": "train_20250212_07000001",
        "task_type": "flux_lora",
        "task_progress": 45.6,
        "estimated_remaining_time": 3600
      }
    ],
    "summary": {
      "total_gpus": 2,
      "available_gpus": 1,
      "busy_gpus": 1,
      "total_memory": 49152,
      "available_memory": 20480
    }
  },
  "timestamp": 1704067200
}
```

### 2.2 获取可用 GPU 列表

**接口路径**: `/gpu/available`

**请求方法**: `GET`

**查询参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| min_memory | integer | 否 | 最小可用显存 (MB) |
| task_type | string | 否 | 任务类型 (可选: training, preprocessing) |

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "available_gpus": [
      {
        "gpu_id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total": 24576,
        "memory_free": 16384,
        "status": "available",
        "recommended_for": ["flux", "sdxl", "hunyuan-video"] // 推荐任务类型
      }
    ]
  },
  "timestamp": 1704067200
}
```

---

## 3. 训练任务 API

### 3.1 创建训练任务

**接口路径**: `/training/start`

**请求方法**: `POST`

**描述**: 选择 GPU 并开始训练任务

**请求参数**:

```json
{
  "gpu_id": 0,
  "model_type": "wan",
  "description": "Wan LoRA 训练 - ODEO 数据集",
  "dataset": {
    "resolutions": [480],
    "enable_ar_bucket": true,
    "ar_buckets": [0.5, 0.563, 0.75],
    "frame_buckets": [1, 29, 49, 97],
    "directories": [
      {
        "path_id": "/mnt/disk0/train_data/3",
        "num_repeats": 5
      }
    ]
  },
  "config": {
    "epochs": 60,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "warmup_steps": 20,
    "eval": {
      "eval_every_n_epochs": 1,
      "eval_before_first_step": true,
      "eval_micro_batch_size_per_gpu": 1,
      "eval_gradient_accumulation_steps": 1
    },
    "model": {
      "ckpt_path": "/mnt/disk0/pretrained_models/Wan2.1-I2V-14B-480P",
      "dtype": "bfloat16",
      "transformer_dtype": "float8",
      "timestep_sample_method": "uniform"
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
      "weight_decay": 0.01,
      "eps": 1e-8
    },
    "save": {
      "save_every_n_epochs": 5,
      "checkpoint_every_n_epochs": 10,
      "save_dtype": "bfloat16"
    },
    "optimization": {
      "activation_checkpointing": true,
      "partition_method": "parameters",
      "caching_batch_size": 1,
      "steps_per_print": 1,
      "video_clip_mode": "single_beginning"
    }
  }
}
```

**请求示例**:

```bash
curl -X POST http://localhost:5000/api/v1/training/start \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_id": 0,
    "model_type": "wan",
    "description": "Wan LoRA 训练 - ODEO 数据集",
    "dataset": {
      "resolutions": [480],
      "enable_ar_bucket": true,
      "ar_buckets": [0.5, 0.563, 0.75],
      "frame_buckets": [1, 29, 49, 97],
      "directories": [
        {
          "path": "/mnt/disk0/train_data/3",
          "num_repeats": 5
        }
      ]
    },
    "config": {
      "epochs": 60,
      "micro_batch_size_per_gpu": 1,
      "pipeline_stages": 1,
      "gradient_accumulation_steps": 1,
      "gradient_clipping": 1.0,
      "warmup_steps": 20,
      "model": {
        "ckpt_path": "/mnt/disk0/pretrained_models/Wan2.1-I2V-14B-480P",
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
        "betas": [0.9, 0.99]
      },
      "optimization": {
        "activation_checkpointing": true,
        "caching_batch_size": 1,
        "video_clip_mode": "single_beginning"
      }
    }
  }'
```

**响应示例** (成功):

```json
{
  "code": 201,
  "message": "训练任务已创建",
  "data": {
    "task_id": "train_20250212_07000001",
    "status": "queued",
    "gpu_id": 0,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "model_type": "flux",
    "config": { ... },
    "description": "Flux LoRA 训练 - 风景数据集",
    "created_at": "2025-02-12T07:00:00Z",
    "estimated_start_time": "2025-02-12T07:00:05Z",
    "estimated_duration": 7200,
    "checkpoints": {
      "save_every_n_epochs": 2,
      "last_saved": null,
      "next_save": "epoch 2"
    }
  },
  "timestamp": 1704067200
}
```

**响应示例** (GPU 忙碌):

```json
{
  "code": 409,
  "message": "GPU 0 已被占用",
  "data": {
    "gpu_id": 0,
    "current_task": "train_20250212_06000001",
    "estimated_available_time": "2025-02-12T09:30:00Z",
    "available_gpus": [1, 2]
  },
  "timestamp": 1704067200
}
```

### 3.2 停止训练任务

**接口路径**: `/training/stop/{task_id}`

**请求方法**: `POST`

**描述**: 停止正在运行的训练任务

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| force | boolean | 否 | 是否强制停止 (默认: false) |

**响应示例**:

```json
{
  "code": 200,
  "message": "训练任务已停止",
  "data": {
    "task_id": "train_20250212_07000001",
    "status": "stopped",
    "stopped_at": "2025-02-12T08:30:00Z",
    "checkpoint_saved": true,
    "last_checkpoint": "epoch 5",
    "training_time": 5400
  },
  "timestamp": 1704067200
}
```

### 3.3 获取训练任务列表

**接口路径**: `/training/list`

**请求方法**: `GET`

**查询参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| status | string | 否 | 过滤状态 (queued, running, completed, failed, stopped) |
| gpu_id | integer | 否 | 过滤 GPU |
| limit | integer | 否 | 返回数量限制 (默认: 50) |
| offset | integer | 否 | 偏移量 (默认: 0) |

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "tasks": [
      {
        "task_id": "train_20250212_07000001",
        "status": "running",
        "gpu_id": 0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "model_type": "flux",
        "description": "Flux LoRA 训练 - 风景数据集",
        "progress": {
          "current_epoch": 3,
          "total_epochs": 10,
          "current_step": 150,
          "total_steps": 500,
          "epoch_progress": 30.0,
          "overall_progress": 30.0,
          "eta_seconds": 3600
        },
        "metrics": {
          "current_loss": 0.245,
          "best_loss": 0.198,
          "learning_rate": 5e-05,
          "grad_norm": 0.85
        },
        "created_at": "2025-02-12T07:00:00Z",
        "started_at": "2025-02-12T07:00:05Z",
        "updated_at": "2025-02-12T07:30:00Z"
      }
    ],
    "pagination": {
      "total": 15,
      "limit": 50,
      "offset": 0,
      "has_more": false
    }
  },
  "timestamp": 1704067200
}
```

### 3.4 获取训练任务详情

**接口路径**: `/training/{task_id}`

**请求方法**: `GET`

**描述**: 获取指定训练任务的详细信息

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "train_20250212_07000001",
    "status": "running",
    "gpu_id": 0,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "model_type": "flux",
    "config": {
      "output_dir": "/data/training_runs/exp1",
      "epochs": 10,
      "micro_batch_size_per_gpu": 1,
      "optimizer": {
        "type": "adamw8bitkahan",
        "lr": 5e-5
      },
      "adapter": {
        "type": "lora",
        "rank": 16
      }
    },
    "description": "Flux LoRA 训练 - 风景数据集",
    "progress": {
      "current_epoch": 3,
      "total_epochs": 10,
      "current_step": 150,
      "total_steps": 500,
      "epoch_progress": 30.0,
      "overall_progress": 30.0,
      "eta_seconds": 3600
    },
    "metrics": {
      "current_loss": 0.245,
      "best_loss": 0.198,
      "learning_rate": 5e-05,
      "grad_norm": 0.85,
      "epoch_losses": [0.456, 0.312, 0.245],
      "step_losses": [...]
    },
    "system_stats": {
      "gpu_memory": {
        "total": 24576,
        "used": 18432,
        "free": 6144,
        "utilization": 75
      },
      "gpu_utilization": 98,
      "gpu_temperature": 72,
      "gpu_power": 380
    },
    "checkpoints": [
      {
        "epoch": 2,
        "step": 100,
        "path": "/data/training_runs/exp1/epoch2",
        "timestamp": "2025-02-12T07:45:00Z",
        "loss": 0.312
      }
    ],
    "created_at": "2025-02-12T07:00:00Z",
    "started_at": "2025-02-12T07:00:05Z",
    "updated_at": "2025-02-12T07:30:00Z",
    "logs_path": "/data/training_runs/exp1/logs/train.log"
  },
  "timestamp": 1704067200
}
```

---

## 4. 训练进度查询 API

### 4.1 获取训练进度

**接口路径**: `/training/{task_id}/progress`

**请求方法**: `GET`

**描述**: 获取训练任务的实时进度信息

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "train_20250212_07000001",
    "status": "running",
    "progress": {
      "current_epoch": 3,
      "total_epochs": 10,
      "current_step": 150,
      "total_steps": 500,
      "current_batch": 5,
      "total_batches": 10,
      "epoch_progress": 30.0,
      "overall_progress": 30.0,
      "eta_seconds": 3600,
      "eta_formatted": "1h 0m 0s",
      "training_time": 1800,
      "training_time_formatted": "0h 30m 0s"
    },
    "metrics": {
      "current_loss": 0.245,
      "best_loss": 0.198,
      "learning_rate": 5e-05,
      "grad_norm": 0.85,
      "epoch_loss": 0.245,
      "best_epoch": 3
    },
    "speed": {
      "steps_per_second": 0.083,
      "samples_per_second": 0.083,
      "time_per_step": 12.0
    },
    "updated_at": "2025-02-12T07:30:00Z"
  },
  "timestamp": 1704067200
}
```

### 4.2 获取训练指标历史

**接口路径**: `/training/{task_id}/metrics`

**请求方法**: `GET`

**查询参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| metric | string | 否 | 指标名称 (loss, lr, grad_norm 等) |
| type | string | 否 | 类型 (epoch, step) |
| limit | integer | 否 | 数据点数量限制 |

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "train_20250212_07000001",
    "metric": "loss",
    "type": "epoch",
    "data": [
      {
        "epoch": 1,
        "value": 0.456,
        "timestamp": "2025-02-12T07:15:00Z"
      },
      {
        "epoch": 2,
        "value": 0.312,
        "timestamp": "2025-02-12T07:30:00Z"
      },
      {
        "epoch": 3,
        "value": 0.245,
        "timestamp": "2025-02-12T07:45:00Z"
      }
    ]
  },
  "timestamp": 1704067200
}
```

---

## 5. 日志查询 API

### 5.1 获取训练日志

**接口路径**: `/training/{task_id}/logs`

**请求方法**: `GET`

**查询参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| type | string | 否 | 日志类型 (train, eval, system, all) |
| level | string | 否 | 日志级别 (DEBUG, INFO, WARNING, ERROR) |
| tail | integer | 否 | 返回最后 N 行 |
| since | string | 否 | 返回指定时间之后的日志 (ISO 8601 格式) |

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "train_20250212_07000001",
    "log_file": "/data/training_runs/exp1/logs/train.log",
    "total_lines": 1250,
    "logs": [
      {
        "timestamp": "2025-02-12T07:00:05Z",
        "level": "INFO",
        "type": "train",
        "message": "Starting training on GPU 0",
        "epoch": null,
        "step": null
      },
      {
        "timestamp": "2025-02-12T07:00:10Z",
        "level": "INFO",
        "type": "train",
        "message": "Loading dataset from /data/datasets/train_dataset.toml",
        "epoch": null,
        "step": null
      },
      {
        "timestamp": "2025-02-12T07:15:00Z",
        "level": "INFO",
        "type": "train",
        "message": "Epoch 1/10 - Step 50/100 - Loss: 0.456 - LR: 5e-05 - Grad Norm: 1.25",
        "epoch": 1,
        "step": 50
      }
    ]
  },
  "timestamp": 1704067200
}
```

### 5.2 获取实时日志流

**接口路径**: `/training/{task_id}/logs/stream`

**请求方法**: `GET`

**Content-Type**: `text/event-stream`

**描述**: 通过 Server-Sent Events 获取实时日志流

**响应示例**:

```
data: {
  "timestamp": "2025-02-12T07:30:00Z",
  "level": "INFO",
  "type": "train",
  "message": "Epoch 3/10 - Step 150/500 - Loss: 0.245 - LR: 5e-05 - Grad Norm: 0.85",
  "epoch": 3,
  "step": 150
}

data: {
  "timestamp": "2025-02-12T07:30:12Z",
  "level": "INFO",
  "type": "eval",
  "message": "Evaluation - Loss: 0.231 - Eval time: 45.2s",
  "epoch": 3,
  "step": 150
}
```

### 5.3 搜索日志

**接口路径**: `/training/{task_id}/logs/search`

**请求方法**: `GET`

**查询参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| keyword | string | 是 | 搜索关键词 |
| type | string | 否 | 日志类型 |
| level | string | 否 | 日志级别 |
| start_time | string | 否 | 开始时间 |
| end_time | string | 否 | 结束时间 |
| limit | integer | 否 | 结果数量限制 |

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "train_20250212_07000001",
    "query": {
      "keyword": "Loss",
      "type": "train",
      "limit": 100
    },
    "total_matches": 50,
    "results": [
      {
        "timestamp": "2025-02-12T07:15:00Z",
        "level": "INFO",
        "type": "train",
        "message": "Epoch 1/10 - Step 50/100 - Loss: 0.456 - LR: 5e-05",
        "line_number": 125,
        "matched_text": "Loss: 0.456"
      }
    ]
  },
  "timestamp": 1704067200
}
```

---

## 6. 模型下载 API

### 6.1 列出可用模型

**接口路径**: `/models/list`

**请求方法**: `GET`

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "models": [
      {
        "id": "flux",
        "name": "Flux",
        "type": "image",
        "supports_lora": true,
        "supports_full_finetune": true,
        "supports_fp8": true,
        "min_vram_gb": 12,
        "recommended_vram_gb": 24,
        "config_path": "/data/models/flux/config.json"
      },
      {
        "id": "hunyuan-video",
        "name": "HunyuanVideo",
        "type": "video",
        "supports_lora": true,
        "supports_full_finetune": false,
        "supports_fp8": true,
        "min_vram_gb": 24,
        "recommended_vram_gb": 48,
        "config_path": "/data/models/hunyuan-video/config.json"
      }
    ]
  },
  "timestamp": 1704067200
}
```

### 6.2 下载模型

**接口路径**: `/models/download`

**请求方法**: `POST`

**请求参数**:

```json
{
  "model_id": "flux",
  "version": "1.0",
  "components": ["vae", "text_encoder", "diffusion_model"]
}
```

**响应示例**:

```json
{
  "code": 202,
  "message": "模型下载已开始",
  "data": {
    "download_id": "dl_20250212_08000001",
    "model_id": "flux",
    "status": "downloading",
    "progress": 15.5,
    "speed": "125 MB/s",
    "eta": "2m 30s",
    "components": [
      {
        "name": "vae",
        "size": "335 MB",
        "downloaded": true,
        "path": "/data/models/flux/vae/"
      },
      {
        "name": "text_encoder",
        "size": "980 MB",
        "downloaded": true,
        "path": "/data/models/flux/text_encoder/"
      },
      {
        "name": "diffusion_model",
        "size": "5.2 GB",
        "downloaded": false,
        "progress": 25.0
      }
    ]
  },
  "timestamp": 1704067200
}
```

---

## 7. 系统监控 API

### 7.1 系统概览

**接口路径**: `/system/overview`

**请求方法**: `GET`

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "system": {
      "cpu_count": 32,
      "cpu_usage": 45.2,
      "memory_total": 128000,
      "memory_used": 64000,
      "memory_available": 64000,
      "disk_total": 2000000,
      "disk_used": 800000,
      "disk_free": 1200000
    },
    "gpus": {
      "total": 4,
      "available": 2,
      "busy": 2
    },
    "tasks": {
      "total": 5,
      "queued": 1,
      "running": 2,
      "completed": 2,
      "failed": 0
    },
    "uptime": 86400,
    "version": "1.0.0"
  },
  "timestamp": 1704067200
}
```

---

## 8. 错误处理

### 8.1 常见错误响应

**参数验证失败** (422):

```json
{
  "code": 422,
  "message": "参数验证失败",
  "data": {
    "errors": [
      {
        "field": "gpu_id",
        "message": "GPU ID 必须为整数"
      },
      {
        "field": "model_type",
        "message": "不支持的模型类型"
      }
    ]
  },
  "timestamp": 1704067200
}
```

**资源不存在** (404):

```json
{
  "code": 404,
  "message": "训练任务不存在",
  "data": {
    "task_id": "train_20250212_99999999"
  },
  "timestamp": 1704067200
}
```

**GPU 不足** (507):

```json
{
  "code": 507,
  "message": "可用 GPU 不足",
  "data": {
    "required_gpus": 2,
    "available_gpus": 0,
    "busy_gpus": [0, 1, 2, 3],
    "recommended_action": "等待当前任务完成或减少 pipeline_stages"
  },
  "timestamp": 1704067200
}
```

### 8.2 错误码完整列表

| 错误码 | HTTP 状态码 | 说明 |
|--------|-------------|------|
| 200 | 200 | 成功 |
| 201 | 201 | 已创建 |
| 202 | 202 | 已接受 (异步处理) |
| 400 | 400 | 请求参数错误 |
| 401 | 401 | 未授权 |
| 403 | 403 | 禁止访问 |
| 404 | 404 | 资源不存在 |
| 409 | 409 | 资源冲突 |
| 422 | 422 | 数据验证失败 |
| 429 | 429 | 请求过于频繁 |
| 500 | 500 | 服务器内部错误 |
| 502 | 502 | 网关错误 |
| 507 | 507 | 资源不足 |

---

## 9. WebSocket 连接 (可选)

### 9.1 训练进度推送

**WebSocket 端点**: `ws://localhost:5000/api/v1/ws/training/{task_id}`

**消息格式**:

```json
{
  "type": "progress", // progress, log, metrics, status
  "timestamp": 1704067200,
  "data": {
    "task_id": "train_20250212_07000001",
    "status": "running",
    "progress": 45.6,
    "current_epoch": 4,
    "total_epochs": 10,
    "current_loss": 0.245
  }
}
```

---

## 10. 认证示例

### 10.1 API Key 认证

**请求头**:

```
Authorization: Bearer sk_live_1234567890abcdef
Content-Type: application/json
```

### 10.2 生成 API Key

**接口路径**: `/auth/generate-key`

**请求方法**: `POST`

**请求参数**:

```json
{
  "user_id": "user_001",
  "permissions": ["training", "preprocessing", "model_download"]
}
```

**响应示例**:

```json
{
  "code": 201,
  "message": "API Key 已生成",
  "data": {
    "api_key": "sk_live_1234567890abcdef",
    "user_id": "user_001",
    "permissions": ["training", "preprocessing", "model_download"],
    "created_at": "2025-02-12T08:00:00Z",
    "expires_at": null
  },
  "timestamp": 1704067200
}
```

---

## 11. 限流和配额

### 11.1 限流规则

- **认证接口**: 60 次/分钟
- **训练任务**: 10 次/小时
- **预处理**: 20 次/小时
- **GPU 查询**: 无限制
- **日志查询**: 100 次/分钟

### 11.2 配额检查

**响应头**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067260
```

**超限响应** (429):

```json
{
  "code": 429,
  "message": "请求频率超限",
  "data": {
    "limit": 100,
    "remaining": 0,
    "reset_time": "2025-02-12T08:01:00Z",
    "retry_after": 60
  },
  "timestamp": 1704067200
}
```

---

## 12. SDK 示例

### 12.1 Python SDK

```python
import requests

class DiffusionPipeClient:
    def __init__(self, api_key, base_url="http://localhost:5000/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_available_gpus(self):
        response = requests.get(
            f"{self.base_url}/gpu/available",
            headers=self.headers
        )
        return response.json()

    def start_training(self, gpu_id, model_type, config):
        response = requests.post(
            f"{self.base_url}/training/start",
            headers=self.headers,
            json={
                "gpu_id": gpu_id,
                "model_type": model_type,
                "config": config
            }
        )
        return response.json()

    def get_training_progress(self, task_id):
        response = requests.get(
            f"{self.base_url}/training/{task_id}/progress",
            headers=self.headers
        )
        return response.json()

    def get_logs(self, task_id, tail=100):
        response = requests.get(
            f"{self.base_url}/training/{task_id}/logs",
            headers=self.headers,
            params={"tail": tail}
        )
        return response.json()

# 使用示例
client = DiffusionPipeClient("sk_live_1234567890abcdef")

# 1. 获取可用 GPU
gpus = client.get_available_gpus()
print(f"可用 GPU: {gpus['data']['available_gpus']}")

# 2. 启动训练
task = client.start_training(
    gpu_id=0,
    model_type="flux",
    config={
        "epochs": 10,
        "micro_batch_size_per_gpu": 1,
        "adapter": {"type": "lora", "rank": 16}
    }
)
task_id = task['data']['task_id']

# 3. 查询进度
progress = client.get_training_progress(task_id)
print(f"训练进度: {progress['data']['progress']['overall_progress']}%")

# 4. 获取日志
logs = client.get_logs(task_id, tail=50)
for log in logs['data']['logs']:
    print(f"{log['timestamp']} [{log['level']}] {log['message']}")
```

---

## 13. 部署说明

### 13.1 环境变量

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=5000
API_WORKERS=4

# 认证
API_SECRET_KEY=your-secret-key-here
API_KEY_EXPIRE_DAYS=30

# 数据库
DATABASE_URL=postgresql://user:password@localhost/diffusion_pipe

# Redis (用于任务队列)
REDIS_URL=redis://localhost:6379/0

# 存储路径
STORAGE_ROOT=/data
MODELS_ROOT=/data/models
TRAINING_OUTPUT_ROOT=/data/training_runs
PREPROCESSED_DATA_ROOT=/data/preprocessed

# 日志
LOG_LEVEL=INFO
LOG_FILE=/var/log/diffusion-pipe/api.log

# GPU 监控
GPU_MONITOR_INTERVAL=5
```

### 13.2 启动命令

```bash
# 开发模式
python app.py

# 生产模式 (Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Docker 部署
docker-compose up -d
```

---

## 14. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2025-02-12 | 初始版本 |
| | | - 预处理 API |
| | | - GPU 管理 |
| | | - 训练任务 |
| | | - 进度查询 |
| | | - 日志查询 |

---

## 15. 支持和联系

- **文档版本**: v1.0
- **最后更新**: 2025-02-12
- **维护者**: Diffusion-Pipe Team
- **技术支持**: [support@diffusion-pipe.ai](mailto:support@diffusion-pipe.ai)

---

**注意事项**:
1. 所有时间戳使用 ISO 8601 格式 (UTC)
2. 文件路径均为服务器端绝对路径
3. GPU 内存单位为 MB
4. 温度单位为摄氏度
5. 功率单位为瓦特
6. 训练时间估算仅供参考
7. 日志文件可能占用大量磁盘空间，建议定期清理
8. 大型数据集预处理可能需要较长时间，建议异步处理
9. 训练任务一旦启动无法暂停，只能停止
10. 请确保有足够的磁盘空间存储模型和日志
