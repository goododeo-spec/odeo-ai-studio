# 数据集预处理 API 文档

## 概述

数据集预处理 API 用于将视频数据集转换为适合 LoRA 训练的格式，包括：
- 视频格式转换（16fps MP4）
- 首帧提取
- 提示词生成（使用 Florence-2 模型）
- 千问优化提示词

## 基础信息

- **基础 URL**: `http://localhost:8080/api/v1/preprocessing`
- **数据格式**: JSON
- **认证方式**: 无

## API 端点

### 1. 检查数据集是否存在

检查指定名称的数据集是否已存在。

**请求**:
```http
GET /api/v1/preprocessing/check/{dataset_name}
```

**路径参数**:
- `dataset_name`: 数据集名称

**响应示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "exists": true,
        "dataset_name": "my_dataset",
        "directory": "/mnt/disk0/lora_outputs/my_dataset",
        "created_at": "2024-01-01T12:00:00",
        "video_count": 100,
        "message": "数据集已存在"
    }
}
```

### 2. 获取所有数据集列表

获取所有已存在的数据集列表。

**请求**:
```http
GET /api/v1/preprocessing/datasets
```

**响应示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "datasets": [
            {
                "dataset_name": "my_dataset",
                "directory": "/mnt/disk0/lora_outputs/my_dataset",
                "created_at": "2024-01-01T12:00:00",
                "video_count": 100,
                "file_count": 200,
                "status": "completed"
            }
        ],
        "total": 1
    }
}
```

### 3. 获取指定数据集信息

获取指定数据集的详细信息。

**请求**:
```http
GET /api/v1/preprocessing/datasets/{dataset_name}
```

**路径参数**:
- `dataset_name`: 数据集名称

**响应示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "dataset_name": "my_dataset",
        "directory": "/mnt/disk0/lora_outputs/my_dataset",
        "created_at": "2024-01-01T12:00:00",
        "video_count": 100,
        "file_count": 200,
        "status": "completed",
        "last_preprocessing_id": "uuid-string"
    }
}
```

### 4. 开始数据集预处理

开始一个新的数据集预处理任务。

**请求**:
```http
POST /api/v1/preprocessing/start
Content-Type: application/json

{
    "dataset_name": "my_dataset",           // 数据集名称（必填）
    "video_directory": "/path/to/videos",   // 视频目录路径（必填）
    "prompt_prefix": "A high quality",      // 提示词前缀（可选）
    "caption_method": "extra_mixed",        // 提示词生成方法（可选，默认extra_mixed）
    "use_qwen_optimize": true,              // 是否使用千问优化（可选，默认true）
    "qwen_api_key": "sk-..."                // 千问API密钥（可选）
}
```

**参数说明**:
- `dataset_name`: 数据集名称，必须唯一
- `video_directory`: 包含视频文件的目录路径
- `prompt_prefix`: 提示词前缀，会添加到所有生成的提示词前
- `caption_method`: 提示词生成方法，可选值：
  - `tags`: 生成标签
  - `simple`: 简单描述
  - `detailed`: 详细描述
  - `extra`: 额外详细描述
  - `mixed`: 混合描述
  - `extra_mixed`: 额外混合描述（推荐）
  - `analyze`: 分析
- `use_qwen_optimize`: 是否使用千问优化提示词
- `qwen_api_key`: 千问API密钥，如果不提供会从环境变量 `DASHSCOPE_API_KEY` 读取

**响应示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "preprocessing_id": "uuid-string",
        "dataset_name": "my_dataset",
        "status": "pending",
        "message": "预处理任务已创建",
        "output_directory": "/mnt/disk0/lora_outputs/my_dataset"
    }
}
```

### 5. 获取预处理任务状态

获取指定预处理任务的当前状态和进度。

**请求**:
```http
GET /api/v1/preprocessing/status/{task_id}
```

**路径参数**:
- `task_id`: 预处理任务ID

**响应示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "preprocessing_id": "uuid-string",
        "dataset_name": "my_dataset",
        "status": "running",
        "output_directory": "/mnt/disk0/lora_outputs/my_dataset",
        "progress": {
            "total_videos": 100,
            "processed_videos": 45,
            "current_video": "video_045.mp4",
            "current_step": "处理视频 45/100: video_045.mp4",
            "progress_percent": 45.0,
            "start_time": "2024-01-01T12:00:00",
            "estimated_remaining_time": 3300
        }
    }
}
```

**状态说明**:
- `pending`: 等待开始
- `running`: 正在处理
- `completed`: 已完成
- `failed`: 处理失败
- `cancelled`: 已取消

## 使用示例

### Python 示例

```python
import requests
import time

# 1. 检查数据集是否存在
response = requests.get('http://localhost:8080/api/v1/preprocessing/check/my_dataset')
if response.json()['data']['exists']:
    print("数据集已存在")
    exit(1)

# 2. 开始预处理
response = requests.post('http://localhost:8080/api/v1/preprocessing/start', json={
    'dataset_name': 'my_dataset',
    'video_directory': '/path/to/my/videos',
    'prompt_prefix': 'A high quality',
    'caption_method': 'extra_mixed',
    'use_qwen_optimize': True,
    'qwen_api_key': 'sk-your-api-key'
})

task_id = response.json()['data']['preprocessing_id']
print(f"预处理任务已创建，ID: {task_id}")

# 3. 轮询任务状态
while True:
    response = requests.get(f'http://localhost:8080/api/v1/preprocessing/status/{task_id}')
    data = response.json()['data']

    status = data['status']
    progress = data['progress']

    print(f"状态: {status}")
    print(f"进度: {progress['progress_percent']:.1f}%")
    print(f"当前步骤: {progress['current_step']}")

    if status in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(5)
```

### cURL 示例

```bash
# 检查数据集
curl http://localhost:8080/api/v1/preprocessing/check/my_dataset

# 开始预处理
curl -X POST http://localhost:8080/api/v1/preprocessing/start \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "my_dataset",
    "video_directory": "/path/to/videos",
    "prompt_prefix": "A high quality",
    "caption_method": "extra_mixed",
    "use_qwen_optimize": true
  }'

# 获取任务状态
curl http://localhost:8080/api/v1/preprocessing/status/{task_id}
```

## 输出结构

预处理完成后，在 `/mnt/disk0/lora_outputs/{dataset_name}/` 目录下会生成以下文件：

```
/mnt/disk0/lora_outputs/my_dataset/
├── 1.mp4          # 第1个视频（16fps）
├── 1.txt          # 第1个视频的提示词
├── 2.mp4          # 第2个视频（16fps）
├── 2.txt          # 第2个视频的提示词
└── ...
```

每个视频文件对应一个文本文件，文本文件包含该视频的提示词。

## 注意事项

1. **数据集名称必须唯一**：如果尝试创建已存在的数据集，API会返回409错误
2. **视频目录必须存在**：如果视频目录不存在，API会返回400错误
3. **千问API密钥**：如果不提供，会从环境变量 `DASHSCOPE_API_KEY` 读取
4. **模型路径**：Florence-2 模型必须预先下载到 `/mnt/disk0/pretrained_models/Florence-2-base-PromptGen-v2.0/`
5. **依赖工具**：系统必须安装 ffmpeg
6. **输出目录**：所有数据集存储在 `/mnt/disk0/lora_outputs/` 目录下，按数据集名称分目录

## 错误码

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `409`: 数据集已存在
- `500`: 服务器内部错误
