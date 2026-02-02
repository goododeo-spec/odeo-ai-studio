# 数据集预处理 API 实现总结

## 完成的工作

### 1. 创建数据模型 (`/root/diffusion-pipe/api/models/preprocessing.py`)

定义了所有相关的数据结构：
- `PreprocessingRequest`: 预处理请求参数
- `PreprocessingProgress`: 预处理进度信息
- `PreprocessingResult`: 预处理结果
- `DatasetInfo`: 数据集信息
- `PreprocessingResponse`: API响应格式
- `DatasetCheckResponse`: 数据集检查响应

### 2. 实现预处理服务 (`/root/diffusion-pipe/api/services/preprocessing_service.py`)

核心功能实现：
- **数据集检查**: `check_dataset_exists()` - 检查数据集是否已存在
- **任务创建**: `create_preprocessing_task()` - 创建异步预处理任务
- **状态查询**: `get_task_status()` - 获取任务实时状态
- **数据集管理**: `get_all_datasets()`, `get_dataset()` - 管理数据集列表

预处理流程（异步执行）：
1. 验证输入参数和目录
2. 初始化千问客户端（可选）
3. 扫描视频文件
4. 循环处理每个视频：
   - 转换视频为16fps MP4
   - 提取首帧
   - 使用Florence-2生成提示词
   - 使用千问优化提示词（可选）
   - 保存提示词到txt文件
5. 清理临时文件（首帧图片）

### 3. 创建 API 路由 (`/root/diffusion-pipe/api/routes/preprocessing.py`)

实现了5个REST API端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/v1/preprocessing/check/<dataset_name>` | GET | 检查数据集是否存在 |
| `/api/v1/preprocessing/datasets` | GET | 获取所有数据集列表 |
| `/api/v1/preprocessing/datasets/<dataset_name>` | GET | 获取指定数据集信息 |
| `/api/v1/preprocessing/start` | POST | 开始新的预处理任务 |
| `/api/v1/preprocessing/status/<task_id>` | GET | 获取任务状态和进度 |

### 4. 更新应用配置 (`/root/diffusion-pipe/api/app.py`)

- 导入预处理路由蓝图
- 注册预处理蓝图到Flask应用

### 5. 创建测试脚本 (`/root/diffusion-pipe/api/test_preprocessing_api.py`)

完整的测试用例：
- 数据集存在性检查
- 数据集列表获取
- 重复数据集创建验证
- 新任务创建
- 任务状态查询
- Flask路由注册验证

### 6. 编写 API 文档 (`/root/diffusion-pipe/api/PREPROCESSING_API.md`)

详细的文档包括：
- 完整的API端点说明
- 请求/响应示例
- 参数说明
- Python和cURL使用示例
- 输出结构说明
- 注意事项和错误码

## 核心特性

### ✅ 数据集去重
- 通过数据集名称保证唯一性
- 创建前检查是否存在
- 已存在返回409错误

### ✅ 异步处理
- 任务创建后立即返回task_id
- 后台异步执行预处理
- 实时查询任务状态和进度

### ✅ 进度追踪
- 实时显示处理进度百分比
- 当前处理的视频文件名
- 当前执行的具体步骤
- 预估剩余时间

### ✅ 灵活配置
- 7种提示词生成方法可选
- 支持千问API优化（可选）
- 支持自定义提示词前缀
- 支持自定义千问API密钥

### ✅ 错误处理
- 输入参数验证
- 文件存在性检查
- 异常捕获和错误信息返回
- 任务状态追踪（pending/running/completed/failed/cancelled）

### ✅ 数据持久化
- 所有数据集存储在 `/mnt/disk0/lora_outputs/`
- 按数据集名称分目录
- 自动加载已存在的数据集

## 测试结果

所有测试用例通过：
- ✅ 数据集检查功能正常
- ✅ 获取数据集列表成功（发现2个已存在数据集）
- ✅ 重复数据集检查正确抛出异常
- ✅ 新任务创建成功
- ✅ 任务状态查询正常
- ✅ Flask应用路由注册成功（5个预处理路由）

## 使用方式

### 启动 API 服务

```bash
# 激活环境
conda activate lora

# 启动服务
cd /root/diffusion-pipe/api
python run.py

# 或使用makefile
make dev
make background
```

### API 调用示例

```python
import requests

# 1. 检查数据集
resp = requests.get('http://localhost:8080/api/v1/preprocessing/check/my_dataset')

# 2. 开始预处理
resp = requests.post('http://localhost:8080/api/v1/preprocessing/start', json={
    'dataset_name': 'my_dataset',
    'video_directory': '/path/to/videos',
    'caption_method': 'extra_mixed',
    'use_qwen_optimize': True
})

task_id = resp.json()['data']['preprocessing_id']

# 3. 轮询状态
while True:
    resp = requests.get(f'http://localhost:8080/api/v1/preprocessing/status/{task_id}')
    data = resp.json()['data']
    print(f"进度: {data['progress']['progress_percent']:.1f}%")

    if data['status'] in ['completed', 'failed']:
        break
```

## 文件结构

```
/root/diffusion-pipe/api/
├── models/
│   └── preprocessing.py          # 数据模型
├── services/
│   └── preprocessing_service.py  # 预处理服务
├── routes/
│   └── preprocessing.py          # API路由
├── app.py                        # Flask应用（已更新）
├── test_preprocessing_api.py     # 测试脚本
├── PREPROCESSING_API.md          # API文档
└── PREPROCESSING_API_SUMMARY.md  # 本文件
```

## 输出示例

预处理完成后的目录结构：
```
/mnt/disk0/lora_outputs/my_dataset/
├── 1.mp4
├── 1.txt    # "A high quality, ..."
├── 2.mp4
├── 2.txt    # "A high quality, ..."
└── ...
```

每个视频文件对应一个提示词文本文件，可用于LoRA训练。

## 总结

数据集预处理API已完全实现并测试通过，提供了完整的REST API接口，支持异步处理、进度追踪、错误处理等企业级功能。API设计遵循RESTful规范，文档完善，易于集成和使用。
