# ODEO AI Studio

基于 diffusion-pipe 的视频 LoRA 训练与推理平台，提供现代化的 Web 界面。

## 功能特性

- 🎬 **视频数据处理**: 视频上传、帧提取、自动标注
- 🚀 **模型训练**: 支持 Wan2.1/2.2 视频 LoRA 训练
- 🎨 **推理测试**: LoRA 模型选择、参数配置、视频生成
- 📊 **实时监控**: GPU 状态、训练进度、Loss 曲线
- 💾 **任务管理**: 草稿保存、任务复制、历史记录

## 系统要求

- Python 3.10+
- NVIDIA GPU (支持 CUDA)
- 推荐: 24GB+ 显存 (A100/4090/3090 等)

## 快速开始

### 1. 安装依赖

```bash
cd api
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp ../env.example ../.env

# 编辑配置
vim ../.env
```

主要配置项:
- `STORAGE_ROOT`: 数据存储目录
- `MODELS_ROOT`: 预训练模型目录

### 3. 下载预训练模型

从 HuggingFace 下载 Wan2.1 模型:
```bash
# 使用 huggingface-cli
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./pretrained_models/Wan2.1-I2V-14B-480P

# 或使用镜像 (国内推荐)
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download ...
```

### 4. 启动服务

```bash
python run.py
```

访问: http://localhost:8080

## 目录结构

```
api/
├── app.py              # Flask 应用入口
├── config.py           # 配置文件
├── run.py              # 启动脚本
├── routes/             # API 路由
├── services/           # 业务逻辑
├── models/             # 数据模型
├── templates/          # 前端页面
├── static/             # 静态资源
│   ├── css/           # 样式文件
│   └── js/            # JavaScript
└── utils/              # 工具函数
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| STORAGE_ROOT | 数据存储根目录 | ./data |
| MODELS_ROOT | 预训练模型目录 | ./pretrained_models |
| SECRET_KEY | Flask 密钥 | dev-secret-key |
| LOG_LEVEL | 日志级别 | INFO |

## API 文档

### 训练相关
- `POST /api/v1/training/create` - 创建训练任务
- `GET /api/v1/training/<task_id>` - 获取任务状态
- `POST /api/v1/training/<task_id>/stop` - 停止训练

### 推理相关
- `POST /api/v1/inference/create` - 创建推理任务
- `GET /api/v1/inference/task/<task_id>` - 获取推理状态

### GPU 状态
- `GET /api/v1/gpu/status` - 获取 GPU 状态

## 技术栈

- **后端**: Flask + Python
- **前端**: HTML/CSS/JS (单页应用)
- **训练**: DeepSpeed + PyTorch
- **推理**: ComfyUI (可选)

## 许可证

MIT License
