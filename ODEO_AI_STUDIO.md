# ODEO AI Studio - Wan2.1 视频 LoRA 训练与推理平台

## 项目概述

ODEO AI Studio 是一个基于 Web 的 AI 视频生成工具，专注于 Wan2.1/Wan2.2 视频动作 LoRA 的训练与推理。项目基于 `diffusion-pipe` 训练框架，提供了完整的 GUI 界面。

### 核心功能

1. **模型训练**
   - 支持 Wan2.1-I2V-14B-480P 模型的 LoRA 训练
   - 多 GPU 任务调度与管理
   - 训练参数可视化配置
   - 实时训练进度和 Loss 曲线监控
   - 训练任务草稿保存与历史管理

2. **数据处理**
   - 视频上传与预处理
   - 自动提示词生成
   - 视频帧提取与预览
   - AR Buckets 和 Frame Buckets 自动配置

3. **推理测试**
   - 基于 ComfyUI + WanVideoWrapper 的推理
   - 训练任务 LoRA 选择（任务 → Epoch 两级选择）
   - 测试图库管理（分类文件夹）
   - 推理参数配置
   - 图片尺寸自动适配

4. **GPU 监控**
   - 8×A100 GPU 实时状态监控
   - 显存使用率可视化
   - 任务分配状态显示

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ODEO AI Studio                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (SPA)                                            │
│  ├── index.html (主页面)                                    │
│  ├── main.css (样式)                                       │
│  └── app.js (交互逻辑)                                      │
├─────────────────────────────────────────────────────────────┤
│  Backend (Flask API)                                       │
│  ├── routes/                                               │
│  │   ├── training.py (训练任务 API)                         │
│  │   ├── inference.py (推理 API)                           │
│  │   ├── gallery.py (图库 API)                             │
│  │   ├── gpu.py (GPU 状态 API)                             │
│  │   └── preprocess.py (数据预处理 API)                     │
│  ├── services/                                             │
│  │   ├── training_service.py (训练服务)                     │
│  │   ├── inference_service.py (推理服务)                    │
│  │   ├── gallery_service.py (图库服务)                      │
│  │   └── gpu_service.py (GPU 监控服务)                      │
│  └── utils/                                                │
│      ├── training_wrapper.py (DeepSpeed 训练包装器)         │
│      └── wan_inference.py (Wan2.1 推理包装器)               │
├─────────────────────────────────────────────────────────────┤
│  Training Core (diffusion-pipe)                            │
│  ├── train.py (主训练脚本)                                  │
│  ├── models/ (Wan 模型定义)                                 │
│  └── utils/dataset.py (数据集处理)                          │
├─────────────────────────────────────────────────────────────┤
│  Inference Core (ComfyUI + WanVideoWrapper)                │
│  └── custom_nodes/ComfyUI-WanVideoWrapper/                 │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
/home/disk2/diffusion-pipe/
├── api/                          # Web API 服务
│   ├── app.py                    # Flask 应用入口
│   ├── run.py                    # 服务启动脚本
│   ├── config.py                 # 配置文件
│   ├── routes/                   # API 路由
│   ├── services/                 # 业务逻辑
│   ├── models/                   # 数据模型
│   ├── utils/                    # 工具函数
│   ├── templates/                # HTML 模板
│   │   └── index.html
│   └── static/                   # 静态资源
│       ├── css/main.css
│       └── js/app.js
├── train.py                      # 主训练脚本
├── models/                       # 模型定义
├── examples/                     # 配置示例
│   ├── wan_odeo.toml            # 训练配置
│   └── wan_odeo_data.toml       # 数据集配置
└── submodules/                   # 子模块
    └── ComfyUI/                  # ComfyUI 推理

/home/disk2/lora_training/        # 训练数据与输出
├── datasets/                     # 数据集目录
├── outputs/                      # 训练输出
│   ├── train_*/                 # 训练任务目录
│   │   └── */epoch*/            # Epoch 检查点
│   ├── inference/               # 推理输出
│   └── tasks.json               # 任务持久化
└── gallery/                      # 测试图库
    └── */                       # 分类文件夹

/home/disk2/comfyui/              # ComfyUI 推理服务
└── custom_nodes/
    └── ComfyUI-WanVideoWrapper/ # Wan 视频包装器
        └── example_workflows/   # 推理 Workflow
```

## API 端点

### 训练相关
- `POST /api/v1/training/create` - 创建训练任务
- `GET /api/v1/training/list` - 获取任务列表
- `GET /api/v1/training/<task_id>` - 获取任务详情
- `POST /api/v1/training/<task_id>/stop` - 停止任务
- `POST /api/v1/training/draft` - 保存草稿
- `POST /api/v1/training/copy/<task_id>` - 复制任务
- `DELETE /api/v1/training/delete/<task_id>` - 删除任务

### 推理相关
- `GET /api/v1/inference/tasks-with-loras` - 获取训练任务及其 LoRA
- `POST /api/v1/inference/create` - 创建推理任务
- `GET /api/v1/inference/task/<task_id>` - 获取推理状态
- `GET /api/v1/inference/tasks` - 获取推理历史

### 图库相关
- `GET /api/v1/gallery/folders` - 获取文件夹列表
- `POST /api/v1/gallery/folders` - 创建文件夹
- `DELETE /api/v1/gallery/folders/<name>` - 删除文件夹
- `GET /api/v1/gallery/images` - 获取图片列表
- `POST /api/v1/gallery/images` - 上传图片
- `GET /api/v1/gallery/images/<id>/info` - 获取图片信息（含尺寸）

### GPU 状态
- `GET /api/v1/gpu/status` - 获取 GPU 状态

### 数据预处理
- `POST /api/v1/preprocess/videos` - 处理视频
- `GET /api/v1/preprocess/ar-buckets` - 获取 AR Buckets
- `GET /api/v1/preprocess/frame/<filename>` - 获取视频帧

## 快速启动

```bash
# 启动服务
cd /home/disk2/diffusion-pipe/api
python run.py

# 或使用启动脚本
/home/disk2/start_api.sh
```

访问地址：http://120.48.186.83:8080

## 依赖环境

- Python 3.13
- PyTorch 2.10.0+cu128
- DeepSpeed
- Flask
- ComfyUI + WanVideoWrapper
- 8×NVIDIA A100 80GB

## 作者

ODEO AI Team
