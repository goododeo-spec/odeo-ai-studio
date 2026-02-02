# ODEO AI Studio - 项目总结与复盘

## 一、项目概述

### 项目名称
ODEO AI Studio - Wan2.1/2.2 视频 LoRA 训练与推理平台

### 项目目标
基于 diffusion-pipe 构建一个完整的视频 LoRA 训练工具，提供现代化的 Web 界面，支持：
- 视频数据处理与标注
- LoRA 模型训练管理
- 推理测试与结果展示
- GPU 资源监控

### 技术栈
- **后端**: Flask + Python 3.13
- **前端**: 原生 HTML/CSS/JS (单页应用)
- **训练**: DeepSpeed + PyTorch
- **推理**: ComfyUI + WanVideoWrapper
- **模型**: Wan2.1-I2V-14B-480P

---

## 二、项目架构

```
diffusion-pipe/
├── api/                          # Web API 服务
│   ├── app.py                    # Flask 应用入口
│   ├── config.py                 # 配置文件
│   ├── run.py                    # 启动脚本
│   ├── routes/                   # API 路由
│   │   ├── training.py           # 训练任务 API
│   │   ├── inference.py          # 推理 API
│   │   ├── gallery.py            # 图库管理 API
│   │   ├── gpu.py                # GPU 状态 API
│   │   └── preprocess.py         # 数据处理 API
│   ├── services/                 # 业务逻辑层
│   │   ├── training_service.py   # 训练服务
│   │   ├── inference_service.py  # 推理服务
│   │   ├── gallery_service.py    # 图库服务
│   │   └── gpu_service.py        # GPU 监控服务
│   ├── models/                   # 数据模型
│   ├── templates/                # HTML 模板
│   │   └── index.html            # 主页面 (SPA)
│   ├── static/                   # 静态资源
│   │   ├── css/main.css          # 样式表
│   │   └── js/app.js             # 前端逻辑
│   └── utils/                    # 工具模块
│       └── wan_inference.py      # 推理脚本
├── train.py                      # 训练主脚本
├── utils/                        # 训练工具
│   └── dataset.py                # 数据集处理
├── models/                       # 模型定义
│   ├── wan/                      # Wan 模型
│   └── base.py                   # 基础类
└── examples/                     # 配置示例
    ├── wan_odeo.toml             # 训练配置
    └── wan_odeo_data.toml        # 数据集配置
```

---

## 三、功能模块

### 1. 模型训练
- 创建训练任务（自动生成任务名前缀）
- 视频数据上传与预处理
- 训练参数配置（epoch、batch_size、learning_rate 等）
- GPU 选择与监控
- 训练进度实时显示
- Loss 曲线可视化
- 训练日志查看
- 草稿保存功能
- 任务复制/删除

### 2. 推理测试
- LoRA 模型选择（按训练任务分组）
- 多 epoch 版本选择
- 图库系统（分类文件夹管理）
- 测试图片多选
- 自动识别图片尺寸
- 推理参数配置
- 结果视频预览

### 3. 数据处理
- 视频上传
- 帧提取与预览
- 自动/手动标注
- AR Buckets 自动计算
- Frame Buckets 配置

### 4. GPU 监控
- 8 卡 A100 状态显示
- 显存使用率
- 任务占用情况

---

## 四、问题复盘

### 4.1 环境与依赖问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Flask 安装失败 | 默认 pip 源不可用 | 使用清华/阿里云镜像 |
| torch 导入失败 | 服务启动时加载 | 实现懒加载机制 |
| deepspeed 安装失败 | 网络问题 | 切换到阿里云镜像 |
| wandb/tensorboard/multiprocess 缺失 | 依赖不完整 | 逐一安装 |
| safetensors 文件损坏 | 下载中断 | 使用 aria2c 多线程下载 |

### 4.2 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 训练一直显示"创建中" | GPU 状态获取超时 | 异步启动训练 |
| "模型类型不能为空" | 前端未传 model_type | 添加隐藏字段 |
| DeepSpeed GPU 分配错误 | CUDA_VISIBLE_DEVICES 冲突 | 使用 --include 参数 |
| ModuleNotFoundError: utils.common | sys.path 配置错误 | 修复 models/base.py 路径 |
| IndexError: dataset out of range | num_proc > dataset_size | 动态调整 num_proc |
| EADDRINUSE: port 29500 | DeepSpeed 端口冲突 | 动态分配端口 |

### 4.3 GPU 问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| GPU 只显示 2 个 | pynvml 初始化失败 | 改用 nvidia-smi 命令 |
| GPU 显存显示 24GB | MB/GB 单位混乱 | 统一使用 GB 并添加属性 |
| GPU 状态刷新后变化 | 缓存机制问题 | 实现 2 秒缓存 |
| Invalid device id | CUDA 映射问题 | 使用映射后的 device 0 |

### 4.4 前端问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 界面无法加载 | 服务未启动/端口问题 | 检查进程和端口 |
| 刷新后回到首页 | 页面状态未保存 | localStorage 持久化 |
| 弹窗不显示 | CSS .modal-overlay 缺失 | 添加样式 |
| renderUploadedVideos undefined | 函数未定义 | 添加占位函数 |
| 训练日志为空 | 未实现日志轮询 | 添加实时日志获取 |

### 4.5 下载问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型下载慢 | 国内网络限制 | 使用 hf-mirror.com |
| WanVideoWrapper 下载失败 | GitHub 访问受限 | 使用 codeload.github.com |
| wget 断点续传失败 | 服务器不支持 Range | 重新下载 |

---

## 五、关键技术实现

### 5.1 DeepSpeed GPU 指定
```python
# 错误方式：--num_gpus=1 会默认使用 GPU0
# 正确方式：使用 --include 明确指定 GPU
cmd = [
    'deepspeed',
    f'--include=localhost:{gpu_id}',
    '--master_port', str(master_port),
    'train.py',
    '--config', config_path
]
env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
```

### 5.2 GPU 状态获取（nvidia-smi）
```python
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
     '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True, timeout=10
)
```

### 5.3 训练任务持久化
```python
# 保存到 tasks.json
def _save_tasks(self):
    tasks_data = []
    for task in self._tasks.values():
        data = task.to_dict()
        data['raw_videos'] = getattr(task, 'raw_videos', [])
        data['processed_videos'] = getattr(task, 'processed_videos', [])
        tasks_data.append(data)
    with open(self._tasks_file, 'w') as f:
        json.dump(tasks_data, f, indent=2, ensure_ascii=False)
```

### 5.4 Loss 曲线可视化
```javascript
// 使用 Chart.js
function updateLossChart(stepLosses) {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    if (!State.lossChart) {
        State.lossChart = new Chart(ctx, {
            type: 'line',
            data: { datasets: [{ label: 'Loss', data: stepLosses }] }
        });
    } else {
        State.lossChart.data.datasets[0].data = stepLosses;
        State.lossChart.update();
    }
}
```

---

## 六、部署信息

### 服务器配置
- GPU: 8 × NVIDIA A100 80GB
- 操作系统: Linux
- Python: 3.13
- CUDA: 12.8

### 目录结构
```
/home/disk1/pretrained_models/    # 预训练模型
/home/disk2/diffusion-pipe/       # 项目代码
/home/disk2/lora_training/        # 训练输出
    ├── outputs/                  # 训练结果
    ├── datasets/                 # 数据集
    └── gallery/                  # 图库
/home/disk2/comfyui/              # ComfyUI (推理)
```

### 启动命令
```bash
# 启动服务
/home/disk2/start_api.sh

# 或手动启动
cd /home/disk2/diffusion-pipe/api
python run.py

# 访问地址
http://120.48.186.83:8080
```

---

## 七、待优化项

1. **推理服务**: 集成真正的 ComfyUI API 推理
2. **进程监控**: 训练卡死自动重启
3. **多任务队列**: 支持任务排队
4. **模型管理**: 模型下载/删除/版本管理
5. **用户系统**: 多用户支持
6. **数据备份**: 定期备份任务数据

---

## 八、版本历史

- **v1.0** (2026-01-27): 初始版本
- **v1.1** (2026-01-28): 添加训练功能
- **v1.2** (2026-01-29): 添加推理测试
- **v1.3** (2026-01-29): 添加图库系统
- **v1.4** (2026-01-30): 修复 GPU 显示问题
