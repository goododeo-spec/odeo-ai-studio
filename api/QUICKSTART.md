# Diffusion-Pipe API 快速入门指南

## 概述

本文档将帮助您快速上手 Diffusion-Pipe 的 Flask API 服务，实现 GPU 管理、训练任务调度等功能。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /root/diffusion-pipe/api
make install
```

### 2. 启动 API 服务

```bash
# 开发模式 (调试开启)
make dev

# 或后台运行
make background
```

### 3. 测试 API

```bash
# 运行自动化测试
make test

# 或运行演示脚本
python demo.py
```

## 📁 项目结构

```
api/
├── app.py              # Flask 应用主文件
├── config.py           # 配置管理
├── run.py              # 启动脚本
├── Makefile            # 便捷命令
├── requirements.txt    # 依赖包
│
├── routes/             # API 路由
│   └── gpu.py         # GPU 管理接口
│
├── services/           # 业务逻辑
│   └── gpu_service.py # GPU 监控服务
│
├── models/             # 数据模型
│   └── gpu.py         # GPU 数据结构
│
├── utils/              # 工具函数
│   └── common.py      # 通用工具
│
└── tests/              # 测试文件
```

## 🔧 核心功能

### GPU 状态监控

API 实时监控 GPU 的：
- 显存使用情况 (总量/已用/可用)
- GPU 利用率
- 显存利用率
- 温度 (GPU/显存)
- 功耗 (当前/限制)
- 当前任务状态
- 驱动版本

### 内存优化

- 后台线程持续监控 (5 秒间隔)
- 10 秒缓存机制
- 异步任务注册/注销
- 任务进度跟踪

## 📡 API 接口

### 1. 获取所有 GPU 状态

```bash
curl http://localhost:8080/api/v1/gpu/status
```

**响应:**
```json
{
  "code": 200,
  "data": {
    "gpus": [
      {
        "gpu_id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "memory": {
          "total": 24576,
          "used": 8192,
          "free": 16384,
          "utilization": 33
        },
        "utilization_gpu": 75,
        "temperature": {
          "gpu": 65
        },
        "status": "available"
      }
    ],
    "summary": {
      "total_gpus": 2,
      "available_gpus": 1,
      "busy_gpus": 1
    }
  }
}
```

### 2. 获取可用 GPU

```bash
# 所有可用 GPU
curl http://localhost:8080/api/v1/gpu/available

# 至少 10GB 显存的可用 GPU
curl "http://localhost:8080/api/v1/gpu/available?min_memory=10000"
```

### 3. 获取 GPU 详情

```bash
curl http://localhost:8080/api/v1/gpu/0/details
```

### 4. GPU 汇总

```bash
curl http://localhost:8080/api/v1/gpu/summary
```

## 🐍 Python 客户端示例

### 基础用法

```python
import requests

API_BASE = "http://localhost:8080/api/v1"

# 1. 获取所有 GPU
response = requests.get(f"{API_BASE}/gpu/status")
data = response.json()

print(f"发现 {len(data['data']['gpus'])} 个 GPU")

# 2. 查找可用 GPU
available_gpus = [
    gpu for gpu in data['data']['gpus']
    if gpu['status'] == 'available'
]

if available_gpus:
    gpu = available_gpus[0]
    print(f"\n使用 GPU {gpu['gpu_id']}: {gpu['name']}")
    print(f"可用显存: {gpu['memory']['free']} MB")
```

### 高级筛选

```python
# 筛选可用显存 > 10GB 的 GPU
response = requests.get(
    f"{API_BASE}/gpu/available",
    params={"min_memory": 10000}
)

gpus = response.json()['data']['available_gpus']

# 选择最空闲的 GPU
if gpus:
    best_gpu = min(gpus, key=lambda x: x['memory']['utilization'])
    print(f"选择 GPU {best_gpu['gpu_id']} (显存利用率 {best_gpu['memory']['utilization']}%)")
```

### 实时监控

```python
import time

for i in range(10):
    response = requests.get(f"{API_BASE}/gpu/status")
    data = response.json()

    for gpu in data['data']['gpus']:
        print(f"GPU {gpu['gpu_id']}: 利用率 {gpu['utilization_gpu']}%")

    time.sleep(5)  # 每 5 秒监控一次
```

## 🛠️ 常用命令

```bash
# 安装依赖
make install

# 启动服务
make dev

# 后台运行
make background

# 运行测试
make test

# 查看日志
make logs

# 停止服务
make stop

# 查看状态
make status

# 清理日志
make clean

# 查看帮助
make help
```

## 📊 性能特性

### 缓存机制
- GPU 信息缓存 10 秒
- 减少 NVML 调用开销
- 提升 API 响应速度

### 异步监控
- 后台线程持续监控
- 不阻塞主 API 请求
- 实时更新 GPU 状态

### 内存效率
- 按需获取 GPU 信息
- 避免重复查询
- 智能缓存更新

## 🔍 故障排查

### 1. NVML 初始化失败

**症状:**
```
Warning: NVML initialization failed
```

**解决方案:**
```bash
# 检查驱动
nvidia-smi

# 检查 CUDA
nvcc --version

# API 会自动降级到模拟数据
```

### 2. 连接被拒绝

**症状:**
```
requests.exceptions.ConnectionError
```

**解决方案:**
```bash
# 检查服务是否运行
make status

# 查看日志
make logs

# 重新启动
make dev
```

### 3. GPU 信息不准确

**原因:**
- 监控有延迟
- 任务状态更新慢

**解决方案:**
```python
# 等待几秒后重试
time.sleep(10)

# 强制刷新
response = requests.get(f"{API_BASE}/gpu/status")
```

## 📈 扩展开发

### 添加新接口

1. **创建路由** (`routes/example.py`)

```python
from flask import Blueprint
from services.example_service import example_service

example_bp = Blueprint('example', url_prefix='/api/v1/example')

@example_bp.route('/test', methods=['GET'])
def test():
    data = example_service.get_data()
    return jsonify({"data": data})
```

2. **注册蓝图** (`app.py`)

```python
from routes import example
app.register_blueprint(example.example_bp)
```

### 扩展 GPU 监控

```python
# services/gpu_service.py
def _get_gpu_additional_info(self, handle):
    """获取额外 GPU 信息"""
    info = {}

    try:
        # 风扇转速
        info['fan_speed'] = pynvml.nvmlDeviceGetFanSpeed(handle)

        # 时钟频率
        info['graphics_clock'] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_GRAPHICS
        )

        # 显存频率
        info['memory_clock'] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM
        )
    except pynvml.NVMLError:
        pass

    return info
```

## 🎯 最佳实践

### 1. 错误处理

```python
try:
    response = requests.get(f"{API_BASE}/gpu/status")
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"API 请求失败: {e}")
    # 使用缓存数据或默认值
```

### 2. 缓存策略

```python
# 缓存 GPU 列表 (30 秒)
GPU_CACHE_TTL = 30

cached_gpus = None
last_update = 0

def get_cached_gpus():
    global cached_gpus, last_update
    now = time.time()

    if now - last_update > GPU_CACHE_TTL:
        response = requests.get(f"{API_BASE}/gpu/status")
        cached_gpus = response.json()
        last_update = now

    return cached_gpus
```

### 3. 监控告警

```python
def check_gpu_health():
    response = requests.get(f"{API_BASE}/gpu/status")
    data = response.json()

    for gpu in data['data']['gpus']:
        # 温度告警
        if gpu['temperature']['gpu'] > 80:
            print(f"警告: GPU {gpu['gpu_id']} 温度过高 ({gpu['temperature']['gpu']}°C)")

        # 显存不足告警
        if gpu['memory']['free'] < 1000:  # 小于 1GB
            print(f"警告: GPU {gpu['gpu_id']} 显存不足")
```

## 📚 更多资源

- **完整 API 文档**: `/root/diffusion-pipe/TRAINING_API.md`
- **README**: `/root/diffusion-pipe/api/README.md`
- **项目架构**: `/root/diffusion-pipe/ARCHITECTURE.md`

## 🆘 获取帮助

```bash
# 查看帮助
make help

# 查看日志
make logs

# 运行测试
make test
```

---

**快速链接:**
- [API 文档](./README.md)
- [完整训练 API 设计](../TRAINING_API.md)
- [项目架构](../ARCHITECTURE.md)
