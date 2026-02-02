# Diffusion-Pipe Training API

Diffusion-Pipe 训练框架的 Flask 后端 API 服务，提供 GPU 管理、训练任务调度等功能。

## 功能特性

- ✅ GPU 状态监控 - 实时获取 GPU 显存、利用率、温度、功耗
- ✅ 可用 GPU 查询 - 根据显存要求筛选可用 GPU
- ✅ 任务管理 - 训练任务注册、进度跟踪
- ✅ RESTful API - 标准化的 API 接口
- ✅ 实时监控 - 后台线程持续监控 GPU 状态
- ✅ CORS 支持 - 支持跨域请求
- ✅ 详细日志 - 完整的请求和错误日志

## 目录结构

```
api/
├── app.py                 # Flask 应用主文件
├── config.py              # 配置管理
├── run.py                 # 启动脚本
├── requirements.txt       # 依赖包
├── routes/                # API 路由
│   ├── gpu.py            # GPU 相关接口
│   └── __init__.py
├── services/              # 业务逻辑
│   ├── gpu_service.py    # GPU 管理服务
│   └── __init__.py
├── models/                # 数据模型
│   ├── gpu.py           # GPU 数据模型
│   └── __init__.py
├── utils/                 # 工具函数
│   ├── common.py        # 通用工具
│   └── __init__.py
├── tests/                 # 测试文件
└── README.md             # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
cd /root/diffusion-pipe/api
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
# 开发模式
python run.py

# 或指定端口
PORT=8080 python run.py

# 后台运行
nohup python run.py > api.log 2>&1 &
```

### 3. 测试 API

```bash
# 运行测试脚本
python test_gpu_api.py
```

### 4. 查看 API 文档

访问: http://localhost:8080/

## API 接口

### GPU 管理

#### 1. 获取所有 GPU 状态

```bash
GET /api/v1/gpu/status
```

**响应示例:**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "gpus": [
      {
        "gpu_id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "memory": {
          "total": 24576,
          "used": 8192,
          "free": 16384,
          "utilization": 33,
          "total_gb": 24.0,
          "free_gb": 16.0
        },
        "utilization_gpu": 75,
        "temperature": {
          "gpu": 65,
          "memory": null
        },
        "power_usage": 250,
        "power_limit": 450,
        "status": "available",
        "current_task": null
      }
    ],
    "summary": {
      "total_gpus": 2,
      "available_gpus": 1,
      "busy_gpus": 1,
      "total_memory": 49152,
      "available_memory": 20480
    }
  }
}
```

#### 2. 获取可用 GPU 列表

```bash
GET /api/v1/gpu/available
GET /api/v1/gpu/available?min_memory=10000  # 至少 10GB 显存
GET /api/v1/gpu/available?task_type=training
```

#### 3. 获取指定 GPU 详情

```bash
GET /api/v1/gpu/{gpu_id}/details
```

#### 4. 获取 GPU 汇总信息

```bash
GET /api/v1/gpu/summary
```

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PORT` | 8080 | API 服务端口 |
| `FLASK_ENV` | development | 运行环境 |
| `FLASK_DEBUG` | False | 调试模式 |
| `LOG_LEVEL` | INFO | 日志级别 |
| `CORS_ORIGINS` | * | 允许的跨域源 |

## 配置说明

### config.py

```python
# GPU 监控间隔 (秒)
GPU_MONITOR_INTERVAL = 5

# 最大并发任务数
MAX_CONCURRENT_TASKS = 4

# 存储路径
STORAGE_ROOT = Path('/data')
MODELS_ROOT = STORAGE_ROOT / 'models'
TRAINING_OUTPUT_ROOT = STORAGE_ROOT / 'training_runs'
```

## 使用示例

### Python 客户端

```python
import requests

API_BASE = "http://localhost:8080/api/v1"

# 1. 获取所有 GPU 状态
response = requests.get(f"{API_BASE}/gpu/status")
gpus = response.json()['data']['gpus']

# 2. 查找可用 GPU
available_gpus = [
    gpu for gpu in gpus
    if gpu['status'] == 'available' and gpu['memory']['free'] >= 10000
]

# 3. 选择 GPU 并开始训练
if available_gpus:
    gpu = available_gpus[0]
    print(f"使用 GPU {gpu['gpu_id']}: {gpu['name']}")
```

### curl 测试

```bash
# 获取所有 GPU 状态
curl http://localhost:8080/api/v1/gpu/status

# 获取可用 GPU (至少 10GB 显存)
curl http://localhost:8080/api/v1/gpu/available?min_memory=10000

# 获取 GPU 0 详情
curl http://localhost:8080/api/v1/gpu/0/details
```

## 故障排查

### 1. NVML 初始化失败

```
Warning: NVML initialization failed: CUDA driver version is insufficient
```

**解决方案:**
- 确保已安装 NVIDIA 驱动
- 确保 CUDA 版本匹配
- 使用模拟数据 (会自动降级)

### 2. 无法连接到 API

```
requests.exceptions.ConnectionError
```

**解决方案:**
- 检查 API 服务是否启动: `ps aux | grep run.py`
- 检查端口是否被占用: `netstat -tulpn | grep 8080`
- 查看日志: `tail -f api.log`

### 3. GPU 信息不准确

**原因:**
- GPU 监控有 10 秒缓存
- 任务状态可能延迟更新

**解决方案:**
- 等待几秒后重试
- 重启 API 服务

## 开发指南

### 添加新接口

1. 在 `routes/` 创建路由文件
2. 在 `services/` 实现业务逻辑
3. 在 `models/` 定义数据模型
4. 在 `app.py` 注册蓝图

示例:

```python
# routes/example.py
from flask import Blueprint
from services.example_service import example_service

example_bp = Blueprint('example', url_prefix='/api/v1/example')

@example_bp.route('/test', methods=['GET'])
def test():
    data = example_service.get_data()
    return jsonify({"data": data})

# app.py
from routes import example
app.register_blueprint(example.example_bp)
```

### 扩展 GPU 监控

```python
# services/gpu_service.py
def _get_additional_gpu_info(self, handle):
    # 添加更多 GPU 信息
    info = {}
    try:
        # 风扇转速
        info['fan_speed'] = pynvml.nvmlDeviceGetFanSpeed(handle)
        # 时钟频率
        info['graphics_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    except pynvml.NVMLError:
        pass
    return info
```

## 生产部署

### 使用 Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 'app:app'
```

### 使用 Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
```

### 使用 systemd

```ini
# /etc/systemd/system/diffusion-pipe-api.service
[Unit]
Description=Diffusion-Pipe API
After=network.target

[Service]
Type=simple
User=api
WorkingDirectory=/root/diffusion-pipe/api
ExecStart=/usr/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl enable diffusion-pipe-api
sudo systemctl start diffusion-pipe-api
sudo systemctl status diffusion-pipe-api
```

## 日志管理

### 日志位置

- 开发模式: 控制台输出
- 生产模式: `/data/logs/api.log`

### 日志轮转

- 最大文件: 10MB
- 保留数量: 10 个文件

### 查看日志

```bash
# 实时日志
tail -f /data/logs/api.log

# 错误日志
grep ERROR /data/logs/api.log

# 最近 100 行
tail -100 /data/logs/api.log
```

## 性能优化

### 1. 缓存 GPU 信息

- 默认缓存 10 秒
- 可通过 `GPU_MONITOR_INTERVAL` 调整

### 2. 异步处理

- GPU 监控在后台线程运行
- 不阻塞主 API 请求

### 3. 连接池

- 使用 requests.Session
- 复用 TCP 连接

## 安全注意事项

### 1. 生产环境配置

```python
# config.py
SECRET_KEY = os.environ.get('SECRET_KEY')  # 必须设置
DEBUG = False
TESTING = False
```

### 2. API 认证

建议添加 API Key 认证:

```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if not api_key or api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated
```

### 3. 限流

使用 Flask-Limiter:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@gpu_bp.route('/status')
@limiter.limit("10/minute")
def get_status():
    pass
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

GPL-3.0

## 联系方式

- 项目主页: https://github.com/tdrussell/diffusion-pipe
- 问题反馈: https://github.com/tdrussell/diffusion-pipe/issues

## 更新日志

### v1.0.0 (2025-02-12)

- 初始版本
- 实现 GPU 状态监控
- 实现可用 GPU 查询
- 实现 GPU 详细信息获取
- 添加测试脚本
- 完善文档
