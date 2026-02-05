# Gunicorn 配置文件 - 优化大文件上传
import multiprocessing
import os

# 设置环境变量（确保所有 worker 都能访问）
os.environ.setdefault('STORAGE_ROOT', '/home/disk2/lora_training')
os.environ.setdefault('MODELS_ROOT', '/home/disk1/pretrained_models')
os.environ.setdefault('TRAINING_OUTPUT_ROOT', '/home/disk2/lora_training/outputs')
os.environ.setdefault('DATASET_PATH', '/home/disk2/lora_training/datasets')
os.environ.setdefault('RAW_PATH', '/home/disk2/lora_training/raw')
os.environ.setdefault('GALLERY_ROOT', '/home/disk2/lora_training/gallery')
os.environ.setdefault('INFERENCE_OUTPUT_ROOT', '/home/disk2/lora_training/outputs/inference')
os.environ.setdefault('LORA_ROOT', '/home/disk2/lora_training/outputs')

# 绑定地址
bind = "0.0.0.0:8080"

# Worker 配置
# 使用单 worker 避免多进程内存不同步问题
# 训练/推理等长时间任务使用后台线程处理，不影响性能
workers = 1
worker_class = "gevent"  # 使用 gevent 异步 worker
worker_connections = 1000  # 每个 worker 的最大并发连接数

# 超时配置（大文件上传需要更长时间）
timeout = 600  # 10 分钟超时
graceful_timeout = 60
keepalive = 5

# 请求配置
limit_request_line = 0  # 不限制请求行长度
limit_request_fields = 100
limit_request_field_size = 0  # 不限制请求字段大小

# 日志
accesslog = "-"  # 输出到标准输出
errorlog = "-"
loglevel = "info"

# 进程名
proc_name = "odeo-ai-studio"

# 预加载应用（加快启动）
preload_app = False

# 最大请求数后重启 worker（防止内存泄漏）
max_requests = 1000
max_requests_jitter = 100
