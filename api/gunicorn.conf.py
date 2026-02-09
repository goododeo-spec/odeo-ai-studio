# Gunicorn 配置文件 - 优化大文件上传
import multiprocessing
import os
from pathlib import Path

# 项目根目录（gunicorn.conf.py 位于 api/ 下，项目根在上一级）
_PROJECT_ROOT = str(Path(__file__).parent.parent)

# 设置环境变量默认值（确保所有 worker 都能访问）
# 这些值仅在环境变量未设置时生效，实际部署时应通过 .env 或 start_api.sh 设置
_storage = os.environ.get('STORAGE_ROOT', os.path.join(_PROJECT_ROOT, 'data'))
os.environ.setdefault('STORAGE_ROOT', _storage)
os.environ.setdefault('MODELS_ROOT', os.path.join(_PROJECT_ROOT, 'pretrained_models'))
os.environ.setdefault('TRAINING_OUTPUT_ROOT', os.path.join(_storage, 'outputs'))
os.environ.setdefault('DATASET_PATH', os.path.join(_storage, 'datasets'))
os.environ.setdefault('RAW_PATH', os.path.join(_storage, 'raw'))
os.environ.setdefault('GALLERY_ROOT', os.path.join(_storage, 'gallery'))
os.environ.setdefault('INFERENCE_OUTPUT_ROOT', os.path.join(_storage, 'outputs', 'inference'))
os.environ.setdefault('LORA_ROOT', os.path.join(_storage, 'outputs'))

# 绑定地址
bind = "0.0.0.0:8080"

# Worker 配置
# 使用单 worker 避免多进程内存不同步问题
# 训练/推理等长时间任务使用后台线程处理，不影响性能
workers = 1
# 注意: 不能使用 gevent worker！
# gevent 的 monkey.patch_all() 会把 threading.Thread 转为 greenlet，
# 导致应用中的后台线程（队列处理、LoRA监控、状态同步等）在 worker 初始化阶段卡死。
# gthread 使用真正的线程，与应用的 threading 模型完全兼容。
worker_class = "gthread"  # 使用线程 worker，兼容 threading.Thread
threads = 8  # 每个 worker 的线程数，支持并发请求

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

# 禁用 max_requests：训练子进程的 stdout 管道绑定在 worker 上，
# worker 重启会导致管道断裂（SIGPIPE），静默杀死训练进程。
# 即使后续改为日志直写文件，也建议保持禁用以确保稳定性。
max_requests = 0
max_requests_jitter = 0
