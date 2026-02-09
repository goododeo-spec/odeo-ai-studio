#!/usr/bin/env python
"""
WSGI 入口文件 - 用于 gunicorn 启动

注意: 不再使用 gevent monkey patching。
应用使用 threading.Thread 实现后台任务（队列处理、LoRA监控、状态同步等），
gevent 的 monkey.patch_all() 会将 Thread 转为 greenlet，
导致 worker 初始化阶段卡死。
现在使用 gthread worker，与标准 threading 完全兼容。
"""
import os
import sys
from pathlib import Path

# 添加 api 目录到 Python 路径
api_root = Path(__file__).parent
sys.path.insert(0, str(api_root))

from app import create_app

# 获取配置环境
config_name = os.environ.get('FLASK_ENV', 'development')

# 创建应用实例
app = create_app(config_name)

if __name__ == '__main__':
    app.run()
