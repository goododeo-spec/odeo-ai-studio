#!/usr/bin/env python
"""
WSGI 入口文件 - 用于 gunicorn 启动
"""
# Gevent monkey patching - 必须在所有其他导入之前
from gevent import monkey
monkey.patch_all()

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
