#!/usr/bin/env python
"""
API 启动脚本
"""
import os
import sys
from pathlib import Path

# 添加 api 目录到 Python 路径（确保api的models和utils优先）
api_root = Path(__file__).parent
sys.path.insert(0, str(api_root))

from app import create_app

def main():
    """主函数"""
    # 获取配置环境
    config_name = os.environ.get('FLASK_ENV', 'development')

    # 创建应用
    app = create_app(config_name)

    # 启动配置
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # 打印启动信息
    print("=" * 60)
    print("Diffusion-Pipe Training API")
    print("=" * 60)
    print(f"环境: {config_name}")
    print(f"调试模式: {debug}")
    print(f"端口: {port}")
    print(f"API 文档: http://localhost:{port}/")
    print("=" * 60)

    # 启动服务器
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

if __name__ == '__main__':
    main()
