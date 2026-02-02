"""
Flask 应用主文件
LoRA Training Studio - 现代化训练管理平台
"""
import os
import sys
from pathlib import Path

# 确保 api 目录在 sys.path 最前面（而不是项目根目录）
api_root = Path(__file__).parent
if str(api_root) not in sys.path:
    sys.path.insert(0, str(api_root))

from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

from config import config
from services.gpu_service import gpu_service
from routes.gpu import gpu_bp
from routes.preprocessing import preprocessing_bp
from routes.training import training_bp
from routes.preprocess import preprocess_bp
from routes.inference import inference_bp
from routes.gallery import gallery_bp

def create_app(config_name='default'):
    """
    创建 Flask 应用

    Args:
        config_name: 配置名称

    Returns:
        Flask 应用实例
    """
    # 设置静态文件和模板路径
    app = Flask(
        __name__,
        static_folder='static',
        static_url_path='/static',
        template_folder='templates'
    )

    # 加载配置
    app.config.from_object(config[config_name])

    # 配置 CORS
    CORS(
        app,
        origins=app.config.get('CORS_ORIGINS', ['*']),
        methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allow_headers=['Content-Type', 'Authorization']
    )

    # 配置日志
    _setup_logging(app)

    # 注册蓝图
    _register_blueprints(app)

    # 注册错误处理器
    _register_error_handlers(app)

    # 启动 GPU 监控
    gpu_service.start_monitoring(
        interval=app.config.get('GPU_MONITOR_INTERVAL', 5)
    )

    # 添加健康检查端点
    @app.route('/health')
    def health_check():
        """健康检查端点"""
        return jsonify({
            "status": "healthy",
            "version": "2.0.0"
        })

    # 添加API信息端点
    @app.route('/api')
    def api_info():
        """API 信息端点"""
        return jsonify({
            "name": "LoRA Training Studio API",
            "version": "2.0.0",
            "status": "running",
            "endpoints": {
                "training": "/api/v1/training",
                "gpu": "/api/v1/gpu",
                "preprocessing": "/api/v1/preprocessing"
            }
        })

    # 前端页面路由 - 返回 SPA 入口
    @app.route('/')
    def index():
        """主页 - 训练管理控制台"""
        return render_template('index.html')

    # 处理前端路由（SPA 支持）
    @app.route('/<path:path>')
    def catch_all(path):
        """
        捕获所有前端路由，返回 SPA 入口页面
        排除 API 和静态文件路径
        """
        # 如果是API请求或静态文件，不处理
        if path.startswith('api/') or path.startswith('static/'):
            return jsonify({"code": 404, "message": "Not found"}), 404
        # 返回 SPA 入口
        return render_template('index.html')

    return app

def _setup_logging(app):
    """配置日志"""
    if not app.debug and not app.testing:
        # 创建日志目录
        log_dir = app.config.get('LOG_FILE', Path('logs/api.log')).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # 文件处理器
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('API 启动')

def _register_blueprints(app):
    """注册蓝图"""
    app.register_blueprint(gpu_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(preprocess_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(gallery_bp)

def _register_error_handlers(app):
    """注册错误处理器"""

    @app.errorhandler(404)
    def not_found(error):
        """404 错误处理器"""
        return jsonify({
            "code": 404,
            "message": "请求的资源不存在",
            "data": None
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """500 错误处理器"""
        app.logger.error(f"服务器内部错误: {str(error)}")
        return jsonify({
            "code": 500,
            "message": "服务器内部错误",
            "data": None
        }), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        """未处理异常处理器"""
        app.logger.error(f"未处理异常: {str(error)}", exc_info=True)
        return jsonify({
            "code": 500,
            "message": "服务器内部错误",
            "data": None
        }), 500

# 创建应用实例
app = create_app(os.environ.get('FLASK_ENV', 'development'))

if __name__ == '__main__':
    # 开发模式启动
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config.get('DEBUG', False)
    )
