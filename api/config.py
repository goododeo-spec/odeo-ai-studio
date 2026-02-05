"""
API 配置文件
"""
import os
from pathlib import Path

class Config:
    """基础配置"""
    # Flask 配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # 上传配置 - 优化大文件上传速度
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10GB 最大上传限制
    UPLOAD_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB 缓冲区

    # API 配置
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'

    # 存储路径 (通过环境变量配置，支持不同部署环境)
    STORAGE_ROOT = Path(os.environ.get('STORAGE_ROOT', './data'))
    MODELS_ROOT = Path(os.environ.get('MODELS_ROOT', './pretrained_models'))
    TRAINING_OUTPUT_ROOT = STORAGE_ROOT / 'outputs'
    PREPROCESSED_DATA_ROOT = STORAGE_ROOT / 'datasets'
    DATASETS_ROOT = STORAGE_ROOT / 'datasets'
    GALLERY_ROOT = STORAGE_ROOT / 'gallery'

    # GPU 监控配置
    GPU_MONITOR_INTERVAL = 5  # 秒
    GPU_STATUS_CHECK_TIMEOUT = 10  # 秒

    # 任务配置
    MAX_CONCURRENT_TASKS = 4
    TASK_CLEANUP_INTERVAL = 3600  # 1小时清理一次过期任务

    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = STORAGE_ROOT / 'logs' / 'api.log'

    # CORS 配置
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True
    TESTING_STORAGE = Path('/tmp/test_diffusion_pipe')

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
