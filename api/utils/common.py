"""
通用工具函数
"""
from datetime import datetime
from typing import Dict, Any, Optional
import psutil

# 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def create_response(
    code: int = 200,
    message: str = "success",
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建标准 API 响应格式

    Args:
        code: 状态码
        message: 消息
        data: 数据

    Returns:
        标准响应格式
    """
    return {
        "code": code,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now().timestamp())
    }

def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息

    Returns:
        系统信息字典
    """
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total,
        "memory_used": psutil.virtual_memory().used,
        "memory_available": psutil.virtual_memory().available,
        "disk_total": psutil.disk_usage('/').total,
        "disk_used": psutil.disk_usage('/').used,
        "disk_free": psutil.disk_usage('/').free
    }

def format_bytes(bytes_value: int) -> str:
    """
    格式化字节数

    Args:
        bytes_value: 字节数

    Returns:
        格式化后的字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def format_duration(seconds: int) -> str:
    """
    格式化时间 duration

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def check_cuda_available() -> bool:
    """
    检查 CUDA 是否可用

    Returns:
        是否可用
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()

def get_cuda_version() -> Optional[str]:
    """
    获取 CUDA 版本

    Returns:
        CUDA 版本字符串
    """
    if not TORCH_AVAILABLE:
        return None
    if check_cuda_available():
        return torch.version.cuda
    return None

def get_cudnn_version() -> Optional[str]:
    """
    获取 cuDNN 版本

    Returns:
        cuDNN 版本字符串
    """
    if not TORCH_AVAILABLE:
        return None
    if check_cuda_available():
        return torch.backends.cudnn.version()
    return None
