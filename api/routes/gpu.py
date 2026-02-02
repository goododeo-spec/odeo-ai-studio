"""
GPU 相关 API 路由
"""
from flask import Blueprint, request, jsonify
from typing import Optional

from services.gpu_service import gpu_service
from utils.common import create_response
from models.gpu import GPUStatus

gpu_bp = Blueprint('gpu', __name__, url_prefix='/api/v1/gpu')

@gpu_bp.route('/status', methods=['GET'])
def get_gpu_status():
    """
    获取所有 GPU 状态

    返回所有 GPU 的实时状态信息，包括显存、利用率、温度等
    """
    try:
        gpu_response = gpu_service.get_all_gpus()

        # 转换为可序列化格式
        data = {
            "gpus": [
                {
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "memory": {
                        "total": gpu.memory.total,
                        "used": gpu.memory.used,
                        "free": gpu.memory.free,
                        "utilization": gpu.memory.utilization,
                        "total_gb": gpu.memory_gb,
                        "free_gb": gpu.free_memory_gb
                    },
                    "utilization_gpu": gpu.utilization,
                    "temperature": {
                        "gpu": gpu.temperature.gpu,
                        "memory": gpu.temperature.memory
                    },
                    "power_usage": gpu.power.current if gpu.power else None,
                    "power_limit": gpu.power.limit if gpu.power else None,
                    "power_utilization": gpu.power.usage_percent if gpu.power else None,
                    "status": gpu.status.value,
                    "current_task": {
                        "task_id": gpu.current_task.task_id if gpu.current_task else None,
                        "task_type": gpu.current_task.task_type.value if gpu.current_task else None,
                        "progress": gpu.current_task.progress if gpu.current_task else None,
                        "estimated_remaining_time": gpu.current_task.estimated_remaining_time if gpu.current_task else None
                    } if gpu.current_task else None,
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version
                }
                for gpu in gpu_response.gpus
            ],
            "summary": gpu_response.summary,
            "timestamp": gpu_response.timestamp.isoformat()
        }

        return jsonify(create_response(code=200, message="success", data=data))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取 GPU 状态失败: {str(e)}",
            data=None
        )), 500

@gpu_bp.route('/available', methods=['GET'])
def get_available_gpus():
    """
    获取可用 GPU 列表

    查询参数:
        min_memory: 最小可用显存 (MB)
        task_type: 任务类型 (training, preprocessing)

    返回当前可用的 GPU 列表
    """
    try:
        # 获取查询参数
        min_memory = request.args.get('min_memory', type=int)
        task_type = request.args.get('task_type', type=str)

        # 获取可用 GPU
        gpu_response = gpu_service.get_available_gpus(
            min_memory=min_memory,
            task_type=task_type
        )

        # 转换为可序列化格式
        data = {
            "available_gpus": [
                {
                    "gpu_id": gpu.gpu_id,
                    "name": gpu.name,
                    "memory_total": gpu.memory.total,
                    "memory_free": gpu.memory.free,
                    "memory_total_gb": gpu.memory_gb,
                    "memory_free_gb": gpu.free_memory_gb,
                    "status": gpu.status.value,
                    "utilization_gpu": gpu.utilization,
                    "temperature_gpu": gpu.temperature.gpu,
                    "driver_version": gpu.driver_version
                }
                for gpu in gpu_response.available_gpus
            ],
            "total_available": len(gpu_response.available_gpus),
            "timestamp": gpu_response.timestamp.isoformat()
        }

        return jsonify(create_response(code=200, message="success", data=data))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取可用 GPU 失败: {str(e)}",
            data=None
        )), 500

@gpu_bp.route('/<int:gpu_id>/details', methods=['GET'])
def get_gpu_details(gpu_id: int):
    """
    获取指定 GPU 详细信息

    Args:
        gpu_id: GPU ID

    返回指定 GPU 的详细信息
    """
    try:
        gpu_response = gpu_service.get_all_gpus()

        # 查找指定 GPU
        gpu_info = None
        for gpu in gpu_response.gpus:
            if gpu.gpu_id == gpu_id:
                gpu_info = gpu
                break

        if not gpu_info:
            return jsonify(create_response(
                code=404,
                message=f"GPU {gpu_id} 不存在",
                data=None
            )), 404

        # 转换为可序列化格式
        data = {
            "gpu_id": gpu_info.gpu_id,
            "name": gpu_info.name,
            "memory": {
                "total": gpu_info.memory.total,
                "used": gpu_info.memory.used,
                "free": gpu_info.memory.free,
                "utilization": gpu_info.memory.utilization,
                "total_gb": gpu_info.memory_gb,
                "free_gb": gpu_info.free_memory_gb
            },
            "utilization": {
                "gpu": gpu_info.utilization,
                "memory": gpu_info.memory.utilization
            },
            "temperature": {
                "gpu": gpu_info.temperature.gpu,
                "memory": gpu_info.temperature.memory
            },
            "power": {
                "current": gpu_info.power.current if gpu_info.power else None,
                "limit": gpu_info.power.limit if gpu_info.power else None,
                "usage_percent": gpu_info.power.usage_percent if gpu_info.power else None
            },
            "status": gpu_info.status.value,
            "current_task": {
                "task_id": gpu_info.current_task.task_id if gpu_info.current_task else None,
                "task_type": gpu_info.current_task.task_type.value if gpu_info.current_task else None,
                "progress": gpu_info.current_task.progress if gpu_info.current_task else None,
                "start_time": gpu_info.current_task.start_time.isoformat() if gpu_info.current_task else None,
                "estimated_remaining_time": gpu_info.current_task.estimated_remaining_time if gpu_info.current_task else None
            } if gpu_info.current_task else None,
            "driver_version": gpu_info.driver_version,
            "cuda_version": gpu_info.cuda_version,
            "is_available": gpu_info.is_available,
            "timestamp": gpu_response.timestamp.isoformat()
        }

        return jsonify(create_response(code=200, message="success", data=data))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取 GPU {gpu_id} 详细信息失败: {str(e)}",
            data=None
        )), 500

@gpu_bp.route('/summary', methods=['GET'])
def get_gpu_summary():
    """
    获取 GPU 汇总信息

    返回所有 GPU 的汇总统计信息
    """
    try:
        gpu_response = gpu_service.get_all_gpus()
        summary = gpu_response.summary

        # 添加额外统计信息
        summary["timestamp"] = gpu_response.timestamp.isoformat()
        summary["updated_at"] = gpu_response.timestamp.isoformat()

        return jsonify(create_response(code=200, message="success", data=summary))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取 GPU 汇总信息失败: {str(e)}",
            data=None
        )), 500
