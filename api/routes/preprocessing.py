"""
预处理相关 API 路由
"""
from flask import Blueprint, request, jsonify
from typing import Optional

from services.preprocessing_service import preprocessing_service
from utils.common import create_response
from models.preprocessing import (
    PreprocessingRequest, PreprocessingResponse, PreprocessingListResponse,
    DatasetCheckResponse, CaptionMethod
)

preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/api/v1/preprocessing')

@preprocessing_bp.route('/start', methods=['POST'])
def start_preprocessing():
    """
    开始数据集预处理

    请求体:
    {
        "dataset_name": "my_dataset",  # 数据集名称（必填）
        "video_directory": "/path/to/videos",  # 视频目录路径（必填）
        "prompt_prefix": "A high quality",  # 提示词前缀（可选）
        "caption_method": "extra_mixed",  # 提示词生成方法（可选，默认extra_mixed）
        "use_qwen_optimize": true,  # 是否使用千问优化（可选，默认true）
        "qwen_api_key": "sk-..."  # 千问API密钥（可选）
    }

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "preprocessing_id": "uuid",
            "dataset_name": "my_dataset",
            "status": "pending",
            "message": "预处理任务已创建"
        }
    }
    """
    try:
        data = request.get_json()

        # 验证必填参数
        if not data:
            return jsonify(create_response(
                code=400,
                message="请求体不能为空",
                data=None
            )), 400

        dataset_name = data.get('dataset_name')
        video_directory = data.get('video_directory')

        if not dataset_name:
            return jsonify(create_response(
                code=400,
                message="数据集名称不能为空",
                data=None
            )), 400

        if not video_directory:
            return jsonify(create_response(
                code=400,
                message="视频目录路径不能为空",
                data=None
            )), 400

        # 验证数据集是否已存在
        exists, _ = preprocessing_service.check_dataset_exists(dataset_name)
        if exists:
            return jsonify(create_response(
                code=409,
                message=f"数据集 '{dataset_name}' 已存在",
                data=None
            )), 409

        # 验证视频目录是否存在
        import os
        if not os.path.exists(video_directory):
            return jsonify(create_response(
                code=400,
                message=f"视频目录不存在: {video_directory}",
                data=None
            )), 400

        # 解析参数
        prompt_prefix = data.get('prompt_prefix')
        caption_method_str = data.get('caption_method', 'extra_mixed')
        use_qwen_optimize = data.get('use_qwen_optimize', True)
        qwen_api_key = data.get('qwen_api_key')

        # 验证caption_method
        try:
            caption_method = CaptionMethod(caption_method_str)
        except ValueError:
            valid_methods = [m.value for m in CaptionMethod]
            return jsonify(create_response(
                code=400,
                message=f"无效的caption_method: {caption_method_str}，可选值: {valid_methods}",
                data=None
            )), 400

        # 创建预处理请求
        request_obj = PreprocessingRequest(
            dataset_name=dataset_name,
            video_directory=video_directory,
            prompt_prefix=prompt_prefix,
            caption_method=caption_method,
            use_qwen_optimize=use_qwen_optimize,
            qwen_api_key=qwen_api_key
        )

        # 创建预处理任务
        task_id = preprocessing_service.create_preprocessing_task(request_obj)

        response = PreprocessingResponse(
            preprocessing_id=task_id,
            dataset_name=dataset_name,
            message="预处理任务已创建",
            data={
                "task_id": task_id,
                "output_directory": f"/mnt/disk0/lora_outputs/{dataset_name}"
            }
        )

        return jsonify(create_response(
            code=200,
            message="success",
            data={
                "preprocessing_id": response.preprocessing_id,
                "dataset_name": response.dataset_name,
                "status": response.status.value,
                "message": response.message,
                "output_directory": response.data["output_directory"]
            }
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=400,
            message=str(e),
            data=None
        )), 400
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"创建预处理任务失败: {str(e)}",
            data=None
        )), 500

@preprocessing_bp.route('/status/<task_id>', methods=['GET'])
def get_preprocessing_status(task_id: str):
    """
    获取预处理任务状态

    路径参数:
        task_id: 预处理任务ID

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "preprocessing_id": "uuid",
            "dataset_name": "my_dataset",
            "status": "running",
            "progress": {
                "total_videos": 100,
                "processed_videos": 45,
                "current_video": "video_045.mp4",
                "current_step": "处理视频 45/100: video_045.mp4",
                "progress_percent": 45.0
            },
            "output_directory": "/mnt/disk0/lora_outputs/my_dataset"
        }
    }
    """
    try:
        result = preprocessing_service.get_task_status(task_id)

        if not result:
            return jsonify(create_response(
                code=404,
                message=f"任务 {task_id} 不存在",
                data=None
            )), 404

        response_data = {
            "preprocessing_id": task_id,
            "dataset_name": result.dataset_name,
            "status": result.status.value,
            "output_directory": result.output_directory,
            "progress": {
                "total_videos": result.progress.total_videos,
                "processed_videos": result.progress.processed_videos,
                "current_video": result.progress.current_video,
                "current_step": result.progress.current_step,
                "progress_percent": result.progress.progress_percent,
                "start_time": result.progress.start_time.isoformat(),
                "estimated_remaining_time": result.progress.estimated_remaining_time
            }
        }

        if result.error_message:
            response_data["error_message"] = result.error_message

        if result.output_files:
            response_data["output_files"] = result.output_files

        if result.end_time:
            response_data["end_time"] = result.end_time.isoformat()

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取任务状态失败: {str(e)}",
            data=None
        )), 500

@preprocessing_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """
    获取所有数据集列表

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "datasets": [
                {
                    "dataset_name": "my_dataset",
                    "directory": "/mnt/disk0/lora_outputs/my_dataset",
                    "created_at": "2024-01-01T00:00:00",
                    "video_count": 100,
                    "file_count": 200,
                    "status": "completed"
                }
            ],
            "total": 1
        }
    }
    """
    try:
        datasets = preprocessing_service.get_all_datasets()

        response_data = {
            "datasets": [
                {
                    "dataset_name": d.dataset_name,
                    "directory": d.directory,
                    "created_at": d.created_at.isoformat(),
                    "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                    "video_count": d.video_count,
                    "file_count": d.file_count,
                    "status": d.status.value,
                    "last_preprocessing_id": d.last_preprocessing_id
                }
                for d in datasets
            ],
            "total": len(datasets)
        }

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取数据集列表失败: {str(e)}",
            data=None
        )), 500

@preprocessing_bp.route('/datasets/<dataset_name>', methods=['GET'])
def get_dataset(dataset_name: str):
    """
    获取指定数据集信息

    路径参数:
        dataset_name: 数据集名称

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "dataset_name": "my_dataset",
            "directory": "/mnt/disk0/lora_outputs/my_dataset",
            "created_at": "2024-01-01T00:00:00",
            "video_count": 100,
            "file_count": 200,
            "status": "completed",
            "last_preprocessing_id": "uuid"
        }
    }
    """
    try:
        dataset = preprocessing_service.get_dataset(dataset_name)

        if not dataset:
            return jsonify(create_response(
                code=404,
                message=f"数据集 '{dataset_name}' 不存在",
                data=None
            )), 404

        response_data = {
            "dataset_name": dataset.dataset_name,
            "directory": dataset.directory,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
            "video_count": dataset.video_count,
            "file_count": dataset.file_count,
            "status": dataset.status.value,
            "last_preprocessing_id": dataset.last_preprocessing_id
        }

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取数据集信息失败: {str(e)}",
            data=None
        )), 500

@preprocessing_bp.route('/check/<dataset_name>', methods=['GET'])
def check_dataset(dataset_name: str):
    """
    检查数据集是否存在

    路径参数:
        dataset_name: 数据集名称

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "exists": true,
            "dataset_name": "my_dataset",
            "directory": "/mnt/disk0/lora_outputs/my_dataset",
            "created_at": "2024-01-01T00:00:00",
            "video_count": 100,
            "message": "数据集已存在"
        }
    }
    """
    try:
        exists, dataset = preprocessing_service.check_dataset_exists(dataset_name)

        response_data = {
            "exists": exists,
            "dataset_name": dataset_name,
            "message": "数据集已存在" if exists else "数据集不存在"
        }

        if dataset:
            response_data.update({
                "directory": dataset.directory,
                "created_at": dataset.created_at.isoformat(),
                "video_count": dataset.video_count
            })

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"检查数据集失败: {str(e)}",
            data=None
        )), 500
