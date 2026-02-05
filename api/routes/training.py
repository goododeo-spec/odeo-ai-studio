"""
训练任务相关 API 路由
"""
from flask import Blueprint, request, jsonify
from typing import Optional

from services.training_service import training_service
from utils.common import create_response
from models.training import (
    TrainingTaskRequest, TrainingStatus, StopTaskRequest
)

training_bp = Blueprint('training', __name__, url_prefix='/api/v1/training')


def _get_system_stats(task):
    """获取系统状态，兼容字典和对象"""
    if not task.system_stats:
        return None
    
    stats = task.system_stats
    
    # 如果是字典
    if isinstance(stats, dict):
        return {
            "gpu_memory": stats.get('gpu_memory'),
            "gpu_utilization": stats.get('gpu_utilization'),
            "gpu_temperature": stats.get('gpu_temperature'),
            "gpu_power": stats.get('gpu_power')
        }
    
    # 如果是对象
    try:
        return {
            "gpu_memory": {
                "total": stats.gpu_memory.total if stats.gpu_memory else None,
                "used": stats.gpu_memory.used if stats.gpu_memory else None,
                "free": stats.gpu_memory.free if stats.gpu_memory else None,
                "utilization": stats.gpu_memory.utilization if stats.gpu_memory else None
            } if stats.gpu_memory else None,
            "gpu_utilization": stats.gpu_utilization,
            "gpu_temperature": stats.gpu_temperature,
            "gpu_power": stats.gpu_power
        }
    except Exception:
        return None

@training_bp.route('/start', methods=['POST'])
def start_training():
    """
    创建训练任务

    请求体:
    {
        "gpu_id": 0,
        "model_type": "wan",
        "description": "Wan LoRA 训练 - ODEO 数据集",
        "dataset": {
            "resolutions": [480],
            "enable_ar_bucket": true,
            "ar_buckets": [0.5, 0.563, 0.75],
            "frame_buckets": [1, 29, 49, 97],
            "directory": [
                {
                    "path": "/mnt/disk0/train_data/3",
                    "num_repeats": 5
                }
            ]
        },
        "config": {
            "epochs": 60,
            "micro_batch_size_per_gpu": 1,
            "pipeline_stages": 1,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            "warmup_steps": 20,
            "model": {
                "ckpt_path": "/mnt/disk0/pretrained_models/Wan2.1-I2V-14B-480P",
                "dtype": "bfloat16",
                "transformer_dtype": "float8"
            },
            "adapter": {
                "type": "lora",
                "rank": 32,
                "dtype": "bfloat16"
            },
            "optimizer": {
                "type": "adamw_optimi",
                "lr": 5e-5,
                "betas": [0.9, 0.99]
            },
            "optimization": {
                "activation_checkpointing": true,
                "caching_batch_size": 1,
                "video_clip_mode": "single_beginning"
            }
        }
    }

    返回:
    {
        "code": 201,
        "message": "训练任务已创建",
        "data": {
            "task_id": "train_20250212_07000001",
            "status": "queued",
            "gpu_id": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "model_type": "flux",
            "config": { ... },
            "description": "Flux LoRA 训练 - 风景数据集",
            "created_at": "2025-02-12T07:00:00Z",
            "estimated_start_time": "2025-02-12T07:00:05Z",
            "estimated_duration": 7200,
            "checkpoints": {
                "save_every_n_epochs": 2,
                "last_saved": null,
                "next_save": "epoch 2"
            }
        }
    }
    """
    import sys
    print("[Route] start_training 收到请求", file=sys.stderr, flush=True)
    try:
        data = request.get_json()
        print(f"[Route] 请求数据: task_id={data.get('task_id')}, gpu_id={data.get('gpu_id')}", file=sys.stderr, flush=True)

        # 验证必填参数
        if not data:
            return jsonify(create_response(
                code=400,
                message="请求体不能为空",
                data=None
            )), 400

        gpu_id = data.get('gpu_id')
        model_type = data.get('model_type')

        if gpu_id is None:
            return jsonify(create_response(
                code=400,
                message="GPU ID 不能为空",
                data=None
            )), 400

        if not model_type:
            return jsonify(create_response(
                code=400,
                message="模型类型不能为空",
                data=None
            )), 400

        # 创建训练任务请求
        request_obj = TrainingTaskRequest(
            gpu_id=gpu_id,
            model_type=model_type,
            description=data.get('description'),
            dataset=data.get('dataset'),
            config=data.get('config'),
            raw_videos=data.get('raw_videos', []),
            processed_videos=data.get('processed_videos', [])
        )

        # 创建训练任务，传递前端指定的 task_id（如果有）
        response = training_service.create_training_task(request_obj, data.get('task_id'))

        # 转换为字典
        response_data = {
            "task_id": response.task_id,
            "status": response.status.value,
            "gpu_id": response.gpu_id,
            "gpu_name": response.gpu_name,
            "model_type": response.model_type,
            "config": response.config,
            "description": response.description,
            "created_at": response.created_at.isoformat() if response.created_at else None,
            "estimated_start_time": response.estimated_start_time.isoformat() if response.estimated_start_time else None,
            "estimated_duration": response.estimated_duration,
            "checkpoints": response.checkpoints
        }

        return jsonify(create_response(
            code=201,
            message="训练任务已创建",
            data=response_data
        )), 201

    except ValueError as e:
        return jsonify(create_response(
            code=400,
            message=str(e),
            data=None
        )), 400
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"创建训练任务失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/stop/<task_id>', methods=['POST'])
def stop_training(task_id):
    """
    停止训练任务

    请求参数:
        task_id: 任务ID

    请求体:
    {
        "force": false
    }

    返回:
    {
        "code": 200,
        "message": "训练任务已停止",
        "data": {
            "task_id": "train_20250212_07000001",
            "status": "stopped",
            "stopped_at": "2025-02-12T08:30:00Z",
            "checkpoint_saved": true,
            "last_checkpoint": "epoch 5",
            "training_time": 5400
        }
    }
    """
    try:
        data = request.get_json() or {}
        force = data.get('force', False)

        # 停止训练任务
        response = training_service.stop_training_task(task_id, force=force)

        # 转换为字典
        response_data = {
            "task_id": response.task_id,
            "status": response.status.value,
            "stopped_at": response.stopped_at.isoformat() if response.stopped_at else None,
            "checkpoint_saved": response.checkpoint_saved,
            "last_checkpoint": response.last_checkpoint,
            "training_time": response.training_time
        }

        return jsonify(create_response(
            code=200,
            message="训练任务已停止",
            data=response_data
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"停止训练任务失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/restart/<task_id>', methods=['POST'])
def restart_training(task_id):
    """
    重新提交失败的训练任务
    
    会清除旧日志并重新开始训练
    """
    try:
        response = training_service.restart_task(task_id)
        
        response_data = {
            "task_id": response.task_id,
            "status": response.status.value,
            "gpu_id": response.gpu_id,
            "gpu_name": response.gpu_name,
            "queue_position": getattr(response, 'queue_position', 0)
        }
        
        return jsonify(create_response(
            code=200,
            message="训练任务已重新提交",
            data=response_data
        ))
    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"重新提交失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/list', methods=['GET'])
def list_training():
    """
    获取训练任务列表

    查询参数:
        status: 过滤状态 (queued, running, completed, failed, stopped)
        gpu_id: 过滤GPU
        limit: 返回数量限制 (默认: 50)
        offset: 偏移量 (默认: 0)

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "tasks": [...],
            "pagination": {
                "total": 15,
                "limit": 50,
                "offset": 0,
                "has_more": false
            }
        }
    }
    """
    try:
        # 获取查询参数
        status_str = request.args.get('status')
        gpu_id_str = request.args.get('gpu_id')
        limit_str = request.args.get('limit', '50')
        offset_str = request.args.get('offset', '0')

        # 转换参数
        status = None
        if status_str:
            try:
                status = TrainingStatus(status_str)
            except ValueError:
                return jsonify(create_response(
                    code=400,
                    message=f"无效的状态值: {status_str}",
                    data=None
                )), 400

        gpu_id = None
        if gpu_id_str:
            try:
                gpu_id = int(gpu_id_str)
            except ValueError:
                return jsonify(create_response(
                    code=400,
                    message="GPU ID必须是整数",
                    data=None
                )), 400

        try:
            limit = int(limit_str)
            offset = int(offset_str)
        except ValueError:
            return jsonify(create_response(
                code=400,
                message="limit和offset必须是整数",
                data=None
            )), 400

        # 获取任务列表
        response = training_service.list_training_tasks(
            status=status,
            gpu_id=gpu_id,
            limit=limit,
            offset=offset
        )

        # 转换为字典
        tasks_data = []
        for task in response.tasks:
            task_dict = {
                "task_id": task.task_id,
                "status": task.status.value,
                "gpu_id": task.gpu_id,
                "gpu_name": task.gpu_name,
                "model_type": task.model_type,
                "description": task.description,
                "progress": {
                    "current_epoch": task.progress.current_epoch if task.progress else None,
                    "total_epochs": task.progress.total_epochs if task.progress else None,
                    "current_step": task.progress.current_step if task.progress else None,
                    "total_steps": task.progress.total_steps if task.progress else None,
                    "epoch_progress": task.progress.epoch_progress if task.progress else None,
                    "overall_progress": task.progress.overall_progress if task.progress else None,
                    "eta_seconds": task.progress.eta_seconds if task.progress else None
                } if task.progress else None,
                "metrics": {
                    "current_loss": task.metrics.current_loss if task.metrics else None,
                    "best_loss": task.metrics.best_loss if task.metrics else None,
                    "learning_rate": task.metrics.learning_rate if task.metrics else None,
                    "grad_norm": task.metrics.grad_norm if task.metrics else None
                } if task.metrics else None,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "updated_at": task.updated_at.isoformat() if task.updated_at else None
            }
            tasks_data.append(task_dict)

        response_data = {
            "tasks": tasks_data,
            "pagination": response.pagination
        }

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取训练任务列表失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/<task_id>', methods=['GET'])
def get_training(task_id):
    """
    获取训练任务详情

    路径参数:
        task_id: 任务ID

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": "train_20250212_07000001",
            "status": "running",
            "gpu_id": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "model_type": "flux",
            "config": { ... },
            "description": "Flux LoRA 训练 - 风景数据集",
            "progress": { ... },
            "metrics": { ... },
            "system_stats": { ... },
            "checkpoints": [...],
            "created_at": "2025-02-12T07:00:00Z",
            "started_at": "2025-02-12T07:00:05Z",
            "updated_at": "2025-02-12T07:30:00Z",
            "logs_path": "/data/training_runs/exp1/logs/train.log"
        }
    }
    """
    try:
        # 获取训练任务
        task = training_service.get_training_task(task_id)

        # 转换为字典
        task_dict = {
            "task_id": task.task_id,
            "status": task.status.value,
            "gpu_id": task.gpu_id,
            "gpu_name": task.gpu_name,
            "model_type": task.model_type,
            "config": task.config,
            "description": task.description,
            "progress": {
                "current_epoch": task.progress.current_epoch if task.progress else None,
                "total_epochs": task.progress.total_epochs if task.progress else None,
                "current_step": task.progress.current_step if task.progress else None,
                "total_steps": task.progress.total_steps if task.progress else None,
                "current_batch": task.progress.current_batch if task.progress else None,
                "total_batches": task.progress.total_batches if task.progress else None,
                "epoch_progress": task.progress.epoch_progress if task.progress else None,
                "overall_progress": task.progress.overall_progress if task.progress else None,
                "eta_seconds": task.progress.eta_seconds if task.progress else None,
                "training_time": task.progress.training_time if task.progress else None
            } if task.progress else None,
            "metrics": {
                "current_loss": task.metrics.current_loss if task.metrics else None,
                "best_loss": task.metrics.best_loss if task.metrics else None,
                "learning_rate": task.metrics.learning_rate if task.metrics else None,
                "grad_norm": task.metrics.grad_norm if task.metrics else None,
                "epoch_loss": task.metrics.epoch_loss if task.metrics else None,
                "best_epoch": task.metrics.best_epoch if task.metrics else None,
                "epoch_losses": task.metrics.epoch_losses if task.metrics else [],
                "step_losses": task.metrics.step_losses if task.metrics else []
            } if task.metrics else None,
            "system_stats": _get_system_stats(task),
            "checkpoints": [
                {
                    "epoch": cp.epoch,
                    "step": cp.step,
                    "path": cp.path,
                    "timestamp": cp.timestamp.isoformat() if cp.timestamp else None,
                    "loss": cp.loss
                }
                for cp in task.checkpoints
            ],
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "logs_path": task.logs_path,
            "error_message": task.error_message,
            "dataset": task.dataset,
            "raw_videos": getattr(task, 'raw_videos', []),
            "processed_videos": getattr(task, 'processed_videos', [])
        }

        return jsonify(create_response(
            code=200,
            message="success",
            data=task_dict
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取训练任务详情失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/<task_id>/progress', methods=['GET'])
def get_training_progress(task_id):
    """
    获取训练进度

    路径参数:
        task_id: 任务ID

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": "train_20250212_07000001",
            "status": "running",
            "progress": { ... },
            "metrics": { ... },
            "speed": { ... },
            "updated_at": "2025-02-12T07:30:00Z"
        }
    }
    """
    try:
        # 获取训练进度
        response = training_service.get_training_progress(task_id)

        # 转换为字典
        response_data = {
            "task_id": response.task_id,
            "status": response.status.value,
            "progress": {
                "current_epoch": response.progress.current_epoch if response.progress else None,
                "total_epochs": response.progress.total_epochs if response.progress else None,
                "current_step": response.progress.current_step if response.progress else None,
                "total_steps": response.progress.total_steps if response.progress else None,
                "current_batch": response.progress.current_batch if response.progress else None,
                "total_batches": response.progress.total_batches if response.progress else None,
                "epoch_progress": response.progress.epoch_progress if response.progress else None,
                "overall_progress": response.progress.overall_progress if response.progress else None,
                "eta_seconds": response.progress.eta_seconds if response.progress else None,
                "training_time": response.progress.training_time if response.progress else None
            } if response.progress else None,
            "metrics": {
                "current_loss": response.metrics.current_loss if response.metrics else None,
                "best_loss": response.metrics.best_loss if response.metrics else None,
                "learning_rate": response.metrics.learning_rate if response.metrics else None,
                "grad_norm": response.metrics.grad_norm if response.metrics else None,
                "epoch_loss": response.metrics.epoch_loss if response.metrics else None,
                "best_epoch": response.metrics.best_epoch if response.metrics else None
            } if response.metrics else None,
            "speed": {
                "steps_per_second": response.speed.steps_per_second if response.speed else None,
                "samples_per_second": response.speed.samples_per_second if response.speed else None,
                "time_per_step": response.speed.time_per_step if response.speed else None
            } if response.speed else None,
            "updated_at": response.updated_at.isoformat() if response.updated_at else None
        }

        return jsonify(create_response(
            code=200,
            message="success",
            data=response_data
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取训练进度失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/<task_id>/metrics', methods=['GET'])
def get_training_metrics(task_id):
    """
    获取训练指标历史

    路径参数:
        task_id: 任务ID

    查询参数:
        metric: 指标名称 (loss, lr, grad_norm 等)
        type: 类型 (epoch, step)
        limit: 数据点数量限制

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": "train_20250212_07000001",
            "metric": "loss",
            "type": "epoch",
            "data": [...]
        }
    }
    """
    try:
        # 获取查询参数
        metric = request.args.get('metric', 'loss')
        type_ = request.args.get('type', 'epoch')
        limit_str = request.args.get('limit', '100')

        try:
            limit = int(limit_str)
        except ValueError:
            return jsonify(create_response(
                code=400,
                message="limit必须是整数",
                data=None
            )), 400

        # 获取指标历史
        response = training_service.get_metrics_history(
            task_id=task_id,
            metric=metric,
            type_=type_,
            limit=limit
        )

        return jsonify(create_response(
            code=200,
            message="success",
            data={
                "task_id": response.task_id,
                "metric": response.metric,
                "type": response.type,
                "data": response.data
            }
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取训练指标历史失败: {str(e)}",
            data=None
        )), 500

@training_bp.route('/<task_id>/logs', methods=['GET'])
def get_training_logs(task_id):
    """
    获取训练日志

    路径参数:
        task_id: 任务ID

    查询参数:
        type: 日志类型 (train, eval, system, all)
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        tail: 返回最后 N 行
        since: 返回指定时间之后的日志

    返回:
    {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": "train_20250212_07000001",
            "log_file": "/data/training_runs/exp1/logs/train.log",
            "total_lines": 1250,
            "logs": [...]
        }
    }
    """
    try:
        # 获取查询参数
        type_ = request.args.get('type', 'all')
        level = request.args.get('level')
        tail_str = request.args.get('tail')
        since = request.args.get('since')

        tail = None
        if tail_str:
            try:
                tail = int(tail_str)
            except ValueError:
                return jsonify(create_response(
                    code=400,
                    message="tail必须是整数",
                    data=None
                )), 400

        # 获取日志
        response = training_service.get_training_logs(
            task_id=task_id,
            type_=type_,
            level=level,
            tail=tail,
            since=since
        )

        # 转换为字典
        logs_data = []
        for log in response.logs:
            log_dict = {
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "level": log.level.value if log.level else None,
                "type": log.type.value if log.type else None,
                "message": log.message,
                "epoch": log.epoch,
                "step": log.step
            }
            logs_data.append(log_dict)

        return jsonify(create_response(
            code=200,
            message="success",
            data={
                "task_id": response.task_id,
                "log_file": response.log_file,
                "total_lines": response.total_lines,
                "logs": logs_data
            }
        ))

    except ValueError as e:
        return jsonify(create_response(
            code=404,
            message=str(e),
            data=None
        )), 404
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取训练日志失败: {str(e)}",
            data=None
        )), 500


@training_bp.route('/draft', methods=['POST'])
def save_draft():
    """
    保存训练草稿

    请求体:
    {
        "description": "任务名称",
        "model_type": "wan",
        "gpu_id": 0,
        "config": { ... },
        "dataset": { ... },
        "raw_videos": ["1.mp4", "2.mp4"],
        "processed_videos": [{"filename": "1.mp4", "caption": "..."}]
    }

    返回:
    {
        "code": 201,
        "message": "草稿已保存",
        "data": { "task_id": "draft_xxx" }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify(create_response(
                code=400,
                message="请求体不能为空",
                data=None
            )), 400

        # 保存草稿
        response = training_service.save_draft(data)

        return jsonify(create_response(
            code=201,
            message="草稿已保存",
            data={
                "task_id": response.task_id,
                "description": response.description,
                "status": "draft",
                "created_at": response.created_at.isoformat() if response.created_at else None
            }
        )), 201

    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"保存草稿失败: {str(e)}",
            data=None
        )), 500


@training_bp.route('/copy', methods=['POST'])
def copy_task():
    """
    复制训练任务

    请求体:
    {
        "task_id": "原任务ID",
        "new_name": "新任务名称"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(code=400, message="请求体不能为空", data=None)), 400

        task_id = data.get('task_id')
        new_name = data.get('new_name')

        if not task_id:
            return jsonify(create_response(code=400, message="缺少 task_id", data=None)), 400
        if not new_name:
            return jsonify(create_response(code=400, message="缺少 new_name", data=None)), 400

        # 复制任务
        response = training_service.copy_task(task_id, new_name)

        return jsonify(create_response(
            code=201,
            message="任务复制成功",
            data={
                "task_id": response.task_id,
                "description": response.description,
                "status": response.status.value,
                "created_at": response.created_at.isoformat() if response.created_at else None
            }
        )), 201

    except ValueError as e:
        return jsonify(create_response(code=404, message=str(e), data=None)), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=f"复制任务失败: {str(e)}", data=None)), 500


@training_bp.route('/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """
    删除训练任务
    """
    try:
        training_service.delete_task(task_id)
        return jsonify(create_response(code=200, message="任务已删除", data={"task_id": task_id}))

    except ValueError as e:
        return jsonify(create_response(code=404, message=str(e), data=None)), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=f"删除任务失败: {str(e)}", data=None)), 500
