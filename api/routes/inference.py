"""
推理 API 路由

Workflow 集成说明 (wanvideo_2_1_14B_I2V_odeo.json):
- 节点 58 (LoadImage) ← 图库选择的图片
- 节点 71 (WanVideoLoraSelect) ← 选择的 LoRA 模型
- 节点 81 (TextToLowercase) ← 触发词输入
- 节点 30 (VHS_VideoCombine) → 输出结果
"""
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import subprocess
import json

from services.inference_service import inference_service, OUTPUT_ROOT, TEST_IMAGES_ROOT
from utils.common import create_response

inference_bp = Blueprint('inference', __name__, url_prefix='/api/v1/inference')

# Workflow 配置
WORKFLOW_PATH = Path("/home/disk2/comfyui/user/default/workflows/wanvideo_2_1_14B_I2V_odeo.json")


@inference_bp.route('/workflow-info', methods=['GET'])
def get_workflow_info():
    """
    获取 Workflow 节点映射信息
    
    返回工作流中关键节点的映射关系，便于前端理解参数对应关系
    """
    try:
        workflow_info = {
            "workflow_path": str(WORKFLOW_PATH),
            "workflow_name": "wanvideo_2_1_14B_I2V_odeo",
            "description": "Wan2.1 14B Image-to-Video 推理工作流",
            "node_mapping": {
                "input_image": {
                    "node_id": "58",
                    "node_type": "LoadImage",
                    "parameter": "image_path",
                    "description": "输入图片（从图库选择）"
                },
                "lora_model": {
                    "node_id": "71",
                    "node_type": "WanVideoLoraSelect",
                    "parameters": {
                        "lora_path": "LoRA 文件路径",
                        "lora_strength": "LoRA 强度 (0-1)"
                    },
                    "description": "用户训练的 LoRA 模型"
                },
                "trigger_word": {
                    "node_id": "81",
                    "node_type": "TextToLowercase",
                    "parameter": "trigger_word",
                    "description": "触发词（会自动转为小写）"
                },
                "output_video": {
                    "node_id": "30",
                    "node_type": "VHS_VideoCombine",
                    "output": "output_path",
                    "description": "输出视频文件"
                }
            },
            "additional_nodes": {
                "auto_caption": {
                    "node_id": "77",
                    "node_type": "AILab_QwenVL",
                    "description": "QwenVL 自动图片描述（可选）"
                },
                "prompt_concat": {
                    "node_id": "79",
                    "node_type": "easy promptConcat",
                    "description": "拼接触发词和自动描述"
                },
                "distill_lora": {
                    "node_id": "69",
                    "node_type": "WanVideoLoraSelect",
                    "description": "加速推理的 distill LoRA"
                }
            },
            "default_values": {
                "num_frames": 81,
                "num_steps": 4,
                "guidance_scale": 1.0,
                "lora_strength": 1.0,
                "use_auto_caption": True
            }
        }
        
        # 检查工作流文件是否存在
        workflow_info["workflow_exists"] = WORKFLOW_PATH.exists()
        
        return jsonify(create_response(
            code=200,
            message="success",
            data=workflow_info
        ))
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取工作流信息失败: {str(e)}"
        )), 500


@inference_bp.route('/loras', methods=['GET'])
def list_loras():
    """获取可用的 LoRA 列表"""
    try:
        loras = inference_service.list_available_loras()
        return jsonify(create_response(
            code=200,
            message="success",
            data={'loras': loras, 'total': len(loras)}
        ))
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取 LoRA 列表失败: {str(e)}"
        )), 500


@inference_bp.route('/tasks-with-loras', methods=['GET'])
def list_tasks_with_loras():
    """获取训练任务及其 LoRA（按任务分组）"""
    try:
        tasks = inference_service.list_training_tasks_with_loras()
        return jsonify(create_response(
            code=200,
            message="success",
            data={'tasks': tasks, 'total': len(tasks)}
        ))
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取任务列表失败: {str(e)}"
        )), 500


@inference_bp.route('/test-images', methods=['GET'])
def list_test_images():
    """获取测试图片列表"""
    try:
        images = inference_service.list_test_images()
        return jsonify(create_response(
            code=200,
            message="success",
            data={'images': images, 'total': len(images)}
        ))
    except Exception as e:
        return jsonify(create_response(
            code=500,
            message=f"获取测试图片失败: {str(e)}"
        )), 500


@inference_bp.route('/image/<path:filename>', methods=['GET'])
def get_test_image(filename):
    """获取测试图片"""
    try:
        image_path = TEST_IMAGES_ROOT / filename
        if not image_path.exists():
            return jsonify(create_response(code=404, message="图片不存在")), 404
        return send_file(str(image_path))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/video-frame/<path:filename>', methods=['GET'])
def get_video_frame(filename):
    """获取视频第一帧"""
    try:
        video_path = TEST_IMAGES_ROOT / filename
        if not video_path.exists():
            return jsonify(create_response(code=404, message="视频不存在")), 404
        
        # 使用 ffmpeg 提取第一帧
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', 'select=eq(n\\,0)',
            '-vframes', '1',
            tmp_path
        ]
        subprocess.run(cmd, capture_output=True)
        
        if Path(tmp_path).exists():
            return send_file(tmp_path, mimetype='image/jpeg')
        else:
            return jsonify(create_response(code=500, message="提取帧失败")), 500
            
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/create', methods=['POST'])
def create_inference():
    """
    创建推理任务
    
    Workflow 节点映射 (wanvideo_2_1_14B_I2V_odeo.json):
    - 节点 58 (LoadImage) ← image_path (图库选择的图片)
    - 节点 71 (WanVideoLoraSelect) ← lora_path, lora_strength (选择的 LoRA 模型)
    - 节点 81 (TextToLowercase) ← trigger_word (触发词输入)
    - 节点 30 (VHS_VideoCombine) → 输出结果
    
    请求体:
    {
        "lora_path": "/path/to/lora.safetensors",  # 节点 71: LoRA 文件路径
        "trigger_word": "dancing",                  # 节点 81: 触发词
        "image_path": "/path/to/image.png",         # 节点 58: 输入图片路径
        "lora_strength": 1.0,                       # 节点 71: LoRA 强度 (0-1)
        "use_auto_caption": true,                   # 是否使用 QwenVL 自动描述图片
        "num_frames": 81,                           # 生成帧数
        "num_steps": 4,                             # 采样步数
        "guidance_scale": 1.0,                      # CFG scale
        "seed": -1                                  # 随机种子
    }
    
    或批量创建（多图片）:
    {
        "lora_path": "/path/to/lora.safetensors",
        "trigger_word": "dancing",
        "image_paths": ["/path/1.png", "/path/2.png"],  # 多张图片（节点58）
        "lora_strength": 1.0
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(code=400, message="请求体不能为空")), 400
        
        # 验证必要参数 - 节点 71: LoRA
        if not data.get('lora_path'):
            return jsonify(create_response(code=400, message="lora_path 不能为空（节点 71）")), 400
        
        # 节点 81: 触发词（prompt 或 trigger_word）
        trigger_word = data.get('trigger_word') or data.get('prompt')
        if not trigger_word:
            return jsonify(create_response(code=400, message="trigger_word 不能为空（节点 81）")), 400
        data['trigger_word'] = trigger_word
        
        # 可选参数：是否使用自动描述
        if 'use_auto_caption' not in data:
            data['use_auto_caption'] = True  # 默认启用
        
        # 判断是批量还是单个任务 - 节点 58: 图片
        image_paths = data.get('image_paths', [])
        if image_paths and len(image_paths) > 1:
            # 批量创建任务
            tasks = inference_service.create_batch_inference_tasks(data)
            return jsonify(create_response(
                code=201,
                message=f"已创建 {len(tasks)} 个推理任务",
                data={
                    'tasks': [t.to_dict() for t in tasks],
                    'batch_id': tasks[0].batch_id if tasks else None
                }
            )), 201
        else:
            # 单个任务
            if image_paths:
                data['image_path'] = image_paths[0]
            
            if not data.get('image_path'):
                return jsonify(create_response(code=400, message="image_path 不能为空")), 400
            
            task = inference_service.create_inference_task(data)
            
            return jsonify(create_response(
                code=201,
                message="推理任务已创建",
                data=task.to_dict()
            )), 201
        
    except ValueError as e:
        return jsonify(create_response(code=400, message=str(e))), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(create_response(code=500, message=f"创建任务失败: {str(e)}")), 500


@inference_bp.route('/create-pending', methods=['POST'])
def create_pending_inference():
    """
    创建等待 LoRA 生成的推理任务
    
    请求体:
    {
        "training_task_id": "train_xxx",   # 训练任务 ID
        "epochs": [5, 10, 15],              # 需要推理的 epoch 列表
        "trigger_word": "dancing",          # 触发词
        "image_paths": ["/path/1.png"],     # 图片路径列表
        "lora_strength": 1.0
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(code=400, message="请求体不能为空")), 400
        
        training_task_id = data.get('training_task_id')
        epochs = data.get('epochs', [])
        trigger_word = data.get('trigger_word', '')
        image_paths = data.get('image_paths', [])
        lora_strength = data.get('lora_strength', 1.0)
        batch_id = data.get('batch_id') or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not training_task_id:
            return jsonify(create_response(code=400, message="training_task_id 不能为空")), 400
        if not epochs:
            return jsonify(create_response(code=400, message="epochs 不能为空")), 400
        if not image_paths:
            return jsonify(create_response(code=400, message="image_paths 不能为空")), 400
        
        created_tasks = []
        pending_tasks = []
        
        for epoch in epochs:
            # 检查 LoRA 是否已存在
            lora_path = inference_service._find_lora_path(training_task_id, epoch)
            
            for idx, image_path in enumerate(image_paths):
                params = {
                    'trigger_word': trigger_word,
                    'image_path': image_path,
                    'lora_strength': lora_strength,
                    'batch_id': batch_id,
                    'image_index': idx
                }
                
                if lora_path:
                    # LoRA 已存在，直接创建推理任务
                    task_data = {
                        'lora_path': lora_path,
                        'trigger_word': trigger_word,
                        'image_path': image_path,
                        'lora_strength': lora_strength,
                        'batch_id': batch_id
                    }
                    task = inference_service.create_inference_task(task_data)
                    created_tasks.append(task.to_dict())
                else:
                    # LoRA 未生成，添加到等待队列
                    pending_id = inference_service.add_pending_inference(training_task_id, epoch, params)
                    pending_tasks.append({
                        'pending_id': pending_id,
                        'training_task_id': training_task_id,
                        'epoch': epoch,
                        'image_path': image_path
                    })
        
        return jsonify(create_response(
            code=201,
            message=f"已创建 {len(created_tasks)} 个推理任务，{len(pending_tasks)} 个等待中",
            data={
                'created_tasks': created_tasks,
                'pending_tasks': pending_tasks,
                'batch_id': batch_id
            }
        )), 201
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(create_response(code=500, message=f"创建任务失败: {str(e)}")), 500


@inference_bp.route('/task/<task_id>', methods=['GET'])
def get_inference_task(task_id):
    """获取推理任务状态"""
    try:
        task = inference_service.get_task(task_id)
        if not task:
            return jsonify(create_response(code=404, message="任务不存在")), 404
        
        return jsonify(create_response(
            code=200,
            message="success",
            data=task.to_dict()
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/tasks', methods=['GET'])
def list_inference_tasks():
    """获取推理任务列表
    
    Query params:
    - limit: 最大返回数量 (默认 50)
    - batch_id: 按批次ID筛选
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        batch_id = request.args.get('batch_id')
        
        tasks = inference_service.list_tasks(limit=limit, batch_id=batch_id)
        
        return jsonify(create_response(
            code=200,
            message="success",
            data={
                'tasks': [t.to_dict() for t in tasks],
                'total': len(tasks)
            }
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/task/<task_id>/stop', methods=['POST'])
def stop_inference_task(task_id):
    """停止推理任务"""
    try:
        success = inference_service.stop_task(task_id)
        if success:
            return jsonify(create_response(code=200, message="任务已停止"))
        else:
            return jsonify(create_response(code=400, message="无法停止该任务")), 400
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/task/<task_id>', methods=['DELETE'])
def delete_inference_task(task_id):
    """删除推理任务（停止+删除数据+删除输出目录）"""
    try:
        success = inference_service.delete_task(task_id)
        if success:
            return jsonify(create_response(code=200, message="任务已删除"))
        else:
            return jsonify(create_response(code=404, message="任务不存在")), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/tasks/bulk-delete', methods=['POST'])
def bulk_delete_inference_tasks():
    """批量删除推理任务
    
    请求体:
    {
        "task_ids": ["id1", "id2", ...]           # 方式1: 指定任务ID列表
    }
    或
    {
        "after_date": "2026-02-07",                # 方式2: 按日期+状态筛选
        "status": "failed",
        "keep_one_per_group": true                  # 每组(lora+image)保留最早的一个
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(code=400, message="请求体不能为空")), 400
        
        task_ids_to_delete = []
        
        if 'task_ids' in data:
            # 方式1: 直接指定
            task_ids_to_delete = data['task_ids']
        elif 'after_date' in data:
            # 方式2: 按条件筛选
            after_date = data['after_date']
            status_filter = data.get('status', 'failed')
            keep_one = data.get('keep_one_per_group', False)
            
            all_tasks = inference_service.list_tasks(limit=10000)
            filtered = []
            for t in all_tasks:
                if t.created_at and t.created_at.strftime('%Y-%m-%d') >= after_date:
                    if t.status.value == status_filter:
                        filtered.append(t)
            
            if keep_one:
                # 按 lora_path+image_path 分组，每组保留最早的
                from collections import defaultdict
                groups = defaultdict(list)
                for t in filtered:
                    key = f"{t.lora_path}|{t.image_path}"
                    groups[key].append(t)
                
                for key, tasks in groups.items():
                    if len(tasks) > 1:
                        # 按创建时间排序，保留最早的
                        tasks.sort(key=lambda x: x.created_at or datetime.min)
                        for t in tasks[1:]:
                            task_ids_to_delete.append(t.task_id)
            else:
                task_ids_to_delete = [t.task_id for t in filtered]
        else:
            return jsonify(create_response(code=400, message="需要提供 task_ids 或 after_date")), 400
        
        if not task_ids_to_delete:
            return jsonify(create_response(code=200, message="没有需要删除的任务", data={"deleted": 0}))
        
        deleted = inference_service.bulk_delete_tasks(task_ids_to_delete)
        
        return jsonify(create_response(
            code=200,
            message=f"已删除 {deleted} 个任务",
            data={"deleted": deleted, "requested": len(task_ids_to_delete)}
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/task/<task_id>/retry', methods=['POST'])
def retry_inference_task(task_id):
    """重试失败的推理任务 - 重置原任务状态，不创建新任务"""
    try:
        task = inference_service.retry_task(task_id)
        if not task:
            # 区分任务不存在还是状态不对
            existing = inference_service.get_task(task_id)
            if not existing:
                return jsonify(create_response(code=404, message="任务不存在")), 404
            return jsonify(create_response(code=400, message="只能重试失败的任务")), 400
        
        return jsonify(create_response(
            code=200,
            message="任务已重置，正在重新推理",
            data=task.to_dict()
        ))
        
    except ValueError as e:
        return jsonify(create_response(code=400, message=str(e))), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(create_response(code=500, message=f"重试失败: {str(e)}")), 500


@inference_bp.route('/output/<task_id>/video', methods=['GET'])
def get_output_video(task_id):
    """获取推理输出视频"""
    try:
        task = inference_service.get_task(task_id)
        if not task:
            return jsonify(create_response(code=404, message="任务不存在")), 404
        
        if not task.output_path or not Path(task.output_path).exists():
            return jsonify(create_response(code=404, message="视频不存在")), 404
        
        return send_file(task.output_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@inference_bp.route('/upload-image', methods=['POST'])
def upload_test_image():
    """上传测试图片"""
    try:
        if 'file' not in request.files:
            return jsonify(create_response(code=400, message="没有上传文件")), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(create_response(code=400, message="文件名为空")), 400
        
        # 保存到测试图片目录
        upload_dir = TEST_IMAGES_ROOT / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        import time
        filename = f"{int(time.time())}_{file.filename}"
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        return jsonify(create_response(
            code=200,
            message="上传成功",
            data={
                'path': str(filepath),
                'url': f"/api/v1/inference/image/uploads/{filename}"
            }
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500
