"""
推理 API 路由
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import subprocess

from services.inference_service import inference_service, OUTPUT_ROOT, TEST_IMAGES_ROOT
from utils.common import create_response

inference_bp = Blueprint('inference', __name__, url_prefix='/api/v1/inference')


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
    """创建推理任务"""
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(code=400, message="请求体不能为空")), 400
        
        # 验证必要参数
        if not data.get('lora_path'):
            return jsonify(create_response(code=400, message="lora_path 不能为空")), 400
        if not data.get('prompt'):
            return jsonify(create_response(code=400, message="prompt 不能为空")), 400
        
        task = inference_service.create_inference_task(data)
        
        return jsonify(create_response(
            code=201,
            message="推理任务已创建",
            data=task.to_dict()
        )), 201
        
    except ValueError as e:
        return jsonify(create_response(code=400, message=str(e))), 400
    except Exception as e:
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
    """获取推理任务列表"""
    try:
        limit = request.args.get('limit', 50, type=int)
        tasks = inference_service.list_tasks(limit=limit)
        
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
