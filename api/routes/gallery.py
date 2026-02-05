"""
图库 API 路由
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path

from services.gallery_service import gallery_service, GALLERY_ROOT
from utils.common import create_response

gallery_bp = Blueprint('gallery', __name__, url_prefix='/api/v1/gallery')


@gallery_bp.route('/folders', methods=['GET'])
def list_folders():
    """获取文件夹列表"""
    try:
        folders = gallery_service.list_folders()
        return jsonify(create_response(
            code=200,
            message="success",
            data={'folders': folders, 'total': len(folders)}
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/folders', methods=['POST'])
def create_folder():
    """创建文件夹"""
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify(create_response(code=400, message="文件夹名称不能为空")), 400
        
        folder = gallery_service.create_folder(name)
        return jsonify(create_response(
            code=201,
            message="文件夹创建成功",
            data=folder
        )), 201
    except ValueError as e:
        return jsonify(create_response(code=400, message=str(e))), 400
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/folders/<name>', methods=['DELETE'])
def delete_folder(name):
    """删除文件夹"""
    try:
        if gallery_service.delete_folder(name):
            return jsonify(create_response(code=200, message="文件夹已删除"))
        else:
            return jsonify(create_response(code=404, message="文件夹不存在")), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/images', methods=['GET'])
def list_images():
    """获取图片列表"""
    try:
        folder = request.args.get('folder')
        images = gallery_service.list_images(folder)
        return jsonify(create_response(
            code=200,
            message="success",
            data={'images': images, 'total': len(images)}
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/images', methods=['POST'])
def upload_image():
    """上传图片"""
    try:
        if 'file' not in request.files:
            return jsonify(create_response(code=400, message="没有上传文件")), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(create_response(code=400, message="文件名为空")), 400
        
        folder = request.form.get('folder', 'default')
        
        result = gallery_service.upload_image(folder, file.read(), file.filename)
        return jsonify(create_response(
            code=200,
            message="上传成功",
            data=result
        ))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/images/<path:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """删除图片"""
    try:
        if gallery_service.delete_image(image_id):
            return jsonify(create_response(code=200, message="图片已删除"))
        else:
            return jsonify(create_response(code=404, message="图片不存在")), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/image/<path:image_id>', methods=['GET'])
def get_image(image_id):
    """获取图片文件（原图）"""
    try:
        image_path = GALLERY_ROOT / image_id
        if not image_path.exists():
            return jsonify(create_response(code=404, message="图片不存在")), 404
        return send_file(str(image_path))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/thumbnail/<path:image_id>', methods=['GET'])
def get_thumbnail(image_id):
    """获取图片缩略图（用于快速加载显示）"""
    try:
        thumb_path = gallery_service.get_thumbnail(image_id)
        if thumb_path and thumb_path.exists():
            return send_file(str(thumb_path), mimetype='image/jpeg')
        
        # 如果缩略图不存在，返回原图
        image_path = GALLERY_ROOT / image_id
        if not image_path.exists():
            return jsonify(create_response(code=404, message="图片不存在")), 404
        return send_file(str(image_path))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500


@gallery_bp.route('/images/<path:image_id>/info', methods=['GET'])
def get_image_info(image_id):
    """获取图片信息（包含尺寸）"""
    try:
        info = gallery_service.get_image_info(image_id)
        if info:
            return jsonify(create_response(code=200, message="success", data=info))
        else:
            return jsonify(create_response(code=404, message="图片不存在")), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e))), 500
