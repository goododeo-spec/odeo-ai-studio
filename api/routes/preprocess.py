"""
数据处理相关 API 路由
支持视频上传、转换、帧提取、提示词生成
"""
import os
import subprocess
import base64
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils.common import create_response

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/v1/preprocess')

DATASET_PATH = os.environ.get('DATASET_PATH', './data/datasets')
RAW_PATH = os.environ.get('RAW_PATH', './data/datasets/raw')

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(RAW_PATH, exist_ok=True)

# 千问API配置
QWEN_API_KEY = "sk-cebe1cdb99ed44a69d41f194c25ece92"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_video_info(video_path):
    """获取视频的宽高和宽高比"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                width = int(parts[0])
                height = int(parts[1])
                ar = round(width / height, 3) if height > 0 else 1.0
                return {'width': width, 'height': height, 'aspect_ratio': ar}
    except Exception as e:
        print(f"获取视频信息失败: {e}")
    return {'width': 0, 'height': 0, 'aspect_ratio': 1.0}


def get_video_files(directory):
    """获取目录中的所有视频文件（包含宽高比信息）"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
    videos = []
    if not os.path.exists(directory):
        return videos
    for file in os.listdir(directory):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_path = os.path.join(directory, file)
            info = get_video_info(video_path)
            videos.append({
                'filename': file,
                'path': video_path,
                'size': os.path.getsize(video_path),
                'width': info['width'],
                'height': info['height'],
                'aspect_ratio': info['aspect_ratio']
            })
    return sorted(videos, key=lambda x: x['filename'])


def get_processed_videos():
    """获取已处理的视频及其提示词"""
    videos = []
    if not os.path.exists(DATASET_PATH):
        return videos
    for file in sorted(os.listdir(DATASET_PATH)):
        if file.endswith('.mp4'):
            txt_path = os.path.join(DATASET_PATH, file.replace('.mp4', '.txt'))
            caption = ''
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            videos.append({
                'id': len(videos),
                'filename': file,
                'caption': caption
            })
    return videos


def extract_frame(video_path, frame_number=0):
    """从视频中提取指定帧"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_number})",
            "-vframes", "1",
            "-q:v", "2",
            tmp_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return tmp_path
    except Exception as e:
        print(f"提取帧失败: {e}")
    
    return None


def optimize_with_qwen(text, prompt_prefix=""):
    """使用千问文本模型优化提示词"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
        
        # 构建优化请求
        system_prompt = """你是视频训练提示词生成助手。根据用户提供的触发词，生成一段适合视频生成训练的英文描述。

要求：
1. 描述应该自然流畅，像是在描述一个视频片段
2. 包含动作、姿态、环境等元素
3. 50-80个英文单词
4. 直接输出英文描述，不要有任何前缀或解释"""

        user_prompt = f"触发词: {text}\n\n请生成视频训练提示词："
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200
        )
        
        caption = response.choices[0].message.content.strip()
        
        # 清理输出
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        
        # 添加前缀
        if prompt_prefix and prompt_prefix.strip():
            caption = f"{prompt_prefix.strip()}, {caption}"
        
        return caption
        
    except Exception as e:
        print(f"千问API调用失败: {e}")
        # 返回默认模板
        base = prompt_prefix.strip() if prompt_prefix else "A person"
        return f"{base} performing a dynamic action, captured in smooth motion with natural lighting and a clean background."


def generate_default_caption(index, prompt_prefix=""):
    """生成默认提示词模板"""
    base = prompt_prefix.strip() if prompt_prefix else "A person"
    templates = [
        f"{base} performing a dynamic movement with natural fluidity.",
        f"{base} in motion, demonstrating smooth and controlled action.",
        f"{base} moving gracefully with expressive body language.",
        f"{base} executing a characteristic pose with confident energy."
    ]
    return templates[index % len(templates)]


# ============================================
# 原始视频管理
# ============================================

@preprocess_bp.route('/raw', methods=['GET'])
def list_raw_videos():
    """获取原始视频列表（包含宽高比信息）"""
    try:
        videos = get_video_files(RAW_PATH)
        # 计算唯一的宽高比列表（用于AR buckets）
        ar_set = set()
        for v in videos:
            if v.get('aspect_ratio', 0) > 0:
                ar_set.add(v['aspect_ratio'])
        ar_buckets = sorted(list(ar_set))
        
        return jsonify(create_response(code=200, message="success", data={
            'videos': videos, 
            'total': len(videos),
            'ar_buckets': ar_buckets
        }))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/ar-buckets', methods=['GET'])
def get_ar_buckets():
    """获取处理后视频的宽高比列表，用于自动填充AR buckets"""
    try:
        videos = get_video_files(DATASET_PATH)
        ar_set = set()
        for v in videos:
            if v.get('aspect_ratio', 0) > 0:
                ar_set.add(v['aspect_ratio'])
        ar_buckets = sorted(list(ar_set))
        
        # 如果没有处理过的视频，从原始视频获取
        if not ar_buckets:
            raw_videos = get_video_files(RAW_PATH)
            for v in raw_videos:
                if v.get('aspect_ratio', 0) > 0:
                    ar_set.add(v['aspect_ratio'])
            ar_buckets = sorted(list(ar_set))
        
        return jsonify(create_response(code=200, message="success", data={
            'ar_buckets': ar_buckets,
            'count': len(ar_buckets)
        }))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/raw/file/<filename>', methods=['GET'])
def get_raw_video(filename):
    """获取原始视频文件"""
    try:
        video_path = os.path.join(RAW_PATH, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="文件不存在", data=None)), 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/raw/<filename>', methods=['DELETE'])
def delete_raw_video(filename):
    """删除原始视频"""
    try:
        video_path = os.path.join(RAW_PATH, filename)
        if os.path.exists(video_path):
            os.remove(video_path)
            return jsonify(create_response(code=200, message="已删除", data={'filename': filename}))
        return jsonify(create_response(code=404, message="文件不存在", data=None)), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/upload', methods=['POST'])
def upload_videos():
    """上传视频到原始目录"""
    try:
        if 'files' not in request.files:
            return jsonify(create_response(code=400, message="没有上传文件", data=None)), 400
        
        files = request.files.getlist('files')
        uploaded = []
        
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename) or file.filename
                filepath = os.path.join(RAW_PATH, filename)
                file.save(filepath)
                uploaded.append(filename)
        
        return jsonify(create_response(code=200, message=f"已上传 {len(uploaded)} 个文件", data={'files': uploaded, 'count': len(uploaded)}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


# ============================================
# 帧提取预览
# ============================================

@preprocess_bp.route('/frame/<filename>', methods=['GET'])
def get_frame(filename):
    """提取视频的指定帧"""
    try:
        frame_number = int(request.args.get('frame', 0))
        video_path = os.path.join(RAW_PATH, filename)
        
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404
        
        frame_path = extract_frame(video_path, frame_number)
        
        if frame_path and os.path.exists(frame_path):
            response = send_file(frame_path, mimetype='image/jpeg')
            # 延迟删除临时文件
            @response.call_on_close
            def cleanup():
                try:
                    os.remove(frame_path)
                except:
                    pass
            return response
        else:
            return jsonify(create_response(code=500, message="无法提取帧", data=None)), 500
            
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


# ============================================
# 视频处理
# ============================================

@preprocess_bp.route('/videos', methods=['POST'])
def process_videos():
    """处理视频：转换格式、生成提示词"""
    try:
        data = request.get_json() or {}
        
        input_dir = data.get('input_dir', RAW_PATH)
        output_dir = data.get('output_dir', DATASET_PATH)
        prompt_prefix = data.get('prompt_prefix', '')
        frame_number = data.get('frame_number', 30)
        fps = data.get('fps', 16)
        use_qwen = data.get('use_qwen', True)
        caption_method = data.get('caption_method', 'qwen')
        
        if not os.path.exists(input_dir):
            return jsonify(create_response(code=400, message=f"输入目录不存在: {input_dir}", data=None)), 400
        
        os.makedirs(output_dir, exist_ok=True)
        video_files = get_video_files(input_dir)
        
        if not video_files:
            return jsonify(create_response(code=400, message="没有找到视频文件", data=None)), 400
        
        processed = []
        errors = []
        
        for i, video in enumerate(video_files, 1):
            video_path = video['path']
            output_video = os.path.join(output_dir, f"{i}.mp4")
            output_txt = os.path.join(output_dir, f"{i}.txt")
            
            try:
                # 1. 转换视频
                cmd_convert = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    output_video
                ]
                subprocess.run(cmd_convert, check=True, capture_output=True, timeout=300)
                
                # 2. 生成提示词
                if caption_method == 'qwen' and use_qwen and prompt_prefix:
                    # 使用千问文本模型优化触发词
                    caption = optimize_with_qwen(prompt_prefix, "")
                elif prompt_prefix:
                    # 使用用户提供的触发词生成默认模板
                    caption = generate_default_caption(i, prompt_prefix)
                else:
                    # 完全默认模板
                    caption = generate_default_caption(i, "")
                
                # 3. 保存提示词
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(caption.strip())
                
                processed.append({
                    'index': i,
                    'original': video['filename'],
                    'output': f"{i}.mp4",
                    'caption': caption[:100] + '...' if len(caption) > 100 else caption
                })
                
            except subprocess.TimeoutExpired:
                errors.append({'file': video['filename'], 'error': '处理超时'})
            except subprocess.CalledProcessError as e:
                errors.append({'file': video['filename'], 'error': f"FFmpeg错误"})
            except Exception as e:
                errors.append({'file': video['filename'], 'error': str(e)[:100]})
        
        return jsonify(create_response(
            code=200,
            message=f"处理完成: {len(processed)} 成功, {len(errors)} 失败",
            data={'processed': processed, 'errors': errors, 'total': len(video_files)}
        ))
        
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


# ============================================
# 已处理视频管理
# ============================================

@preprocess_bp.route('/list', methods=['GET'])
def list_processed_videos():
    """获取已处理的视频列表"""
    try:
        videos = get_processed_videos()
        return jsonify(create_response(code=200, message="success", data={'videos': videos, 'total': len(videos)}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/caption', methods=['PUT'])
def update_caption():
    """更新视频的提示词"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        caption = data.get('caption', '')
        
        if not filename:
            return jsonify(create_response(code=400, message="文件名不能为空", data=None)), 400
        
        txt_path = os.path.join(DATASET_PATH, filename.replace('.mp4', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        
        return jsonify(create_response(code=200, message="已保存", data={'filename': filename}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    """获取处理后的视频文件"""
    try:
        video_path = os.path.join(DATASET_PATH, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/video/<filename>', methods=['DELETE'])
def delete_video(filename):
    """删除视频及其提示词文件"""
    try:
        video_path = os.path.join(DATASET_PATH, filename)
        txt_path = os.path.join(DATASET_PATH, filename.replace('.mp4', '.txt'))
        
        deleted = []
        if os.path.exists(video_path):
            os.remove(video_path)
            deleted.append(filename)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        
        if not deleted:
            return jsonify(create_response(code=404, message="文件不存在", data=None)), 404
        
        return jsonify(create_response(code=200, message="已删除", data={'deleted': deleted}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500
