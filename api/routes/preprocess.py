"""
数据处理相关 API 路由
支持视频上传、转换、帧提取、提示词生成
"""
import os
import subprocess
import base64
import tempfile
import requests
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils.common import create_response

preprocess_bp = Blueprint('preprocess', __name__, url_prefix='/api/v1/preprocess')

# 基础路径
BASE_DATASET_PATH = os.environ.get('DATASET_PATH', '/home/disk2/lora_training/datasets')
BASE_RAW_PATH = os.environ.get('RAW_PATH', '/home/disk2/lora_training/raw')

# 全局路径（向后兼容）
DATASET_PATH = BASE_DATASET_PATH
RAW_PATH = BASE_RAW_PATH

os.makedirs(BASE_DATASET_PATH, exist_ok=True)
os.makedirs(BASE_RAW_PATH, exist_ok=True)


def get_task_raw_path(task_id=None):
    """获取任务特定的原始视频路径"""
    if task_id:
        path = os.path.join(BASE_RAW_PATH, task_id)
        os.makedirs(path, exist_ok=True)
        return path
    return BASE_RAW_PATH


def get_task_dataset_path(task_id=None):
    """获取任务特定的处理后视频路径"""
    if task_id:
        path = os.path.join(BASE_DATASET_PATH, task_id)
        os.makedirs(path, exist_ok=True)
        return path
    return BASE_DATASET_PATH

# 千问 VL API 配置（从环境变量获取）
QWEN_VL_API_KEY = os.environ.get('QWEN_VL_API_KEY', '')
QWEN_VL_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


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


def video_to_base64(video_path):
    """将视频文件转换为 base64 数据 URL"""
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # 获取文件大小（MB）
        size_mb = len(video_data) / (1024 * 1024)
        print(f"[Preprocess] 视频大小: {size_mb:.2f} MB", flush=True)
        
        # 转为 base64
        b64_data = base64.b64encode(video_data).decode('utf-8')
        return f"data:video/mp4;base64,{b64_data}"
    except Exception as e:
        print(f"[Preprocess] 视频转 base64 失败: {e}", flush=True)
        return None


def describe_video_with_qwen_vl(video_path, trigger_word=""):
    """使用 Qwen VL API 描述视频内容（帧提取模式）"""
    print(f"[Preprocess] 开始处理视频（帧提取模式）: {video_path}", flush=True)
    return _describe_video_by_frames(video_path, trigger_word)


def _extract_video_frames(video_path, num_frames=8):
    """从视频中均匀提取指定数量的帧，返回 base64 JPEG 列表"""
    frames = []
    try:
        # 获取视频总帧数
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        total_frames = int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else 100
        
        # 计算采样间隔
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(step * i) for i in range(num_frames)]
        
        print(f"[Preprocess] 视频总帧数: {total_frames}, 采样帧: {frame_indices}", flush=True)
        
        for idx in frame_indices:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", f"select=eq(n\\,{idx})",
                    "-vframes", "1",
                    "-q:v", "3",
                    tmp_path
                ]
                subprocess.run(cmd, capture_output=True, timeout=15)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    with open(tmp_path, 'rb') as f:
                        frame_b64 = base64.b64encode(f.read()).decode('utf-8')
                    frames.append(f"data:image/jpeg;base64,{frame_b64}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        print(f"[Preprocess] 成功提取 {len(frames)} 帧", flush=True)
    except Exception as e:
        print(f"[Preprocess] 帧提取失败: {e}", flush=True)
    
    return frames


def _describe_video_by_frames(video_path, trigger_word=""):
    """使用帧提取模式描述视频（主要打标方式）"""
    
    headers = {
        "Authorization": f"Bearer {QWEN_VL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    text_prompt = """These are frames extracted from a video. Based on all frames, describe the character's appearance, hairstyle, body type, clothing, as well as the background environment and objects.
- Do NOT describe poses or actions.
Output rules: English only, single paragraph, ≤300 characters, no explanations/titles/lists/JSON/prefixes, nothing beyond the description itself."""
    
    # 多轮尝试：先用 8 帧，若被内容审查拦截则逐步减少帧数重试
    for attempt, num_frames in enumerate([8, 4, 1], start=1):
        try:
            frames = _extract_video_frames(video_path, num_frames=num_frames)
            if not frames:
                print(f"[Preprocess] 无法提取帧 (尝试 {attempt}, {num_frames}帧)", flush=True)
                continue
            
            content = []
            for frame_url in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": frame_url}
                })
            content.append({"type": "text", "text": text_prompt})
            
            payload = {
                "model": "qwen-vl-max",
                "messages": [{"role": "user", "content": content}]
            }
            
            print(f"[Preprocess] 调用 Qwen VL API (帧提取模式, 尝试 {attempt}, {len(frames)}帧)", flush=True)
            response = requests.post(QWEN_VL_API_URL, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if description:
                    if trigger_word and trigger_word.strip():
                        description = f"{trigger_word.strip()}, {description}"
                    print(f"[Preprocess] 帧提取模式描述 (尝试 {attempt}): {description[:100]}...", flush=True)
                    return description
                else:
                    print(f"[Preprocess] API 返回空描述 (尝试 {attempt})", flush=True)
            else:
                error_text = response.text[:500]
                print(f"[Preprocess] 帧提取模式 API 错误 (尝试 {attempt}): {response.status_code}", flush=True)
                print(f"[Preprocess] 错误详情: {error_text}", flush=True)
                
                # 如果是内容审查拦截，减少帧数重试
                if "data_inspection_failed" in error_text or "inappropriate" in error_text:
                    print(f"[Preprocess] ⚠ 内容审查拦截，将减少帧数重试...", flush=True)
                    continue
                else:
                    # 其他错误不再重试
                    break
                    
        except Exception as e:
            print(f"[Preprocess] 帧提取模式失败 (尝试 {attempt}): {e}", flush=True)
    
    print(f"[Preprocess] ✗ 所有尝试均失败: {os.path.basename(video_path)}", flush=True)
    return None


def generate_default_caption(trigger_word=""):
    """生成默认提示词"""
    if trigger_word and trigger_word.strip():
        return f"{trigger_word.strip()}, a person in dynamic motion with natural movement."
    return "A person performing fluid movements with natural body language."


# ============================================
# 原始视频管理
# ============================================

@preprocess_bp.route('/raw', methods=['GET'])
def list_raw_videos():
    """获取原始视频列表（包含宽高比信息）"""
    try:
        task_id = request.args.get('task_id')
        raw_path = get_task_raw_path(task_id)
        
        videos = get_video_files(raw_path)
        # 计算唯一的宽高比列表（用于AR buckets）
        ar_set = set()
        for v in videos:
            if v.get('aspect_ratio', 0) > 0:
                ar_set.add(v['aspect_ratio'])
        ar_buckets = sorted(list(ar_set))
        
        return jsonify(create_response(code=200, message="success", data={
            'videos': videos, 
            'total': len(videos),
            'ar_buckets': ar_buckets,
            'task_id': task_id
        }))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/ar-buckets', methods=['GET'])
def get_ar_buckets():
    """获取处理后视频的宽高比列表，用于自动填充AR buckets"""
    try:
        task_id = request.args.get('task_id')
        dataset_path = get_task_dataset_path(task_id)
        raw_path = get_task_raw_path(task_id)
        
        videos = get_video_files(dataset_path)
        ar_set = set()
        for v in videos:
            if v.get('aspect_ratio', 0) > 0:
                ar_set.add(v['aspect_ratio'])
        ar_buckets = sorted(list(ar_set))
        
        # 如果没有处理过的视频，从原始视频获取
        if not ar_buckets:
            raw_videos = get_video_files(raw_path)
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
        task_id = request.args.get('task_id')
        raw_path = get_task_raw_path(task_id)
        video_path = os.path.join(raw_path, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="文件不存在", data=None)), 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/raw/<filename>', methods=['DELETE'])
def delete_raw_video(filename):
    """删除原始视频"""
    try:
        task_id = request.args.get('task_id')
        raw_path = get_task_raw_path(task_id)
        video_path = os.path.join(raw_path, filename)
        if os.path.exists(video_path):
            os.remove(video_path)
            return jsonify(create_response(code=200, message="已删除", data={'filename': filename}))
        return jsonify(create_response(code=404, message="文件不存在", data=None)), 404
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/raw/clear', methods=['POST'])
def clear_raw_videos():
    """清除所有原始视频"""
    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        raw_path = get_task_raw_path(task_id)
        
        count = 0
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
        if os.path.exists(raw_path):
            for file in os.listdir(raw_path):
                if os.path.splitext(file)[1].lower() in video_extensions:
                    os.remove(os.path.join(raw_path, file))
                    count += 1
        return jsonify(create_response(code=200, message=f"已清除 {count} 个视频", data={'count': count}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/upload', methods=['POST'])
def upload_videos():
    """上传视频到原始目录（优化大文件上传速度）"""
    try:
        if 'files' not in request.files:
            return jsonify(create_response(code=400, message="没有上传文件", data=None)), 400
        
        task_id = request.form.get('task_id')
        raw_path = get_task_raw_path(task_id)
        
        files = request.files.getlist('files')
        uploaded = []
        
        # 使用流式写入优化大文件上传
        CHUNK_SIZE = 16 * 1024 * 1024  # 16MB 分块
        
        for file in files:
            if file.filename:
                # 保持原始文件名，只做基本的安全处理
                filename = file.filename.replace('/', '_').replace('\\', '_')
                if not filename:
                    filename = secure_filename(file.filename) or 'video.mp4'
                filepath = os.path.join(raw_path, filename)
                
                # 流式写入文件，避免一次性加载到内存
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = file.stream.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                
                uploaded.append(filename)
        
        return jsonify(create_response(code=200, message=f"已上传 {len(uploaded)} 个文件", data={'files': uploaded, 'count': len(uploaded), 'task_id': task_id}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


# ============================================
# 帧提取预览 & 视频缩略图
# ============================================

# 视频缩略图缓存目录
VIDEO_THUMBNAIL_DIR = Path(BASE_RAW_PATH) / ".thumbnails"
VIDEO_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

def get_video_thumbnail_path(video_path, task_id=None):
    """获取视频缩略图的缓存路径"""
    import hashlib
    # 使用视频路径和修改时间生成唯一hash
    mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
    hash_input = f"{video_path}:{mtime}"
    path_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    if task_id:
        task_thumb_dir = Path(BASE_RAW_PATH) / task_id / ".thumbnails"
        task_thumb_dir.mkdir(parents=True, exist_ok=True)
        return task_thumb_dir / f"{path_hash}.jpg"
    return VIDEO_THUMBNAIL_DIR / f"{path_hash}.jpg"

def generate_video_thumbnail(video_path, task_id=None):
    """生成视频缩略图（第一帧）"""
    thumb_path = get_video_thumbnail_path(video_path, task_id)
    
    # 如果缩略图已存在且比视频新，直接返回
    if thumb_path.exists():
        if thumb_path.stat().st_mtime >= os.path.getmtime(video_path):
            return str(thumb_path)
    
    try:
        # 使用 ffmpeg 提取第一帧并压缩
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "scale='min(320,iw)':'-1'",  # 缩放到最大320宽度
            "-vframes", "1",
            "-q:v", "5",  # JPEG 质量
            str(thumb_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=10)
        
        if thumb_path.exists():
            return str(thumb_path)
    except Exception as e:
        print(f"[Preprocess] 生成视频缩略图失败: {e}")
    
    return None

@preprocess_bp.route('/thumbnail/<filename>', methods=['GET'])
def get_video_thumbnail(filename):
    """获取视频缩略图（用于快速预览）"""
    try:
        task_id = request.args.get('task_id')
        raw_path = get_task_raw_path(task_id)
        video_path = os.path.join(raw_path, filename)
        
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404
        
        thumb_path = generate_video_thumbnail(video_path, task_id)
        
        if thumb_path and os.path.exists(thumb_path):
            return send_file(thumb_path, mimetype='image/jpeg')
        else:
            # 如果缩略图生成失败，返回第一帧
            frame_path = extract_frame(video_path, 0)
            if frame_path and os.path.exists(frame_path):
                response = send_file(frame_path, mimetype='image/jpeg')
                @response.call_on_close
                def cleanup():
                    try:
                        os.remove(frame_path)
                    except:
                        pass
                return response
            return jsonify(create_response(code=500, message="无法生成缩略图", data=None)), 500
            
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500

@preprocess_bp.route('/frame/<filename>', methods=['GET'])
def get_frame(filename):
    """提取视频的指定帧"""
    try:
        frame_number = int(request.args.get('frame', 0))
        task_id = request.args.get('task_id')
        raw_path = get_task_raw_path(task_id)
        video_path = os.path.join(raw_path, filename)
        
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
    """处理视频：转换格式、使用 Qwen VL 生成提示词"""
    try:
        data = request.get_json() or {}
        
        # 支持任务特定路径
        task_id = data.get('task_id')
        input_dir = get_task_raw_path(task_id)
        output_dir = get_task_dataset_path(task_id)
        trigger_word = data.get('trigger_word', data.get('prompt_prefix', ''))
        fps = data.get('fps', 16)
        use_qwen_vl = data.get('use_qwen_vl', True)
        
        print(f"[Preprocess] 开始处理视频")
        print(f"[Preprocess] 任务 ID: {task_id}")
        print(f"[Preprocess] 输入目录: {input_dir}")
        print(f"[Preprocess] 输出目录: {output_dir}")
        print(f"[Preprocess] 触发词: {trigger_word}")
        print(f"[Preprocess] 使用 Qwen VL: {use_qwen_vl}")
        
        if not os.path.exists(input_dir):
            return jsonify(create_response(code=400, message=f"输入目录不存在: {input_dir}", data=None)), 400
        
        os.makedirs(output_dir, exist_ok=True)
        video_files = get_video_files(input_dir)
        
        print(f"[Preprocess] 找到 {len(video_files)} 个视频文件")
        
        if not video_files:
            return jsonify(create_response(code=400, message=f"没有找到视频文件，目录: {input_dir}", data=None)), 400
        
        # 清理旧的输出文件
        for f in os.listdir(output_dir):
            if f.endswith('.mp4') or f.endswith('.txt'):
                try:
                    os.remove(os.path.join(output_dir, f))
                except:
                    pass
        
        processed = []
        errors = []
        
        for i, video in enumerate(video_files, 1):
            video_path = video['path']
            output_video = os.path.join(output_dir, f"{i}.mp4")
            output_txt = os.path.join(output_dir, f"{i}.txt")
            
            print(f"[Preprocess] 处理视频 {i}/{len(video_files)}: {video['filename']}")
            
            try:
                # 1. 转换视频为指定帧率（默认 16fps）
                print(f"[Preprocess] 转换视频为 {fps}fps...")
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
                print(f"[Preprocess] 视频转换完成: {output_video}")
                
                # 2. 使用帧提取模式调用 Qwen VL 生成描述
                caption = None
                caption_failed = False
                if use_qwen_vl:
                    caption = describe_video_with_qwen_vl(output_video, trigger_word)
                
                # 如果 Qwen VL 失败，标记失败并使用空描述占位
                if not caption:
                    caption_failed = True
                    caption = generate_default_caption(trigger_word)
                    print(f"[Preprocess] ⚠ 视频 {video['filename']} 打标失败，使用默认文案", flush=True)
                
                # 3. 保存提示词
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(caption.strip())
                
                processed.append({
                    'index': i,
                    'original': video['filename'],
                    'output': f"{i}.mp4",
                    'caption': caption[:100] + '...' if len(caption) > 100 else caption,
                    'caption_failed': caption_failed
                })
                
            except subprocess.TimeoutExpired:
                errors.append({'file': video['filename'], 'error': '处理超时'})
            except subprocess.CalledProcessError as e:
                errors.append({'file': video['filename'], 'error': f"FFmpeg错误: {e.stderr[:100] if e.stderr else ''}"})
            except Exception as e:
                errors.append({'file': video['filename'], 'error': str(e)[:100]})
        
        print(f"[Preprocess] 处理完成: {len(processed)} 成功, {len(errors)} 失败")
        
        return jsonify(create_response(
            code=200,
            message=f"处理完成: {len(processed)} 成功, {len(errors)} 失败",
            data={'processed': processed, 'errors': errors, 'total': len(video_files)}
        ))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


# ============================================
# 已处理视频管理
# ============================================

@preprocess_bp.route('/list', methods=['GET'])
def list_processed_videos():
    """获取已处理的视频列表"""
    try:
        task_id = request.args.get('task_id')
        dataset_path = get_task_dataset_path(task_id)
        
        videos = []
        if os.path.exists(dataset_path):
            for file in sorted(os.listdir(dataset_path)):
                if file.endswith('.mp4'):
                    txt_path = os.path.join(dataset_path, file.replace('.mp4', '.txt'))
                    caption = ''
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    videos.append({
                        'filename': file,
                        'caption': caption,
                        'path': os.path.join(dataset_path, file)
                    })
        
        return jsonify(create_response(code=200, message="success", data={'videos': videos, 'total': len(videos), 'task_id': task_id}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/caption', methods=['PUT'])
def update_caption():
    """更新视频的提示词"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        caption = data.get('caption', '')
        task_id = data.get('task_id')
        
        if not filename:
            return jsonify(create_response(code=400, message="文件名不能为空", data=None)), 400
        
        dataset_path = get_task_dataset_path(task_id)
        txt_path = os.path.join(dataset_path, filename.replace('.mp4', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        
        return jsonify(create_response(code=200, message="已保存", data={'filename': filename}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/translate', methods=['POST'])
def translate_caption():
    """翻译提示词（英译中）"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify(create_response(code=400, message="文本不能为空", data=None)), 400

        headers = {
            "Authorization": f"Bearer {QWEN_VL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen-plus",
            "messages": [
                {"role": "system", "content": "你是一个翻译助手。将用户给出的英文翻译为中文，只输出翻译结果，不要任何解释。"},
                {"role": "user", "content": text}
            ]
        }
        response = requests.post(QWEN_VL_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            translated = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return jsonify(create_response(code=200, message="翻译成功", data={'translated': translated}))
        else:
            return jsonify(create_response(code=500, message=f"翻译API错误: {response.status_code}", data=None)), 500
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/recaption', methods=['POST'])
def recaption_video():
    """重新打标：使用帧提取模式调用 QwenVL 重新生成提示词"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        trigger_word = data.get('trigger_word', '')
        task_id = data.get('task_id')

        if not filename:
            return jsonify(create_response(code=400, message="文件名不能为空", data=None)), 400

        dataset_path = get_task_dataset_path(task_id)
        video_path = os.path.join(dataset_path, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404

        print(f"[Preprocess] 重新打标（帧提取模式）: {filename}", flush=True)

        # 使用帧提取模式调用 QwenVL 重新打标
        caption = describe_video_with_qwen_vl(video_path, trigger_word)
        if not caption:
            return jsonify(create_response(code=500, message="Qwen VL 打标失败，请重试", data={'filename': filename})), 500

        # 保存提示词
        txt_path = os.path.join(dataset_path, filename.replace('.mp4', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())

        return jsonify(create_response(code=200, message="重新打标完成", data={'filename': filename, 'caption': caption}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/batch-replace-trigger', methods=['POST'])
def batch_replace_trigger():
    """批量替换已处理视频的触发词前缀"""
    try:
        data = request.get_json()
        old_trigger = data.get('old_trigger', '').strip()
        new_trigger = data.get('new_trigger', '').strip()
        task_id = data.get('task_id')

        if not new_trigger:
            return jsonify(create_response(code=400, message="新触发词不能为空", data=None)), 400

        dataset_path = get_task_dataset_path(task_id)
        updated = 0
        for file in sorted(os.listdir(dataset_path)):
            if file.endswith('.txt'):
                txt_path = os.path.join(dataset_path, file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # 替换旧触发词前缀
                if old_trigger and content.startswith(old_trigger):
                    content = content[len(old_trigger):].lstrip(', ').lstrip()
                    content = f"{new_trigger}, {content}" if content else new_trigger
                elif old_trigger == '':
                    # 没有旧触发词，直接在前面加
                    content = f"{new_trigger}, {content}" if content else new_trigger
                else:
                    # 旧触发词不匹配，也直接替换第一个逗号前的部分
                    parts = content.split(',', 1)
                    if len(parts) > 1:
                        content = f"{new_trigger}, {parts[1].strip()}"
                    else:
                        content = f"{new_trigger}, {content}"

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated += 1

        return jsonify(create_response(code=200, message=f"已更新 {updated} 个文件", data={'updated': updated}))
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    """获取处理后的视频文件"""
    try:
        task_id = request.args.get('task_id')
        dataset_path = get_task_dataset_path(task_id)
        video_path = os.path.join(dataset_path, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/processed-thumb/<filename>', methods=['GET'])
def get_processed_video_thumbnail(filename):
    """获取处理后视频的缩略图"""
    try:
        task_id = request.args.get('task_id')
        dataset_path = get_task_dataset_path(task_id)
        video_path = os.path.join(dataset_path, filename)
        
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message="视频不存在", data=None)), 404
        
        # 缩略图缓存目录
        thumb_dir = Path(dataset_path) / ".thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        import hashlib
        mtime = os.path.getmtime(video_path)
        hash_input = f"{video_path}:{mtime}"
        path_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        thumb_path = thumb_dir / f"{path_hash}.jpg"
        
        # 如果缩略图不存在或过期，重新生成
        if not thumb_path.exists() or thumb_path.stat().st_mtime < mtime:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "scale='min(320,iw)':'-1'",
                "-vframes", "1",
                "-q:v", "5",
                str(thumb_path)
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)
        
        if thumb_path.exists():
            return send_file(str(thumb_path), mimetype='image/jpeg')
        else:
            return jsonify(create_response(code=500, message="无法生成缩略图", data=None)), 500
            
    except Exception as e:
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/video/<filename>', methods=['DELETE'])
def delete_video(filename):
    """删除视频及其提示词文件"""
    try:
        task_id = request.args.get('task_id')
        dataset_path = get_task_dataset_path(task_id)
        video_path = os.path.join(dataset_path, filename)
        txt_path = os.path.join(dataset_path, filename.replace('.mp4', '.txt'))
        
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


@preprocess_bp.route('/regenerate-caption', methods=['POST'])
def regenerate_caption():
    """重新反推单个视频的提示词（帧提取模式）"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        trigger_word = data.get('trigger_word', '')
        task_id = data.get('task_id')
        
        if not filename:
            return jsonify(create_response(code=400, message="缺少文件名", data=None)), 400
        
        # 视频文件路径
        dataset_path = get_task_dataset_path(task_id)
        video_path = os.path.join(dataset_path, filename)
        if not os.path.exists(video_path):
            return jsonify(create_response(code=404, message=f"视频文件不存在: {filename}", data=None)), 404
        
        print(f"[Preprocess] 重新反推视频（帧提取模式）: {filename}", flush=True)
        
        # 使用帧提取模式生成描述
        caption = describe_video_with_qwen_vl(video_path, trigger_word)
        
        if not caption:
            return jsonify(create_response(code=500, message="Qwen VL 反推失败，请重试", data={'filename': filename})), 500
        
        # 保存提示词
        txt_path = os.path.join(dataset_path, filename.replace('.mp4', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        
        print(f"[Preprocess] 反推完成: {caption[:50]}...", flush=True)
        
        return jsonify(create_response(code=200, message="反推成功", data={'caption': caption}))
        
    except Exception as e:
        print(f"[Preprocess] 重新反推失败: {e}", flush=True)
        return jsonify(create_response(code=500, message=str(e), data=None)), 500


@preprocess_bp.route('/translate', methods=['POST'])
def translate_text():
    """将英文翻译为简体中文"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify(create_response(code=400, message="没有要翻译的文本", data=None)), 400
        
        # 使用千问 API 进行翻译
        if not QWEN_VL_API_KEY:
            return jsonify(create_response(code=500, message="API Key 未配置", data=None)), 500
        
        headers = {
            "Authorization": f"Bearer {QWEN_VL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen-plus",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的翻译助手。请将用户输入的英文直接翻译成简体中文，只输出翻译结果，不要添加任何解释或其他内容。"
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
        
        response = requests.post(QWEN_VL_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            return jsonify(create_response(code=200, message="翻译成功", data={'translated_text': translated_text}))
        else:
            print(f"[Translate] API 错误: {response.status_code} - {response.text}")
            return jsonify(create_response(code=500, message=f"翻译API错误: {response.status_code}", data=None)), 500
            
    except requests.Timeout:
        return jsonify(create_response(code=500, message="翻译请求超时", data=None)), 500
    except Exception as e:
        print(f"[Translate] 翻译失败: {e}")
        return jsonify(create_response(code=500, message=str(e), data=None)), 500
