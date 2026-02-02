"""
数据集预处理服务
"""
import os
import sys
import subprocess
import json
import uuid
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# 延迟加载可选依赖
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from unittest.mock import patch
    from transformers.dynamic_module_utils import get_imports
    from transformers import AutoModelForCausalLM, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None
    AutoProcessor = None

# 确保api目录在sys.path中（而不是项目根目录）
api_root = Path(__file__).parent.parent
if str(api_root) not in sys.path:
    sys.path.insert(0, str(api_root))

from models.preprocessing import (
    PreprocessingRequest, PreprocessingResult, PreprocessingProgress,
    PreprocessingStatus, DatasetInfo, CaptionMethod
)

# 基础输出目录
BASE_OUTPUT_DIR = "/mnt/disk0/lora_outputs"

class PreprocessingService:
    """数据集预处理服务"""

    def __init__(self):
        """初始化预处理服务"""
        self._tasks: Dict[str, PreprocessingResult] = {}  # 预处理任务映射
        self._datasets: Dict[str, DatasetInfo] = {}  # 数据集映射
        self._lock = threading.Lock()

        # 确保基础输出目录存在
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

        # 加载已存在的数据集
        self._load_existing_datasets()

    def _load_existing_datasets(self):
        """加载已存在的数据集"""
        if not os.path.exists(BASE_OUTPUT_DIR):
            return

        for dataset_name in os.listdir(BASE_OUTPUT_DIR):
            dataset_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
            if os.path.isdir(dataset_dir):
                # 统计文件数量
                video_count = 0
                file_count = 0
                for file in os.listdir(dataset_dir):
                    if file.endswith('.mp4'):
                        video_count += 1
                    if file.endswith('.txt'):
                        file_count += 1

                # 获取创建时间
                created_at = datetime.fromtimestamp(os.path.getctime(dataset_dir))

                self._datasets[dataset_name] = DatasetInfo(
                    dataset_name=dataset_name,
                    directory=dataset_dir,
                    created_at=created_at,
                    video_count=video_count,
                    file_count=file_count,
                    status=PreprocessingStatus.COMPLETED
                )

    def check_dataset_exists(self, dataset_name: str) -> tuple[bool, Optional[DatasetInfo]]:
        """检查数据集是否已存在"""
        with self._lock:
            if dataset_name in self._datasets:
                return True, self._datasets[dataset_name]
            return False, None

    def create_preprocessing_task(self, request: PreprocessingRequest) -> str:
        """创建预处理任务"""
        # 检查数据集是否已存在
        exists, dataset_info = self.check_dataset_exists(request.dataset_name)
        if exists:
            raise ValueError(f"数据集 '{request.dataset_name}' 已存在")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建输出目录
        output_dir = os.path.join(BASE_OUTPUT_DIR, request.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # 创建任务结果
        progress = PreprocessingProgress(
            total_videos=0,
            processed_videos=0,
            current_video="",
            current_step="等待开始",
            progress_percent=0.0,
            start_time=datetime.now()
        )

        result = PreprocessingResult(
            dataset_name=request.dataset_name,
            output_directory=output_dir,
            status=PreprocessingStatus.PENDING,
            progress=progress
        )

        with self._lock:
            self._tasks[task_id] = result

        # 异步执行预处理
        threading.Thread(
            target=self._run_preprocessing,
            args=(task_id, request),
            daemon=True
        ).start()

        return task_id

    def get_task_status(self, task_id: str) -> Optional[PreprocessingResult]:
        """获取任务状态"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_datasets(self) -> List[DatasetInfo]:
        """获取所有数据集"""
        with self._lock:
            return list(self._datasets.values())

    def get_dataset(self, dataset_name: str) -> Optional[DatasetInfo]:
        """获取指定数据集"""
        with self._lock:
            return self._datasets.get(dataset_name)

    def _run_preprocessing(self, task_id: str, request: PreprocessingRequest):
        """运行预处理任务"""
        try:
            result = self._tasks[task_id]
            result.status = PreprocessingStatus.RUNNING
            result.progress.current_step = "初始化"

            # 验证输入目录
            if not os.path.exists(request.video_directory):
                raise ValueError(f"视频目录不存在: {request.video_directory}")

            # 初始化千问客户端
            qwen_client = None
            if request.use_qwen_optimize:
                result.progress.current_step = "初始化千问客户端"
                qwen_client = self._init_qwen_client(request.qwen_api_key)

            # 获取视频文件列表
            result.progress.current_step = "扫描视频文件"
            video_files = self._get_video_files(request.video_directory)

            if not video_files:
                raise ValueError(f"在目录 {request.video_directory} 中未找到视频文件")

            result.progress.total_videos = len(video_files)
            result.progress.current_step = f"开始处理 {len(video_files)} 个视频"

            # 处理每个视频
            for i, video_path in enumerate(video_files, 1):
                video_name = os.path.splitext(os.path.basename(video_path))[0]

                result.progress.current_video = video_name
                result.progress.current_step = f"处理视频 {i}/{len(video_files)}: {video_name}"
                result.progress.processed_videos = i - 1
                result.progress.progress_percent = (i - 1) / len(video_files) * 100

                # 1. 转换视频格式为16fps的mp4
                output_video_path = os.path.join(result.output_directory, f"{i}.mp4")
                first_frame_path = os.path.join(result.output_directory, f"{i}.jpg")

                result.progress.current_step = f"转换视频格式: {video_name}"
                self._convert_video(video_path, output_video_path)

                # 2. 提取首帧
                result.progress.current_step = f"提取首帧: {video_name}"
                self._extract_frame(video_path, first_frame_path)

                # 3. 对首帧进行提示词反推
                if os.path.exists(first_frame_path):
                    result.progress.current_step = f"生成提示词: {video_name}"
                    caption = self._generate_caption(
                        first_frame_path,
                        request.caption_method.value
                    )

                    if caption:
                        # 4. 使用千问优化提示词
                        if qwen_client:
                            result.progress.current_step = f"优化提示词: {video_name}"
                            optimized_caption = self._optimize_prompt_with_qwen(caption, qwen_client)
                        else:
                            optimized_caption = caption

                        # 5. 添加前缀并保存为txt文件
                        final_caption = f"{request.prompt_prefix} {optimized_caption}" if request.prompt_prefix else optimized_caption
                        txt_path = os.path.join(result.output_directory, f"{i}.txt")

                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(final_caption.strip())

                # 清理首帧图片
                if os.path.exists(first_frame_path):
                    try:
                        os.remove(first_frame_path)
                    except Exception:
                        pass

            # 清理完成
            result.progress.processed_videos = len(video_files)
            result.progress.progress_percent = 100.0
            result.progress.current_step = "清理临时文件"

            # 清理所有首帧图片
            for i in range(1, len(video_files) + 1):
                first_frame_path = os.path.join(result.output_directory, f"{i}.jpg")
                if os.path.exists(first_frame_path):
                    try:
                        os.remove(first_frame_path)
                    except Exception:
                        pass

            # 任务完成
            result.status = PreprocessingStatus.COMPLETED
            result.progress.current_step = "完成"
            result.end_time = datetime.now()
            result.output_files = [f"{i}.mp4" for i in range(1, len(video_files) + 1)] + \
                                  [f"{i}.txt" for i in range(1, len(video_files) + 1)]

            # 更新数据集信息
            with self._lock:
                self._datasets[request.dataset_name] = DatasetInfo(
                    dataset_name=request.dataset_name,
                    directory=result.output_directory,
                    created_at=datetime.now(),
                    video_count=len(video_files),
                    file_count=len(video_files) * 2,  # mp4 + txt
                    status=PreprocessingStatus.COMPLETED,
                    last_preprocessing_id=task_id
                )

        except Exception as e:
            result.status = PreprocessingStatus.FAILED
            result.error_message = str(e)
            result.progress.current_step = f"错误: {str(e)[:50]}"
            result.end_time = datetime.now()

    def _init_qwen_client(self, api_key: Optional[str] = None):
        """初始化千问客户端"""
        from openai import OpenAI

        # 如果没有提供API密钥，尝试从环境变量获取
        if not api_key:
            api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not api_key:
            print("⚠️  警告: 未找到 DASHSCOPE_API_KEY，千问优化功能将被跳过")
            return None

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            return client
        except Exception as e:
            print(f"❌ 千问客户端初始化失败: {e}")
            return None

    def _get_video_files(self, video_dir: str) -> List[str]:
        """获取视频文件列表"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        video_files = []

        for file in os.listdir(video_dir):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in video_extensions:
                video_files.append(os.path.join(video_dir, file))

        video_files.sort()
        return video_files

    def _convert_video(self, input_path: str, output_path: str):
        """转换视频格式为16fps的mp4"""
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-r", "16",
            "-y",
            output_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def _extract_frame(self, input_path: str, output_path: str):
        """提取视频首帧"""
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", "select=eq(n\\,0)",
            "-vframes", "1",
            "-y",
            output_path
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def _generate_caption(self, image_path: str, caption_method: str) -> str:
        """生成图像提示词"""
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16

        # 模型路径
        model_path = "/mnt/disk0/pretrained_models/Florence-2-base-PromptGen-v2.0"

        if not os.path.exists(model_path):
            print(f"⚠️  模型路径不存在: {model_path}")
            return ""

        try:
            # 加载模型和处理器
            with patch("transformers.dynamic_module_utils.get_imports", self._fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation='sdpa',
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).to(device)

            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # 读取图像
            image = Image.open(image_path).convert("RGB")

            # 设置提示词
            prompt_map = {
                "tags": "<GENERATE_TAGS>",
                "simple": "<CAPTION>",
                "detailed": "<DETAILED_CAPTION>",
                "extra": "<MORE_DETAILED_CAPTION>",
                "mixed": "<MIX_CAPTION>",
                "extra_mixed": "<MIX_CAPTION_PLUS>",
                "analyze": "<ANALYZE>"
            }

            prompt = prompt_map.get(caption_method, "<MIX_CAPTION_PLUS>")

            # 生成提示词
            inputs = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=4,
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            return parsed_answer[prompt]

        except Exception as e:
            print(f"❌ 生成提示词失败: {e}")
            return ""

    def _optimize_prompt_with_qwen(self, caption: str, client) -> str:
        """使用千问优化提示词"""
        if not client:
            return caption

        if not caption or caption.strip() == "":
            return caption

        try:
            prompt = f"""请将以下图像提示词优化为英文视频生成提示词，要求：

1. 保持核心内容不变
2. 将涉及image等图像相关字眼的描述都去掉
3. 保持简洁明了，长度在50-100词之间
5. 直接输出英文提示词，不需要任何解释

图像提示词：
{caption}

请输出优化后的英文提示词："""

            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )

            optimized_caption = completion.choices[0].message.content.strip()

            # 清理输出
            if optimized_caption.startswith('"') and optimized_caption.endswith('"'):
                optimized_caption = optimized_caption[1:-1]
            if optimized_caption.startswith("提示词："):
                optimized_caption = optimized_caption[4:]

            return optimized_caption

        except Exception as e:
            print(f"❌ 千问优化失败: {e}")
            return caption

    def _fixed_get_imports(self, filename):
        """修复transformers动态模块导入问题"""
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        try:
            imports.remove("flash_attn")
        except:
            pass
        return imports

# 全局实例
preprocessing_service = PreprocessingService()
