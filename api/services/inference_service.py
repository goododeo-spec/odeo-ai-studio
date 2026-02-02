"""
推理服务 - Wan2.1 视频 LoRA 推理
"""
import os
import sys
import time
import uuid
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# 配置路径
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "./pretrained_models/Wan2.1-I2V-14B-480P"))
LORA_ROOT = Path(os.environ.get("LORA_ROOT", "./data/outputs"))
OUTPUT_ROOT = Path(os.environ.get("INFERENCE_OUTPUT_ROOT", "./data/outputs/inference"))
TEST_IMAGES_ROOT = Path(os.environ.get("DATASET_PATH", "./data/datasets"))

# 确保输出目录存在
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class InferenceStatus(Enum):
    QUEUED = "queued"
    LOADING = "loading"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceTask:
    task_id: str
    status: InferenceStatus
    lora_path: str
    lora_name: str
    prompt: str
    image_path: Optional[str] = None
    gpu_id: int = 0
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_steps: int = 30
    guidance_scale: float = 5.0
    lora_strength: float = 0.8
    seed: int = -1
    output_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat() if self.created_at else None
        d['started_at'] = self.started_at.isoformat() if self.started_at else None
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d


class InferenceService:
    """推理服务"""
    
    def __init__(self):
        self._tasks: Dict[str, InferenceTask] = {}
        self._task_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        
        # 加载保存的任务
        self._tasks_file = OUTPUT_ROOT / "inference_tasks.json"
        self._load_tasks()
        
        print(f"[Inference] 推理服务初始化完成")
        print(f"[Inference] 模型根目录: {MODELS_ROOT}")
        print(f"[Inference] LoRA 根目录: {LORA_ROOT}")
    
    def _load_tasks(self):
        """加载历史任务"""
        if self._tasks_file.exists():
            try:
                with open(self._tasks_file, 'r') as f:
                    data = json.load(f)
                    for task_data in data:
                        task_id = task_data['task_id']
                        task_data['status'] = InferenceStatus(task_data['status'])
                        task_data['created_at'] = datetime.fromisoformat(task_data['created_at']) if task_data.get('created_at') else None
                        task_data['started_at'] = datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None
                        task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None
                        self._tasks[task_id] = InferenceTask(**task_data)
                        self._task_locks[task_id] = threading.Lock()
                print(f"[Inference] 加载了 {len(self._tasks)} 个历史任务")
            except Exception as e:
                print(f"[Inference] 加载任务失败: {e}")
    
    def _save_tasks(self):
        """保存任务"""
        try:
            tasks_data = [task.to_dict() for task in self._tasks.values()]
            with open(self._tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Inference] 保存任务失败: {e}")
    
    def list_training_tasks_with_loras(self) -> List[Dict[str, Any]]:
        """列出训练任务及其 LoRA（按任务分组）"""
        tasks = {}
        
        for task_dir in LORA_ROOT.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith('train_'):
                continue
            
            task_id = task_dir.name
            
            # 尝试读取任务配置获取名称
            task_name = task_id
            tasks_json = LORA_ROOT / "tasks.json"
            if tasks_json.exists():
                try:
                    with open(tasks_json, 'r') as f:
                        all_tasks = json.load(f)
                        for t in all_tasks:
                            if t.get('task_id') == task_id:
                                task_name = t.get('description', task_id)
                                break
                except:
                    pass
            
            epochs = []
            
            # 查找所有 epoch 目录
            for run_dir in task_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                for epoch_dir in sorted(run_dir.iterdir()):
                    if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch'):
                        continue
                    
                    lora_file = epoch_dir / 'adapter_model.safetensors'
                    if lora_file.exists():
                        epoch_num = int(epoch_dir.name.replace('epoch', ''))
                        epochs.append({
                            'epoch': epoch_num,
                            'name': epoch_dir.name,
                            'path': str(lora_file),
                            'size_mb': lora_file.stat().st_size / (1024 * 1024),
                            'created_at': datetime.fromtimestamp(lora_file.stat().st_mtime).isoformat()
                        })
            
            if epochs:
                epochs.sort(key=lambda x: x['epoch'], reverse=True)
                tasks[task_id] = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'epochs': epochs,
                    'latest_epoch': epochs[0]['epoch'],
                    'created_at': min(e['created_at'] for e in epochs)
                }
        
        # 按创建时间排序
        result = list(tasks.values())
        result.sort(key=lambda x: x['created_at'], reverse=True)
        return result
    
    def list_available_loras(self) -> List[Dict[str, Any]]:
        """列出可用的 LoRA 模型（扁平列表，兼容旧 API）"""
        loras = []
        
        for task_dir in LORA_ROOT.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith('train_'):
                continue
            
            # 查找所有 epoch 目录
            for run_dir in task_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                for epoch_dir in sorted(run_dir.iterdir()):
                    if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch'):
                        continue
                    
                    lora_file = epoch_dir / 'adapter_model.safetensors'
                    if lora_file.exists():
                        # 获取任务信息
                        task_id = task_dir.name
                        epoch = epoch_dir.name
                        
                        # 读取配置
                        config_file = epoch_dir / f"{task_id}.toml"
                        config = {}
                        if config_file.exists():
                            try:
                                import toml
                                config = toml.load(config_file)
                            except:
                                pass
                        
                        loras.append({
                            'id': f"{task_id}/{run_dir.name}/{epoch}",
                            'task_id': task_id,
                            'epoch': epoch,
                            'path': str(lora_file),
                            'name': f"{task_id[-8:]}_{epoch}",
                            'size_mb': lora_file.stat().st_size / (1024 * 1024),
                            'created_at': datetime.fromtimestamp(lora_file.stat().st_mtime).isoformat(),
                            'config': config
                        })
        
        # 按时间排序，最新的在前
        loras.sort(key=lambda x: x['created_at'], reverse=True)
        return loras
    
    def list_test_images(self) -> List[Dict[str, Any]]:
        """列出测试图片库"""
        images = []
        
        # 从数据集目录获取图片
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for img_file in TEST_IMAGES_ROOT.glob(f"**/{ext}"):
                images.append({
                    'id': str(img_file.relative_to(TEST_IMAGES_ROOT)),
                    'path': str(img_file),
                    'name': img_file.name,
                    'url': f"/api/v1/inference/image/{img_file.relative_to(TEST_IMAGES_ROOT)}"
                })
        
        # 从训练数据获取视频第一帧作为测试图
        for ext in ['*.mp4', '*.mov', '*.avi']:
            for video_file in TEST_IMAGES_ROOT.glob(f"**/{ext}"):
                images.append({
                    'id': str(video_file.relative_to(TEST_IMAGES_ROOT)),
                    'path': str(video_file),
                    'name': video_file.stem + "_frame",
                    'url': f"/api/v1/inference/video-frame/{video_file.relative_to(TEST_IMAGES_ROOT)}",
                    'type': 'video'
                })
        
        return images[:50]  # 限制数量
    
    def create_inference_task(self, data: Dict[str, Any]) -> InferenceTask:
        """创建推理任务"""
        task_id = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        lora_path = data.get('lora_path', '')
        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"LoRA 文件不存在: {lora_path}")
        
        task = InferenceTask(
            task_id=task_id,
            status=InferenceStatus.QUEUED,
            lora_path=lora_path,
            lora_name=data.get('lora_name', Path(lora_path).parent.name),
            prompt=data.get('prompt', 'a person dancing'),
            image_path=data.get('image_path'),
            gpu_id=data.get('gpu_id', 0),
            width=data.get('width', 832),
            height=data.get('height', 480),
            num_frames=data.get('num_frames', 81),
            num_steps=data.get('num_steps', 30),
            guidance_scale=data.get('guidance_scale', 5.0),
            lora_strength=data.get('lora_strength', 0.8),
            seed=data.get('seed', -1)
        )
        
        with self._lock:
            self._tasks[task_id] = task
            self._task_locks[task_id] = threading.Lock()
        
        self._save_tasks()
        
        # 异步启动推理
        thread = threading.Thread(target=self._run_inference, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return task
    
    def _run_inference(self, task_id: str):
        """执行推理"""
        task = self._tasks.get(task_id)
        if not task:
            return
        
        try:
            task.status = InferenceStatus.LOADING
            task.started_at = datetime.now()
            self._save_tasks()
            
            print(f"[Inference] 开始推理任务: {task_id}")
            print(f"[Inference] LoRA: {task.lora_path}")
            print(f"[Inference] Prompt: {task.prompt}")
            print(f"[Inference] GPU: {task.gpu_id}")
            
            # 设置输出路径
            output_dir = OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video = output_dir / "output.mp4"
            
            task.status = InferenceStatus.RUNNING
            task.progress = 10.0
            self._save_tasks()
            
            # 使用 ComfyUI 的 Wan 推理
            # 这里我们调用一个独立的推理脚本
            inference_script = Path(__file__).parent.parent / "utils" / "wan_inference.py"
            
            cmd = [
                sys.executable, str(inference_script),
                "--model_path", str(MODELS_ROOT),
                "--lora_path", task.lora_path,
                "--prompt", task.prompt,
                "--output", str(output_video),
                "--width", str(task.width),
                "--height", str(task.height),
                "--num_frames", str(task.num_frames),
                "--num_steps", str(task.num_steps),
                "--guidance_scale", str(task.guidance_scale),
                "--lora_strength", str(task.lora_strength),
                "--seed", str(task.seed if task.seed > 0 else int(time.time()) % 10000),
                "--gpu", str(task.gpu_id)
            ]
            
            if task.image_path:
                cmd.extend(["--image", task.image_path])
            
            print(f"[Inference] 执行命令: {' '.join(cmd)}")
            
            # 执行推理
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(task.gpu_id)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            
            # 读取输出并更新进度
            log_file = output_dir / "inference.log"
            with open(log_file, 'w') as f:
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    
                    # 解析进度
                    if 'step' in line.lower() or 'progress' in line.lower():
                        try:
                            # 尝试解析进度百分比
                            import re
                            match = re.search(r'(\d+)%', line)
                            if match:
                                task.progress = float(match.group(1))
                            else:
                                match = re.search(r'(\d+)/(\d+)', line)
                                if match:
                                    current, total = int(match.group(1)), int(match.group(2))
                                    task.progress = (current / total) * 100
                        except:
                            pass
            
            return_code = process.wait()
            
            if return_code == 0 and output_video.exists():
                task.status = InferenceStatus.COMPLETED
                task.output_path = str(output_video)
                task.progress = 100.0
                print(f"[Inference] 推理完成: {output_video}")
            else:
                task.status = InferenceStatus.FAILED
                task.error_message = f"推理失败，返回码: {return_code}"
                print(f"[Inference] 推理失败: {task.error_message}")
                
                # 读取错误日志
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        error_lines = f.readlines()[-20:]
                        task.error_message += "\n" + "".join(error_lines)
            
        except Exception as e:
            task.status = InferenceStatus.FAILED
            task.error_message = str(e)
            print(f"[Inference] 推理异常: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            task.completed_at = datetime.now()
            self._save_tasks()
    
    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def list_tasks(self, limit: int = 50) -> List[InferenceTask]:
        """列出任务"""
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.created_at or datetime.min, reverse=True)
        return tasks[:limit]
    
    def stop_task(self, task_id: str) -> bool:
        """停止任务"""
        task = self._tasks.get(task_id)
        if task and task.status in [InferenceStatus.QUEUED, InferenceStatus.LOADING, InferenceStatus.RUNNING]:
            task.status = InferenceStatus.FAILED
            task.error_message = "用户手动停止"
            task.completed_at = datetime.now()
            self._save_tasks()
            return True
        return False


# 全局实例
inference_service = InferenceService()
