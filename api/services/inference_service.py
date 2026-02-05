"""
推理服务 - Wan2.1 视频 LoRA 推理
使用 ComfyUI wanvideo_2_1_14B_I2V_odeo workflow
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
GALLERY_ROOT = Path(os.environ.get("GALLERY_ROOT", "./data/gallery"))

# 推理 GPU 范围（4-7）
INFERENCE_GPU_RANGE = range(4, 8)

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
    """
    推理任务数据结构
    
    Workflow 节点映射（wanvideo_2_1_14B_I2V_odeo.json）:
    - 节点 58 (LoadImage) ← image_path (图库选择的图片)
    - 节点 71 (WanVideoLoraSelect) ← lora_path, lora_strength (选择的 LoRA 模型)
    - 节点 81 (TextToLowercase) ← trigger_word (触发词输入)
    - 节点 30 (VHS_VideoCombine) → output_path (输出结果)
    """
    task_id: str
    status: InferenceStatus
    lora_path: str                          # 节点 71: LoRA 文件路径
    lora_name: str
    prompt: str  # 用于兼容
    trigger_word: str = ""                  # 节点 81: 触发词输入
    image_path: Optional[str] = None        # 节点 58: 输入图片路径
    gpu_id: int = 4  # 默认使用 GPU 4
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_steps: int = 4                      # workflow 默认是 4
    guidance_scale: float = 1.0             # workflow 默认是 1
    lora_strength: float = 1.0              # 节点 71: LoRA 强度
    seed: int = -1
    use_auto_caption: bool = True           # 是否使用 QwenVL 自动描述
    output_path: Optional[str] = None       # 节点 30: 输出视频路径
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    batch_id: Optional[str] = None          # 批次ID（多图片时同一批次）
    image_index: int = 0                    # 图片索引
    epoch_index: int = 0                    # epoch索引（多epoch时）
    
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
        self._pending_inferences: List[Dict] = []  # 等待 LoRA 生成的推理任务
        self._monitor_running = True
        
        # 加载保存的任务
        self._tasks_file = OUTPUT_ROOT / "inference_tasks.json"
        self._pending_file = OUTPUT_ROOT / "pending_inferences.json"
        self._load_tasks()
        self._load_pending_inferences()
        
        # 启动轮询监控线程（每 5 分钟检查一次）
        self._start_lora_monitor()
        
        # 启动任务状态同步线程（每 30 秒检查一次）
        self._start_status_sync_monitor()
        
        print(f"[Inference] 推理服务初始化完成")
        print(f"[Inference] 模型根目录: {MODELS_ROOT}")
        print(f"[Inference] LoRA 根目录: {LORA_ROOT}")
        print(f"[Inference] 推理 GPU 范围: {list(INFERENCE_GPU_RANGE)}")
    
    def _load_pending_inferences(self):
        """加载等待中的推理任务"""
        if self._pending_file.exists():
            try:
                with open(self._pending_file, 'r') as f:
                    self._pending_inferences = json.load(f)
                print(f"[Inference] 加载了 {len(self._pending_inferences)} 个等待中的推理任务")
            except Exception as e:
                print(f"[Inference] 加载等待推理任务失败: {e}")
    
    def _save_pending_inferences(self):
        """保存等待中的推理任务"""
        try:
            with open(self._pending_file, 'w') as f:
                json.dump(self._pending_inferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Inference] 保存等待推理任务失败: {e}")
    
    def _start_lora_monitor(self):
        """启动 LoRA 监控线程，每 5 分钟检查一次"""
        def monitor():
            print("[Inference] LoRA 监控线程已启动（每 5 分钟检查一次）")
            while self._monitor_running:
                try:
                    self._check_pending_inferences()
                except Exception as e:
                    print(f"[Inference] 轮询检查错误: {e}")
                time.sleep(300)  # 5 分钟
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _start_status_sync_monitor(self):
        """启动任务状态同步线程，每 30 秒检查一次"""
        def sync_monitor():
            print("[Inference] 任务状态同步线程已启动（每 30 秒检查一次）")
            while self._monitor_running:
                try:
                    self._sync_task_completion_status()
                except Exception as e:
                    print(f"[Inference] 状态同步错误: {e}")
                time.sleep(30)  # 30 秒
        
        sync_thread = threading.Thread(target=sync_monitor, daemon=True)
        sync_thread.start()
    
    def _sync_task_completion_status(self):
        """同步任务完成状态 - 检查输出文件是否存在并更新状态"""
        updated_count = 0
        with self._lock:
            for task_id, task in self._tasks.items():
                # 只检查运行中的任务
                if task.status not in [InferenceStatus.RUNNING, InferenceStatus.LOADING]:
                    continue
                
                # 检查输出文件是否存在
                output_path = OUTPUT_ROOT / task_id / "output.mp4"
                if output_path.exists():
                    task.status = InferenceStatus.COMPLETED
                    task.output_path = str(output_path)
                    task.progress = 100.0
                    if not task.completed_at:
                        task.completed_at = datetime.now()
                    updated_count += 1
                    print(f"[Inference] 状态同步：{task_id} -> completed")
            
            if updated_count > 0:
                self._save_tasks()
                print(f"[Inference] 状态同步完成，更新了 {updated_count} 个任务")
    
    def _check_pending_inferences(self):
        """检查等待中的推理任务，如果对应的 LoRA 已生成则启动推理"""
        if not self._pending_inferences:
            return
        
        completed = []
        for pending in self._pending_inferences:
            task_id = pending.get('training_task_id')
            epoch = pending.get('epoch')
            
            # 检查 LoRA 是否已生成
            lora_path = self._find_lora_path(task_id, epoch)
            if lora_path:
                print(f"[Inference] 检测到 LoRA 已生成: {task_id}/epoch{epoch}")
                # 创建实际的推理任务
                pending['lora_path'] = lora_path
                self._create_inference_from_pending(pending)
                completed.append(pending)
        
        # 移除已完成的
        for c in completed:
            self._pending_inferences.remove(c)
        
        if completed:
            self._save_pending_inferences()
    
    def _find_lora_path(self, task_id: str, epoch: int) -> Optional[str]:
        """查找指定任务和 epoch 的 LoRA 路径"""
        task_dir = LORA_ROOT / task_id
        if not task_dir.exists():
            return None
        
        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir():
                continue
            epoch_dir = run_dir / f'epoch{epoch}'
            lora_file = epoch_dir / 'adapter_model.safetensors'
            if lora_file.exists():
                return str(lora_file)
        return None
    
    def _create_inference_from_pending(self, pending: Dict):
        """从等待任务创建实际推理任务"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1)
            
            # 创建推理任务
            task_id = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            task = InferenceTask(
                task_id=task_id,
                status=InferenceStatus.QUEUED,
                lora_path=pending['lora_path'],
                lora_name=pending.get('lora_name', ''),
                prompt='',
                trigger_word=pending.get('trigger_word', ''),
                image_path=pending.get('image_path'),
                gpu_id=self._find_available_gpu(),
                lora_strength=pending.get('lora_strength', 1.0),
                batch_id=pending.get('batch_id'),
                image_index=pending.get('image_index', 0)
            )
            
            self._tasks[task_id] = task
            self._task_locks[task_id] = threading.Lock()
            self._save_tasks()
            
            # 异步执行
            executor.submit(self._run_inference, task_id)
            print(f"[Inference] 创建推理任务: {task_id}（来自等待队列）")
        except Exception as e:
            print(f"[Inference] 创建等待推理任务失败: {e}")
    
    def add_pending_inference(self, training_task_id: str, epoch: int, params: Dict) -> str:
        """添加等待 LoRA 生成的推理任务"""
        pending_id = f"pending_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        pending = {
            'pending_id': pending_id,
            'training_task_id': training_task_id,
            'epoch': epoch,
            'lora_name': f"epoch{epoch}",
            'trigger_word': params.get('trigger_word', ''),
            'image_path': params.get('image_path'),
            'lora_strength': params.get('lora_strength', 1.0),
            'batch_id': params.get('batch_id'),
            'image_index': params.get('image_index', 0),
            'created_at': datetime.now().isoformat()
        }
        
        self._pending_inferences.append(pending)
        self._save_pending_inferences()
        
        print(f"[Inference] 添加等待推理任务: {pending_id}（等待 {training_task_id}/epoch{epoch}）")
        return pending_id
    
    def _load_tasks(self):
        """加载历史任务"""
        if self._tasks_file.exists():
            try:
                with open(self._tasks_file, 'r') as f:
                    data = json.load(f)
                    updated_count = 0
                    for task_data in data:
                        task_id = task_data['task_id']
                        task_data['status'] = InferenceStatus(task_data['status'])
                        task_data['created_at'] = datetime.fromisoformat(task_data['created_at']) if task_data.get('created_at') else None
                        task_data['started_at'] = datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None
                        task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None
                        
                        # 加载时同步检查完成状态：如果输出文件存在但状态不是完成，则修正
                        if task_data['status'] in [InferenceStatus.RUNNING, InferenceStatus.LOADING]:
                            output_path = OUTPUT_ROOT / task_id / "output.mp4"
                            if output_path.exists():
                                task_data['status'] = InferenceStatus.COMPLETED
                                task_data['output_path'] = str(output_path)
                                task_data['progress'] = 100.0
                                if not task_data.get('completed_at'):
                                    task_data['completed_at'] = datetime.now()
                                updated_count += 1
                        
                        self._tasks[task_id] = InferenceTask(**task_data)
                        self._task_locks[task_id] = threading.Lock()
                
                print(f"[Inference] 加载了 {len(self._tasks)} 个历史任务")
                if updated_count > 0:
                    print(f"[Inference] 自动修正了 {updated_count} 个已完成任务的状态")
                    self._save_tasks()
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
        """列出训练任务及其 LoRA（按任务分组）
        
        支持训练中/排队中的任务，显示已完成和将要生成的所有 epoch
        """
        tasks = {}
        tasks_info = {}  # 存储任务配置信息
        
        # 读取任务配置
        tasks_json = LORA_ROOT / "tasks.json"
        if tasks_json.exists():
            try:
                with open(tasks_json, 'r') as f:
                    for t in json.load(f):
                        tasks_info[t.get('task_id')] = t
            except:
                pass
        
        # 扫描已有 LoRA 的任务
        for task_dir in LORA_ROOT.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith('train_'):
                continue
            
            task_id = task_dir.name
            task_info = tasks_info.get(task_id, {})
            task_name = task_info.get('description', task_id)
            task_status = task_info.get('status', 'unknown')
            total_epochs = task_info.get('config', {}).get('epochs', 60)
            save_every = task_info.get('config', {}).get('save_every_n_epochs', 5)
            
            epochs = []
            existing_epochs = set()
            
            # 查找已有的 epoch
            for run_dir in task_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                for epoch_dir in sorted(run_dir.iterdir()):
                    if not epoch_dir.is_dir() or not epoch_dir.name.startswith('epoch'):
                        continue
                    
                    lora_file = epoch_dir / 'adapter_model.safetensors'
                    if lora_file.exists():
                        epoch_num = int(epoch_dir.name.replace('epoch', ''))
                        existing_epochs.add(epoch_num)
                        epochs.append({
                            'epoch': epoch_num,
                            'name': epoch_dir.name,
                            'path': str(lora_file),
                            'size_mb': lora_file.stat().st_size / (1024 * 1024),
                            'created_at': datetime.fromtimestamp(lora_file.stat().st_mtime).isoformat(),
                            'ready': True
                        })
            
            # 如果任务正在训练或排队中，添加将要生成的 epoch
            if task_status in ['queued', 'running']:
                for e in range(save_every, total_epochs + 1, save_every):
                    if e not in existing_epochs:
                        epochs.append({
                            'epoch': e,
                            'name': f'epoch{e}',
                            'path': None,
                            'size_mb': 0,
                            'created_at': None,
                            'ready': False,
                            'pending': True
                        })
            
            if epochs:
                epochs.sort(key=lambda x: x['epoch'], reverse=True)
                tasks[task_id] = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'task_status': task_status,
                    'epochs': epochs,
                    'latest_epoch': max(e['epoch'] for e in epochs if e.get('ready', False)) if any(e.get('ready') for e in epochs) else 0,
                    'total_epochs': total_epochs,
                    'created_at': min(e['created_at'] for e in epochs if e.get('created_at')) if any(e.get('created_at') for e in epochs) else datetime.now().isoformat()
                }
        
        # 添加尚未产生 LoRA 的训练中/排队中任务
        for task_id, task_info in tasks_info.items():
            if task_id in tasks:
                continue
            task_status = task_info.get('status', 'unknown')
            if task_status in ['queued', 'running']:
                total_epochs = task_info.get('config', {}).get('epochs', 60)
                save_every = task_info.get('config', {}).get('save_every_n_epochs', 5)
                epochs = []
                for e in range(save_every, total_epochs + 1, save_every):
                    epochs.append({
                        'epoch': e,
                        'name': f'epoch{e}',
                        'path': None,
                        'size_mb': 0,
                        'created_at': None,
                        'ready': False,
                        'pending': True
                    })
                if epochs:
                    epochs.sort(key=lambda x: x['epoch'], reverse=True)
                    tasks[task_id] = {
                        'task_id': task_id,
                        'task_name': task_info.get('description', task_id),
                        'task_status': task_status,
                        'epochs': epochs,
                        'latest_epoch': 0,
                        'total_epochs': total_epochs,
                        'created_at': task_info.get('created_at', datetime.now().isoformat())
                    }
        
        # 按创建时间排序
        result = list(tasks.values())
        result.sort(key=lambda x: x['created_at'] or '', reverse=True)
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
    
    def _find_available_gpu(self) -> Optional[int]:
        """查找空闲的推理 GPU（4-7）"""
        try:
            # 使用 nvidia-smi 获取 GPU 状态
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                gpu_usage = {}
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_id = int(parts[0].strip())
                        mem_used = int(parts[1].strip())
                        gpu_usage[gpu_id] = mem_used
                
                # 检查 GPU 4-7 中哪个正在被当前任务使用
                running_gpus = set()
                for task in self._tasks.values():
                    if task.status in [InferenceStatus.RUNNING, InferenceStatus.LOADING]:
                        running_gpus.add(task.gpu_id)
                
                # 找一个未被使用的 GPU（优先选择显存使用最少的）
                candidates = []
                for gpu_id in INFERENCE_GPU_RANGE:
                    if gpu_id not in running_gpus:
                        mem = gpu_usage.get(gpu_id, 0)
                        candidates.append((gpu_id, mem))
                
                if candidates:
                    # 选择显存占用最少的
                    candidates.sort(key=lambda x: x[1])
                    return candidates[0][0]
        except Exception as e:
            print(f"[Inference] 查找空闲 GPU 失败: {e}")
        
        # 默认返回 GPU 4
        return 4
    
    def create_inference_task(self, data: Dict[str, Any]) -> InferenceTask:
        """
        创建推理任务
        
        Workflow 节点映射:
        - 节点 58 (LoadImage) ← data['image_path']
        - 节点 71 (WanVideoLoraSelect) ← data['lora_path'], data['lora_strength']
        - 节点 81 (TextToLowercase) ← data['trigger_word']
        - 节点 30 (VHS_VideoCombine) → output
        """
        task_id = f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # 节点 71: LoRA 验证
        lora_path = data.get('lora_path', '')
        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"LoRA 文件不存在: {lora_path}")
        
        # 自动分配 GPU（4-7）
        gpu_id = self._find_available_gpu()
        
        # 节点 81: 触发词
        trigger_word = data.get('trigger_word', '') or data.get('prompt', 'a person dancing')
        
        task = InferenceTask(
            task_id=task_id,
            status=InferenceStatus.QUEUED,
            lora_path=lora_path,                                    # 节点 71
            lora_name=data.get('lora_name', Path(lora_path).parent.name),
            prompt=trigger_word,  # 兼容
            trigger_word=trigger_word,                              # 节点 81
            image_path=data.get('image_path'),                      # 节点 58
            gpu_id=gpu_id,
            width=data.get('width', 832),
            height=data.get('height', 480),
            num_frames=data.get('num_frames', 81),
            num_steps=data.get('num_steps', 4),                     # workflow 默认 4
            guidance_scale=data.get('guidance_scale', 1.0),         # workflow 默认 1
            lora_strength=data.get('lora_strength', 1.0),           # 节点 71 强度
            seed=data.get('seed', -1),
            use_auto_caption=data.get('use_auto_caption', True),    # 是否使用 QwenVL 自动描述
            batch_id=data.get('batch_id'),
            image_index=data.get('image_index', 0),
            epoch_index=data.get('epoch_index', 0)
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
    
    def create_batch_inference_tasks(self, data: Dict[str, Any]) -> List[InferenceTask]:
        """创建批量推理任务（多图片）"""
        lora_path = data.get('lora_path', '')
        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"LoRA 文件不存在: {lora_path}")
        
        image_paths = data.get('image_paths', [])
        if not image_paths:
            raise ValueError("至少需要选择一张图片")
        
        # 生成批次 ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        tasks = []
        for i, image_path in enumerate(image_paths):
            task_data = {
                **data,
                'image_path': image_path,
                'batch_id': batch_id,
                'image_index': i
            }
            task = self.create_inference_task(task_data)
            tasks.append(task)
        
        return tasks
    
    def _run_inference(self, task_id: str):
        """执行推理 - 使用 ComfyUI workflow"""
        task = self._tasks.get(task_id)
        if not task:
            return
        
        try:
            task.status = InferenceStatus.LOADING
            task.started_at = datetime.now()
            self._save_tasks()
            
            print(f"[Inference] 开始推理任务: {task_id}")
            print(f"[Inference] LoRA: {task.lora_path}")
            print(f"[Inference] 触发词: {task.trigger_word}")
            print(f"[Inference] 图片: {task.image_path}")
            print(f"[Inference] GPU: {task.gpu_id}")
            
            # 检查图片是否存在
            if not task.image_path or not Path(task.image_path).exists():
                raise ValueError(f"图片文件不存在: {task.image_path}")
            
            # 设置输出路径
            output_dir = OUTPUT_ROOT / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video = output_dir / "output.mp4"
            
            task.status = InferenceStatus.RUNNING
            task.progress = 10.0
            self._save_tasks()
            
            # 使用 ComfyUI 推理脚本
            # Workflow 节点映射:
            # - 节点 58 (LoadImage) ← task.image_path
            # - 节点 71 (WanVideoLoraSelect) ← task.lora_path, task.lora_strength
            # - 节点 81 (TextToLowercase) ← task.trigger_word
            # - 节点 30 (VHS_VideoCombine) → output_video
            inference_script = Path(__file__).parent.parent / "utils" / "comfyui_inference.py"
            
            cmd = [
                sys.executable, str(inference_script),
                "--lora_path", task.lora_path,          # 节点 71
                "--trigger_word", task.trigger_word,    # 节点 81
                "--image_path", task.image_path,        # 节点 58
                "--output", str(output_video),          # 节点 30 输出
                "--lora_strength", str(task.lora_strength),
                "--gpu", str(task.gpu_id),
                "--seed", str(task.seed if task.seed > 0 else int(time.time()) % 100000),
                "--num_frames", str(task.num_frames),
                "--steps", str(task.num_steps),
                "--cfg", str(task.guidance_scale)
            ]
            
            # 自动描述选项
            if not task.use_auto_caption:
                cmd.append("--no_auto_caption")
            
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
                    print(f"[Inference] {line.rstrip()}")
                    
                    # 解析进度
                    if 'progress' in line.lower():
                        try:
                            import re
                            match = re.search(r'(\d+)%', line)
                            if match:
                                task.progress = float(match.group(1))
                                self._save_tasks()
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
        task = self._tasks.get(task_id)
        if task:
            return task
        
        # 多进程环境下内存可能不同步，尝试从文件重新加载
        self._load_tasks()
        return self._tasks.get(task_id)
    
    def list_tasks(self, limit: int = 50, batch_id: str = None) -> List[InferenceTask]:
        """列出任务
        
        Args:
            limit: 最大返回数量
            batch_id: 按批次ID筛选，如果提供则只返回该批次的任务
        """
        # 多进程环境下重新加载以获取最新数据
        self._load_tasks()
        tasks = list(self._tasks.values())
        
        # 按 batch_id 筛选
        if batch_id:
            tasks = [t for t in tasks if t.batch_id == batch_id]
        
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
