"""
训练任务管理服务
"""
import os
import sys
import json
import uuid
import threading
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from models.training import (
    TrainingTask, TrainingTaskRequest, TrainingTaskResponse, TrainingListResponse,
    TrainingProgressResponse, MetricsHistoryResponse, LogsResponse, StopTaskResponse,
    TrainingStatus, TrainingProgress, TrainingMetrics, SystemStats, GPUMemory,
    TrainingSpeed, LogEntry, LogLevel, LogType, Checkpoint
)
from services.gpu_service import gpu_service
from utils.training_wrapper import TrainingWrapper

# 训练输出目录
TRAINING_OUTPUT_ROOT = Path(os.environ.get("TRAINING_OUTPUT_ROOT", "./data/outputs"))
TASKS_FILE = TRAINING_OUTPUT_ROOT / "tasks.json"

class TrainingService:
    """训练任务管理服务"""

    def __init__(self):
        """初始化训练服务"""
        self._tasks: Dict[str, TrainingTask] = {}  # 训练任务映射
        self._task_locks: Dict[str, threading.Lock] = {}  # 任务锁
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=10)

        # 确保输出目录存在
        TRAINING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        
        # 加载已保存的任务
        self._load_tasks()
    
    def _load_tasks(self):
        """从文件加载任务列表"""
        if TASKS_FILE.exists():
            try:
                with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                for task_data in tasks_data:
                    try:
                        # 重建任务对象
                        task = TrainingTask(
                            task_id=task_data['task_id'],
                            status=TrainingStatus(task_data['status']),
                            gpu_id=task_data.get('gpu_id', 0),
                            gpu_name=task_data.get('gpu_name', 'GPU 0'),
                            model_type=task_data.get('model_type', 'wan'),
                            description=task_data.get('description', ''),
                            config=task_data.get('config', {}),
                            dataset=task_data.get('dataset'),
                            created_at=datetime.fromisoformat(task_data['created_at']) if task_data.get('created_at') else None,
                            started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                            completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None,
                            logs_path=task_data.get('logs_path')
                        )
                        self._tasks[task.task_id] = task
                        self._task_locks[task.task_id] = threading.Lock()
                    except Exception as e:
                        print(f"加载任务 {task_data.get('task_id')} 失败: {e}")
                print(f"[API] 已加载 {len(self._tasks)} 个历史任务")
                # 同步任务状态（检查运行中的任务是否真的在运行）
                self._sync_task_status()
            except Exception as e:
                print(f"加载任务文件失败: {e}")
    
    def _sync_task_status(self):
        """同步任务状态 - 检查运行中的任务是否真的在运行"""
        import subprocess
        from pathlib import Path
        
        updated = False
        for task_id, task in self._tasks.items():
            if task.status == TrainingStatus.RUNNING:
                # 检查是否有对应的训练进程
                try:
                    result = subprocess.run(
                        ['pgrep', '-f', task_id],
                        capture_output=True, text=True, timeout=5
                    )
                    process_exists = result.returncode == 0 and result.stdout.strip()
                except:
                    process_exists = False
                
                if not process_exists:
                    # 进程不存在，检查日志判断状态
                    log_file = Path(f"/tmp/diffusion_pipe_logs/{task_id}.log")
                    new_status = TrainingStatus.FAILED  # 默认失败
                    
                    if log_file.exists():
                        try:
                            content = log_file.read_text(encoding='utf-8', errors='ignore')
                            if 'TRAINING COMPLETE!' in content:
                                new_status = TrainingStatus.COMPLETED
                            elif 'exits with return code = 1' in content or 'Error' in content or 'Traceback' in content:
                                new_status = TrainingStatus.FAILED
                        except:
                            pass
                    
                    print(f"[API] 任务 {task_id} 状态更新: running -> {new_status.value}")
                    task.status = new_status
                    task.completed_at = datetime.now()
                    updated = True
        
        if updated:
            self._save_tasks()
    
    def _save_tasks(self):
        """保存任务列表到文件"""
        try:
            tasks_data = []
            for task in self._tasks.values():
                tasks_data.append({
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'gpu_id': task.gpu_id,
                    'gpu_name': task.gpu_name,
                    'model_type': task.model_type,
                    'description': task.description,
                    'config': task.config,
                    'dataset': task.dataset,
                    'created_at': task.created_at.isoformat() if task.created_at else None,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'logs_path': task.logs_path
                })
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存任务文件失败: {e}")

    def create_training_task(self, request: TrainingTaskRequest) -> TrainingTaskResponse:
        """创建训练任务（快速返回，不阻塞）"""
        # 生成任务ID
        task_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print(f"[API] 创建训练任务: {task_id}")

        # 使用简单的GPU名称，不查询GPU服务
        gpu_name = f"GPU {request.gpu_id}"

        # 创建训练配置
        output_dir = TRAINING_OUTPUT_ROOT / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = request.config or {}
        config['output_dir'] = str(output_dir)

        # 创建任务对象
        task = TrainingTask(
            task_id=task_id,
            status=TrainingStatus.QUEUED,
            gpu_id=request.gpu_id,
            gpu_name=gpu_name,
            model_type=request.model_type,
            description=request.description,
            config=config,
            dataset=request.dataset,
            created_at=datetime.now(),
            logs_path=str(output_dir / 'logs' / 'train.log'),
            progress=TrainingProgress(
                current_epoch=0,
                total_epochs=config.get('epochs', 60),
                current_step=0,
                total_steps=0,
                overall_progress=0
            )
        )

        # 保存任务
        self._tasks[task_id] = task
        self._task_locks[task_id] = threading.Lock()
        
        # 保存任务到文件
        self._save_tasks()

        print(f"[API] 任务 {task_id} 已创建，状态: QUEUED")

        # 异步启动训练（不阻塞API响应）
        self._executor.submit(self._start_training_task, task_id)

        # 立即返回响应
        return TrainingTaskResponse(
            task_id=task_id,
            status=TrainingStatus.QUEUED,
            gpu_id=request.gpu_id,
            gpu_name=gpu_name,
            model_type=request.model_type,
            config=config,
            description=request.description,
            created_at=task.created_at,
            estimated_start_time=datetime.now() + timedelta(seconds=5),
            estimated_duration=None,
            checkpoints={}
        )

    def save_draft(self, data: dict) -> TrainingTaskResponse:
        """保存训练草稿（不启动训练）"""
        # 生成草稿ID
        task_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print(f"[API] 保存草稿: {task_id}")

        gpu_id = data.get('gpu_id')
        gpu_name = f"GPU {gpu_id}" if gpu_id is not None else "未选择"

        config = data.get('config', {})
        
        # 创建草稿任务对象
        task = TrainingTask(
            task_id=task_id,
            status=TrainingStatus.DRAFT,  # 草稿状态
            gpu_id=gpu_id if gpu_id is not None else -1,
            gpu_name=gpu_name,
            model_type=data.get('model_type', 'wan'),
            description=data.get('description', '未命名草稿'),
            config=config,
            dataset=data.get('dataset'),
            created_at=datetime.now(),
            logs_path=None,
            progress=None
        )
        
        # 保存额外的草稿数据（原始视频、处理后视频等）
        task.raw_videos = data.get('raw_videos', [])
        task.processed_videos = data.get('processed_videos', [])

        # 保存任务
        self._tasks[task_id] = task
        self._task_locks[task_id] = threading.Lock()
        
        # 保存任务到文件
        self._save_tasks()

        print(f"[API] 草稿 {task_id} 已保存")

        return TrainingTaskResponse(
            task_id=task_id,
            status=TrainingStatus.DRAFT,
            gpu_id=task.gpu_id,
            gpu_name=gpu_name,
            model_type=task.model_type,
            config=config,
            description=task.description,
            created_at=task.created_at,
            estimated_start_time=None,
            estimated_duration=None,
            checkpoints={}
        )

    def copy_task(self, task_id: str, new_name: str) -> TrainingTaskResponse:
        """复制任务"""
        # 获取原任务
        original = self._tasks.get(task_id)
        if not original:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 生成新的任务ID
        new_task_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        print(f"[API] 复制任务 {task_id} -> {new_task_id}")
        
        # 复制任务
        new_task = TrainingTask(
            task_id=new_task_id,
            status=TrainingStatus.DRAFT,
            gpu_id=original.gpu_id,
            gpu_name=original.gpu_name,
            model_type=original.model_type,
            description=new_name,
            config=original.config.copy() if original.config else {},
            dataset=original.dataset.copy() if original.dataset else None,
            created_at=datetime.now(),
            logs_path=None,
            progress=None
        )
        
        # 复制视频数据
        if hasattr(original, 'raw_videos'):
            new_task.raw_videos = original.raw_videos.copy() if original.raw_videos else []
        if hasattr(original, 'processed_videos'):
            new_task.processed_videos = original.processed_videos.copy() if original.processed_videos else []
        
        # 保存
        self._tasks[new_task_id] = new_task
        self._task_locks[new_task_id] = threading.Lock()
        self._save_tasks()
        
        return TrainingTaskResponse(
            task_id=new_task_id,
            status=TrainingStatus.DRAFT,
            gpu_id=new_task.gpu_id,
            gpu_name=new_task.gpu_name,
            model_type=new_task.model_type,
            config=new_task.config,
            description=new_task.description,
            created_at=new_task.created_at,
            estimated_start_time=None,
            estimated_duration=None,
            checkpoints={}
        )

    def delete_task(self, task_id: str):
        """删除任务"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 如果任务正在运行，先停止
        if task.status == TrainingStatus.RUNNING:
            try:
                self.stop_training_task(task_id, force=True)
            except:
                pass
        
        # 删除任务
        del self._tasks[task_id]
        if task_id in self._task_locks:
            del self._task_locks[task_id]
        
        self._save_tasks()
        print(f"[API] 任务 {task_id} 已删除")

    def _start_training_task(self, task_id: str):
        """启动训练任务（在后台线程中执行）"""
        task = self._tasks.get(task_id)
        if not task:
            print(f"[{task_id}] 任务不存在")
            return

        print(f"[{task_id}] 开始准备训练任务...")

        try:
            # 更新状态为运行中
            if task_id in self._task_locks:
                with self._task_locks[task_id]:
                    task.status = TrainingStatus.RUNNING
                    task.started_at = datetime.now()
                    task.updated_at = datetime.now()
                    self._save_tasks()
                    # 初始化进度
                    task.progress = TrainingProgress(
                        current_epoch=0,
                        total_epochs=task.config.get('epochs', 60),
                        current_step=0,
                        total_steps=1000,
                        overall_progress=0
                    )
            
            print(f"[{task_id}] 状态已更新为 RUNNING")

            # 尝试注册GPU任务
            try:
                gpu_service.register_task(task.gpu_id, task_id, "training")
            except Exception as e:
                print(f"[{task_id}] 注册GPU任务失败（非致命）: {e}")

            # 创建训练包装器并启动训练
            print(f"[{task_id}] 正在启动训练...")
            wrapper = TrainingWrapper()
            success = wrapper.start_training(
                task_id=task_id,
                gpu_id=task.gpu_id,
                model_type=task.model_type,
                config=task.config,
                dataset=task.dataset,
                output_dir=Path(task.config.get('output_dir', f'/tmp/{task_id}'))
            )

            print(f"[{task_id}] 训练启动结果: {'成功' if success else '失败'}")

            # 注意：实际训练是异步的，这里只是启动
            # 状态更新由训练进程日志解析完成
            if not success:
                if task_id in self._task_locks:
                    with self._task_locks[task_id]:
                        task.status = TrainingStatus.FAILED
                        task.error_message = "训练启动失败"
                        task.updated_at = datetime.now()
                        self._save_tasks()

        except Exception as e:
            print(f"[{task_id}] 训练任务异常: {e}")
            # 处理异常
            if task_id in self._task_locks:
                with self._task_locks[task_id]:
                    task.status = TrainingStatus.FAILED
                    task.error_message = str(e)
                    task.updated_at = datetime.now()
                    task.completed_at = datetime.now()
                    self._save_tasks()

            # 尝试释放GPU
            try:
                gpu_service.unregister_task(task.gpu_id)
            except:
                pass

    def stop_training_task(self, task_id: str, force: bool = False) -> StopTaskResponse:
        """停止训练任务"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")

        with self._task_locks[task_id]:
            if task.status not in [TrainingStatus.RUNNING, TrainingStatus.QUEUED]:
                raise ValueError(f"任务 {task_id} 当前状态为 {task.status.value}，无法停止")

            # 发送停止信号
            if task.status == TrainingStatus.RUNNING:
                # TODO: 实现真正的停止逻辑
                # 这里需要向训练进程发送SIGTERM或SIGKILL
                pass

            task.status = TrainingStatus.STOPPED if force else TrainingStatus.PAUSED
            task.updated_at = datetime.now()
            task.completed_at = datetime.now()
            self._save_tasks()

            # 计算训练时间
            training_time = None
            if task.started_at:
                training_time = int((task.completed_at - task.started_at).total_seconds())

            # 释放GPU
            gpu_service.unregister_task(task.gpu_id)

            return StopTaskResponse(
                task_id=task_id,
                status=task.status,
                stopped_at=task.completed_at,
                checkpoint_saved=True,
                last_checkpoint=None,
                training_time=training_time
            )

    def get_training_task(self, task_id: str) -> TrainingTask:
        """获取训练任务详情"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")

        # 更新进度和指标
        self._update_task_progress(task)

        return task

    def list_training_tasks(
        self,
        status: Optional[TrainingStatus] = None,
        gpu_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> TrainingListResponse:
        """获取训练任务列表"""
        tasks = list(self._tasks.values())

        # 过滤
        if status:
            tasks = [t for t in tasks if t.status == status]
        if gpu_id is not None:
            tasks = [t for t in tasks if t.gpu_id == gpu_id]

        # 排序（按创建时间倒序）
        tasks.sort(key=lambda t: t.created_at or datetime.min, reverse=True)

        # 分页
        total = len(tasks)
        tasks = tasks[offset:offset + limit]

        # 更新进度
        for task in tasks:
            self._update_task_progress(task)

        # 构建分页信息
        pagination = {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }

        return TrainingListResponse(tasks=tasks, pagination=pagination)

    def get_training_progress(self, task_id: str) -> TrainingProgressResponse:
        """获取训练进度"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")

        # 更新进度
        self._update_task_progress(task)

        # 计算速度
        speed = None
        if task.progress and task.progress.training_time:
            steps_per_second = task.progress.current_step / task.progress.training_time if task.progress.training_time > 0 else 0
            speed = TrainingSpeed(
                steps_per_second=steps_per_second,
                samples_per_second=steps_per_second,
                time_per_step=1.0 / steps_per_second if steps_per_second > 0 else None
            )

        return TrainingProgressResponse(
            task_id=task_id,
            status=task.status,
            progress=task.progress,
            metrics=task.metrics,
            speed=speed,
            updated_at=task.updated_at
        )

    def get_metrics_history(
        self,
        task_id: str,
        metric: str = "loss",
        type_: str = "epoch",
        limit: int = 100
    ) -> MetricsHistoryResponse:
        """获取指标历史"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")

        # 获取指标数据
        data = []
        if metric == "loss" and type_ == "epoch" and task.metrics:
            for epoch, loss in enumerate(task.metrics.epoch_losses, 1):
                data.append({
                    "epoch": epoch,
                    "value": loss,
                    "timestamp": (task.created_at or datetime.now()).isoformat()
                })

        # 限制数量
        data = data[-limit:]

        return MetricsHistoryResponse(
            task_id=task_id,
            metric=metric,
            type=type_,
            data=data
        )

    def get_training_logs(
        self,
        task_id: str,
        type_: str = "all",
        level: Optional[str] = None,
        tail: Optional[int] = None,
        since: Optional[str] = None
    ) -> LogsResponse:
        """获取训练日志"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")

        logs = []
        
        # 尝试多个可能的日志路径
        possible_log_files = [
            f"/tmp/diffusion_pipe_logs/{task_id}.log",  # 训练包装器写入的日志
            task.logs_path,  # 任务配置的日志路径
            str(TRAINING_OUTPUT_ROOT / task_id / 'logs' / 'train.log'),
        ]
        
        log_file = None
        for path in possible_log_files:
            if path and os.path.exists(path):
                log_file = path
                break

        if log_file:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                    # 应用过滤条件
                    if tail:
                        lines = lines[-tail:]

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # 尝试解析日志行
                        try:
                            # 格式: [task_id] message 或直接 message
                            message = line
                            if line.startswith('['):
                                # 去掉任务ID前缀
                                idx = line.find(']')
                                if idx > 0:
                                    message = line[idx+1:].strip()
                            
                            logs.append(LogEntry(
                                timestamp=datetime.now(),
                                level=LogLevel.INFO,
                                type=LogType.TRAIN,
                                message=message
                            ))
                        except Exception:
                            logs.append(LogEntry(
                                timestamp=datetime.now(),
                                level=LogLevel.INFO,
                                type=LogType.ALL,
                                message=line
                            ))
            except Exception as e:
                print(f"读取日志文件失败: {e}")

        return LogsResponse(
            task_id=task_id,
            log_file=log_file,
            total_lines=len(logs),
            logs=logs
        )

    def _get_gpu_info(self, gpu_id: int) -> Optional[Dict]:
        """获取GPU信息"""
        try:
            # 从GPU服务获取GPU状态
            gpu_response = gpu_service.get_all_gpus()
            for gpu in gpu_response.gpus:
                if gpu.gpu_id == gpu_id:
                    return {
                        'name': gpu.name,
                        'status': gpu.status.value if gpu.status else 'unknown'
                    }
        except Exception:
            pass
        return None

    def _update_task_progress(self, task: TrainingTask):
        """更新任务进度 - 从日志文件解析真实进度"""
        if task.status not in [TrainingStatus.RUNNING, TrainingStatus.QUEUED]:
            return

        # 初始化进度
        if not task.progress:
            total_epochs = task.config.get('epochs', 60) if task.config else 60
            task.progress = TrainingProgress(
                current_epoch=0,
                total_epochs=total_epochs,
                current_step=0,
                total_steps=0
            )

        # 检查训练进程是否还在运行
        try:
            from api.utils.training_wrapper import training_wrapper
            status = training_wrapper.get_training_status(task.task_id)
            if status:
                if not status.get('running'):
                    # 进程已结束
                    returncode = status.get('returncode')
                    if returncode == 0:
                        task.status = TrainingStatus.COMPLETED
                    else:
                        task.status = TrainingStatus.FAILED
                        task.error_message = f"训练进程退出，返回码: {returncode}"
                    task.completed_at = datetime.now()
                    self._save_tasks()
                    return
        except Exception:
            pass

        # 从日志文件解析进度
        log_file = f"/tmp/diffusion_pipe_logs/{task.task_id}.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in reversed(lines):
                    line = line.strip()
                    # 解析 Epoch 信息: "Epoch 1/60" 或类似格式
                    import re
                    epoch_match = re.search(r'[Ee]poch\s*[:\s]*(\d+)\s*/\s*(\d+)', line)
                    if epoch_match:
                        task.progress.current_epoch = int(epoch_match.group(1))
                        task.progress.total_epochs = int(epoch_match.group(2))
                    
                    # 解析 Step 信息: "Step 100/1000" 或 "step: 100"
                    step_match = re.search(r'[Ss]tep\s*[:\s]*(\d+)\s*(?:/\s*(\d+))?', line)
                    if step_match:
                        task.progress.current_step = int(step_match.group(1))
                        if step_match.group(2):
                            task.progress.total_steps = int(step_match.group(2))
                    
                    # 解析 Loss 信息: "loss: 0.1234" 或 "Loss: 0.1234"
                    loss_match = re.search(r'[Ll]oss\s*[:\s]*([0-9.]+)', line)
                    if loss_match:
                        if not task.metrics:
                            task.metrics = TrainingMetrics(
                                current_loss=float(loss_match.group(1)),
                                avg_loss=float(loss_match.group(1)),
                                learning_rate=task.config.get('optimizer', {}).get('lr', 2e-5) if task.config else 2e-5
                            )
                        else:
                            task.metrics.current_loss = float(loss_match.group(1))

            except Exception as e:
                print(f"解析训练日志失败: {e}")

        # 计算总体进度
        if task.progress.total_epochs > 0:
            task.progress.overall_progress = (task.progress.current_epoch / task.progress.total_epochs) * 100
        if task.progress.total_steps > 0:
            task.progress.epoch_progress = (task.progress.current_step / task.progress.total_steps) * 100

        # 获取GPU内存使用
        self._update_gpu_memory(task)

        task.updated_at = datetime.now()

    def _update_gpu_memory(self, task: TrainingTask):
        """获取GPU内存使用情况"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', f'--id={task.gpu_id}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    used = int(parts[0].strip())
                    total = int(parts[1].strip())
                    # 使用字典存储而不是SystemStats对象
                    if not task.system_stats:
                        task.system_stats = {'gpu_memory': {'used': used, 'total': total}}
                    else:
                        # 如果是对象，转换为字典
                        if hasattr(task.system_stats, '__dict__'):
                            task.system_stats = {'gpu_memory': {'used': used, 'total': total}}
                        else:
                            task.system_stats['gpu_memory'] = {'used': used, 'total': total}
        except Exception as e:
            print(f"获取GPU内存失败: {e}")

# 创建全局训练服务实例
training_service = TrainingService()
