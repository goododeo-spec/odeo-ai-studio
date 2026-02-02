"""
GPU 管理服务 - 稳定版本
始终使用 nvidia-smi 命令获取 GPU 信息
"""
import time
import threading
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from models.gpu import (
    GPUInfo, GPUStatus, GPUStatusResponse, AvailableGPUResponse,
    GPUMemory, GPUTemperature, GPUPower, CurrentTask, TaskType,
    GPUMetrics, GPUHistory
)
from utils.common import format_bytes


class GPUService:
    """GPU 管理服务 - 使用 nvidia-smi"""

    def __init__(self):
        """初始化 GPU 服务"""
        self._gpu_cache: List[GPUInfo] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 2.0  # 缓存2秒
        self._active_tasks: Dict[int, CurrentTask] = {}
        self._lock = threading.Lock()
        
        # 初始化时立即获取一次 GPU 信息
        self._refresh_gpu_cache()
        print(f"GPU Service initialized: found {len(self._gpu_cache)} GPUs")

    def _refresh_gpu_cache(self):
        """刷新 GPU 缓存"""
        try:
            result = subprocess.run(
                ['nvidia-smi', 
                 '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"[GPU] nvidia-smi 返回错误: {result.returncode}")
                print(f"[GPU] stderr: {result.stderr}")
                return
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 7:
                    print(f"[GPU] 解析行失败: {line}")
                    continue
                
                try:
                    gpu_id = int(parts[0])
                    name = parts[1]
                    total_mb = int(parts[2])
                    used_mb = int(parts[3])
                    free_mb = int(parts[4])
                    util = int(parts[5]) if parts[5].isdigit() else 0
                    temp = int(parts[6]) if parts[6].isdigit() else 0
                    
                    # 检查是否有注册的任务
                    with self._lock:
                        current_task = self._active_tasks.get(gpu_id)
                    
                    status = GPUStatus.TRAINING if current_task else GPUStatus.AVAILABLE
                    
                    gpu = GPUInfo(
                        gpu_id=gpu_id,
                        name=name,
                        memory=GPUMemory(
                            total=total_mb,
                            used=used_mb,
                            free=free_mb,
                            utilization=int(used_mb / total_mb * 100) if total_mb > 0 else 0,
                            total_gb=total_mb / 1024,
                            free_gb=free_mb / 1024
                        ),
                        utilization=util,
                        temperature=GPUTemperature(gpu=temp),
                        power=None,
                        status=status,
                        current_task=current_task
                    )
                    gpus.append(gpu)
                except (ValueError, IndexError) as e:
                    print(f"[GPU] 解析错误: {e}, line: {line}")
                    continue
            
            if gpus:
                self._gpu_cache = gpus
                self._cache_time = time.time()
                
        except subprocess.TimeoutExpired:
            print("[GPU] nvidia-smi 超时")
        except FileNotFoundError:
            print("[GPU] nvidia-smi 未找到")
        except Exception as e:
            print(f"[GPU] 获取 GPU 信息失败: {e}")

    def get_all_gpus(self) -> GPUStatusResponse:
        """获取所有 GPU 状态"""
        # 检查缓存是否过期
        if time.time() - self._cache_time > self._cache_ttl:
            self._refresh_gpu_cache()
        
        gpus = self._gpu_cache
        
        # 如果没有 GPU 数据，返回空响应（不使用 mock 数据）
        if not gpus:
            print("[GPU] 警告: 无法获取 GPU 信息")
            return GPUStatusResponse(
                gpus=[],
                summary={
                    "total_gpus": 0,
                    "available_gpus": 0,
                    "busy_gpus": 0,
                    "total_memory": 0,
                    "available_memory": 0,
                    "memory_utilization": 0
                },
                timestamp=datetime.now()
            )
        
        summary = self._calculate_summary(gpus)
        return GPUStatusResponse(
            gpus=gpus,
            summary=summary,
            timestamp=datetime.now()
        )

    def get_available_gpus(
        self,
        min_memory: Optional[int] = None,
        task_type: Optional[str] = None
    ) -> AvailableGPUResponse:
        """获取可用 GPU 列表"""
        gpu_response = self.get_all_gpus()

        available_gpus = []
        for gpu in gpu_response.gpus:
            if gpu.status != GPUStatus.AVAILABLE:
                continue
            if min_memory and gpu.memory.free < min_memory:
                continue
            available_gpus.append(gpu)

        return AvailableGPUResponse(
            available_gpus=available_gpus,
            timestamp=datetime.now()
        )

    def _calculate_summary(self, gpus: List[GPUInfo]) -> Dict[str, Any]:
        """计算汇总信息"""
        total_gpus = len(gpus)
        available_gpus = sum(1 for gpu in gpus if gpu.status == GPUStatus.AVAILABLE)
        busy_gpus = total_gpus - available_gpus

        total_memory = sum(gpu.memory.total for gpu in gpus)
        available_memory = sum(gpu.memory.free for gpu in gpus)

        return {
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "busy_gpus": busy_gpus,
            "total_memory": total_memory,
            "available_memory": available_memory,
            "memory_utilization": 100 - (available_memory / total_memory * 100) if total_memory > 0 else 0
        }

    def register_task(self, gpu_id: int, task_id: str, task_type: str):
        """注册任务到 GPU"""
        with self._lock:
            self._active_tasks[gpu_id] = CurrentTask(
                task_id=task_id,
                task_type=TaskType(task_type),
                start_time=datetime.now()
            )
        # 刷新缓存以反映新状态
        self._refresh_gpu_cache()

    def unregister_task(self, gpu_id: int):
        """从 GPU 注销任务"""
        with self._lock:
            if gpu_id in self._active_tasks:
                del self._active_tasks[gpu_id]
        # 刷新缓存以反映新状态
        self._refresh_gpu_cache()

    def update_task_progress(self, gpu_id: int, progress: float, eta_seconds: Optional[int] = None):
        """更新任务进度"""
        with self._lock:
            if gpu_id in self._active_tasks:
                self._active_tasks[gpu_id].progress = progress
                self._active_tasks[gpu_id].estimated_remaining_time = eta_seconds

    # 兼容旧的监控接口
    def start_monitoring(self, interval: int = 5):
        """启动监控（兼容）"""
        pass

    def stop_monitoring(self):
        """停止监控（兼容）"""
        pass


# 全局实例
gpu_service = GPUService()
