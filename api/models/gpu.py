"""
GPU 相关数据模型
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class GPUStatus(Enum):
    """GPU 状态枚举"""
    AVAILABLE = "available"
    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    UNKNOWN = "unknown"

class TaskType(Enum):
    """任务类型枚举"""
    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    EVALUATION = "evaluation"

@dataclass
class GPUMemory:
    """GPU 显存信息"""
    total: int  # MB
    used: int   # MB
    free: int   # MB
    utilization: int  # 百分比
    total_gb: float = 0.0  # GB (便于前端显示)
    free_gb: float = 0.0   # GB (便于前端显示)

    @property
    def used_percent(self) -> float:
        """使用的显存百分比"""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100

@dataclass
class GPUTemperature:
    """GPU 温度信息"""
    gpu: int  # 摄氏度
    memory: Optional[int] = None  # 显存温度（如果支持）

@dataclass
class GPUPower:
    """GPU 功耗信息"""
    current: int  # 当前功耗 (W)
    limit: int    # 功耗限制 (W)
    usage_percent: float  # 使用百分比

@dataclass
class CurrentTask:
    """当前任务信息"""
    task_id: str
    task_type: TaskType
    progress: Optional[float] = None  # 进度百分比
    start_time: Optional[datetime] = None
    estimated_remaining_time: Optional[int] = None  # 秒

@dataclass
class GPUInfo:
    """GPU 信息"""
    gpu_id: int
    name: str
    memory: GPUMemory
    utilization: int  # GPU 利用率百分比
    temperature: GPUTemperature
    power: Optional[GPUPower] = None
    status: GPUStatus = GPUStatus.UNKNOWN
    current_task: Optional[CurrentTask] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None

    # 计算属性
    @property
    def is_available(self) -> bool:
        """GPU 是否可用"""
        return self.status == GPUStatus.AVAILABLE

    @property
    def memory_gb(self) -> float:
        """显存总量 (GB)"""
        return self.memory.total / 1024.0

    @property
    def free_memory_gb(self) -> float:
        """可用显存 (GB)"""
        return self.memory.free / 1024.0

@dataclass
class GPUStatusResponse:
    """GPU 状态响应"""
    gpus: List[GPUInfo]
    summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AvailableGPUResponse:
    """可用 GPU 响应"""
    available_gpus: List[GPUInfo]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GPUMetrics:
    """GPU 指标"""
    gpu_id: int
    timestamp: datetime
    memory: GPUMemory
    utilization: int
    temperature: GPUTemperature
    power: Optional[GPUPower] = None

@dataclass
class GPUHistory:
    """GPU 历史数据"""
    gpu_id: int
    metrics: List[GPUMetrics]
    start_time: datetime
    end_time: datetime
