"""
训练任务相关数据模型
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class TrainingStatus(Enum):
    """训练状态枚举"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"
    DRAFT = "draft"  # 草稿状态

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogType(Enum):
    """日志类型枚举"""
    TRAIN = "train"
    EVAL = "eval"
    SYSTEM = "system"
    ALL = "all"

@dataclass
class TrainingProgress:
    """训练进度信息"""
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    current_batch: Optional[int] = None
    total_batches: Optional[int] = None
    epoch_progress: float = 0.0  # 当前epoch进度百分比
    overall_progress: float = 0.0  # 总体进度百分比
    eta_seconds: Optional[int] = None  # 预计剩余时间(秒)
    training_time: Optional[int] = None  # 已训练时间(秒)

@dataclass
class TrainingMetrics:
    """训练指标"""
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    epoch_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    epoch_losses: List[float] = field(default_factory=list)
    step_losses: List[float] = field(default_factory=list)

@dataclass
class TrainingSpeed:
    """训练速度"""
    steps_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    time_per_step: Optional[float] = None

@dataclass
class GPUMemory:
    """GPU显存信息"""
    total: int
    used: int
    free: int
    utilization: float

@dataclass
class SystemStats:
    """系统状态统计"""
    gpu_memory: Optional[GPUMemory] = None
    gpu_utilization: Optional[int] = None
    gpu_temperature: Optional[int] = None
    gpu_power: Optional[int] = None

@dataclass
class Checkpoint:
    """检查点信息"""
    epoch: Optional[int] = None
    step: Optional[int] = None
    path: Optional[str] = None
    timestamp: Optional[datetime] = None
    loss: Optional[float] = None

@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: Optional[str] = None
    epochs: Optional[int] = None
    micro_batch_size_per_gpu: Optional[int] = None
    pipeline_stages: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    gradient_clipping: Optional[float] = None
    warmup_steps: Optional[int] = None
    model: Optional[Dict[str, Any]] = None
    adapter: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    save: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    eval: Optional[Dict[str, Any]] = None

@dataclass
class DatasetConfig:
    """数据集配置"""
    resolutions: Optional[List[int]] = None
    enable_ar_bucket: bool = False
    ar_buckets: Optional[List[float]] = None
    frame_buckets: Optional[List[int]] = None
    directory: Optional[List[Dict[str, Any]]] = None

@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    status: TrainingStatus
    gpu_id: int
    gpu_name: Optional[str] = None
    model_type: Optional[str] = None
    description: Optional[str] = None
    config: Optional[TrainingConfig] = None
    dataset: Optional[DatasetConfig] = None
    progress: Optional[TrainingProgress] = None
    metrics: Optional[TrainingMetrics] = None
    system_stats: Optional[SystemStats] = None
    checkpoints: List[Checkpoint] = field(default_factory=list)
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs_path: Optional[str] = None
    error_message: Optional[str] = None
    data_task_id: Optional[str] = None  # 数据目录的任务ID（用于关联上传的视频）

@dataclass
class TrainingTaskRequest:
    """训练任务创建请求"""
    gpu_id: int
    model_type: str
    description: Optional[str] = None
    dataset: Optional[DatasetConfig] = None
    config: Optional[TrainingConfig] = None
    data_task_id: Optional[str] = None  # 数据目录的任务ID（用于关联上传的视频）
    raw_videos: Optional[List[str]] = None  # 原始视频列表
    processed_videos: Optional[List[dict]] = None  # 处理后的视频列表

@dataclass
class TrainingTaskResponse:
    """训练任务响应"""
    task_id: str
    status: TrainingStatus
    gpu_id: int
    gpu_name: Optional[str] = None
    model_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    estimated_start_time: Optional[datetime] = None
    estimated_duration: Optional[int] = None
    checkpoints: Optional[Dict[str, Any]] = None
    queue_position: Optional[int] = None  # 队列位置（GPU 满时）

@dataclass
class TrainingListResponse:
    """训练任务列表响应"""
    tasks: List[TrainingTask]
    pagination: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingProgressResponse:
    """训练进度响应"""
    task_id: str
    status: TrainingStatus
    progress: Optional[TrainingProgress] = None
    metrics: Optional[TrainingMetrics] = None
    speed: Optional[TrainingSpeed] = None
    updated_at: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MetricsHistoryResponse:
    """指标历史响应"""
    task_id: str
    metric: str
    type: str
    data: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LogEntry:
    """日志条目"""
    timestamp: Optional[datetime] = None
    level: Optional[LogLevel] = None
    type: Optional[LogType] = None
    message: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None

@dataclass
class LogsResponse:
    """日志响应"""
    task_id: str
    logs: List[LogEntry]
    log_file: Optional[str] = None
    total_lines: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StopTaskRequest:
    """停止任务请求"""
    force: bool = False

@dataclass
class StopTaskResponse:
    """停止任务响应"""
    task_id: str
    status: TrainingStatus
    stopped_at: Optional[datetime] = None
    checkpoint_saved: bool = False
    last_checkpoint: Optional[str] = None
    training_time: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
