"""
预处理相关数据模型
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class CaptionMethod(Enum):
    """提示词生成方法"""
    TAGS = "tags"
    SIMPLE = "simple"
    DETAILED = "detailed"
    EXTRA = "extra"
    MIXED = "mixed"
    EXTRA_MIXED = "extra_mixed"
    ANALYZE = "analyze"

class PreprocessingStatus(Enum):
    """预处理状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PreprocessingRequest:
    """预处理请求"""
    dataset_name: str  # 数据集名称
    video_directory: str  # 视频目录路径
    prompt_prefix: Optional[str] = None  # 提示词前缀
    caption_method: CaptionMethod = CaptionMethod.EXTRA_MIXED  # 提示词生成方法
    use_qwen_optimize: bool = True  # 是否使用千问优化
    qwen_api_key: Optional[str] = None  # 千问API密钥（可选）

@dataclass
class PreprocessingProgress:
    """预处理进度"""
    total_videos: int  # 总视频数
    processed_videos: int  # 已处理视频数
    current_video: str  # 当前处理的视频名
    current_step: str  # 当前步骤
    progress_percent: float  # 进度百分比
    start_time: datetime  # 开始时间
    estimated_remaining_time: Optional[int] = None  # 预估剩余时间（秒）

@dataclass
class PreprocessingResult:
    """预处理结果"""
    dataset_name: str
    output_directory: str
    status: PreprocessingStatus
    progress: PreprocessingProgress
    error_message: Optional[str] = None
    output_files: Optional[List[str]] = None  # 输出文件列表
    end_time: Optional[datetime] = None

@dataclass
class DatasetInfo:
    """数据集信息"""
    dataset_name: str
    directory: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    video_count: int = 0
    file_count: int = 0
    status: PreprocessingStatus = PreprocessingStatus.PENDING
    last_preprocessing_id: Optional[str] = None

@dataclass
class PreprocessingResponse:
    """预处理响应"""
    preprocessing_id: str  # 预处理任务ID
    dataset_name: str
    status: PreprocessingStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PreprocessingListResponse:
    """预处理列表响应"""
    datasets: List[DatasetInfo]
    total: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DatasetCheckResponse:
    """数据集检查响应"""
    exists: bool
    dataset_name: str
    directory: Optional[str] = None
    created_at: Optional[datetime] = None
    video_count: int = 0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
