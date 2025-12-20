"""RAG 查询任务领域模型与仓储抽象。

本模块用于描述“耗时查询任务”的状态与结果引用（例如结果暂存到 Redis）。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, JsonValue


class QueryTaskStatus(StrEnum):
    """查询任务状态枚举。"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class QueryTaskCreate(BaseModel):
    """创建查询任务所需的最小信息。"""

    task_id: str = Field(..., description="查询任务 ID（通常为 UUID）")
    query: str = Field(..., min_length=1, description="用户查询文本")
    retrieval_mode: str = Field(..., description="检索模式（与 RetrievalMode 的值保持一致）")
    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class QueryTaskRecord(BaseModel):
    """查询任务记录（持久化模型的领域视图）。"""

    task_id: str = Field(..., description="查询任务 ID（通常为 UUID）")
    status: QueryTaskStatus = Field(..., description="任务状态")
    query: str = Field(..., description="用户查询文本")
    retrieval_mode: str = Field(..., description="检索模式")

    result_redis_key: str | None = Field(default=None, description="结果在 Redis 的键（可选）")
    result_preview: str | None = Field(default=None, description="结果预览（可选，便于列表展示）")
    error_message: str | None = Field(default=None, description="失败原因（可选）")

    created_at: datetime = Field(..., description="创建时间（UTC）")
    updated_at: datetime = Field(..., description="更新时间（UTC）")
    processing_started_at: datetime | None = Field(default=None, description="处理开始时间（UTC）")
    processing_finished_at: datetime | None = Field(default=None, description="处理结束时间（UTC）")

    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class QueryTaskNotFoundError(LookupError):
    """根据任务 ID 未找到对应记录时抛出的异常。"""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"未找到查询任务记录: task_id={task_id}")
        self.task_id = task_id


class QueryTaskRepository(ABC):
    """查询任务仓储抽象接口。"""

    @abstractmethod
    async def get(self, task_id: str) -> QueryTaskRecord | None:
        """获取单个查询任务记录。"""

    @abstractmethod
    async def create_pending(self, task: QueryTaskCreate) -> QueryTaskRecord:
        """创建一条 PENDING 查询任务记录。"""

    @abstractmethod
    async def list(
        self,
        *,
        status: QueryTaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QueryTaskRecord]:
        """查询查询任务列表（支持按状态过滤）。"""

    @abstractmethod
    async def update_status(
        self,
        *,
        task_id: str,
        status: QueryTaskStatus,
        error_message: str | None = None,
        processing_started_at: datetime | None = None,
        processing_finished_at: datetime | None = None,
        result_redis_key: str | None = None,
        result_preview: str | None = None,
    ) -> QueryTaskRecord:
        """更新任务状态与结果引用。"""

