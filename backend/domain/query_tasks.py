"""RAG 查询任务领域模型与仓储抽象。

本模块描述耗时查询任务的状态、持久化视图以及仓储接口，便于异步或同步问答统一持久化与检索。
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
    """查询任务创建模型。

    Attributes:
        task_id: 查询任务唯一 ID（通常为 UUID）。
        query: 用户查询文本。
        retrieval_mode: 检索模式（需与 RetrievalMode 的取值保持一致）。
        metadata: 可扩展的元数据字典。
    """

    task_id: str = Field(..., description="查询任务 ID（通常 UUID）")
    query: str = Field(..., min_length=1, description="用户查询文本")
    retrieval_mode: str = Field(..., description="检索模式（需与 RetrievalMode 的取值保持一致）")
    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class QueryTaskRecord(BaseModel):
    """查询任务持久化记录的领域视图。

    Attributes:
        task_id: 查询任务 ID。
        status: 任务状态。
        query: 用户查询文本。
        retrieval_mode: 检索模式。
        result_redis_key: 结果在 Redis 的键（可选）。
        result_preview: 结果预览文本（可选，便于列表展示）。
        error_message: 失败原因（可选）。
        created_at: 创建时间（UTC）。
        updated_at: 更新时间（UTC）。
        processing_started_at: 处理开始时间（UTC，可选）。
        processing_finished_at: 处理结束时间（UTC，可选）。
        metadata: 可扩展元数据。
    """

    task_id: str = Field(..., description="查询任务 ID（通常 UUID）")
    status: QueryTaskStatus = Field(..., description="任务状态")
    query: str = Field(..., description="用户查询文本")
    retrieval_mode: str = Field(..., description="检索模式")

    result_redis_key: str | None = Field(default=None, description="结果在 Redis 的键（可选）")
    result_preview: str | None = Field(default=None, description="结果预览文本（可选，便于列表展示）")
    error_message: str | None = Field(default=None, description="失败原因（可选）")

    created_at: datetime = Field(..., description="创建时间（UTC）")
    updated_at: datetime = Field(..., description="更新时间（UTC）")
    processing_started_at: datetime | None = Field(default=None, description="处理开始时间（UTC，可选）")
    processing_finished_at: datetime | None = Field(default=None, description="处理结束时间（UTC，可选）")

    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class QueryTaskNotFoundError(LookupError):
    """未找到查询任务记录时抛出的异常。

    Attributes:
        task_id: 未找到的任务 ID。
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(f"未找到查询任务记录 task_id={task_id}")
        self.task_id = task_id


class QueryTaskRepository(ABC):
    """查询任务仓储抽象接口。"""

    @abstractmethod
    async def get(self, task_id: str) -> QueryTaskRecord | None:
        """获取单个查询任务记录。

        Args:
            task_id: 查询任务 ID。

        Returns:
            找到则返回记录，否则返回 None。
        """

    @abstractmethod
    async def create_pending(self, task: QueryTaskCreate) -> QueryTaskRecord:
        """创建一条 PENDING 查询任务记录。

        Args:
            task: 查询任务创建信息。

        Returns:
            创建后的任务记录。
        """

    @abstractmethod
    async def list(
        self,
        *,
        status: QueryTaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QueryTaskRecord]:
        """查询任务列表，可按状态过滤。

        Args:
            status: 指定状态则按状态过滤。
            limit: 返回数量上限。
            offset: 偏移量，用于分页。

        Returns:
            查询任务记录列表。
        """

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
        """更新任务状态与结果引用。

        Args:
            task_id: 查询任务 ID。
            status: 更新后的状态。
            error_message: 失败原因（可选）。
            processing_started_at: 处理开始时间（可选）。
            processing_finished_at: 处理结束时间（可选）。
            result_redis_key: 结果在 Redis 的键（可选）。
            result_preview: 结果预览（可选）。

        Returns:
            更新后的任务记录。
        """
