"""API 层请求/响应模型（Pydantic）。

说明：
- API 层模型以“对外契约”为准，允许与 domain 层模型重叠但保持可控。
- 本模块仅包含轻量的 schema，不做业务逻辑。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from ..domain.documents import DocumentRecord, DocumentStatus
from ..domain.models import AnswerResult, RetrievalMode
from ..domain.query_tasks import QueryTaskRecord, QueryTaskStatus


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(default="ok", description="服务状态")


class UploadResponse(BaseModel):
    """上传接口响应。

    说明：
    - 当为新上传时，API 返回 `202 Accepted`，并携带 `task_id`。
    - 当命中秒传（已处理完成）时，API 返回 `200 OK`，`task_id` 为空。
    """

    document: DocumentRecord = Field(..., description="文档记录")
    task_id: str | None = Field(default=None, description="后台任务 ID（可选）")
    duplicated: bool = Field(default=False, description="是否命中去重/秒传")


class DocumentListResponse(BaseModel):
    """文档列表响应。"""

    items: list[DocumentRecord] = Field(default_factory=list, description="文档列表")


class RetryResponse(BaseModel):
    """重试/重处理接口响应。"""

    retried_count: int = Field(..., ge=0, description="触发重试的文档数量")
    task_ids: list[str] = Field(default_factory=list, description="投递的后台任务 ID 列表")


class QueryRequest(BaseModel):
    """同步问答请求。"""

    query: str = Field(..., description="用户问题")
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.FUSION,
        description="检索模式：vector/kg_low/kg_high/kg_mix/fusion",
    )


class QueryResponse(AnswerResult):
    """同步问答响应。"""

    request_id: str | None = Field(default=None, description="请求 ID（可选）")


class CreateQueryTaskRequest(BaseModel):
    """创建异步查询任务请求。"""

    query: str = Field(..., min_length=1, description="用户问题")
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.FUSION,
        description="检索模式：vector/kg_low/kg_high/kg_mix/fusion",
    )


class CreateQueryTaskResponse(BaseModel):
    """创建异步查询任务响应。"""

    task: QueryTaskRecord = Field(..., description="查询任务记录")
    enqueued_task_id: str | None = Field(default=None, description="后台任务 ID（可选）")


class QueryTaskListResponse(BaseModel):
    """查询任务列表响应。"""

    items: list[QueryTaskRecord] = Field(default_factory=list, description="查询任务列表")


class QueryTaskResultResponse(BaseModel):
    """查询任务结果响应（结果来自 Redis）。"""

    task_id: str = Field(..., description="查询任务 ID")
    status: QueryTaskStatus = Field(..., description="任务状态")
    result: dict[str, JsonValue] | None = Field(default=None, description="查询结果（JSON 对象，可为空）")


class DocumentStatusCountsResponse(BaseModel):
    """文档状态统计响应。"""

    pending: int = Field(..., ge=0, description="PENDING 数量")
    processing: int = Field(..., ge=0, description="PROCESSING 数量")
    processed: int = Field(..., ge=0, description="PROCESSED 数量")
    failed: int = Field(..., ge=0, description="FAILED 数量")


class DocumentStatusFilter(BaseModel):
    """用于 OpenAPI 文档的状态枚举说明。"""

    status: DocumentStatus = Field(..., description="文档状态")
