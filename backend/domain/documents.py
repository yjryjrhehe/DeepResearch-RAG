"""文档领域模型与仓储抽象。

本模块定义文档元信息、处理状态以及对外暴露的仓储接口（Repository）。基础设施层需要提供对应实现（例如基于 SQLAlchemy 的 SQLite 仓储）。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue


class DocumentStatus(StrEnum):
    """文档处理状态枚举。"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class DocumentCreate(BaseModel):
    """创建文档记录所需的最小信息。

    Attributes:
        document_id: 文档 ID（SHA256 十六进制字符串）。
        file_path: 文档在本地存储中的路径。
        original_filename: 用户上传的原始文件名。
        file_size_bytes: 文档大小（字节）。
        content_summary: 可选，文档摘要（短文本）。
        metadata: 可扩展的元数据。
    """

    document_id: str = Field(..., description="文档 ID（SHA256 十六进制字符串）")
    file_path: Path = Field(..., description="文档在本地存储中的路径")
    original_filename: str = Field(..., description="用户上传的原始文件名")
    file_size_bytes: int = Field(..., ge=0, description="文档大小（字节）")
    content_summary: str | None = Field(default=None, description="可选：文档摘要（短文本）")
    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class DocumentRecord(BaseModel):
    """文档记录（持久化模型的领域视图）。

    Attributes:
        document_id: 文档 ID（SHA256 十六进制字符串）。
        status: 文档处理状态。
        file_path: 文档在本地存储中的路径。
        original_filename: 用户上传的原始文件名。
        file_size_bytes: 文档大小（字节）。
        content_summary: 可选，文档摘要。
        created_at: 创建时间（UTC）。
        updated_at: 更新时间（UTC）。
        processing_started_at: 处理开始时间（UTC，可选）。
        processing_finished_at: 处理结束时间（UTC，可选）。
        chunks_count: 成功入库的 chunks 数量（可选）。
        error_message: 失败原因（可选）。
        attempt_count: 处理尝试次数（用于重试策略）。
        metadata: 可扩展的元数据。
    """

    document_id: str = Field(..., description="文档 ID（SHA256 十六进制字符串）")
    status: DocumentStatus = Field(..., description="文档处理状态")

    file_path: Path = Field(..., description="文档在本地存储中的路径")
    original_filename: str = Field(..., description="用户上传的原始文件名")
    file_size_bytes: int = Field(..., ge=0, description="文档大小（字节）")
    content_summary: str | None = Field(default=None, description="可选：文档摘要（短文本）")

    created_at: datetime = Field(..., description="创建时间（UTC）")
    updated_at: datetime = Field(..., description="更新时间（UTC）")
    processing_started_at: datetime | None = Field(default=None, description="处理开始时间（UTC，可选）")
    processing_finished_at: datetime | None = Field(default=None, description="处理结束时间（UTC，可选）")

    chunks_count: int | None = Field(default=None, ge=0, description="成功入库的 chunks 数量（可选）")
    error_message: str | None = Field(default=None, description="失败原因（可选）")
    attempt_count: int = Field(default=0, ge=0, description="处理尝试次数（用于重试策略）")

    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展的元数据")


class DocumentNotFoundError(LookupError):
    """根据文档 ID 未找到对应记录时抛出的异常。

    Attributes:
        document_id: 未找到的文档 ID。
    """

    def __init__(self, document_id: str) -> None:
        super().__init__(f"未找到文档记录 document_id={document_id}")
        self.document_id = document_id


class DocumentRepository(ABC):
    """文档仓储抽象接口。"""

    @abstractmethod
    async def get(self, document_id: str) -> DocumentRecord | None:
        """获取单个文档记录。

        Args:
            document_id: 文档 ID（SHA256 十六进制字符串）。

        Returns:
            文档记录；若不存在则返回 None。
        """

    @abstractmethod
    async def create_pending(self, document: DocumentCreate) -> DocumentRecord:
        """创建一条 PENDING 文档记录。

        Args:
            document: 待创建的文档信息。

        Returns:
            创建后的文档记录。

        Raises:
            ValueError: 当文档 ID 已存在且无法创建时抛出。
        """

    @abstractmethod
    async def list_documents(
        self,
        *,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """查询文档列表（支持按状态过滤）。

        Args:
            status: 可选；按状态过滤。
            limit: 返回数量上限。
            offset: 分页偏移量。

        Returns:
            文档记录列表。
        """

    @abstractmethod
    async def update_status(
        self,
        *,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        processing_started_at: datetime | None = None,
        processing_finished_at: datetime | None = None,
        chunks_count: int | None = None,
        increment_attempt: bool = False,
    ) -> DocumentRecord:
        """更新文档状态与处理信息。

        Args:
            document_id: 文档 ID。
            status: 新状态。
            error_message: 可选；失败原因。
            processing_started_at: 可选；处理开始时间。
            processing_finished_at: 可选；处理结束时间。
            chunks_count: 可选；成功入库 chunks 数量。
            increment_attempt: 是否将 attempt_count + 1（通常在进入 PROCESSING 时置 True）。

        Returns:
            更新后的文档记录。

        Raises:
            DocumentNotFoundError: 文档不存在时抛出。
        """

    @abstractmethod
    async def reset_failed_to_pending(self, *, document_id: str) -> DocumentRecord:
        """将 FAILED 文档重置为 PENDING 以便重试。

        Args:
            document_id: 文档 ID。

        Returns:
            更新后的文档记录。

        Raises:
            DocumentNotFoundError: 文档不存在时抛出。
        """

    @abstractmethod
    async def list_retryable_failed(self, *, limit: int = 100) -> list[DocumentRecord]:
        """查询可重试的失败任务。

        Args:
            limit: 返回数量上限。

        Returns:
            FAILED 状态的文档记录列表。
        """
