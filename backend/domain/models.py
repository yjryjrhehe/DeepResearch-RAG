"""核心领域数据模型（Pydantic）。

该模块承载 RAG/摄取流程中会跨层传递的数据结构，例如：
- `DocumentSource`：原始文件来源（路径、文件名、元数据等）
- `DocumentChunk`：切分后的文档块（供检索入库/知识图谱抽取/引用展示）
- `AnswerResult`：问答结果（回答、引用、证据块与图谱上下文）

类型约束：
- 为保证接口稳定并避免 `Any` 漫延，扩展字段统一使用 `pydantic.JsonValue`。
"""

import uuid
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, model_validator


class DocumentSource(BaseModel):
    """待处理的原始文档来源。

    Attributes:
        document_id: 文档唯一 ID，用于把同一份文档的所有 chunks 串联起来。
        file_path: 文档在本地存储中的路径。
        document_name: 文档原始名称；未提供时自动从 `file_path` 推导。
        metadata: 其他可扩展元数据（JSON 兼容）。
    """

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="文档唯一 ID")
    file_path: Path = Field(..., description="文档在存储中的本地路径")
    document_name: str | None = Field(default=None, description="文档原始名称（可选）")
    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展元数据（JSON 兼容）")

    @model_validator(mode="after")
    def _fill_document_name(self) -> "DocumentSource":
        """自动补全 `document_name`。

        Returns:
            当前对象本身（Pydantic after-validator 约定返回 model）。
        """
        if self.document_name is None:
            self.document_name = self.file_path.name
        return self


class DocumentChunk(BaseModel):
    """核心数据单元：文档块（Chunk）。

    说明：
    - 该对象会被用于向量检索入库、知识图谱构建以及问答引用展示。
    - 向量本身不在该模型中传递，由检索仓储实现负责生成与管理。
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="文档块唯一 ID")
    document_id: str = Field(..., description="所属文档 ID")
    document_name: str = Field(..., description="文档原始名称")
    content: str = Field(..., description="文档块内容")

    parent_headings: list[str] = Field(default_factory=list, description="父级标题路径（用于检索）")
    summary: str | None = Field(default=None, description="AI 生成摘要（可选）")
    hypothetical_questions: list[str] = Field(default_factory=list, description="AI 生成的假设性问题（可选）")

    metadata: dict[str, JsonValue] = Field(default_factory=dict, description="可扩展元数据（JSON 兼容）")


class RetrievedChunk(BaseModel):
    """检索系统返回的带分数文档块。"""

    chunk: DocumentChunk = Field(..., description="检索到的文档块")
    search_score: float = Field(..., description="检索阶段分数（例如 BM25 + 向量相似度融合）")
    rerank_score: float | None = Field(default=None, description="重排阶段分数（可选）")


class GraphEntity(BaseModel):
    """知识图谱实体。"""

    name: str = Field(..., description="实体名称（唯一键）")
    type: str = Field(default="Other", description="实体类型")
    description: str = Field(default="", description="实体描述（来自文本抽取）")


class GraphRelation(BaseModel):
    """知识图谱关系。"""

    source: str = Field(..., description="源实体名称")
    target: str = Field(..., description="目标实体名称")
    keywords: list[str] = Field(default_factory=list, description="关系关键词")
    description: str = Field(default="", description="关系描述")
    weight: float = Field(default=1.0, description="关系权重（用于筛选/排序）")


class RetrievalMode(StrEnum):
    """RAG 检索模式。"""

    VECTOR = "vector"
    KG_LOW = "kg_low"
    KG_HIGH = "kg_high"
    KG_MIX = "kg_mix"
    FUSION = "fusion"


class AnswerReference(BaseModel):
    """回答引用的参考文档条目。"""

    reference_id: str = Field(..., description="引用编号（从 1 开始）")
    document_title: str = Field(..., description="文档标题（通常为 document_name）")


class AnswerResult(BaseModel):
    """问答结果（含引用与证据块）。"""

    answer: str = Field(..., description="模型生成的回答（Markdown）")
    references: list[AnswerReference] = Field(default_factory=list, description="参考引用列表")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="用于回答的检索证据块")
    graph_context: dict[str, JsonValue] = Field(default_factory=dict, description="知识图谱检索上下文（JSON 兼容）")


class IngestionStats(BaseModel):
    """文档摄取统计信息。"""

    initial_chunks_count: int = Field(..., ge=0, description="切分产生的初始 chunks 数量")
    stored_chunks_count: int = Field(..., ge=0, description="成功写入检索系统的 chunks 数量")
    kg_tasks_count: int = Field(..., ge=0, description="知识图谱增量更新的 chunk 数量（可为 0）")
