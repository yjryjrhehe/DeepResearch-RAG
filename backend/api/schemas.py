"""
API 层数据模型（Pydantic）。

说明：
- API 层与 domain 层可复用部分模型，但此处仍保留独立的请求结构，
  便于未来做字段裁剪/兼容版本演进。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..domain.models import AnswerResult, RetrievalMode


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="服务状态")


class UploadResponse(BaseModel):
    status: str = Field(default="ok", description="处理状态")
    document_id: str = Field(..., description="文档ID")
    document_name: str = Field(..., description="文档名")


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.FUSION,
        description=(
            "检索模式："
            "vector(仅向量) / kg_low(仅KG低层) / kg_high(仅KG高层) / kg_mix(KG低+高) / fusion(向量+KG融合)"
        ),
    )


class QueryResponse(AnswerResult):
    """问答接口响应（复用 domain.AnswerResult）。"""

    request_id: Optional[str] = Field(default=None, description="请求ID（可选）")
