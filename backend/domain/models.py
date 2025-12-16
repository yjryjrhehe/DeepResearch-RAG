import uuid
from enum import StrEnum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------
# 1. 文档摄入模型：文档解析与入库
# --------------------------------------------------------------------

class DocumentSource(BaseModel):
    """
    描述一个待处理的原始文档来源。

    说明：
    - `document_id` 用于将同一份文档的所有块串联起来；
    - `document_name` 默认从文件名推断，便于引用展示；
    - 其他额外信息放入 `metadata`，避免频繁改动模型结构。
    """
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="文档的唯一ID")
    file_path: Path = Field(..., description="文档在存储中的本地路径")
    
    # 1. 保持类型为 str，但设置 default=None
    #    这允许它在输入时缺失，并由下面的 'before' 验证器填充
    document_name: str = Field(
        default=None, 
        description="文档原始名称 (如果为 None，将自动从 file_path 提取)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元数据")

    @model_validator(mode='before')
    @classmethod
    def set_document_name_from_path(cls, data: Any) -> Any:
        """
        在字段验证之前，如果 document_name 未提供，则从 file_path 提取。
        """
        # 确保我们正在处理一个字典（即，不是从已实例化的对象创建）
        if isinstance(data, dict):
            
            # 2. 检查 document_name 是否未提供或明确为 None
            if data.get('document_name') is None:
                
                # 3. 检查 file_path 是否存在
                file_path_val = data.get('file_path')
                
                if file_path_val:
                    # 4. 从 file_path (可能是 str 或 Path) 提取 .name
                    #    因为这是 'before' 验证器, file_path_val 尚未被 Pydantic 转换为 Path 对象
                    if isinstance(file_path_val, str):
                        data['document_name'] = Path(file_path_val).name
                    elif isinstance(file_path_val, Path):
                        data['document_name'] = file_path_val.name
                    # (如果 file_path_val 是其他类型，让 Pydantic 在后续步骤中正常失败)
        
        return data


class DocumentChunk(BaseModel):
    """
    核心数据单元：文档块。

    说明：
    - 该对象同时服务于：向量库入库、知识图谱构建、问答引用；
    - 向量字段不直接存放在模型中，避免在 API/服务层传播大向量。
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="文本块的唯一ID")
    document_id: str = Field(..., description="所属文档的唯一ID")
    document_name: str = Field(..., description="文档原始名称")
    
    content: str = Field(..., description="文本块的原始内容")
    
    parent_headings: List[str] = Field(default_factory=list, description="所有父标题 (用于检索)")
    summary: Optional[str] = Field(None, description="AI生成的文本块摘要")
    hypothetical_questions: List[str] = Field(default_factory=list, description="AI生成的假设性问题 (用于增强检索)")
    
    # 注意：向量 (Vector) 本身通常不在Pydantic模型中传输，
    # 而是由 ISearchRepository 的实现在存入OpenSearch时生成和管理的。
    # 这里我们只定义业务数据。
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")


# --------------------------------------------------------------------
# 2. 检索模型 (对应流程图2：文档检索)
# --------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """
    从检索系统返回的带分数的文档块。
    """
    chunk: DocumentChunk = Field(..., description="检索到的原始文档块")
    
    # 对应流程图2中“检索”步骤的分数 (向量相似度+BM25)
    search_score: float = Field(..., description="混合检索的原始分数 (例如 BM25 + 向量相似度)")
    
    # 对应流程图2中 "rerank" 步骤的分数
    rerank_score: Optional[float] = Field(None, description="经过Reranker（如Cross-Encoder）重排后的分数")

# --------------------------------------------------------------------
# 3. 知识图谱与问答模型
# --------------------------------------------------------------------

class GraphEntity(BaseModel):
    """知识图谱实体。"""

    name: str = Field(..., description="实体名（唯一键）")
    type: str = Field(default="Other", description="实体类型")
    description: str = Field(default="", description="实体描述（来自文本抽取）")


class GraphRelation(BaseModel):
    """知识图谱关系。"""

    source: str = Field(..., description="源实体名")
    target: str = Field(..., description="目标实体名")
    keywords: List[str] = Field(default_factory=list, description="关系关键词")
    description: str = Field(default="", description="关系描述")
    weight: float = Field(default=1.0, description="关系权重（用于排序/筛选）")


class RetrievalMode(StrEnum):
    """
    RAG 检索模式。

    说明：
    - 该枚举同时用于 services 层编排与 API 入参校验；
    - 采用字符串值，便于前后端传参与日志记录。
    """

    VECTOR = "vector"
    KG_LOW = "kg_low"
    KG_HIGH = "kg_high"
    KG_MIX = "kg_mix"
    FUSION = "fusion"


class AnswerReference(BaseModel):
    """回答引用的参考文档条目。"""

    reference_id: str = Field(..., description="引用编号，从 1 开始")
    document_title: str = Field(..., description="文档标题（通常为 document_name）")


class AnswerResult(BaseModel):
    """
    问答结果（包含回答与引用）。

    说明：
    - `answer` 为 Markdown 文本，末尾包含 `### References` 段落；
    - `references` 是结构化参考列表，便于前端展示与点击跳转；
    - `chunks` 为实际用于回答的 Top-N 文本块（已重排）。
    - `graph_context` 为图谱检索返回的实体/关系与关联 chunk_id，便于前端渲染。
    """

    answer: str = Field(..., description="模型生成的回答（Markdown）")
    references: List[AnswerReference] = Field(default_factory=list, description="参考引用列表")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="用于回答的检索块")
    graph_context: Dict[str, Any] = Field(default_factory=dict, description="知识图谱上下文（实体、关系、chunk_ids 等）")
