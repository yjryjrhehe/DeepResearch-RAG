"""领域层抽象接口定义（DDD）。

本模块集中定义两类接口：
1) 基础设施抽象：由 `backend/infrastructure/` 提供实现（解析、切分、检索、图谱、缓存等）
2) 业务编排抽象：由 `backend/services/` 提供实现（摄取编排、RAG 编排等）

设计目标：
- API/Service 只依赖抽象，不直接依赖具体第三方库；
- 通过接口隔离，便于替换实现（例如更换向量库/图数据库/队列等）。
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import AsyncGenerator

from pydantic import JsonValue

from .models import (
    AnswerResult,
    DocumentChunk,
    DocumentSource,
    GraphEntity,
    GraphRelation,
    IngestionStats,
    RetrievedChunk,
    RetrievalMode,
)


class DocumentParser(ABC):
    """文档解析器抽象（例如 Docling + VLM/LLM 增强）。"""

    @abstractmethod
    async def parse(self, source: DocumentSource) -> str:
        """解析原始文档为 Markdown。

        Args:
            source: 文档来源信息（路径、文件名、元数据等）。

        Returns:
            Markdown 文本。
        """


class TextSplitter(ABC):
    """文本切分器抽象（将 Markdown 切分为可检索的 chunks）。"""

    @abstractmethod
    async def split(self, markdown_content: str, source: DocumentSource) -> list[DocumentChunk]:
        """将 Markdown 文本切分为文档块列表。

        Args:
            markdown_content: 解析得到的 Markdown 内容。
            source: 文档来源信息，用于补全 chunks 的元数据。

        Returns:
            切分后的文档块列表（此时通常尚未进行摘要/假设性问题等增强）。
        """


class PreProcessor(ABC):
    """文档块预处理抽象（摘要、假设性问题、清洗改写等）。"""

    @abstractmethod
    async def preprocess(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        """预处理单个文档块。

        Args:
            chunk: 原始文档块。

        Returns:
            预处理后产生的一个或多个文档块（允许拆分/增强）。
        """

    @abstractmethod
    async def run_concurrent_preprocessing(
        self, chunks: list[DocumentChunk]
    ) -> AsyncGenerator[DocumentChunk, None]:
        """并发预处理所有文档块，并以流式方式产出结果。

        Args:
            chunks: 待预处理的文档块列表。

        Yields:
            预处理完成的文档块。
        """


class SearchRepository(ABC):
    """检索仓储抽象（例如 OpenSearch）。"""

    @abstractmethod
    async def bulk_add_documents(self, chunks: list[DocumentChunk]) -> None:
        """批量写入（或更新）文档块到检索系统。"""

    @abstractmethod
    async def hybrid_search(self, query_text: str, k: int = 5, rrf_k: int = 60) -> list[RetrievedChunk]:
        """执行混合检索（BM25 + 向量 + RRF 等）。

        Args:
            query_text: 查询文本（可为子查询）。
            k: 召回数量。
            rrf_k: RRF 参数。

        Returns:
            带分数的检索结果列表。
        """

    @abstractmethod
    async def hybrid_search_batch(
        self,
        queries: list[str],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[list[RetrievedChunk]]:
        """批量执行混合检索。"""

    @abstractmethod
    async def get_query_embedding(self, text: str) -> list[float] | None:
        """获取查询向量（实现可选使用缓存）。"""

    @abstractmethod
    async def hybrid_search_batch_with_embeddings(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[list[RetrievedChunk]]:
        """批量混合检索（使用外部提供的 query embeddings）。"""

    @abstractmethod
    async def mget_documents(self, chunk_ids: list[str]) -> list[dict[str, JsonValue]]:
        """批量获取原始存储文档（保持输入顺序）。"""

    @abstractmethod
    async def index_graph_entities_relations(
        self,
        chunk: DocumentChunk,
        entities: list[GraphEntity],
        relations: list[GraphRelation],
    ) -> None:
        """将实体/关系写入检索系统的独立索引。"""

    @abstractmethod
    async def vector_search_entities(self, query_text: str, k: int = 10) -> list[dict[str, JsonValue]]:
        """在实体索引中执行向量检索。"""

    @abstractmethod
    async def vector_search_relations(self, query_text: str, k: int = 10) -> list[dict[str, JsonValue]]:
        """在关系索引中执行向量检索。"""


class EmbeddingCache(ABC):
    """Embedding 缓存抽象（例如 Redis）。"""

    @abstractmethod
    async def get(self, key: str) -> list[float] | None:
        """读取缓存 embedding。"""

    @abstractmethod
    async def set(self, key: str, embedding: list[float], ttl_seconds: int) -> None:
        """写入缓存 embedding。"""


class GraphExtractor(ABC):
    """从文档块中抽取实体与关系的抽象接口。"""

    @abstractmethod
    async def extract(self, chunk: DocumentChunk) -> tuple[list[GraphEntity], list[GraphRelation]]:
        """抽取实体与关系。"""


class KeywordExtractor(ABC):
    """从查询中抽取图谱检索关键词的抽象接口。"""

    @abstractmethod
    async def extract(self, query: str) -> tuple[list[str], list[str]]:
        """返回 (high_level_keywords, low_level_keywords)。"""


class GraphRepository(ABC):
    """知识图谱仓储抽象（例如 Neo4j）。"""

    @abstractmethod
    async def upsert_chunk_knowledge(
        self,
        chunk: DocumentChunk,
        entities: list[GraphEntity],
        relations: list[GraphRelation],
    ) -> None:
        """将单个 chunk 的抽取结果写入（或合并）到知识图谱。"""

    @abstractmethod
    async def query_context(
        self,
        entity_candidates: list[str],
        relation_candidates: list[tuple[str, str]],
        top_k_entities: int = 10,
        top_k_relations: int = 10,
        top_k_chunks: int = 20,
    ) -> dict[str, JsonValue]:
        """查询知识图谱上下文（供回答生成使用）。"""

    @abstractmethod
    async def get_entity_subgraph(self, entity_name: str, depth: int = 1) -> dict[str, JsonValue]:
        """获取实体邻域子图（用于 API 展示）。"""


class Reranker(ABC):
    """重排器抽象（例如 TEI reranker）。"""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 8) -> list[RetrievedChunk]:
        """对候选文档块进行重排。"""


class Ingestor(ABC):
    """文档摄取编排服务抽象。"""

    @abstractmethod
    async def pipeline(
        self,
        source: DocumentSource,
        status_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> IngestionStats:
        """执行完整的文档摄取流水线。

        典型步骤：
        1) 解析：DocumentParser
        2) 切分：TextSplitter
        3) 预处理：PreProcessor
        4) 入库：SearchRepository.bulk_add_documents
        5) 图谱：GraphExtractor + GraphRepository
        """


class RAGOrchestrator(ABC):
    """RAG 应用编排服务抽象。"""

    @abstractmethod
    async def ask(
        self,
        query: str,
        retrieval_mode: RetrievalMode = RetrievalMode.FUSION,
    ) -> AnswerResult:
        """对外提供统一的问答入口。"""
