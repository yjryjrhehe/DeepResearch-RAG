from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Callable, Awaitable

from .models import (
    DocumentSource,
    DocumentChunk,
    RetrievedChunk,
    GraphEntity,
    GraphRelation,
    AnswerResult,
    RetrievalMode,
)

# --------------------------------------------------------------------
# 1. 基础设施接口 (由 infrastructure/ 实现)
# --------------------------------------------------------------------


class DocumentParser(ABC):
    """
    对应"docling+VLM"。
    职责：将原始文件（PDF, DOCX等）解析为Markdown。
    """

    @abstractmethod
    async def parse(self, source: DocumentSource) -> str:
        """
        解析原始文档。
        :param source: DocumentSource对象，包含文件路径等信息。
        :return: Markdown 格式的字符串。
        """
        pass


class TextSplitter(ABC):
    """
    对应 "基于markdown结构分块" 和 "基于语义分块"。
    职责：将Markdown文本分割成 DocumentChunk。
    """

    @abstractmethod
    async def split(
        self, markdown_content: str, source: DocumentSource
    ) -> List[DocumentChunk]:
        """
        将Markdown文本分割成块。
        :param markdown_content: 从 IDocumentParser 获得的Markdown内容。
        :param source: 原始文档信息，用于填充块的元数据。
        :return: DocumentChunk 列表 (此时还没有摘要和假设性问题)。
        """
        pass


class PreProcessor(ABC):
    """
    对切分后的文本块进行预处理，如生成摘要、假设性问题等。
    """

    @abstractmethod
    async def preprocess(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """
        将原始文本块转换为多个新的Document对象，便于检索。
        """
        pass
    
    @abstractmethod
    async def run_concurrent_preprocessing(
        self, chunks: List[DocumentChunk]
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        在生产环境中并发处理所有文本块的主函数。
        :param chunks: 所有待处理的块列表。
        :return: 以流式方式产出的处理后块。
        """
        pass


class SearchRepository(ABC):
    """
    对应 OpenSearch 的存储和检索抽象。
    由 infrastructure/repository/opensearch_store.py 实现。
    """

    @abstractmethod
    async def bulk_add_documents(self, chunks: List[DocumentChunk]):
        """
        批量添加（或更新）文档块到 OpenSearch。
        此方法内部应处理向量生成。
        """
        pass

    @abstractmethod
    async def hybrid_search(self, query_text: str, k: int = 5, rrf_k: int = 60):
        """
        执行混合检索。
        对应流程图2的 "检索、去重 (向量相似度+BM25)"。
        :param query: 单个子查询 (sub_query)。
        :return: 检索到的原始文档块列表（带search_score）。
        """
        pass

    @abstractmethod
    async def hybrid_search_batch(
        self, 
        queries: List[str], 
        k: int = 5, 
        rrf_k: int = 60
    ) -> List[List[RetrievedChunk]]:
        """
        批量执行混合检索。
        """
        pass

    @abstractmethod
    async def get_query_embedding(self, text: str) -> Optional[List[float]]:
        """获取 query embedding（实现可选择使用缓存）。"""
        pass

    @abstractmethod
    async def hybrid_search_batch_with_embeddings(
        self,
        queries: List[str],
        embeddings: List[List[float]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[List[RetrievedChunk]]:
        """
        批量执行混合检索（使用外部提供的 query embeddings）。

        说明：
        - `queries[i]` 对应 `embeddings[i]`；
        - 该方法不会在内部再次生成 embedding（除非实现方自行兜底）。
        """
        pass

    @abstractmethod
    async def mget_documents(self, chunk_ids: List[str]) -> List[dict]:
        """批量获取原始存储文档（保持输入顺序）。"""
        pass

    @abstractmethod
    async def index_graph_entities_relations(
        self,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
    ) -> None:
        """将从单个 chunk 抽取的实体/关系分别写入 OpenSearch 的独立索引。"""
        pass

    @abstractmethod
    async def vector_search_entities(self, query_text: str, k: int = 10) -> List[dict]:
        """在实体索引中执行向量检索，返回包含 entity_name 等字段的结果列表。"""
        pass

    @abstractmethod
    async def vector_search_relations(self, query_text: str, k: int = 10) -> List[dict]:
        """在关系索引中执行向量检索，返回包含 source_entity/target_entity 等字段的结果列表。"""
        pass


class EmbeddingCache(ABC):
    """Embedding 缓存抽象（Redis 等实现）。"""

    @abstractmethod
    async def get(self, key: str) -> Optional[List[float]]:
        """读取缓存 embedding。"""
        pass

    @abstractmethod
    async def set(self, key: str, embedding: List[float], ttl_seconds: int) -> None:
        """写入缓存 embedding。"""
        pass


class GraphExtractor(ABC):
    """从文本块中抽取实体与关系的抽象接口。"""

    @abstractmethod
    async def extract(self, chunk: DocumentChunk) -> tuple[List[GraphEntity], List[GraphRelation]]:
        """抽取实体与关系。"""
        pass


class KeywordExtractor(ABC):
    """从查询中抽取图谱检索关键词。"""

    @abstractmethod
    async def extract(self, query: str) -> tuple[List[str], List[str]]:
        """
        返回 (high_level_keywords, low_level_keywords)。
        """
        pass


class GraphRepository(ABC):
    """知识图谱存储抽象（Neo4j 等实现）。"""

    @abstractmethod
    async def upsert_chunk_knowledge(
        self,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
    ) -> None:
        """将单个 chunk 的抽取结果写入/合并到图谱。"""
        pass

    @abstractmethod
    async def query_context(
        self,
        entity_candidates: List[str],
        relation_candidates: List[tuple[str, str]],
        top_k_entities: int = 10,
        top_k_relations: int = 10,
        top_k_chunks: int = 20,
    ) -> dict:
        """
        根据候选实体/关系返回用于回答的图谱上下文。

        建议结构：
        {
          "entities": [{"name":..., "type":..., "description":...}],
          "relations": [{"source":..., "target":..., "keywords":..., "description":..., "weight":...}],
          "chunk_ids": ["..."],
          "entity_to_chunk_ids": {"entity_name": ["chunk_id", ...]}
        }
        """
        pass

    @abstractmethod
    async def get_entity_subgraph(self, entity_name: str, depth: int = 1) -> dict:
        """获取实体邻域子图（用于 API 展示）。"""
        pass


class Reranker(ABC):
    """重排序抽象（TEI 等实现）。"""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_n: int = 8
    ) -> List[RetrievedChunk]:
        """对候选块重排序。"""
        pass


# --------------------------------------------------------------------
# 2. 服务接口 (由 services/ 实现)
# --------------------------------------------------------------------


class Ingestor(ABC):
    """
    文档摄入服务 (业务流程编排)。
    services/ingestion_service.py
    """

    @abstractmethod
    async def pipeline(
        self,
        source: DocumentSource,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """
        编排完整的文档摄入流程（流程图1）。
        1. 调用 DocumentParser
        2. 调用 TextSplitter
        3. 调用 Preprocessor
        4. 调用 SearchRepository.bulk_add_documents
        5. 调用 GraphExtractor + GraphRepository 构建知识图谱
        """
        pass


class RAGOrchestrator(ABC):
    """RAG 应用编排服务：图谱+向量检索、重排、回答生成一体化。"""

    @abstractmethod
    async def ask(self, query: str, retrieval_mode: RetrievalMode = RetrievalMode.FUSION) -> AnswerResult:
        """对外提供的统一问答入口，支持选择检索模式。"""
        pass
