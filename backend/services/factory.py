import logging
from functools import lru_cache

from ..domain.interfaces import Ingestor, RAGOrchestrator
from .ingestion_service import IngestionService
from .rag_service import RagService

# 2. 导入其他基础设施的工厂函数
from ..infrastructure.parse.factory import (
    get_docling_parser,
    get_llm_preprocessor,
    get_markdown_splitter
)
from ..infrastructure.repository.factory import get_opensearch_store
from ..infrastructure.llm.factory import get_rewrite_llm, get_answer_llm, get_reranker
from ..infrastructure.graph.factory import (
    get_graph_extractor,
    get_graph_repository,
    get_keyword_extractor,
)
from ..core.config import settings

log = logging.getLogger(__name__)

@lru_cache()
def get_ingestion_service() -> Ingestor:
    """
    [工厂方法] 组装并获取 IngestionService 单例。
    
    职责：
    1. 调用底层组件的工厂方法获取实例。
    2. 将这些实例注入到 IngestionService 中。
    3. 返回组装好的 Service。
    """
    log.info("正在组装 IngestionService...")
    
    try:
        # 获取依赖实例
        parser_instance = get_docling_parser()
        splitter_instance = get_markdown_splitter()
        preprocessor_instance = get_llm_preprocessor()
        store_instance = get_opensearch_store()
        
        # 注入依赖并实例化
        return IngestionService(
            parser=parser_instance,
            splitter=splitter_instance,
            preprocessor=preprocessor_instance,
            store=store_instance,
            graph_extractor=get_graph_extractor(),
            graph_repo=get_graph_repository(),
            kg_max_concurrency=settings.kg.extract_max_concurrency,
        )
        
    except Exception as e:
        log.error(f"IngestionService 工厂初始化失败: {e}", exc_info=True)
        raise e
    

@lru_cache()
def get_rag_service() -> RAGOrchestrator:
    """[工厂方法] 组装并获取 RAG 编排服务单例。"""
    return RagService(
        search_repo=get_opensearch_store(),
        graph_repo=get_graph_repository(),
        keyword_extractor=get_keyword_extractor(),
        rewrite_llm=get_rewrite_llm(),
        answer_llm=get_answer_llm(),
        reranker=get_reranker(),
    )
