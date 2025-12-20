"""知识图谱相关工厂方法（Neo4j + LLM 抽取器）。"""

from functools import lru_cache

from ...core.config import settings
from ..llm.factory import get_research_llm
from ..cache.factory import get_redis_client
from .llm_extractors import LLMGraphExtractor, LLMKeywordExtractor
from .neo4j_graph import Neo4jGraphRepository


@lru_cache()
def get_graph_repository() -> Neo4jGraphRepository:
    """获取 Neo4j 图谱仓库单例。"""
    return Neo4jGraphRepository(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
    )


@lru_cache()
def get_graph_extractor() -> LLMGraphExtractor:
    """获取 LLM 图谱抽取器单例。"""
    return LLMGraphExtractor(llm=get_research_llm())


@lru_cache()
def get_keyword_extractor() -> LLMKeywordExtractor:
    """获取 LLM 关键词抽取器单例。"""
    redis_client = get_redis_client() if settings.redis.enabled else None
    return LLMKeywordExtractor(
        llm=get_research_llm(),
        redis=redis_client,
        cache_ttl_seconds=settings.redis.embedding_ttl_seconds,
        model_name=settings.research_llm.model,
    )
