from functools import lru_cache

from .opensearch_store import AsyncOpenSearchRAGStore
from ..cache.factory import get_embedding_cache
from ...core.config import settings

@lru_cache()
def get_opensearch_store() -> AsyncOpenSearchRAGStore:
    """
    [工厂方法] 获取 AsyncOpenSearchRAGStore 单例。
    """
    return AsyncOpenSearchRAGStore(
        embedding_cache=get_embedding_cache(),
        embedding_cache_ttl_seconds=settings.redis.embedding_ttl_seconds,
        embedding_model_name=settings.embedding_llm.model,
    )
