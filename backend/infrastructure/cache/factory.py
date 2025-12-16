"""缓存相关工厂方法（Redis）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from redis.asyncio import Redis

from ...core.config import settings
from ...domain.interfaces import EmbeddingCache
from .redis_embedding_cache import RedisEmbeddingCache


@lru_cache()
def get_redis_client() -> Redis:
    """创建 Redis 客户端单例。"""
    return Redis.from_url(settings.redis.url, decode_responses=False)


@lru_cache()
def get_embedding_cache() -> Optional[EmbeddingCache]:
    """根据配置返回 embedding 缓存实现；若禁用则返回 None。"""
    if not settings.redis.enabled:
        return None
    return RedisEmbeddingCache(get_redis_client())

