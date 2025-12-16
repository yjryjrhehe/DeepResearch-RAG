"""
Redis Embedding 缓存实现。

目标：
- 缓存 query 的 embedding 结果，减少重复向量化开销；
- 缓存 key 与模型版本绑定，避免模型切换导致脏数据；
- 提供简单、可复用的异步接口（domain.EmbeddingCache）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, List

from redis.asyncio import Redis

from ...domain.interfaces import EmbeddingCache


def build_embedding_cache_key(*, model: str, text: str) -> str:
    """
    构造 embedding 缓存 key。

    说明：
    - 使用 sha256 避免 key 过长；
    - 以 model 作为命名空间，避免不同模型互相污染。
    """
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"emb:{model}:{digest}"


class RedisEmbeddingCache(EmbeddingCache):
    """Redis 实现的 EmbeddingCache。"""

    def __init__(self, redis: Redis):
        self._redis = redis

    async def get(self, key: str) -> Optional[List[float]]:
        value = await self._redis.get(key)
        if not value:
            return None
        try:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            data = json.loads(value)
            if isinstance(data, list):
                return [float(x) for x in data]
        except Exception:
            return None
        return None

    async def set(self, key: str, embedding: List[float], ttl_seconds: int) -> None:
        payload = json.dumps(embedding, ensure_ascii=False, separators=(",", ":"))
        await self._redis.set(name=key, value=payload, ex=ttl_seconds)

