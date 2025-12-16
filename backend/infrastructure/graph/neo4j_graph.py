"""
Neo4j 知识图谱存储实现。

图谱设计（简化版）：
- (:Entity {name, type, description})
- (:Chunk {chunk_id, document_id, document_name})
- (:Chunk)-[:MENTIONS]->(:Entity)
- (:Entity)-[:RELATED {keywords, description, weight}]->(:Entity)

说明：
1) 以 `Entity.name` 作为实体唯一键；以 `Chunk.chunk_id` 作为文本块唯一键；
2) 关系按无向关系处理：写入时对 (source, target) 做字典序规范化，避免重复边；
3) 动态更新：同一实体/关系出现时，做 MERGE 并叠加/合并属性（weight、keywords 等）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Dict, Any, Awaitable, Callable, TypeVar

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from ...domain.interfaces import GraphRepository
from ...domain.models import DocumentChunk, GraphEntity, GraphRelation

log = logging.getLogger(__name__)

_T = TypeVar("_T")


class Neo4jGraphRepository(GraphRepository):
    """Neo4j 图谱仓库。"""

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: Optional[str] = None,
    ):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        self._database = database or None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _run_with_retry(self, op_name: str, func: Callable[[], Awaitable[_T]]) -> _T:
        """
        对 Neo4j 的连接/路由抖动做有限重试。

        背景：在容器刚启动（Bolt 尚未就绪）或网络瞬态抖动时，Neo4j 驱动可能报
        `ServiceUnavailable: Unable to retrieve routing information` 或 `SessionExpired`。
        这里做短时间重试，避免 API 启动阶段直接失败。
        """
        max_attempts = 10
        delay = 0.5
        max_delay = 5.0

        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except (ServiceUnavailable, SessionExpired) as exc:
                if attempt >= max_attempts:
                    log.error("Neo4j %s 失败，已达最大重试次数: %s", op_name, exc, exc_info=True)
                    raise
                log.warning(
                    "Neo4j %s 失败（第 %s/%s 次），%.1fs 后重试: %s",
                    op_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, max_delay)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            async def _init_once() -> None:
                async with self._driver.session(database=self._database) as session:
                    # 约束与索引：尽量使用 IF NOT EXISTS，避免重复创建报错
                    result = await session.run(
                        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                        "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                    )
                    await result.consume()
                    result = await session.run(
                        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
                    )
                    await result.consume()
                    result = await session.run(
                        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                        "FOR (e:Entity) ON EACH [e.name, e.description]"
                    )
                    await result.consume()

            await self._run_with_retry("初始化约束/索引", _init_once)
            self._initialized = True

    async def verify_connection(self) -> None:
        """
        异步校验 Neo4j 连接与权限，并触发必要的约束/索引初始化。
        """
        await self._ensure_initialized()

        async def _ping() -> None:
            async with self._driver.session(database=self._database) as session:
                result = await session.run("RETURN 1 AS ok")
                await result.consume()

        await self._run_with_retry("连接校验", _ping)

    async def upsert_chunk_knowledge(
        self,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
    ) -> None:
        await self._ensure_initialized()

        entities_payload = [e.model_dump() for e in entities]
        relations_payload = [r.model_dump() for r in relations]

        cypher = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.document_id = $document_id,
            c.document_name = $document_name

        WITH c
        UNWIND $entities AS ent
            MERGE (e:Entity {name: ent.name})
            SET e.type = coalesce(ent.type, e.type, 'Other'),
                e.description = CASE
                    WHEN coalesce(ent.description, '') = '' THEN coalesce(e.description, '')
                    WHEN coalesce(e.description, '') = '' THEN ent.description
                    ELSE e.description
                END
            MERGE (c)-[:MENTIONS]->(e)

        WITH c
        UNWIND $relations AS rel
            WITH c, rel,
                 CASE WHEN rel.source < rel.target THEN rel.source ELSE rel.target END AS src,
                 CASE WHEN rel.source < rel.target THEN rel.target ELSE rel.source END AS tgt
            MATCH (s:Entity {name: src})
            MATCH (t:Entity {name: tgt})
            MERGE (s)-[r:RELATED]->(t)
            SET r.weight = coalesce(r.weight, 0.0) + coalesce(rel.weight, 1.0),
                r.description = CASE
                    WHEN coalesce(rel.description, '') = '' THEN coalesce(r.description, '')
                    WHEN coalesce(r.description, '') = '' THEN rel.description
                    ELSE r.description
                END,
                r.keywords = apoc.coll.toSet(coalesce(r.keywords, []) + coalesce(rel.keywords, []))
        """

        # 说明：keywords 合并依赖 APOC；若未安装 APOC，则回退为简单覆盖。
        # 为了让系统在无 APOC 的 Neo4j 上仍可运行，这里做一次降级尝试。
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    cypher,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    entities=entities_payload,
                    relations=relations_payload,
                )
                await result.consume()
        except Exception:
            cypher_no_apoc = cypher.replace(
                "r.keywords = apoc.coll.toSet(coalesce(r.keywords, []) + coalesce(rel.keywords, []))",
                "r.keywords = coalesce(rel.keywords, r.keywords, [])",
            )
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    cypher_no_apoc,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    entities=entities_payload,
                    relations=relations_payload,
                )
                await result.consume()

    async def query_context(
        self,
        query: str,
        top_k_entities: int = 10,
        top_k_chunks: int = 20,
    ) -> Dict[str, Any]:
        """
        返回图谱上下文：实体、关系、以及关联 chunk_id。
        """
        await self._ensure_initialized()

        # 1) 取 Top 实体
        cypher_entities = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $q) YIELD node, score
        RETURN node { .name, .type, .description } AS entity, score
        ORDER BY score DESC
        LIMIT $top_k_entities
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher_entities, q=query, top_k_entities=top_k_entities
            )
            rows = await result.data()

        entities = [r["entity"] for r in rows if r.get("entity") and r["entity"].get("name")]
        entity_names = [e["name"] for e in entities]
        if not entity_names:
            return {"entities": [], "relations": [], "chunk_ids": []}

        # 2) 聚合 chunk_id（按出现频次排序）
        cypher_chunks = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $names
        RETURN c.chunk_id AS chunk_id, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT $top_k_chunks
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher_chunks, names=entity_names, top_k_chunks=top_k_chunks
            )
            chunk_rows = await result.data()
        chunk_ids = [r["chunk_id"] for r in chunk_rows if r.get("chunk_id")]

        # 3) 抽取实体间关系（仅保留 entity_names 范围内）
        cypher_relations = """
        MATCH (a:Entity)-[r:RELATED]-(b:Entity)
        WHERE a.name IN $names AND b.name IN $names
        RETURN a.name AS source, b.name AS target,
               r.keywords AS keywords, r.description AS description, r.weight AS weight
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher_relations, names=entity_names)
            rel_rows = await result.data()

        seen = set()
        relations: List[Dict[str, Any]] = []
        for r in rel_rows:
            src = r.get("source")
            tgt = r.get("target")
            if not src or not tgt or src == tgt:
                continue
            key = tuple(sorted([src, tgt]))
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "source": src,
                    "target": tgt,
                    "keywords": r.get("keywords") or [],
                    "description": r.get("description") or "",
                    "weight": r.get("weight") or 0.0,
                }
            )

        return {"entities": entities, "relations": relations, "chunk_ids": chunk_ids}

    async def get_entity_subgraph(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        await self._ensure_initialized()
        # 目前提供 depth=1 的轻量子图，便于前端展示与调试
        cypher = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH (e)-[r:RELATED]-(n:Entity)
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        RETURN
            e { .name, .type, .description } AS entity,
            collect(DISTINCT n { .name, .type, .description }) AS neighbors,
            collect(DISTINCT {
                source: e.name,
                target: n.name,
                keywords: r.keywords,
                description: r.description,
                weight: r.weight
            }) AS relations,
            collect(DISTINCT c.chunk_id) AS chunks
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, name=entity_name)
            record = await result.single()
        if not record:
            return {"entity": None, "neighbors": [], "relations": [], "chunks": []}
        return {
            "entity": record.get("entity"),
            "neighbors": [n for n in (record.get("neighbors") or []) if n and n.get("name")],
            "relations": [
                r
                for r in (record.get("relations") or [])
                if r and r.get("source") and r.get("target")
            ],
            "chunks": [c for c in (record.get("chunks") or []) if c],
        }
