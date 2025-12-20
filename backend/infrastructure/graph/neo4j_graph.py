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
from typing import List, Optional, Dict, Awaitable, Callable, TypeVar

from pydantic import JsonValue

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
        entity_candidates: List[str],
        relation_candidates: List[tuple[str, str]],
        top_k_entities: int = 10,
        top_k_relations: int = 10,
        top_k_chunks: int = 20,
    ) -> Dict[str, JsonValue]:
        """
        返回图谱上下文：实体、关系、以及关联 chunk_id。

        检索策略（面向 OpenSearch 候选回查）：
        1) 对候选实体在 Neo4j 中计算节点度（degree），按度排序筛选 Top-K 实体；
        2) 对候选关系按端点度之和排序筛选 Top-K 关系；
        3) 从命中实体出发扩展一跳邻居与关联边；
        4) 汇总与命中实体相关的 chunk_id，并返回结构化上下文。
        """
        await self._ensure_initialized()

        def _dedupe_keep_order(values: List[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for v in values:
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
            return out

        # --- 0) 清洗候选，合并所有实体 和 关系的两端节点，去重 ---
        cleaned_entities = [
            (name or "").strip() for name in (entity_candidates or []) if (name or "").strip()
        ]

        rel_pairs: List[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in relation_candidates or []:
            if not pair or len(pair) != 2:
                continue
            src = (pair[0] or "").strip()
            tgt = (pair[1] or "").strip()
            if not src or not tgt or src == tgt:
                continue
            key = tuple(sorted([src, tgt]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rel_pairs.append(key)

        candidate_names = _dedupe_keep_order(
            cleaned_entities + [s for s, _ in rel_pairs] + [t for _, t in rel_pairs]
        )
        if not candidate_names:
            return {
                "entities": [],
                "relations": [],
                "chunk_ids": [],
                "entity_to_chunk_ids": {},
                "relation_to_chunk_ids": [],
            }

        # --- 1) 回查实体属性与 degree 形成 节点：degree 字典，用于后续对实体和关系排序---
        cypher_entity_degrees = """
        MATCH (e:Entity)
        WHERE e.name IN $names
        RETURN
            e.name AS name,
            e.type AS type,
            e.description AS description,
            COUNT { (e)-[:RELATED]-() } AS degree
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher_entity_degrees, names=candidate_names)
            rows = await result.data()

        entity_map: Dict[str, Dict[str, JsonValue]] = {}
        degree_map: Dict[str, int] = {}
        for r in rows:
            name = r.get("name")
            if not name:
                continue
            degree = int(r.get("degree") or 0)
            degree_map[name] = degree
            entity_map[name] = {
                "name": name,
                "type": r.get("type") or "Other",
                "description": (r.get("description") or "").strip(),
                "degree": degree,
            }

        if not entity_map:
            return {
                "entities": [],
                "relations": [],
                "chunk_ids": [],
                "entity_to_chunk_ids": {},
                "relation_to_chunk_ids": [],
            }

        safe_top_k_entities = max(0, int(top_k_entities or 0))
        safe_top_k_relations = max(0, int(top_k_relations or 0))
        safe_top_k_chunks = max(0, int(top_k_chunks or 0))

        # --- 2) Top-K 实体（按 degree 排序） ---
        sorted_entities = sorted(
            entity_map.values(),
            key=lambda e: (e.get("degree", 0), e.get("name", "")),
            reverse=True,
        )
        top_entities = sorted_entities[:safe_top_k_entities] if safe_top_k_entities else []
        top_entity_names = [e["name"] for e in top_entities if e.get("name")]

        # --- 3) Top-K 关系（按端点 degree 之和排序） ---
        scored_rel_pairs: List[tuple[str, str, int]] = []
        for src, tgt in rel_pairs:
            if src not in degree_map or tgt not in degree_map:
                continue
            scored_rel_pairs.append((src, tgt, degree_map.get(src, 0) + degree_map.get(tgt, 0)))
        scored_rel_pairs.sort(key=lambda x: (x[2], x[0], x[1]), reverse=True)
        top_rel_pairs = [(s, t) for s, t, _ in scored_rel_pairs[:safe_top_k_relations]]
        
        # 把实体 和 关系两端节点 合并 
        seed_names = _dedupe_keep_order(
            top_entity_names + [s for s, _ in top_rel_pairs] + [t for _, t in top_rel_pairs]
        )
        if not seed_names:
            return {
                "entities": [],
                "relations": [],
                "chunk_ids": [],
                "entity_to_chunk_ids": {},
                "relation_to_chunk_ids": [],
            }

        # --- 4) 从命中实体扩展邻居（限制每个 seed 的邻居数量） ---
        neighbor_limit_per_entity = max(1, safe_top_k_relations or safe_top_k_entities or 10)
        cypher_expand_neighbors = """
        UNWIND $seed_names AS name
        MATCH (e:Entity {name: name})
        CALL {
            WITH e
            MATCH (e)-[r:RELATED]-(n:Entity)
            RETURN n, r
            ORDER BY coalesce(r.weight, 0.0) DESC
            LIMIT $neighbor_limit
        }
        RETURN
            e.name AS source,
            n.name AS target,
            n.type AS target_type,
            n.description AS target_description,
            r.keywords AS keywords,
            r.description AS description,
            r.weight AS weight
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher_expand_neighbors,
                seed_names=seed_names,
                neighbor_limit=neighbor_limit_per_entity,
            )
            neighbor_rows = await result.data()

        all_names: List[str] = []
        relations_map: Dict[tuple[str, str], Dict[str, JsonValue]] = {}
        neighbor_names: set[str] = set()
        for r in neighbor_rows:
            src = r.get("source")
            tgt = r.get("target")
            if not src or not tgt or src == tgt:
                continue
            neighbor_names.add(tgt)

            key = tuple(sorted([src, tgt]))
            existing = relations_map.get(key)
            kws = r.get("keywords") or []
            if not isinstance(kws, list):
                kws = [kws]
            kws = [k for k in kws if str(k).strip()]
            desc = (r.get("description") or "").strip()
            weight = float(r.get("weight") or 0.0)
            if existing is None:
                relations_map[key] = {
                    "source": src,
                    "target": tgt,
                    "keywords": kws,
                    "description": desc,
                    "weight": weight,
                }
            else:
                existing_kws = existing.get("keywords") or []
                existing["keywords"] = list({*existing_kws, *kws})
                if not existing.get("description") and desc:
                    existing["description"] = desc
                existing["weight"] = max(float(existing.get("weight") or 0.0), weight)

        all_names = _dedupe_keep_order(seed_names + list(neighbor_names))

        # 补齐邻居节点信息（若未在候选中出现）
        missing_names = [n for n in all_names if n not in entity_map]
        if missing_names:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher_entity_degrees, names=missing_names)
                extra_rows = await result.data()
            for r in extra_rows:
                name = r.get("name")
                if not name:
                    continue
                degree = int(r.get("degree") or 0)
                degree_map[name] = degree
                entity_map[name] = {
                    "name": name,
                    "type": r.get("type") or "Other",
                    "description": (r.get("description") or "").strip(),
                    "degree": degree,
                }

        # --- 5) 回查 Top-K 关系的属性（仅保留 Neo4j 中真实存在的边） ---
        # top_rel_pairs 来自 OpenSearch 关系向量检索的候选对 (source,target)，但候选不一定都成功落到 Neo4j（或被去重/规范化后不存在）
        # 为什么不在第一步直接筛选掉不存在的关系：因为opensearch检索到的结果可能存在真实有效但是未被写入graph的关系，
        # 我们希望这类结果也被利用起来去检索单跳邻居节点，获得更多有用信息。
        if top_rel_pairs:
            cypher_top_relations = """
            UNWIND $pairs AS p
            MATCH (a:Entity {name: p.source})-[r:RELATED]-(b:Entity {name: p.target})
            RETURN
                a.name AS source,
                b.name AS target,
                r.keywords AS keywords,
                r.description AS description,
                r.weight AS weight
            """
            pairs_payload = [{"source": s, "target": t} for s, t in top_rel_pairs]
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher_top_relations, pairs=pairs_payload)
                rel_rows = await result.data()

            for r in rel_rows:
                src = r.get("source")
                tgt = r.get("target")
                if not src or not tgt or src == tgt:
                    continue
                # 把关系当“无向边”去重
                key = tuple(sorted([src, tgt]))
                existing = relations_map.get(key)
                kws = r.get("keywords") or []
                if not isinstance(kws, list):
                    kws = [kws]
                kws = [k for k in kws if str(k).strip()]
                desc = (r.get("description") or "").strip()
                weight = float(r.get("weight") or 0.0) # weight是同样的关系出现次数，越多则说明越重要
                if existing is None:
                    relations_map[key] = {
                        "source": src,
                        "target": tgt,
                        "keywords": kws,
                        "description": desc,
                        "weight": weight,
                    }
                else:
                    existing_kws = existing.get("keywords") or []
                    existing["keywords"] = list({*existing_kws, *kws})
                    if not existing.get("description") and desc:
                        existing["description"] = desc
                    existing["weight"] = max(float(existing.get("weight") or 0.0), weight)

        # --- 6) 关系按 degree_score 排序并截断 ---
        # 把 relations_map 里的候选边做最终筛选
        relations: List[Dict[str, JsonValue]] = []
        for rel in relations_map.values():
            src = rel.get("source")
            tgt = rel.get("target")
            degree_score = int(degree_map.get(src, 0) + degree_map.get(tgt, 0))
            rel["degree_score"] = degree_score
            relations.append(rel)

        relations.sort(
            key=lambda r: (r.get("degree_score", 0), float(r.get("weight") or 0.0)),
            reverse=True,
        )
        if safe_top_k_relations:
            relations = relations[:safe_top_k_relations]

        # --- 7) 关联 chunk_id（基于 seed 实体聚合） ---
        entity_to_chunk_ids: Dict[str, List[str]] = {}
        chunks_per_entity = 3
        cypher_entity_chunks = """
        UNWIND $seed_names AS name
        MATCH (e:Entity {name: name})
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
        RETURN name AS entity_name, collect(DISTINCT c.chunk_id)[0..$limit] AS chunk_ids
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher_entity_chunks, seed_names=seed_names, limit=chunks_per_entity
            )
            chunk_rows = await result.data()
        for r in chunk_rows:
            name = r.get("entity_name")
            chunk_list = [c for c in (r.get("chunk_ids") or []) if c]
            if name:
                entity_to_chunk_ids[name] = chunk_list

        chunk_ids: List[str] = []
        if safe_top_k_chunks:
            cypher_chunks = """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $names
            RETURN c.chunk_id AS chunk_id, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT $top_k_chunks
            """
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    cypher_chunks, names=seed_names, top_k_chunks=safe_top_k_chunks
                )
                chunk_rows = await result.data()
            chunk_ids = [r["chunk_id"] for r in chunk_rows if r.get("chunk_id")]

        # --- 8) （可选）关系相关 chunk_id：同时提及两端实体的 chunk ---
        relation_to_chunk_ids: List[Dict[str, JsonValue]] = []
        if relations:
            pairs_payload = [
                {"source": r.get("source"), "target": r.get("target")}
                for r in relations
                if r.get("source") and r.get("target")
            ]
            cypher_relation_chunks = """
            UNWIND $pairs AS p
            MATCH (a:Entity {name: p.source})
            MATCH (b:Entity {name: p.target})
            MATCH (c:Chunk)-[:MENTIONS]->(a)
            MATCH (c)-[:MENTIONS]->(b)
            RETURN p.source AS source, p.target AS target, collect(DISTINCT c.chunk_id)[0..$limit] AS chunk_ids
            """
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    cypher_relation_chunks, pairs=pairs_payload, limit=3
                )
                rel_chunk_rows = await result.data()
            for r in rel_chunk_rows:
                src = r.get("source")
                tgt = r.get("target")
                cids = [c for c in (r.get("chunk_ids") or []) if c]
                if src and tgt:
                    relation_to_chunk_ids.append({"source": src, "target": tgt, "chunk_ids": cids})

        # --- 9) 输出实体列表（度排序；包含 seed + 邻居） ---
        entities_out = [entity_map[n] for n in all_names if n in entity_map]
        entities_out.sort(
            key=lambda e: (e.get("degree", 0), e.get("name", "")),
            reverse=True,
        )

        return {
            "entities": entities_out,
            "relations": relations,
            "chunk_ids": chunk_ids,
            "entity_to_chunk_ids": entity_to_chunk_ids,
            "relation_to_chunk_ids": relation_to_chunk_ids,
            "seed_entities": seed_names,
        }

    async def get_entity_subgraph(self, entity_name: str, depth: int = 1) -> Dict[str, JsonValue]:
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
