"""
RAG 应用编排服务（图谱 + 向量 + 重排 + 回答）。

设计目标：
1) 同时使用 Neo4j 知识图谱检索与 OpenSearch 向量/混合检索；
2) 使用 TEI 重排器对候选块进行最终排序；
3) 参考 LightRAG 的“回答 + 参考引用”组织方式输出结果。
"""

import logging
import asyncio
import hashlib
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

from pydantic import JsonValue

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..core.config import settings
from ..core.prompts import QUERY_REWRITE_PROMPT, RAG_ANSWER_WITH_REFERENCES_PROMPT
from ..domain.interfaces import (
    RAGOrchestrator,
    SearchRepository,
    GraphRepository,
    KeywordExtractor,
    Reranker,
)
from ..domain.models import (
    RetrievedChunk,
    DocumentChunk,
    AnswerResult,
    AnswerReference,
    RetrievalMode,
)
from redis.asyncio import Redis
from ..infrastructure.cache.redis_embedding_cache import build_embedding_cache_key

log = logging.getLogger(__name__)


def _deduplicate_by_chunk_id(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """按 chunk_id 去重，保留 search_score 更高的一条。"""
    best: Dict[str, RetrievedChunk] = {}
    for c in chunks:
        cid = c.chunk.chunk_id
        if cid not in best or c.search_score > best[cid].search_score:
            best[cid] = c
    return list(best.values())


def _build_references_by_document_title(
    chunks: List[RetrievedChunk],
) -> Tuple[List[AnswerReference], Dict[str, str]]:
    """
    根据 document_name 构建引用列表（参考 LightRAG 的频次优先策略）。
    """
    titles = [c.chunk.document_name for c in chunks if c.chunk.document_name]
    if not titles:
        return [], {}

    counts = Counter(titles)
    first_index: Dict[str, int] = {}
    for idx, title in enumerate(titles):
        first_index.setdefault(title, idx)

    sorted_titles = sorted(counts.keys(), key=lambda t: (-counts[t], first_index[t]))
    title_to_ref = {title: str(i + 1) for i, title in enumerate(sorted_titles)}

    references = [
        AnswerReference(reference_id=title_to_ref[title], document_title=title)
        for title in sorted_titles
    ]
    return references, title_to_ref


def _build_context(
    *,
    graph_context: Dict[str, JsonValue],
    chunks: List[RetrievedChunk],
    title_to_ref: Dict[str, str],
) -> str:
    """构建给 LLM 的上下文文本（包含图谱数据与文本块）。"""
    lines: List[str] = []

    # 1) Knowledge Graph Data
    lines.append("[Knowledge Graph Data]")
    entities = graph_context.get("entities") or []
    relations = graph_context.get("relations") or []
    entity_to_chunk_ids = graph_context.get("entity_to_chunk_ids") or {}
    if entities:
        lines.append("- Entities:")
        for e in entities[: settings.kg.query_top_k_entities]:
            name = e.get("name", "")
            etype = e.get("type", "Other")
            desc = (e.get("description") or "").strip()
            degree = e.get("degree")
            degree_str = f" | degree: {degree}" if isinstance(degree, (int, float)) else ""
            lines.append(f"  - {name} ({etype}){degree_str}: {desc}")
            if isinstance(entity_to_chunk_ids, dict):
                cids = entity_to_chunk_ids.get(name) or []
                if cids:
                    lines.append(f"    - chunks: {', '.join([str(c) for c in cids if str(c).strip()])}")
    else:
        lines.append("- Entities: (无)")

    if relations:
        lines.append("- Relations:")
        for r in relations[: settings.kg.query_top_k_entities]:
            src = r.get("source", "")
            tgt = r.get("target", "")
            kws = r.get("keywords") or []
            if isinstance(kws, list):
                kw_str = ", ".join([str(k) for k in kws if str(k).strip()])
            else:
                kw_str = str(kws)
            desc = (r.get("description") or "").strip()
            lines.append(f"  - {src} <-> {tgt} | 关键词: {kw_str} | {desc}")
    else:
        lines.append("- Relations: (无)")

    # 2) Reference Document List
    lines.append("\n[Reference Document List]")
    for title, ref_id in sorted(title_to_ref.items(), key=lambda x: int(x[1])):
        lines.append(f"- [{ref_id}] {title}")

    # 3) Document Chunks
    lines.append("\n[Document Chunks]")
    for c in chunks:
        title = c.chunk.document_name or "未知文档"
        ref_id = title_to_ref.get(title, "")
        lines.extend(
            [
                f"- chunk_id: {c.chunk.chunk_id}",
                f"  document_title: {title}",
                f"  reference_id: {ref_id}",
                "  content:",
                f"  {c.chunk.content.strip()}",
            ]
        )

    return "\n".join(lines)


class RagService(RAGOrchestrator):
    """RAG 应用编排服务实现。"""

    def __init__(
        self,
        *,
        search_repo: SearchRepository,
        graph_repo: GraphRepository,
        keyword_extractor: KeywordExtractor,
        rewrite_llm: BaseChatModel,
        answer_llm: BaseChatModel,
        reranker: Reranker,
        redis: Redis | None = None,
        cache_ttl_seconds: int = 0,
        rewrite_model_name: str = "",
    ):
        self._search_repo = search_repo
        self._graph_repo = graph_repo
        self._keyword_extractor = keyword_extractor
        self._reranker = reranker
        self._answer_llm = answer_llm
        self._redis = redis
        self._cache_ttl_seconds = max(0, int(cache_ttl_seconds or 0))
        self._rewrite_model_name = (rewrite_model_name or "").strip() or "rewrite"

        self._rewrite_chain = (
            ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
            | rewrite_llm
            | StrOutputParser()
        )

    def _make_cache_key(self, prefix: str, query: str) -> str:
        digest = hashlib.sha256((query or "").strip().encode("utf-8")).hexdigest()
        return f"{prefix}:{self._rewrite_model_name}:{digest}"

    async def _get_cached_rewrites(self, query: str) -> Optional[List[str]]:
        """
        仅从 Redis 读取 query 改写缓存；未命中返回 None。

        用于 `_vector_retrieve` 显式走“Redis 优先”路径。
        """
        if not (self._redis and self._cache_ttl_seconds > 0 and query and query.strip()):
            return None
        try:
            cache_key = self._make_cache_key("qrew", query)
            cached = await self._redis.get(cache_key)
            if not cached:
                return None
            if isinstance(cached, (bytes, bytearray)):
                cached = cached.decode("utf-8")
            data = json.loads(cached)
            if isinstance(data, dict) and isinstance(data.get("rewrites"), list):
                rewrites = [str(x).strip() for x in data["rewrites"] if str(x).strip()]
                return rewrites
        except Exception:
            return None
        return None

    async def _rewrite_query(self, query: str) -> List[str]:
        """生成查询变体（失败则返回空列表）。"""
        if self._redis and self._cache_ttl_seconds > 0 and query and query.strip():
            try:
                cache_key = self._make_cache_key("qrew", query)
                cached = await self._redis.get(cache_key)
                if cached:
                    if isinstance(cached, (bytes, bytearray)):
                        cached = cached.decode("utf-8")
                    data = json.loads(cached)
                    if isinstance(data, dict) and isinstance(data.get("rewrites"), list):
                        return [str(x).strip() for x in data["rewrites"] if str(x).strip()]
            except Exception:
                pass
        try:
            result = await self._rewrite_chain.ainvoke({"question": query})
            rewrites = [line.strip() for line in str(result).split("\n") if line.strip()]
            if self._redis and self._cache_ttl_seconds > 0 and query and query.strip():
                try:
                    cache_key = self._make_cache_key("qrew", query)
                    payload = json.dumps(
                        {"query": query, "rewrites": rewrites},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    await self._redis.set(name=cache_key, value=payload, ex=self._cache_ttl_seconds)
                except Exception:
                    pass
            return rewrites
        except Exception as e:
            log.warning(f"查询改写失败，降级为仅使用原始查询: {e}")
            return []

    async def _vector_retrieve(self, query: str) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        # 1) Redis 优先：尝试读取 query 改写结果
        rewrites = await self._get_cached_rewrites(query)
        if rewrites is None:
            rewrites = await self._rewrite_query(query)

        queries = [query, *rewrites]

        # 2) Redis 优先：尝试读取改写结果对应的 embedding；未命中则补齐向量化
        embeddings: List[Optional[List[float]]] = [None for _ in queries]
        if self._redis and self._cache_ttl_seconds > 0:
            try:
                keys = [
                    build_embedding_cache_key(model=settings.embedding_llm.model, text=q)
                    for q in queries
                ]
                cached_values = await self._redis.mget(keys)
                for i, value in enumerate(cached_values or []):
                    if not value:
                        continue
                    try:
                        if isinstance(value, (bytes, bytearray)):
                            value = value.decode("utf-8")
                        data = json.loads(value)
                        if isinstance(data, list) and data:
                            embeddings[i] = [float(x) for x in data]
                    except Exception:
                        continue
            except Exception:
                pass

        missing_indices = [i for i, emb in enumerate(embeddings) if emb is None]
        if missing_indices:
            tasks = [self._search_repo.get_query_embedding(queries[i]) for i in missing_indices]
            computed = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, res in zip(missing_indices, computed):
                if isinstance(res, Exception) or res is None:
                    continue
                embeddings[idx] = res

        # 3) 记录向量检索的 query/改写结果（用于调试/审计）
        if self._redis and self._cache_ttl_seconds > 0 and query.strip():
            try:
                cache_key = self._make_cache_key("vecq", query)
                payload = json.dumps(
                    {"query": query, "vector_queries": queries},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                await self._redis.set(name=cache_key, value=payload, ex=self._cache_ttl_seconds)
            except Exception:
                pass

        # 4) 使用预取到的 embedding 执行批量混合检索（避免重复向量化）
        filtered_queries: List[str] = []
        filtered_embeddings: List[List[float]] = []
        for q, emb in zip(queries, embeddings):
            if q and q.strip() and emb:
                filtered_queries.append(q)
                filtered_embeddings.append(emb)

        if not filtered_queries:
            return []

        results = await self._search_repo.hybrid_search_batch_with_embeddings(
            queries=filtered_queries, embeddings=filtered_embeddings, k=10, rrf_k=60
        )
        flat = [c for sub in results for c in sub]
        return _deduplicate_by_chunk_id(flat)

    async def _graph_retrieve(
        self,
        query: str,
        *,
        use_low_level: bool = True,
        use_high_level: bool = True,
    ) -> Tuple[Dict[str, JsonValue], List[RetrievedChunk]]:
        high, low = await self._keyword_extractor.extract(query)
        keywords: List[str] = []
        if use_low_level:
            keywords.extend(low)
        if use_high_level:
            keywords.extend(high)
        graph_query = " ".join(keywords).strip() or query

        # 1) 先从 OpenSearch 的实体/关系向量索引分别检索候选
        entity_task = self._search_repo.vector_search_entities(graph_query, k=30)
        relation_task = self._search_repo.vector_search_relations(graph_query, k=30)
        entity_results, relation_results = await asyncio.gather(entity_task, relation_task)

        entity_candidates: List[str] = []
        for r in entity_results or []:
            name = (r.get("entity_name") or "").strip()
            if name:
                entity_candidates.append(name)

        relation_candidates: List[tuple[str, str]] = []
        for r in relation_results or []:
            src = (r.get("source_entity") or "").strip()
            tgt = (r.get("target_entity") or "").strip()
            if src and tgt and src != tgt:
                relation_candidates.append((src, tgt))

        # 2) 再回查 Neo4j：按节点度排序筛选实体/关系，并从命中实体扩展邻居
        graph_context = await self._graph_repo.query_context(
            entity_candidates,
            relation_candidates,
            top_k_entities=settings.kg.query_top_k_entities,
            top_k_relations=settings.kg.query_top_k_entities,
            top_k_chunks=settings.kg.query_top_k_chunks,
        )
        chunk_ids = graph_context.get("chunk_ids", [])
        if not chunk_ids:
            return graph_context, []

        docs = await self._search_repo.mget_documents(chunk_ids)
        retrieved: List[RetrievedChunk] = []
        for doc in docs:
            try:
                chunk = DocumentChunk(
                    chunk_id=doc.get("chunk_id", ""),
                    document_id=doc.get("document_id", ""),
                    document_name=doc.get("document_name", ""),
                    content=doc.get("content", ""),
                    summary=doc.get("summary", ""),
                    metadata=doc.get("metadata", {}),
                )
                retrieved.append(RetrievedChunk(chunk=chunk, search_score=0.01, rerank_score=None))
            except Exception:
                continue

        return graph_context, retrieved

    async def ask(self, query: str, retrieval_mode: RetrievalMode = RetrievalMode.FUSION) -> AnswerResult:
        if not query or not query.strip():
            return AnswerResult(answer="问题为空，无法回答。", references=[], chunks=[], graph_context={})

        graph_context: Dict[str, JsonValue] = {}
        vector_chunks: List[RetrievedChunk] = []
        graph_chunks: List[RetrievedChunk] = []

        # 1) 按检索模式获取候选块
        if retrieval_mode == RetrievalMode.VECTOR:
            vector_chunks = await self._vector_retrieve(query)
        elif retrieval_mode == RetrievalMode.KG_LOW:
            graph_context, graph_chunks = await self._graph_retrieve(
                query, use_low_level=True, use_high_level=False
            )
        elif retrieval_mode == RetrievalMode.KG_HIGH:
            graph_context, graph_chunks = await self._graph_retrieve(
                query, use_low_level=False, use_high_level=True
            )
        elif retrieval_mode == RetrievalMode.KG_MIX:
            graph_context, graph_chunks = await self._graph_retrieve(
                query, use_low_level=True, use_high_level=True
            )
        else:  # RetrievalMode.FUSION（默认）
            vector_task = self._vector_retrieve(query)
            graph_task = self._graph_retrieve(query, use_low_level=True, use_high_level=True)
            vector_chunks, (graph_context, graph_chunks) = await asyncio.gather(vector_task, graph_task)

        # 2) 融合 + 截断候选（避免 TEI 压力过大）
        candidates = _deduplicate_by_chunk_id([*vector_chunks, *graph_chunks])
        candidates.sort(key=lambda x: x.search_score, reverse=True)
        candidates = candidates[:30]

        if not candidates:
            prompt = RAG_ANSWER_WITH_REFERENCES_PROMPT.format(
                context="[Knowledge Graph Data]\n(无)\n\n[Document Chunks]\n(无可用上下文)",
                query=query,
            )
            resp = await self._answer_llm.ainvoke(prompt)
            return AnswerResult(
                answer=getattr(resp, "content", "") or "",
                references=[],
                chunks=[],
                graph_context=graph_context,
            )

        # 3) 重排
        try:
            reranked = await self._reranker.rerank(query=query, chunks=candidates, top_n=8)
        except Exception as e:
            log.error(f"重排失败，降级为按 search_score 排序: {e}", exc_info=True)
            reranked = sorted(candidates, key=lambda x: x.search_score, reverse=True)[:8]

        # 4) 构建引用映射
        references, title_to_ref = _build_references_by_document_title(reranked)
        for c in reranked:
            title = c.chunk.document_name or ""
            c.chunk.metadata = dict(c.chunk.metadata or {})
            c.chunk.metadata["reference_id"] = title_to_ref.get(title, "")

        # 5) 回答生成（包含图谱上下文）
        context = _build_context(graph_context=graph_context, chunks=reranked, title_to_ref=title_to_ref)
        prompt = RAG_ANSWER_WITH_REFERENCES_PROMPT.format(context=context, query=query)
        resp = await self._answer_llm.ainvoke(prompt)

        return AnswerResult(
            answer=getattr(resp, "content", "") or "",
            references=references[:5],
            chunks=reranked,
            graph_context=graph_context,
        )
