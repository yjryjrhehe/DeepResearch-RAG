"""
RAG 应用编排服务（图谱 + 向量 + 重排 + 回答）。

设计目标：
1) 同时使用 Neo4j 知识图谱检索与 OpenSearch 向量/混合检索；
2) 使用 TEI 重排器对候选块进行最终排序；
3) 参考 LightRAG 的“回答 + 参考引用”组织方式输出结果。
"""

from __future__ import annotations

import logging
import asyncio
from collections import Counter
from typing import List, Dict, Tuple, Any

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
    graph_context: Dict[str, Any],
    chunks: List[RetrievedChunk],
    title_to_ref: Dict[str, str],
) -> str:
    """构建给 LLM 的上下文文本（包含图谱数据与文本块）。"""
    lines: List[str] = []

    # 1) Knowledge Graph Data
    lines.append("[Knowledge Graph Data]")
    entities = graph_context.get("entities") or []
    relations = graph_context.get("relations") or []
    if entities:
        lines.append("- Entities:")
        for e in entities[: settings.kg.query_top_k_entities]:
            name = e.get("name", "")
            etype = e.get("type", "Other")
            desc = (e.get("description") or "").strip()
            lines.append(f"  - {name} ({etype}): {desc}")
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
    ):
        self._search_repo = search_repo
        self._graph_repo = graph_repo
        self._keyword_extractor = keyword_extractor
        self._reranker = reranker
        self._answer_llm = answer_llm

        self._rewrite_chain = (
            ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
            | rewrite_llm
            | StrOutputParser()
        )

    async def _rewrite_query(self, query: str) -> List[str]:
        """生成查询变体（失败则返回空列表）。"""
        try:
            result = await self._rewrite_chain.ainvoke({"question": query})
            return [line.strip() for line in str(result).split("\n") if line.strip()]
        except Exception as e:
            log.warning(f"查询改写失败，降级为仅使用原始查询: {e}")
            return []

    async def _vector_retrieve(self, query: str) -> List[RetrievedChunk]:
        queries = [query]
        queries.extend(await self._rewrite_query(query))
        results = await self._search_repo.hybrid_search_batch(queries=queries, k=10, rrf_k=60)
        flat = [c for sub in results for c in sub]
        return _deduplicate_by_chunk_id(flat)

    async def _graph_retrieve(
        self,
        query: str,
        *,
        use_low_level: bool = True,
        use_high_level: bool = True,
    ) -> Tuple[Dict[str, Any], List[RetrievedChunk]]:
        high, low = await self._keyword_extractor.extract(query)
        keywords: List[str] = []
        if use_low_level:
            keywords.extend(low)
        if use_high_level:
            keywords.extend(high)
        graph_query = " ".join(keywords).strip() or query

        graph_context = await self._graph_repo.query_context(
            graph_query,
            top_k_entities=settings.kg.query_top_k_entities,
            top_k_chunks=settings.kg.query_top_k_chunks,
        )
        chunk_ids = graph_context.get("chunk_ids") or []
        if not chunk_ids:
            return graph_context, []

        docs = await self._search_repo.mget_documents(chunk_ids)
        retrieved: List[RetrievedChunk] = []
        for doc in docs:
            try:
                chunk = DocumentChunk(
                    chunk_id=doc.get("chunk_id"),
                    document_id=doc.get("document_id"),
                    document_name=doc.get("document_name", ""),
                    content=doc.get("content", ""),
                    summary=doc.get("summary"),
                    metadata=doc.get("metadata", {}) or {},
                )
                retrieved.append(RetrievedChunk(chunk=chunk, search_score=0.01, rerank_score=None))
            except Exception:
                continue

        return graph_context, retrieved

    async def ask(self, query: str, retrieval_mode: RetrievalMode = RetrievalMode.FUSION) -> AnswerResult:
        if not query or not query.strip():
            return AnswerResult(answer="问题为空，无法回答。", references=[], chunks=[], graph_context={})

        graph_context: Dict[str, Any] = {}
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
