import asyncio
import hashlib
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import jieba

# --- OpenSearch 异步客户端 ---
from opensearchpy import TransportError
from opensearchpy._async.client import AsyncOpenSearch
from opensearchpy._async.helpers.actions import async_bulk

# --- 项目核心模块 ---
# 导入配置 (config)
from ...core.config import settings
# 导入日志 (logging)
from ...core.logging import setup_logging
from ..llm.factory import get_embedding_model
from ...domain.models import DocumentChunk, RetrievedChunk, GraphEntity, GraphRelation
from ...domain.interfaces import SearchRepository, EmbeddingCache
from .mappings import (
    get_entity_opensearch_mapping,
    get_opensearch_mapping,
    get_relation_opensearch_mapping,
)
from ..cache.redis_embedding_cache import build_embedding_cache_key

# === 日志配置 ===
setup_logging() 
log = logging.getLogger(__name__)

# === 从配置中获取 Embedding 维度 ===
EMBEDDING_DIM = settings.embedding_llm.dimension

class AsyncOpenSearchRAGStore(SearchRepository):
    """
    一个用于 RAG 系统的 异步 OpenSearch 存储和检索类。
    """

    def __init__(
        self,
        *,
        embedding_cache: Optional[EmbeddingCache] = None,
        embedding_cache_ttl_seconds: int = 0,
        embedding_model_name: str = "",
    ):
        """
        初始化 (同步)。
        从 `settings` 模块加载配置。
        """
        # 从 config 模块导入 (使用 settings.opensearch.*)
        self.index_name = settings.opensearch.index_name
        self.entity_index_name = settings.opensearch.entity_index_name
        self.relation_index_name = settings.opensearch.relation_index_name
        self.host = settings.opensearch.host
        self.port = settings.opensearch.port
        
        # 使用 logging
        log.info(f"正在初始化 AsyncOpenSearchRAGStore...")
        log.info(f"目标索引 (chunks): {self.index_name}")
        log.info(f"目标索引 (entities): {self.entity_index_name}")
        log.info(f"目标索引 (relations): {self.relation_index_name}")
        log.info(f"OpenSearch 地址: {self.host}:{self.port}")
        log.info(f"Embedding 维度: {EMBEDDING_DIM}")

        # [Async Change] 实例化 AsyncOpenSearch 客户端
        self.client = AsyncOpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=settings.opensearch.auth,  
            use_ssl=settings.opensearch.use_ssl,
            verify_certs=settings.opensearch.verify_certs,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=60,
            max_retry=3,
            retry_on_timeout=True
        )
        
        # 使用 liteLLM 客户端
        self.embedding_client = get_embedding_model()
        self.embedding_cache = embedding_cache
        self.embedding_cache_ttl_seconds = int(embedding_cache_ttl_seconds or 0)
        self.embedding_model_name = embedding_model_name or "embedding"
        log.info("Embedding 客户端 (liteLLM) 已链接。")
        log.info("Jieba 分词器已准备就绪。")
        log.info(f"AsyncOpenSearchRAGStore (索引: {self.index_name}) 已初始化。")

    async def verify_connection(self):
        """
        异步检查与 OpenSearch 的连接。
        """
        try:
            if not await self.client.ping():
                log.error(f"无法 Ping 通 OpenSearch (在 {self.host}:{self.port})。")
                raise ConnectionError(f"无法连接到 OpenSearch (在 {self.host}:{self.port})。")
            log.info(f"成功连接到 OpenSearch (在 {self.host}:{self.port})。")
        except Exception as e:
            log.error(f"连接到 OpenSearch 失败: {e}", exc_info=True)
            raise

    # --- 异步 Embedding 封装 ---

    async def _get_embedding_async(self, text: str) -> List[float]:
        if not text: 
            return None
        try:
            return await self.embedding_client.aembed_query(text)
        except Exception as e:
            log.error(f"获取单个 embedding (aembed_query) 失败: {e}", exc_info=True)
            return None

    async def _get_query_embedding_async(self, text: str) -> List[float]:
        """
        获取 query embedding（带 Redis 缓存）。

        注意：仅用于“查询向量化”，避免把文档块 embedding 写入缓存造成膨胀。
        """
        if not text or not text.strip():
            return None

        if self.embedding_cache and self.embedding_cache_ttl_seconds > 0:
            cache_key = build_embedding_cache_key(model=self.embedding_model_name, text=text)
            cached = await self.embedding_cache.get(cache_key)
            if cached is not None:
                return cached

        embedding = await self._get_embedding_async(text)
        if embedding is None:
            return None

        if self.embedding_cache and self.embedding_cache_ttl_seconds > 0:
            try:
                await self.embedding_cache.set(cache_key, embedding, self.embedding_cache_ttl_seconds)
            except Exception:
                # 缓存失败不影响主流程
                pass

        return embedding

    async def get_query_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取 query embedding（复用内部缓存逻辑）。

        供 services 层在检索前显式预取/复用 Redis 缓存。
        """
        return await self._get_query_embedding_async(text)

    async def _get_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            
        if not valid_texts:
            return [None for _ in texts]
        
        results: List[Optional[List[float]]] = [None for _ in texts]
        
        try:
            embeddings = await self.embedding_client.aembed_documents(valid_texts)
            
            for i, embedding in enumerate(embeddings):
                original_index = valid_indices[i]
                results[original_index] = embedding
                
            return results
            
        except Exception as e:
            log.error(f"获取批量 embeddings (aembed_documents) 失败: {e}", exc_info=True)
            raise e

    # --- Jieba 分词封装 ---

    def _tokenize_with_jieba_sync(self, text: str) -> str:
        tokens = jieba.cut_for_search(text) 
        return " ".join(tokens)

    async def _tokenize_with_jieba_async(self, text: str) -> str:
        if not text:
            return ""
        return await asyncio.to_thread(self._tokenize_with_jieba_sync, text)
    
    # 数据转换
    def _convert_to_retrieved_chunk(self, source: Dict[str, Any], score: float) -> RetrievedChunk:
        """
        [内部辅助] 将 OpenSearch 的 _source 字典和分数转换为 RetrievedChunk 对象。
        
        注意：这里显式构建 DocumentChunk，通常不包含 embedding 向量字段，
        以减少后续流程的内存开销。
        """
        # 显式映射字段，避免传入多余的 OpenSearch 内部字段 (如 vectors)
        doc_chunk = DocumentChunk(
            chunk_id=source.get("chunk_id"),
            document_id=source.get("document_id"),
            document_name=source.get("document_name", ""),
            content=source.get("content", ""),
            summary=source.get("summary"),
            metadata=source.get("metadata", {})
        )
        
        return RetrievedChunk(
            chunk=doc_chunk,
            search_score=score,
            rerank_score=None # 此时还没重排序
        )

    # --- 索引管理 (DDL) ---

    async def _create_index_if_missing(self, index_name: str, mapping_body: Dict[str, Any]) -> None:
        if await self.client.indices.exists(index=index_name):
            log.warning(f"索引 '{index_name}' 已存在。")
            return
        try:
            await self.client.indices.create(index=index_name, body=mapping_body)
            log.info(f"索引 '{index_name}' 创建成功。")
        except TransportError as e:
            log.error(f"创建索引 '{index_name}' 时出错: {e.status_code} {e.info}", exc_info=True)
        except Exception as e:
            log.error(f"创建索引 '{index_name}' 时发生未知错误: {e}", exc_info=True)

    async def create_index(self):
        """
        显式创建索引的方法。应在应用启动时调用。
        """
        await self._create_index_if_missing(self.index_name, get_opensearch_mapping())
        await self._create_index_if_missing(self.entity_index_name, get_entity_opensearch_mapping())
        await self._create_index_if_missing(self.relation_index_name, get_relation_opensearch_mapping())

    # --- 文档读取（供图谱检索回查） ---

    async def mget_documents(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取文档。

        说明：
        - 返回列表与输入 `chunk_ids` 顺序一致；
        - 未找到的条目会被跳过（不填充 None），调用方可自行比对。
        """
        if not chunk_ids:
            return []

        try:
            response = await self.client.mget(index=self.index_name, body={"ids": chunk_ids})
            docs_map = {
                doc["_id"]: doc["_source"]
                for doc in response.get("docs", [])
                if doc.get("found") and doc.get("_source")
            }
            return [docs_map[cid] for cid in chunk_ids if cid in docs_map]
        except Exception as e:
            log.error(f"mget 批量获取失败: {e}", exc_info=True)
            return []

    # --- 图谱索引（实体/关系） ---

    def _make_stable_id(self, namespace: str, raw: str) -> str:
        digest = hashlib.sha256((raw or "").encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    async def index_graph_entities_relations(
        self,
        chunk: DocumentChunk,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
    ) -> None:
        """
        将 LLM 从单个 chunk 抽取的实体/关系分别写入 OpenSearch 的独立索引。

        - 实体文本拼接格式：entity_name、entity_type、entity_description
        - 关系文本拼接格式：source_entity、target_entity、relationship_keywords、relationship_description
        """
        if not chunk or (not entities and not relations):
            return

        entity_docs: List[Dict[str, Any]] = []
        for e in entities or []:
            name = (e.name or "").strip()
            if not name:
                continue
            entity_type = (e.type or "Other").strip() or "Other"
            entity_description = (e.description or "").strip()
            entity_text = f"{name}、{entity_type}、{entity_description}"
            entity_docs.append(
                {
                    "entity_id": self._make_stable_id(
                        "ent",
                        f"{chunk.chunk_id}|{name}|{entity_type}|{entity_description}",
                    ),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "entity_name": name,
                    "entity_type": entity_type,
                    "entity_description": entity_description,
                    "entity_text": entity_text,
                    "metadata": dict(chunk.metadata or {}),
                }
            )

        relation_docs: List[Dict[str, Any]] = []
        for r in relations or []:
            src = (r.source or "").strip()
            tgt = (r.target or "").strip()
            if not src or not tgt or src == tgt:
                continue
            kw_list = [str(k).strip() for k in (r.keywords or []) if str(k).strip()]
            kw_str = ",".join(kw_list)
            rel_desc = (r.description or "").strip()
            relation_text = f"{src}、{tgt}、{kw_str}、{rel_desc}"
            relation_docs.append(
                {
                    "relation_id": self._make_stable_id(
                        "rel",
                        f"{chunk.chunk_id}|{src}|{tgt}|{kw_str}|{rel_desc}",
                    ),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "source_entity": src,
                    "target_entity": tgt,
                    "relationship_keywords": kw_list,
                    "relationship_description": rel_desc,
                    "relation_text": relation_text,
                    "metadata": dict(chunk.metadata or {}),
                }
            )

        # 并发向量化
        entity_texts = [d["entity_text"] for d in entity_docs]
        relation_texts = [d["relation_text"] for d in relation_docs]
        (entity_embeddings, relation_embeddings) = await asyncio.gather(
            self._get_embeddings_batch_async(entity_texts),
            self._get_embeddings_batch_async(relation_texts),
        )

        actions: List[Dict[str, Any]] = []
        for doc, emb in zip(entity_docs, entity_embeddings):
            if emb is None:
                continue
            doc["embedding_entity_text"] = emb
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.entity_index_name,
                    "_id": doc["entity_id"],
                    "_source": doc,
                }
            )
        for doc, emb in zip(relation_docs, relation_embeddings):
            if emb is None:
                continue
            doc["embedding_relation_text"] = emb
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.relation_index_name,
                    "_id": doc["relation_id"],
                    "_source": doc,
                }
            )

        if not actions:
            return

        try:
            success_count, errors = await async_bulk(
                self.client,
                actions,
                chunk_size=min(settings.opensearch.bulk_chunk_size, 500),
                max_chunk_bytes=10 * 1024 * 1024,
                raise_on_error=False,
                max_retries=3,
            )
            if errors:
                log.error(
                    "图谱索引写入存在失败: success=%s failed=%s",
                    success_count,
                    len(errors),
                )
        except Exception as e:
            log.error(f"图谱索引写入失败 (chunk_id={chunk.chunk_id}): {e}", exc_info=True)

    async def vector_search_entities(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []
        query_embedding = await self._get_query_embedding_async(query_text)
        if not query_embedding:
            return []

        query = {
            "size": k,
            "_source": [
                "entity_id",
                "chunk_id",
                "document_id",
                "document_name",
                "entity_name",
                "entity_type",
                "entity_description",
            ],
            "query": {
                "knn": {
                    "embedding_entity_text": {
                        "vector": query_embedding,
                        "k": k,
                    }
                }
            },
        }

        try:
            response = await self.client.search(index=self.entity_index_name, body=query)
            hits = response.get("hits", {}).get("hits", []) or []
            results: List[Dict[str, Any]] = []
            for h in hits:
                src = h.get("_source") or {}
                if not src:
                    continue
                results.append({"score": float(h.get("_score") or 0.0), **src})
            return results
        except Exception as e:
            log.error(f"实体向量检索失败: {e}", exc_info=True)
            return []

    async def vector_search_relations(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []
        query_embedding = await self._get_query_embedding_async(query_text)
        if not query_embedding:
            return []

        query = {
            "size": k,
            "_source": [
                "relation_id",
                "chunk_id",
                "document_id",
                "document_name",
                "source_entity",
                "target_entity",
                "relationship_keywords",
                "relationship_description",
            ],
            "query": {
                "knn": {
                    "embedding_relation_text": {
                        "vector": query_embedding,
                        "k": k,
                    }
                }
            },
        }

        try:
            response = await self.client.search(index=self.relation_index_name, body=query)
            hits = response.get("hits", {}).get("hits", []) or []
            results: List[Dict[str, Any]] = []
            for h in hits:
                src = h.get("_source") or {}
                if not src:
                    continue
                results.append({"score": float(h.get("_score") or 0.0), **src})
            return results
        except Exception as e:
            log.error(f"关系向量检索失败: {e}", exc_info=True)
            return []

    # --- 高并发检索算法 ---

    async def bm25_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = await self._tokenize_with_jieba_async(query_text)
        log.debug(f"[BM25] 原始查询: '{query_text}', Jieba分词: '{tokenized_query}'")
        
        query = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": tokenized_query, 
                    "type": "best_fields",
                    "fields": [
                        "content_tokenized^3", 
                        "content^2",
                        "hypothetical_questions_merged^2",
                        "summary^1.5",
                        "parent_headings_merged^1.5",
                        "document_name^1.0"
                    ]
                }
            }
        }
        try:
            response = await self.client.search(
                index=self.index_name,
                body=query
            )
            return response['hits']['hits']
        except TransportError as e:
            log.error(f"BM25 (multi_match) 检索时出错: {e.status_code} {e.info}", exc_info=True)
            return []

    async def _base_vector_search(self, field_name: str, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        query = {
            "size": k,
            "query": {
                "knn": {
                    field_name: {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        }
        try:
            response = await self.client.search(
                index=self.index_name,
                body=query
            )
            return response['hits']['hits']
        except TransportError as e:
            log.error(f"向量检索字段 '{field_name}' 时出错: {e.status_code} {e.info}", exc_info=False) 
            return []

    def _rrf_fuse(self, 
                  results_lists: List[List[Dict[str, Any]]], 
                  k_constant: int = 60) -> List[Tuple[str, float]]:
        """
        使用 RRF 融合多路召回结果。
        
        返回值从 List[str] 改为 List[Tuple[str, float]]。
        返回 (doc_id, rrf_score) 的列表，以便保留分数信息。
        """
        fused_scores: Dict[str, float] = {}
        
        for results in results_lists:
            for rank, doc in enumerate(results, 1):
                doc_id = doc['_id'] 
                # RRF 公式: 1 / (k + rank)
                rrf_score = 1.0 / (k_constant + rank)
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                fused_scores[doc_id] += rrf_score
        
        # 按分数降序排序
        sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs

    async def hybrid_search(
        self, 
        query_text: str, 
        k: int = 5, 
        rrf_k: int = 60
    ) -> List[RetrievedChunk]: # [修改] 返回类型变更
        """
        [异步] 高并发混合搜索 (BM25 + 4路向量)。
        返回标准的 RetrievedChunk 列表。
        """
        log.info(f"--- 开始 *异步* 混合搜索 (5路召回) (查询: '{query_text}') ---")

        if not query_text or not query_text.strip():
            return []
        
        # 1. 并发获取 query embedding 和 BM25 结果
        try:
            (query_embedding, bm25_results) = await asyncio.gather(
                self._get_query_embedding_async(query_text),
                self.bm25_search(query_text, k=k*2) 
            )
        except Exception as e:
            log.error(f"混合搜索第一阶段失败: {e}", exc_info=True)
            return []

        # 2. 并发执行向量搜索
        vector_tasks = [
            self._base_vector_search("embedding_content", query_embedding, k=k*2),
            self._base_vector_search("embedding_parent_headings", query_embedding, k=k*2),
            self._base_vector_search("embedding_summary", query_embedding, k=k*2),
            self._base_vector_search("embedding_hypothetical_questions", query_embedding, k=k*2)
        ]
        
        try:
            (
                vec_content_results, vec_headings_results, 
                vec_summary_results, vec_questions_results
            ) = await asyncio.gather(*vector_tasks)
        except Exception as e:
            log.error(f"混合搜索第二阶段失败: {e}", exc_info=True)
            # 降级策略：如果向量搜索失败，仅使用 BM25
            vec_content_results, vec_headings_results, vec_summary_results, vec_questions_results = [], [], [], []

        # 3. RRF 融合 (获取 ID 和 RRF 分数)
        all_results_lists = [
            bm25_results, vec_content_results, vec_headings_results, 
            vec_summary_results, vec_questions_results
        ]
        
        # 获取 [(id, score), ...]
        fused_results_with_score = self._rrf_fuse(all_results_lists, k_constant=rrf_k)
        
        # 截取 Top K
        top_k_results = fused_results_with_score[:k]
        
        if not top_k_results:
            log.warning("混合搜索未找到任何结果。")
            return []

        top_k_ids = [item[0] for item in top_k_results]
        # 创建一个 id -> score 的映射，方便后续组装
        score_map = {item[0]: item[1] for item in top_k_results}

        log.debug(f"RRF 融合后 Top-{k} ID: {top_k_ids}")
        
        # 4. 异步 mget 批量获取文档详情 (fetching full document content)
        try:
            response = await self.client.mget(
                index=self.index_name,
                body={"ids": top_k_ids}
            )
            
            # 5. 组装为 RetrievedChunk 对象列表
            retrieved_chunks = []
            
            # 创建临时字典以按 ID 查找 mget 结果
            docs_map = {doc['_id']: doc['_source'] for doc in response['docs'] if doc.get('found', False)}
            
            # 按照 RRF 排序的顺序构建结果
            for doc_id in top_k_ids:
                if doc_id in docs_map:
                    source = docs_map[doc_id]
                    score = score_map[doc_id]
                    
                    # 转换并添加
                    ret_chunk = self._convert_to_retrieved_chunk(source, score)
                    retrieved_chunks.append(ret_chunk)
            
            log.info(f"--- 混合搜索成功，返回 {len(retrieved_chunks)} 个 RetrievedChunk ---")
            return retrieved_chunks
            
        except TransportError as e:
            log.error(f"混合搜索 (mget) 时出错: {e.status_code} {e.info}", exc_info=True)
            return []

    async def hybrid_search_with_embedding(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        [异步] 混合搜索（使用外部提供的 query embedding）。

        说明：
        - 用于 services 层先从 Redis 取出 embedding 后的检索路径；
        - 若未提供 embedding，则降级为内部获取。
        """
        log.info(f"--- 开始 *异步* 混合搜索 (预置向量) (查询: '{query_text}') ---")

        if not query_text or not query_text.strip():
            return []

        if not query_embedding:
            query_embedding = await self._get_query_embedding_async(query_text)
        if not query_embedding:
            return []

        # 并发执行 BM25 和向量搜索
        bm25_task = self.bm25_search(query_text, k=k * 2)
        vector_tasks = [
            self._base_vector_search("embedding_content", query_embedding, k=k * 2),
            self._base_vector_search("embedding_parent_headings", query_embedding, k=k * 2),
            self._base_vector_search("embedding_summary", query_embedding, k=k * 2),
            self._base_vector_search("embedding_hypothetical_questions", query_embedding, k=k * 2),
        ]

        try:
            (
                bm25_results,
                vec_content_results,
                vec_headings_results,
                vec_summary_results,
                vec_questions_results,
            ) = await asyncio.gather(bm25_task, *vector_tasks)
        except Exception as e:
            log.error(f"混合搜索 (预置向量) 召回阶段失败: {e}", exc_info=True)
            return []

        all_results_lists = [
            bm25_results,
            vec_content_results,
            vec_headings_results,
            vec_summary_results,
            vec_questions_results,
        ]

        fused_results_with_score = self._rrf_fuse(all_results_lists, k_constant=rrf_k)
        top_k_results = fused_results_with_score[:k]
        if not top_k_results:
            return []

        top_k_ids = [item[0] for item in top_k_results]
        score_map = {item[0]: item[1] for item in top_k_results}

        try:
            response = await self.client.mget(index=self.index_name, body={"ids": top_k_ids})
            docs_map = {
                doc["_id"]: doc["_source"]
                for doc in response.get("docs", [])
                if doc.get("found") and doc.get("_source")
            }

            retrieved_chunks: List[RetrievedChunk] = []
            for doc_id in top_k_ids:
                if doc_id not in docs_map:
                    continue
                retrieved_chunks.append(self._convert_to_retrieved_chunk(docs_map[doc_id], score_map[doc_id]))
            return retrieved_chunks
        except TransportError as e:
            log.error(f"混合搜索 (预置向量 mget) 时出错: {e.status_code} {e.info}", exc_info=True)
            return []

    # --- 批量操作 ---

    async def _generate_bulk_actions_async(
        self, 
        documents: List[DocumentChunk]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        all_content = [doc.content for doc in documents]
        all_headings = [" ".join(doc.parent_headings) for doc in documents]
        all_summaries = [doc.summary or "" for doc in documents]
        all_questions = [" ".join(doc.hypothetical_questions) for doc in documents]

        log.info(f"批量处理 {len(documents)} 个文档：开始并发执行 Embedding (4批) 和 Jieba (1批)...")

        try:
            tasks = [
                self._get_embeddings_batch_async(all_content),
                self._get_embeddings_batch_async(all_headings),
                self._get_embeddings_batch_async(all_summaries),
                self._get_embeddings_batch_async(all_questions),
                asyncio.gather(*[self._tokenize_with_jieba_async(content) for content in all_content])
            ]
            
            (
                all_emb_content,
                all_emb_headings,
                all_emb_summaries,
                all_emb_questions,
                all_tokenized_content
            ) = await asyncio.gather(*tasks)

        except Exception as e:
            log.error(f"批量处理 (gather) 失败: {e}", exc_info=True)
            raise 

        log.info("Embedding 和 Jieba 处理完毕，开始 yield...")

        for i, doc in enumerate(documents):
            
            doc_body = {
                "chunk_id": doc.chunk_id,
                "document_id": doc.document_id,
                "document_name": doc.document_name,
                "content": doc.content,
                "content_tokenized": all_tokenized_content[i],
                "parent_headings_merged": all_headings[i], 
                "summary": doc.summary,
                "hypothetical_questions_merged": all_questions[i], 
                "embedding_content": all_emb_content[i],
                "embedding_parent_headings": all_emb_headings[i],
                "embedding_summary": all_emb_summaries[i],
                "embedding_hypothetical_questions": all_emb_questions[i],
                "metadata": doc.metadata
            }
            
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc.chunk_id, 
                "_source": doc_body
            }
            
            yield action 

    async def bulk_add_documents(self, documents: List[DocumentChunk]):
        if not documents:
            log.warning("没有要添加的文档。")
            return

        log.info(f"--- 开始 *异步* 批量导入 {len(documents)} 个文档 ---")
        
        try:
            actions_generator = self._generate_bulk_actions_async(documents)
            
            log.info("使用 'helpers.async_bulk' 开始导入...")
            
            success_count, errors = await async_bulk(
                self.client, 
                actions_generator, 
                # 修改点：从 settings.opensearch 读取
                chunk_size=settings.opensearch.bulk_chunk_size, 
                max_chunk_bytes=10 * 1024 * 1024, # 关键：限制单次请求最大为 10MB
                raise_on_error=False,          # 建议设为 False，避免单个失败炸掉整个流程
                max_retries=3
            )

            log.info(f"批量导入完成。成功: {success_count}, 失败: {len(errors)}")
            if errors:
                log.error("--- 批量导入错误示例 (最多显示5条) ---")
                for i, err in enumerate(errors[:5]):
                    log.error(json.dumps(err, indent=2, ensure_ascii=False))

        except Exception as e:
            log.error(f"批量导入过程中发生严重错误: {e}", exc_info=True)
        
        finally:
            log.info("正在执行手动刷新 (refresh)...")
            try:
                await self.client.indices.refresh(index=self.index_name)
                log.info("--- 批量导入流程结束 (已刷新) ---")
            except TransportError as e:
                log.error(f"刷新索引 {self.index_name} 失败: {e.status_code} {e.info}", exc_info=True)

    # --- 异步批量查询 ---

    async def hybrid_search_batch(
        self, 
        queries: List[str], 
        k: int = 5, 
        rrf_k: int = 60
    ) -> List[List[RetrievedChunk]]:
        if not queries:
            return []
            
        log.info(f"--- 开始 *异步* 批量混合搜索 (共 {len(queries)} 个查询) ---")
        
        tasks = [
            self.hybrid_search(query, k=k, rrf_k=rrf_k)
            for query in queries
        ]
        
        try:
            all_results = await asyncio.gather(*tasks)
            log.info(f"--- *异步* 批量混合搜索完成 ---")
            return all_results
            
        except Exception as e:
            log.error(f"批量混合搜索过程中发生错误: {e}", exc_info=True)
            return [[] for _ in queries]

    async def hybrid_search_batch_with_embeddings(
        self,
        queries: List[str],
        embeddings: List[List[float]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[List[RetrievedChunk]]:
        if not queries:
            return []
        if len(queries) != len(embeddings):
            raise ValueError("queries 与 embeddings 长度不一致。")

        tasks = [
            self.hybrid_search_with_embedding(query_text=q, query_embedding=emb, k=k, rrf_k=rrf_k)
            for q, emb in zip(queries, embeddings)
        ]
        try:
            return await asyncio.gather(*tasks)
        except Exception as e:
            log.error(f"批量混合搜索 (预置向量) 过程中发生错误: {e}", exc_info=True)
            return [[] for _ in queries]
