import logging
import httpx
from typing import List, Optional

# 导入数据模型
from ...domain.models import RetrievedChunk
from ...domain.interfaces import Reranker

# 初始化日志
log = logging.getLogger(__name__)

class TEIRerankerClient:
    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_concurrency: int = 50
    ):
        """
        初始化 TEI Reranker 客户端。
        """
        self.url = f"{base_url.rstrip('/')}/rerank"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.timeout = timeout
        self.max_concurrency = max_concurrency
            
        log.info(f"TEI Reranker 客户端已初始化，目标地址: {self.url}，超时: {self.timeout}s")

    # ==================== 1. 单条处理方法 ====================
    async def arerank(
        self, 
        query: str, 
        chunks: List[RetrievedChunk], 
        top_n: Optional[int] = None, 
        truncate: bool = True,
        client: Optional[httpx.AsyncClient] = None
    ) -> List[RetrievedChunk]:
        """
        [异步] 对单组查询和文档块列表进行重排序。
        """
        if not chunks:
            return []

        # 1. 提取文本
        texts = [c.chunk.content for c in chunks]
        payload = {"query": query, "texts": texts, "truncate": truncate}

        try:
            async def _send_request(ac: httpx.AsyncClient):
                return await ac.post(
                    self.url, 
                    json=payload, 
                    headers=self.headers,
                    timeout=self.timeout
                )

            if client:
                response = await _send_request(client)
            else:
                async with httpx.AsyncClient() as temp_client:
                    response = await _send_request(temp_client)
            
            response.raise_for_status()
            
            # 2. 处理响应
            return self._process_response(response.json(), chunks, top_n)

        except httpx.HTTPStatusError as e:
            log.error(f"异步重排序 API 返回错误状态码: {e.response.status_code} - {e.response.text}", exc_info=True)
            return []
        except httpx.RequestError as e:
            log.error(f"异步重排序网络连接错误: {e}", exc_info=True)
            return []
        except Exception as e:
            log.error(f"异步重排序发生未预期的错误: {e}", exc_info=True)
            return []

    # ==================== 2. 辅助方法 ====================

    def _process_response(
        self, 
        api_results: List[dict], 
        original_chunks: List[RetrievedChunk], 
        top_n: Optional[int]
    ) -> List[RetrievedChunk]:
        """
        处理 API 返回结果，更新 RetrievedChunk 的分数并排序。
        
        :param api_results: TEI API 返回的 JSON 列表 [{'index': 0, 'score': 0.9}, ...]
        :param original_chunks: 原始的 RetrievedChunk 列表
        :return: 更新分数并排序后的 RetrievedChunk 列表
        """
        # 创建副本以避免修改原始列表的顺序（虽然修改内部对象的属性是引用传递）
        # 建议：如果不想副作用影响外部，应该 deepcopy，但为了性能通常直接修改对象
        processed_chunks = []
        
        try:
            for item in api_results:
                idx = item['index']
                score = item['score']
                
                if 0 <= idx < len(original_chunks):
                    chunk_obj = original_chunks[idx]
                    # [核心修改] 更新 rerank_score
                    chunk_obj.rerank_score = score
                    processed_chunks.append(chunk_obj)
                else:
                    log.warning(f"API返回的索引 {idx} 超出原始列表范围，已忽略。")

            # [核心修改] 根据 rerank_score 进行降序排序
            # 注意：处理可能有部分文档未被 API 返回的情况（虽然少见），未返回的将被丢弃
            processed_chunks.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -1.0, reverse=True)

            if top_n is not None:
                return processed_chunks[:top_n]
            return processed_chunks

        except KeyError as e:
            log.error(f"API 响应格式解析失败，缺失字段: {e}", exc_info=True)
            return []
        except Exception as e:
            log.error(f"处理响应数据时发生未知错误: {e}", exc_info=True)
            return []


class TEIReranker(Reranker):
    """
    TEI 重排器适配层（满足 domain.Reranker 的异步接口）。

    说明：
    - 具体 HTTP 调用复用 `TEIRerankerClient.arerank`；
    - 统一在此处固定 `truncate=True`，避免上下文过长导致 TEI 失败。
    """

    def __init__(self, client: TEIRerankerClient):
        self._client = client

    async def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_n: int = 8
    ) -> List[RetrievedChunk]:
        return await self._client.arerank(
            query=query, chunks=chunks, top_n=top_n, truncate=True
        )
