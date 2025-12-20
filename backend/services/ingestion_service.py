"""文档摄取服务（业务编排）。

该服务负责把“解析 -> 切分 -> 预处理 -> 入库 -> 图谱增量更新”等步骤串起来，
并以可复用的方法拆分关键阶段，便于被后台任务（Worker）调用。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ..core.logging import setup_logging
from ..domain.interfaces import (
    DocumentParser,
    GraphExtractor,
    GraphRepository,
    Ingestor,
    PreProcessor,
    SearchRepository,
    TextSplitter,
)
from ..domain.models import DocumentChunk, DocumentSource, IngestionStats

setup_logging()
log = logging.getLogger(__name__)


class IngestionService(Ingestor):
    """文档摄取编排服务实现。"""

    BATCH_SIZE = 50

    def __init__(
        self,
        *,
        parser: DocumentParser,
        splitter: TextSplitter,
        preprocessor: PreProcessor,
        store: SearchRepository,
        graph_extractor: GraphExtractor | None = None,
        graph_repo: GraphRepository | None = None,
        kg_max_concurrency: int = 3,
    ) -> None:
        """初始化摄取服务。

        Args:
            parser: 文档解析器。
            splitter: 文本切分器。
            preprocessor: 文档块预处理器。
            store: 检索仓储（用于 chunks 写入/索引）。
            graph_extractor: 可选；图谱抽取器。
            graph_repo: 可选；图谱仓储。
            kg_max_concurrency: 图谱增量更新的最大并发数。
        """

        self._parser = parser
        self._splitter = splitter
        self._preprocessor = preprocessor
        self._store = store
        self._graph_extractor = graph_extractor
        self._graph_repo = graph_repo
        self._kg_max_concurrency = max(1, int(kg_max_concurrency or 1))

    async def pipeline(
        self,
        source: DocumentSource,
        status_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> IngestionStats:
        """执行完整的文档摄取流水线。

        Args:
            source: 文档来源。
            status_callback: 可选；用于上报处理日志的回调（例如 SSE/日志聚合）。

        Returns:
            本次摄取的统计信息。

        Raises:
            FileNotFoundError: 源文件不存在。
            ValueError: 解析/切分/入库结果为空时抛出。
            Exception: 处理过程中出现的其他未预期异常。
        """

        await self._emit(f"--- [开始处理] 文档: {source.document_name} ---", status_callback)

        markdown = await self.parse_to_markdown(source, status_callback=status_callback)
        initial_chunks = await self.split_markdown(markdown, source, status_callback=status_callback)
        return await self.preprocess_and_store(initial_chunks, status_callback=status_callback)

    async def parse_to_markdown(
        self,
        source: DocumentSource,
        *,
        status_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """解析文档为 Markdown。"""

        await self._emit("步骤 1/4: 正在解析文档...", status_callback)
        markdown = await self._parser.parse(source)
        if not markdown.strip():
            raise ValueError("解析结果为空，无法继续处理。")
        await self._emit(f"步骤 1/4: 解析成功，内容长度={len(markdown)}", status_callback)
        return markdown

    async def split_markdown(
        self,
        markdown: str,
        source: DocumentSource,
        *,
        status_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[DocumentChunk]:
        """将 Markdown 切分为初始 chunks。"""

        await self._emit("步骤 2/4: 正在切分文本...", status_callback)
        chunks = await self._splitter.split(markdown, source)
        if not chunks:
            raise ValueError("切分结果为空，无法继续处理。")
        await self._emit(f"步骤 2/4: 切分成功，chunks={len(chunks)}", status_callback)
        return chunks

    async def preprocess_and_store(
        self,
        initial_chunks: list[DocumentChunk],
        *,
        status_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> IngestionStats:
        """并发预处理并批量写入检索仓储，同时可选更新知识图谱。"""

        await self._emit(
            f"步骤 3-4/4: 并发预处理并分批写入 (batch_size={self.BATCH_SIZE})...",
            status_callback,
        )

        processed_buffer: list[DocumentChunk] = []
        stored_chunks_count = 0

        kg_tasks: list[asyncio.Task[None]] = []
        kg_semaphore = asyncio.Semaphore(self._kg_max_concurrency)

        async def _upsert_kg(enriched_chunk: DocumentChunk) -> None:
            if not self._graph_extractor or not self._graph_repo:
                return
            async with kg_semaphore:
                entities, relations = await self._graph_extractor.extract(enriched_chunk)
                await asyncio.gather(
                    self._graph_repo.upsert_chunk_knowledge(enriched_chunk, entities, relations),
                    self._store.index_graph_entities_relations(enriched_chunk, entities, relations),
                )

        async for enriched_chunk in self._preprocessor.run_concurrent_preprocessing(initial_chunks):
            processed_buffer.append(enriched_chunk)

            if self._graph_extractor and self._graph_repo:
                kg_tasks.append(asyncio.create_task(_upsert_kg(enriched_chunk)))

            if len(processed_buffer) >= self.BATCH_SIZE:
                await self._store.bulk_add_documents(processed_buffer)
                stored_chunks_count += len(processed_buffer)
                await self._emit(
                    f"  -> 已写入 {len(processed_buffer)} chunks (累计: {stored_chunks_count})",
                    status_callback,
                )
                processed_buffer.clear()

        if processed_buffer:
            await self._store.bulk_add_documents(processed_buffer)
            stored_chunks_count += len(processed_buffer)
            await self._emit(
                f"  -> 已写入剩余 {len(processed_buffer)} chunks (累计: {stored_chunks_count})",
                status_callback,
            )

        if stored_chunks_count == 0:
            raise ValueError("入库结果为空：未写入任何 chunks。")

        if kg_tasks:
            await self._emit(
                f"步骤 4/4: 等待知识图谱增量更新完成 (tasks={len(kg_tasks)})...",
                status_callback,
            )
            results = await asyncio.gather(*kg_tasks, return_exceptions=True)
            failed = [r for r in results if isinstance(r, Exception)]
            if failed:
                await self._emit_error(f"知识图谱更新存在失败任务数: {len(failed)}", status_callback)
            else:
                await self._emit("知识图谱增量更新完成。", status_callback)

        await self._emit(
            f"处理完成：stored_chunks={stored_chunks_count}, kg_tasks={len(kg_tasks)}",
            status_callback,
        )

        return IngestionStats(
            initial_chunks_count=len(initial_chunks),
            stored_chunks_count=stored_chunks_count,
            kg_tasks_count=len(kg_tasks),
        )

    async def _emit(
        self,
        message: str,
        status_callback: Callable[[str], Awaitable[None]] | None,
    ) -> None:
        """输出日志并可选回调到上层。"""

        log.info(message)
        if status_callback is not None:
            await status_callback(message)

    async def _emit_error(
        self,
        message: str,
        status_callback: Callable[[str], Awaitable[None]] | None,
    ) -> None:
        """输出错误日志并可选回调到上层。"""

        log.error(message)
        if status_callback is not None:
            await status_callback(message)

