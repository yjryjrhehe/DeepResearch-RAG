"""后台任务定义（Taskiq）。

说明：
- 任务函数应尽量保持“薄”，把业务逻辑委托给 services 层；
- 任务函数负责状态流转、异常捕获与错误信息写回，确保可观测性与可重试性。
"""

import json
import logging

from pydantic import JsonValue

from ..core.config import settings
from ..domain.documents import DocumentNotFoundError, DocumentStatus
from ..domain.models import DocumentSource, RetrievalMode
from ..domain.query_tasks import QueryTaskNotFoundError, QueryTaskStatus
from ..infrastructure.cache.factory import get_redis_client
from ..infrastructure.db.factory import utc_now
from ..infrastructure.documents.factory import get_document_repository
from ..infrastructure.query_tasks.factory import get_query_task_repository
from ..infrastructure.task_queue.broker import broker
from ..services.factory import get_ingestion_service, get_rag_service
from ..core.logging import setup_logging

log = logging.getLogger(__name__)
setup_logging()


@broker.task
async def process_document_ingestion(document_id: str) -> None:
    """处理文档摄取任务。

    逻辑：
    1) 更新 documents 状态为 PROCESSING
    2) 从数据库加载文件路径，调用摄取服务
    3) 成功：更新为 PROCESSED 并记录统计信息
    4) 失败：捕获异常、记录错误信息并更新为 FAILED

    Args:
        document_id: 文档 ID（SHA256 十六进制字符串）。

    Raises:
        DocumentNotFoundError: 文档记录不存在时抛出（便于排查）。
    """

    document_repo = get_document_repository()
    doc = await document_repo.get(document_id)
    if doc is None:
        raise DocumentNotFoundError(document_id)

    await document_repo.update_status(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        processing_started_at=utc_now(),
        increment_attempt=True,
    )

    try:
        ingestion_service = get_ingestion_service()
        source = DocumentSource(
            document_id=document_id,
            file_path=doc.file_path,
            document_name=doc.original_filename,
            metadata=doc.metadata,
        )
        stats = await ingestion_service.pipeline(source)

        await document_repo.update_status(
            document_id=document_id,
            status=DocumentStatus.PROCESSED,
            processing_finished_at=utc_now(),
            chunks_count=stats.stored_chunks_count,
        )
    except Exception as exc:
        log.error("文档处理失败 document_id=%s: %s", document_id, exc, exc_info=True)
        await document_repo.update_status(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            error_message=str(exc),
            processing_finished_at=utc_now(),
        )
        raise


@broker.task
async def process_rag_query(task_id: str) -> None:
    """处理耗时 RAG 查询任务（结果暂存至 Redis）。

    Args:
        task_id: 查询任务 ID。

    Raises:
        QueryTaskNotFoundError: 任务记录不存在时抛出。
        RuntimeError: Redis 未启用且无法暂存结果时抛出。
    """

    query_repo = get_query_task_repository()
    task = await query_repo.get(task_id)
    if task is None:
        raise QueryTaskNotFoundError(task_id)

    await query_repo.update_status(
        task_id=task_id,
        status=QueryTaskStatus.PROCESSING,
        processing_started_at=utc_now(),
    )

    try:
        retrieval_mode = RetrievalMode(task.retrieval_mode)
        rag_service = get_rag_service()
        result = await rag_service.ask(task.query, retrieval_mode=retrieval_mode)

        if not settings.redis.enabled:
            raise RuntimeError("Redis 未启用，无法暂存查询结果。")
        redis = get_redis_client()

        payload: dict[str, JsonValue] = result.model_dump()
        result_key = f"rag:query_task:{task_id}"
        await redis.set(result_key, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        preview = result.answer[:200]
        await query_repo.update_status(
            task_id=task_id,
            status=QueryTaskStatus.PROCESSED,
            processing_finished_at=utc_now(),
            result_redis_key=result_key,
            result_preview=preview,
        )
    except Exception as exc:
        log.error("查询任务失败 task_id=%s: %s", task_id, exc, exc_info=True)
        await query_repo.update_status(
            task_id=task_id,
            status=QueryTaskStatus.FAILED,
            error_message=str(exc),
            processing_finished_at=utc_now(),
        )
        raise
