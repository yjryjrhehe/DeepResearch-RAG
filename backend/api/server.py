"""FastAPI 服务入口。

提供的核心能力：
- 文档上传：计算 SHA256 去重，创建 PENDING 记录并投递后台任务
- 文档管理：列表/详情/重试失败/手动重处理
- 同步问答：直接调用 RAG 编排服务
- 异步问答（可选）：创建 query_tasks 并投递后台任务，结果暂存 Redis
"""

import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import JsonValue, TypeAdapter
from redis.asyncio import Redis

from ..core.config import settings
from ..core.logging import setup_logging
from ..domain.documents import DocumentCreate, DocumentRepository, DocumentStatus
from ..domain.query_tasks import QueryTaskCreate, QueryTaskRepository, QueryTaskStatus
from ..domain.task_queue import TaskQueueService
from ..domain.interfaces import GraphRepository
from ..infrastructure.db import init_db
from ..infrastructure.db.factory import utc_now
from ..infrastructure.documents.factory import get_document_repository
from ..infrastructure.graph.factory import get_graph_repository
from ..infrastructure.query_tasks.factory import get_query_task_repository
from ..infrastructure.repository.factory import get_opensearch_store
from ..infrastructure.task_queue.broker import broker
from ..infrastructure.task_queue.factory import get_task_queue_service
from ..services.factory import get_rag_service
from .schemas import (
    CreateQueryTaskRequest,
    CreateQueryTaskResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QueryTaskListResponse,
    QueryTaskResultResponse,
    RetryResponse,
    UploadResponse,
)

setup_logging()

UPLOAD_DIR = Path("uploads")
TEMP_UPLOAD_DIR = UPLOAD_DIR / "_tmp"
MAX_FILE_SIZE_MB = 300
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


async def _startup() -> None:
    """应用启动初始化。

    - 初始化 SQLite 表结构
    - 连接/创建 OpenSearch 索引
    - 校验 Neo4j 连接（可选）
    - 启动 Taskiq broker client（用于投递任务）
    """

    await init_db()
    await broker.startup()

    store = get_opensearch_store()
    if hasattr(store, "verify_connection"):
        await store.verify_connection()
    if hasattr(store, "create_index"):
        await store.create_index()

    graph_repo = get_graph_repository()
    if hasattr(graph_repo, "verify_connection"):
        await graph_repo.verify_connection()


async def _shutdown() -> None:
    """应用关闭阶段清理。"""

    await broker.shutdown()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI 生命周期管理。"""

    await _startup()
    try:
        yield
    finally:
        await _shutdown()


app = FastAPI(title="DeepResearch-RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def _save_upload_file_and_calc_sha256(
    upload_file: UploadFile,
    destination: Path,
) -> tuple[str, int]:
    """保存上传文件到指定路径，并计算 SHA256。

    Args:
        upload_file: FastAPI 上传文件对象。
        destination: 保存路径（临时文件）。

    Returns:
        (sha256_hex, file_size_bytes)

    Raises:
        HTTPException: 文件过大或保存失败时抛出。
    """

    sha256 = hashlib.sha256()
    file_size_bytes = 0

    try:
        async with aiofiles.open(destination, "wb") as out_file:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                file_size_bytes += len(chunk)
                if file_size_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，超过限制({MAX_FILE_SIZE_MB}MB)",
                    )
                sha256.update(chunk)
                await out_file.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc

    return sha256.hexdigest(), file_size_bytes


def _safe_original_filename(filename: str | None) -> str:
    """将上传文件名归一化为安全的文件名（防路径穿越）。"""

    if not filename:
        return "upload.bin"
    return Path(filename).name


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查。"""

    return HealthResponse()


@app.post("/api/documents/upload")
async def upload_and_enqueue(
    file: UploadFile = File(...),
    document_repo: DocumentRepository = Depends(get_document_repository),
    queue: TaskQueueService = Depends(get_task_queue_service),
) -> JSONResponse:
    """上传文件并投递后台摄取任务。

    去重逻辑：
    - 读取文件流计算 SHA256（作为 document_id）
    - 若数据库已存在且状态正常，则返回已有记录（秒传/复用）
    - 若为新文件，则落盘、创建 PENDING 记录、投递任务，返回 202
    """

    original_filename = _safe_original_filename(file.filename)
    suffix = Path(original_filename).suffix
    temp_path = TEMP_UPLOAD_DIR / f"{uuid.uuid4().hex}.upload"

    document_id, file_size_bytes = await _save_upload_file_and_calc_sha256(file, temp_path)

    existing = await document_repo.get(document_id)
    if existing is not None:
        temp_path.unlink(missing_ok=True)

        if existing.status == DocumentStatus.PROCESSED:
            payload = UploadResponse(document=existing, duplicated=True)
            return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))

        if existing.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
            payload = UploadResponse(document=existing, duplicated=True)
            return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))

        if existing.status == DocumentStatus.FAILED:
            reset_doc = await document_repo.reset_failed_to_pending(document_id=document_id)
            task_id = await queue.enqueue_document_ingestion(document_id=document_id)
            payload = UploadResponse(document=reset_doc, task_id=task_id, duplicated=True)
            return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))

    final_path = UPLOAD_DIR / f"{document_id}{suffix}"
    if final_path.exists():
        temp_path.unlink(missing_ok=True)
    else:
        os.replace(temp_path, final_path)

    try:
        created = await document_repo.create_pending(
            DocumentCreate(
                document_id=document_id,
                file_path=final_path,
                original_filename=original_filename,
                file_size_bytes=file_size_bytes,
                content_summary=None,
                metadata={},
            )
        )
    except ValueError:
        # 并发场景下可能被其他请求先插入：按“已存在”逻辑返回即可
        created = await document_repo.get(document_id)
        if created is None:
            raise HTTPException(status_code=500, detail="创建文档记录失败：并发插入后仍未找到记录。")

    task_id = await queue.enqueue_document_ingestion(document_id=document_id)
    payload = UploadResponse(document=created, task_id=task_id, duplicated=False)
    return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    status: DocumentStatus | None = Query(default=None, description="按状态过滤（可选）"),
    limit: int = Query(default=100, ge=1, le=500, description="返回数量上限"),
    offset: int = Query(default=0, ge=0, description="分页偏移量"),
    document_repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    """获取文档列表（支持状态过滤）。"""

    items = await document_repo.list_documents(status=status, limit=limit, offset=offset)
    return DocumentListResponse(items=items)


@app.get("/api/documents/{document_id}", response_model=UploadResponse)
async def get_document(
    document_id: str,
    document_repo: DocumentRepository = Depends(get_document_repository),
) -> UploadResponse:
    """获取单个文档详情。"""

    doc = await document_repo.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return UploadResponse(document=doc, duplicated=True)


@app.post("/api/documents/retry_failed", response_model=RetryResponse)
async def retry_failed_documents(
    limit: int = Query(default=100, ge=1, le=1000, description="本次重试的最大数量"),
    document_repo: DocumentRepository = Depends(get_document_repository),
    queue: TaskQueueService = Depends(get_task_queue_service),
) -> RetryResponse:
    """重试所有失败文档（FAILED -> PENDING 并投递任务）。"""

    failed_docs = await document_repo.list_retryable_failed(limit=limit)
    task_ids: list[str] = []
    for doc in failed_docs:
        reset_doc = await document_repo.reset_failed_to_pending(document_id=doc.document_id)
        task_ids.append(await queue.enqueue_document_ingestion(document_id=reset_doc.document_id))
    return RetryResponse(retried_count=len(task_ids), task_ids=task_ids)


@app.post("/api/documents/{document_id}/reprocess", response_model=RetryResponse)
async def reprocess_document(
    document_id: str,
    document_repo: DocumentRepository = Depends(get_document_repository),
    queue: TaskQueueService = Depends(get_task_queue_service),
) -> RetryResponse:
    """手动触发指定文档的重新处理。"""

    doc = await document_repo.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    if doc.status == DocumentStatus.FAILED:
        doc = await document_repo.reset_failed_to_pending(document_id=document_id)
    elif doc.status == DocumentStatus.PROCESSED:
        # 允许对已完成文档重跑：先重置为 PENDING
        doc = await document_repo.update_status(document_id=document_id, status=DocumentStatus.PENDING)
    elif doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="文档正在处理中，无法重处理。")

    task_id = await queue.enqueue_document_ingestion(document_id=document_id)
    return RetryResponse(retried_count=1, task_ids=[task_id])


@app.post("/api/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    query_repo: QueryTaskRepository = Depends(get_query_task_repository),
) -> QueryResponse:
    """即时响应问答接口（立刻执行并持久化记录与结果）。"""

    task_id = str(uuid.uuid4())
    record = await query_repo.create_pending(
        QueryTaskCreate(
            task_id=task_id,
            query=req.query,
            retrieval_mode=req.retrieval_mode.value,
            metadata={},
        )
    )

    await query_repo.update_status(
        task_id=task_id,
        status=QueryTaskStatus.PROCESSING,
        processing_started_at=utc_now(),
    )

    try:
        rag = get_rag_service()
        result = await rag.ask(req.query, retrieval_mode=req.retrieval_mode)

        result_key: str | None = None
        if settings.redis.enabled:
            redis = await _get_redis()
            result_payload = result.model_dump()
            result_key = f"rag:query_task:{task_id}"
            await redis.set(result_key, json.dumps(result_payload, ensure_ascii=False).encode("utf-8"))

        preview = result.answer[:200]
        record = await query_repo.update_status(
            task_id=task_id,
            status=QueryTaskStatus.PROCESSED,
            processing_finished_at=utc_now(),
            result_redis_key=result_key,
            result_preview=preview,
        )

        return QueryResponse(task=record, result=result, request_id=str(uuid.uuid4()))
    except Exception as exc:
        await query_repo.update_status(
            task_id=task_id,
            status=QueryTaskStatus.FAILED,
            error_message=str(exc),
            processing_finished_at=utc_now(),
        )
        raise


@app.post("/api/query/tasks", response_model=CreateQueryTaskResponse)
async def create_query_task(
    req: CreateQueryTaskRequest,
    query_repo: QueryTaskRepository = Depends(get_query_task_repository),
    queue: TaskQueueService = Depends(get_task_queue_service),
) -> JSONResponse:
    """创建异步查询任务并投递后台执行。"""

    task_id = str(uuid.uuid4())
    record = await query_repo.create_pending(
        QueryTaskCreate(
            task_id=task_id,
            query=req.query,
            retrieval_mode=req.retrieval_mode.value,
            metadata={},
        )
    )
    enqueued_id = await queue.enqueue_rag_query(task_id=task_id)
    payload = CreateQueryTaskResponse(task=record, enqueued_task_id=enqueued_id)
    return JSONResponse(status_code=202, content=payload.model_dump(mode="json"))


@app.get("/api/query/tasks", response_model=QueryTaskListResponse)
async def list_query_tasks(
    status: QueryTaskStatus | None = Query(default=None, description="按状态过滤（可选）"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    query_repo: QueryTaskRepository = Depends(get_query_task_repository),
) -> QueryTaskListResponse:
    """获取查询任务列表。"""

    items = await query_repo.list(status=status, limit=limit, offset=offset)
    return QueryTaskListResponse(items=items)


@app.get("/api/query/tasks/{task_id}", response_model=CreateQueryTaskResponse)
async def get_query_task(
    task_id: str,
    query_repo: QueryTaskRepository = Depends(get_query_task_repository),
) -> CreateQueryTaskResponse:
    """获取单个查询任务记录（不含结果）。"""

    record = await query_repo.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="查询任务不存在")
    return CreateQueryTaskResponse(task=record, enqueued_task_id=None)


@app.get("/api/query/tasks/{task_id}/result", response_model=QueryTaskResultResponse)
async def get_query_task_result(
    task_id: str,
    query_repo: QueryTaskRepository = Depends(get_query_task_repository),
) -> QueryTaskResultResponse:
    """获取查询任务结果（从 Redis 读取）。"""

    record = await query_repo.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="查询任务不存在")
    if record.status != QueryTaskStatus.PROCESSED:
        raise HTTPException(status_code=409, detail=f"查询任务未完成: status={record.status}")
    if record.result_redis_key is None:
        raise HTTPException(status_code=404, detail="查询任务结果不存在（缺少 result_redis_key）")
    if not settings.redis.enabled:
        raise HTTPException(status_code=503, detail="Redis 未启用，无法读取查询结果。")

    redis = await _get_redis()
    raw = await redis.get(record.result_redis_key)
    if raw is None:
        raise HTTPException(status_code=404, detail="查询结果已过期或不存在")

    try:
        decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        raw_obj: object = json.loads(decoded)
        result = _JSON_OBJECT_ADAPTER.validate_python(raw_obj)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解析 Redis 结果失败: {exc}") from exc

    return QueryTaskResultResponse(task_id=task_id, status=record.status, result=result)


async def _get_redis() -> Redis:
    """获取 Redis 客户端（仅在需要时初始化）。"""

    if not settings.redis.enabled:
        raise RuntimeError("Redis 未启用")
    # 延迟 import，避免在未启用 Redis 时触发连接依赖。
    from ..infrastructure.cache.factory import get_redis_client

    return get_redis_client()


@app.get("/api/graph/entity/{entity_name}")
async def graph_entity(
    entity_name: str,
    graph_repo: GraphRepository = Depends(get_graph_repository),
):
    """获取实体邻域子图（用于可视化/调试）。"""

    return await graph_repo.get_entity_subgraph(entity_name, depth=1)
