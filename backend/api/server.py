"""
FastAPI 服务入口。

接口概览：
- GET  /api/health                健康检查
- POST /api/documents/upload      上传并摄入文档（支持 SSE 流式日志）
- POST /api/query                 图谱+向量检索 + 重排 + LLM 回答（含引用）
- GET  /api/graph/entity/{name}   查询实体子图（调试/可视化）
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiofiles
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..core.logging import setup_logging
from ..domain.models import DocumentSource
from ..domain.interfaces import Ingestor, RAGOrchestrator, GraphRepository
from ..services.factory import get_ingestion_service, get_rag_service
from ..infrastructure.graph.factory import get_graph_repository
from ..infrastructure.repository.factory import get_opensearch_store
from .schemas import HealthResponse, UploadResponse, QueryRequest, QueryResponse


setup_logging()

async def _startup() -> None:
    """
    启动时初始化外部依赖：
    - OpenSearch：连通性检查 + 索引创建（若不存在）
    - Neo4j：保持惰性初始化（首次使用时自动建约束/全文索引）
    """
    store = get_opensearch_store()
    if hasattr(store, "verify_connection"):
        await store.verify_connection()
    if hasattr(store, "create_index"):
        await store.create_index()

    graph_repo = get_graph_repository()
    if hasattr(graph_repo, "verify_connection"):
        await graph_repo.verify_connection()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    - 启动阶段：初始化外部依赖（OpenSearch / Neo4j）
    - 关闭阶段：当前无显式资源回收（依赖进程退出回收）；如后续需要可在此处补充 close()
    """
    await _startup()
    yield


app = FastAPI(title="DeepResearch-RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
MAX_FILE_SIZE_MB = 300
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def _save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """异步保存上传文件，并做大小限制。"""
    file_size = 0
    try:
        async with aiofiles.open(destination, "wb") as out_file:
            while content := await upload_file.read(1024 * 1024):
                file_size += len(content)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"文件过大，超过限制 ({MAX_FILE_SIZE_MB}MB)")
                await out_file.write(content)
    except HTTPException:
        if os.path.exists(destination):
            os.remove(destination)
        raise
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/documents/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    stream: bool = Query(default=True, description="是否以 SSE 流式返回处理日志"),
    ingestion_service: Ingestor = Depends(get_ingestion_service),
):
    """
    上传文件并触发摄入流程。

    - `stream=true`：返回 `text/event-stream`，事件类型为 `log`/`error`；
    - `stream=false`：等待流程结束后返回 JSON。
    """
    safe_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    await _save_upload_file(file, file_path)

    source = DocumentSource(
        file_path=file_path,
        document_name=file.filename,
        document_id=str(uuid.uuid4()),
    )

    if not stream:
        await ingestion_service.pipeline(source)
        return UploadResponse(document_id=source.document_id, document_name=source.document_name)

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        stop_signal = object()

        async def status_callback(message: str) -> None:
            payload = json.dumps({"message": message}, ensure_ascii=False)
            await queue.put(f"event: log\ndata: {payload}\n\n")

        async def run_pipeline() -> None:
            try:
                await status_callback(f"文件: {file.filename} 上传成功，开始解析与入库...")
                await ingestion_service.pipeline(source, status_callback=status_callback)
                await status_callback("摄入流程完成。")
            except Exception as e:
                payload = json.dumps({"error": str(e)}, ensure_ascii=False)
                await queue.put(f"event: error\ndata: {payload}\n\n")
            finally:
                await queue.put(stop_signal)

        task = asyncio.create_task(run_pipeline())

        try:
            while True:
                data = await queue.get()
                if data is stop_signal:
                    break
                yield data
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    rag: RAGOrchestrator = Depends(get_rag_service),
) -> QueryResponse:
    result = await rag.ask(req.query, retrieval_mode=req.retrieval_mode)
    return QueryResponse(**result.model_dump(), request_id=str(uuid.uuid4()))


@app.get("/api/graph/entity/{entity_name}")
async def graph_entity(
    entity_name: str,
    graph_repo: GraphRepository = Depends(get_graph_repository),
):
    return await graph_repo.get_entity_subgraph(entity_name, depth=1)
