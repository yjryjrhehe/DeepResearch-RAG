"""上传接口（去重+入队）集成测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.api.server import app
from backend.domain.documents import DocumentCreate, DocumentStatus
from backend.domain.task_queue import TaskQueueService
from backend.infrastructure.db.models import Base
from backend.infrastructure.documents.sqlalchemy_document_repository import (
    SqlAlchemyDocumentRepository,
)
from backend.infrastructure.documents.factory import get_document_repository
from backend.infrastructure.task_queue.factory import get_task_queue_service


class StubTaskQueueService(TaskQueueService):
    """用于测试的任务队列替身。"""

    def __init__(self) -> None:
        self.enqueued_documents: list[str] = []
        self.enqueued_queries: list[str] = []

    async def enqueue_document_ingestion(self, document_id: str) -> str:
        self.enqueued_documents.append(document_id)
        return f"task:{document_id[:8]}"

    async def enqueue_rag_query(self, task_id: str) -> str:
        self.enqueued_queries.append(task_id)
        return f"task:{task_id}"


@pytest_asyncio.fixture
async def session_factory(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    """创建独立的 SQLite 测试库，并返回 Session 工厂。"""

    db_path = tmp_path / "api.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def repo(session_factory: async_sessionmaker[AsyncSession]) -> SqlAlchemyDocumentRepository:
    """提供文档仓储实例。"""

    return SqlAlchemyDocumentRepository(session_factory=session_factory)


@pytest_asyncio.fixture
async def queue() -> StubTaskQueueService:
    """提供队列服务替身。"""

    return StubTaskQueueService()


@pytest_asyncio.fixture(autouse=True)
async def override_dependencies(repo: SqlAlchemyDocumentRepository, queue: StubTaskQueueService, tmp_path: Path):
    """覆盖 FastAPI 依赖，避免外部服务（Redis/OpenSearch/Neo4j）。"""

    app.dependency_overrides[get_document_repository] = lambda: repo
    app.dependency_overrides[get_task_queue_service] = lambda: queue

    # 将上传目录切换到临时目录，避免污染工作区
    import backend.api.server as server_module

    server_module.UPLOAD_DIR = tmp_path / "uploads"
    server_module.TEMP_UPLOAD_DIR = server_module.UPLOAD_DIR / "_tmp"
    server_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server_module.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_new_file_creates_pending_and_enqueues(repo: SqlAlchemyDocumentRepository, queue: StubTaskQueueService) -> None:
    """新上传：创建 PENDING 并投递任务（202）。"""

    content = b"hello"
    document_id = hashlib.sha256(content).hexdigest()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/documents/upload", files={"file": ("a.txt", content)})

    assert resp.status_code == 202
    body = resp.json()
    assert body["duplicated"] is False
    assert body["task_id"] == f"task:{document_id[:8]}"
    assert body["document"]["document_id"] == document_id
    assert body["document"]["status"] == DocumentStatus.PENDING
    assert queue.enqueued_documents == [document_id]

    stored = await repo.get(document_id)
    assert stored is not None
    assert stored.file_path.exists()


@pytest.mark.asyncio
async def test_upload_duplicate_processed_returns_200(repo: SqlAlchemyDocumentRepository, queue: StubTaskQueueService, tmp_path: Path) -> None:
    """秒传：已 PROCESSED 的同内容文件直接返回 200，不重复入队。"""

    content = b"dup"
    document_id = hashlib.sha256(content).hexdigest()

    file_path = tmp_path / "stored.txt"
    file_path.write_bytes(content)

    await repo.create_pending(
        DocumentCreate(
            document_id=document_id,
            file_path=file_path,
            original_filename="stored.txt",
            file_size_bytes=file_path.stat().st_size,
            content_summary=None,
            metadata={},
        )
    )
    await repo.update_status(document_id=document_id, status=DocumentStatus.PROCESSED, chunks_count=1)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/documents/upload", files={"file": ("dup.txt", content)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicated"] is True
    assert body["task_id"] is None
    assert body["document"]["document_id"] == document_id
    assert body["document"]["status"] == DocumentStatus.PROCESSED
    assert queue.enqueued_documents == []

