"""文档仓储（SQLite/SQLAlchemy）单元测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.domain.documents import DocumentCreate, DocumentStatus
from backend.infrastructure.db.models import Base
from backend.infrastructure.documents.sqlalchemy_document_repository import (
    SqlAlchemyDocumentRepository,
)


@pytest_asyncio.fixture
async def session_factory(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    """创建独立的 SQLite 测试库，并返回 Session 工厂。"""

    db_path = tmp_path / "test.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_get_update_and_reset(session_factory: async_sessionmaker[AsyncSession], tmp_path: Path) -> None:
    """覆盖：创建、查询、状态更新、失败重置。"""

    repo = SqlAlchemyDocumentRepository(session_factory=session_factory)

    document_id = "a" * 64
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"dummy")

    created = await repo.create_pending(
        DocumentCreate(
            document_id=document_id,
            file_path=file_path,
            original_filename="doc.pdf",
            file_size_bytes=file_path.stat().st_size,
            content_summary=None,
            metadata={"ext": ".pdf"},
        )
    )
    assert created.document_id == document_id
    assert created.status == DocumentStatus.PENDING

    fetched = await repo.get(document_id)
    assert fetched is not None
    assert fetched.document_id == document_id

    started_at = datetime.now(timezone.utc)
    updated = await repo.update_status(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        processing_started_at=started_at,
        increment_attempt=True,
    )
    assert updated.status == DocumentStatus.PROCESSING
    assert updated.attempt_count == 1
    assert updated.processing_started_at == started_at

    failed = await repo.update_status(
        document_id=document_id,
        status=DocumentStatus.FAILED,
        error_message="boom",
    )
    assert failed.status == DocumentStatus.FAILED
    assert failed.error_message == "boom"

    reset = await repo.reset_failed_to_pending(document_id=document_id)
    assert reset.status == DocumentStatus.PENDING
    assert reset.error_message is None


@pytest.mark.asyncio
async def test_list_retryable_failed(session_factory: async_sessionmaker[AsyncSession], tmp_path: Path) -> None:
    """覆盖：失败任务查询。"""

    repo = SqlAlchemyDocumentRepository(session_factory=session_factory)

    ok_id = "b" * 64
    failed_id = "c" * 64

    ok_path = tmp_path / "ok.txt"
    ok_path.write_text("ok", encoding="utf-8")
    failed_path = tmp_path / "failed.txt"
    failed_path.write_text("failed", encoding="utf-8")

    await repo.create_pending(
        DocumentCreate(
            document_id=ok_id,
            file_path=ok_path,
            original_filename="ok.txt",
            file_size_bytes=ok_path.stat().st_size,
            content_summary=None,
            metadata={},
        )
    )

    await repo.create_pending(
        DocumentCreate(
            document_id=failed_id,
            file_path=failed_path,
            original_filename="failed.txt",
            file_size_bytes=failed_path.stat().st_size,
            content_summary=None,
            metadata={},
        )
    )
    await repo.update_status(
        document_id=failed_id,
        status=DocumentStatus.FAILED,
        error_message="x",
    )

    failed_list = await repo.list_retryable_failed(limit=10)
    assert [d.document_id for d in failed_list] == [failed_id]

