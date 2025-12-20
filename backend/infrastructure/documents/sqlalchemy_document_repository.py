"""基于 SQLAlchemy 的文档仓储实现。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...domain.documents import (
    DocumentCreate,
    DocumentNotFoundError,
    DocumentRecord,
    DocumentRepository,
    DocumentStatus,
)
from ..db.factory import dumps_metadata, loads_metadata, utc_now
from ..db.models import DocumentOrm
from ..db.time import ensure_utc


class SqlAlchemyDocumentRepository(DocumentRepository):
    """使用 SQLite + SQLAlchemy 的文档仓储实现。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, document_id: str) -> DocumentRecord | None:
        async with self._session_factory() as session:
            orm_obj = await session.get(DocumentOrm, document_id)
            if orm_obj is None:
                return None
            return _to_domain_document(orm_obj)

    async def create_pending(self, document: DocumentCreate) -> DocumentRecord:
        orm_obj = DocumentOrm(
            document_id=document.document_id,
            status=DocumentStatus.PENDING.value,
            file_path=str(document.file_path),
            original_filename=document.original_filename,
            file_size_bytes=document.file_size_bytes,
            content_summary=document.content_summary,
            created_at=utc_now(),
            updated_at=utc_now(),
            metadata_json=dumps_metadata(document.metadata),
        )
        async with self._session_factory() as session:
            session.add(orm_obj)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ValueError(f"document_id 已存在: {document.document_id}") from exc
            await session.refresh(orm_obj)
            return _to_domain_document(orm_obj)

    async def list(
        self,
        *,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        stmt = select(DocumentOrm).order_by(DocumentOrm.updated_at.desc())
        if status is not None:
            stmt = stmt.where(DocumentOrm.status == status.value)
        stmt = stmt.limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = list(result.scalars().all())
        return [_to_domain_document(r) for r in records]

    async def update_status(
        self,
        *,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        processing_started_at: datetime | None = None,
        processing_finished_at: datetime | None = None,
        chunks_count: int | None = None,
        increment_attempt: bool = False,
    ) -> DocumentRecord:
        async with self._session_factory() as session:
            orm_obj = await session.get(DocumentOrm, document_id)
            if orm_obj is None:
                raise DocumentNotFoundError(document_id)

            orm_obj.status = status.value
            orm_obj.updated_at = utc_now()

            if increment_attempt:
                orm_obj.attempt_count += 1

            if error_message is not None:
                orm_obj.error_message = error_message
            if processing_started_at is not None:
                orm_obj.processing_started_at = processing_started_at
            if processing_finished_at is not None:
                orm_obj.processing_finished_at = processing_finished_at
            if chunks_count is not None:
                orm_obj.chunks_count = chunks_count

            await session.commit()
            await session.refresh(orm_obj)
            return _to_domain_document(orm_obj)

    async def reset_failed_to_pending(self, *, document_id: str) -> DocumentRecord:
        async with self._session_factory() as session:
            orm_obj = await session.get(DocumentOrm, document_id)
            if orm_obj is None:
                raise DocumentNotFoundError(document_id)

            orm_obj.status = DocumentStatus.PENDING.value
            orm_obj.error_message = None
            orm_obj.processing_started_at = None
            orm_obj.processing_finished_at = None
            orm_obj.updated_at = utc_now()

            await session.commit()
            await session.refresh(orm_obj)
            return _to_domain_document(orm_obj)

    async def list_retryable_failed(self, *, limit: int = 100) -> list[DocumentRecord]:
        stmt = (
            select(DocumentOrm)
            .where(DocumentOrm.status == DocumentStatus.FAILED.value)
            .order_by(DocumentOrm.updated_at.desc())
            .limit(limit)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = list(result.scalars().all())
        return [_to_domain_document(r) for r in records]


def _to_domain_document(orm_obj: DocumentOrm) -> DocumentRecord:
    """将 ORM 对象转换为领域模型。"""

    metadata: dict[str, JsonValue] = loads_metadata(orm_obj.metadata_json)
    return DocumentRecord(
        document_id=orm_obj.document_id,
        status=DocumentStatus(orm_obj.status),
        file_path=Path(orm_obj.file_path),
        original_filename=orm_obj.original_filename,
        file_size_bytes=orm_obj.file_size_bytes,
        content_summary=orm_obj.content_summary,
        created_at=ensure_utc(orm_obj.created_at) or utc_now(),
        updated_at=ensure_utc(orm_obj.updated_at) or utc_now(),
        processing_started_at=ensure_utc(orm_obj.processing_started_at),
        processing_finished_at=ensure_utc(orm_obj.processing_finished_at),
        chunks_count=orm_obj.chunks_count,
        error_message=orm_obj.error_message,
        attempt_count=orm_obj.attempt_count,
        metadata=metadata,
    )
