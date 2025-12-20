"""文档仓储工厂方法（SQLAlchemy）。"""

from __future__ import annotations

from functools import lru_cache

from ...domain.documents import DocumentRepository
from ..db.factory import get_session_factory
from .sqlalchemy_document_repository import SqlAlchemyDocumentRepository


@lru_cache
def get_document_repository() -> DocumentRepository:
    """创建并缓存文档仓储实例。"""

    return SqlAlchemyDocumentRepository(session_factory=get_session_factory())

