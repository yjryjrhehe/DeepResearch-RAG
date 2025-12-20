"""基于 SQLAlchemy 的查询任务仓储实现。"""

from datetime import datetime

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...domain.query_tasks import (
    QueryTaskCreate,
    QueryTaskNotFoundError,
    QueryTaskRecord,
    QueryTaskRepository,
    QueryTaskStatus,
)
from ..db.factory import dumps_metadata, loads_metadata, utc_now
from ..db.models import QueryTaskOrm
from ..db.time import ensure_utc


class SqlAlchemyQueryTaskRepository(QueryTaskRepository):
    """使用 SQLite + SQLAlchemy 的查询任务仓储实现。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """初始化仓储。

        Args:
            session_factory: 已配置好的异步 Session 工厂。
        """
        self._session_factory = session_factory

    async def get(self, task_id: str) -> QueryTaskRecord | None:
        """获取单个查询任务记录。"""
        async with self._session_factory() as session:
            orm_obj = await session.get(QueryTaskOrm, task_id)
            if orm_obj is None:
                return None
            return _to_domain_query_task(orm_obj)

    async def create_pending(self, task: QueryTaskCreate) -> QueryTaskRecord:
        """创建一条 PENDING 状态的查询任务记录。"""
        orm_obj = QueryTaskOrm(
            task_id=task.task_id,
            status=QueryTaskStatus.PENDING.value,
            query=task.query,
            retrieval_mode=task.retrieval_mode,
            created_at=utc_now(),
            updated_at=utc_now(),
            metadata_json=dumps_metadata(task.metadata),
        )
        async with self._session_factory() as session:
            session.add(orm_obj)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ValueError(f"task_id 已存在 {task.task_id}") from exc
            await session.refresh(orm_obj)
            return _to_domain_query_task(orm_obj)

    async def list(
        self,
        *,
        status: QueryTaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QueryTaskRecord]:
        """查询任务列表，支持按状态过滤。"""
        stmt = select(QueryTaskOrm).order_by(QueryTaskOrm.updated_at.desc())
        if status is not None:
            stmt = stmt.where(QueryTaskOrm.status == status.value)
        stmt = stmt.limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            records = list(result.scalars().all())
        return [_to_domain_query_task(r) for r in records]

    async def update_status(
        self,
        *,
        task_id: str,
        status: QueryTaskStatus,
        error_message: str | None = None,
        processing_started_at: datetime | None = None,
        processing_finished_at: datetime | None = None,
        result_redis_key: str | None = None,
        result_preview: str | None = None,
    ) -> QueryTaskRecord:
        """更新任务状态与结果引用。"""
        async with self._session_factory() as session:
            orm_obj = await session.get(QueryTaskOrm, task_id)
            if orm_obj is None:
                raise QueryTaskNotFoundError(task_id)

            orm_obj.status = status.value
            orm_obj.updated_at = utc_now()

            if error_message is not None:
                orm_obj.error_message = error_message
            if processing_started_at is not None:
                orm_obj.processing_started_at = processing_started_at
            if processing_finished_at is not None:
                orm_obj.processing_finished_at = processing_finished_at
            if result_redis_key is not None:
                orm_obj.result_redis_key = result_redis_key
            if result_preview is not None:
                orm_obj.result_preview = result_preview

            await session.commit()
            await session.refresh(orm_obj)
            return _to_domain_query_task(orm_obj)


def _to_domain_query_task(orm_obj: QueryTaskOrm) -> QueryTaskRecord:
    """将 ORM 对象转换为领域模型。"""

    metadata: dict[str, JsonValue] = loads_metadata(orm_obj.metadata_json)
    return QueryTaskRecord(
        task_id=orm_obj.task_id,
        status=QueryTaskStatus(orm_obj.status),
        query=orm_obj.query,
        retrieval_mode=orm_obj.retrieval_mode,
        result_redis_key=orm_obj.result_redis_key,
        result_preview=orm_obj.result_preview,
        error_message=orm_obj.error_message,
        created_at=ensure_utc(orm_obj.created_at) or utc_now(),
        updated_at=ensure_utc(orm_obj.updated_at) or utc_now(),
        processing_started_at=ensure_utc(orm_obj.processing_started_at),
        processing_finished_at=ensure_utc(orm_obj.processing_finished_at),
        metadata=metadata,
    )
