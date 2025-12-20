"""查询任务仓储工厂方法（SQLAlchemy）。"""

from functools import lru_cache

from ...domain.query_tasks import QueryTaskRepository
from ..db.factory import get_session_factory
from .sqlalchemy_query_task_repository import SqlAlchemyQueryTaskRepository


@lru_cache
def get_query_task_repository() -> QueryTaskRepository:
    """创建并缓存查询任务仓储实例。"""

    return SqlAlchemyQueryTaskRepository(session_factory=get_session_factory())

