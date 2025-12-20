"""数据库基础设施（SQLAlchemy + SQLite）。

该包负责：
- 异步 Engine/Session 的创建与管理
- ORM 表模型定义（documents/query_tasks）
- 提供初始化表结构的入口（create_all）
"""

from .factory import get_async_engine, get_session_factory, init_db

__all__ = [
    "get_async_engine",
    "get_session_factory",
    "init_db",
]

