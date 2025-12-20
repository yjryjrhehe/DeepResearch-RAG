"""数据库 Engine/Session 工厂与初始化逻辑。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ...core.config import settings
from .models import Base
from .time import utc_now

_METADATA_ADAPTER = TypeAdapter(dict[str, JsonValue])


def ensure_sqlite_parent_dir(database_url: str) -> None:
    """确保 SQLite 数据库文件的父目录存在。

    Args:
        database_url: SQLAlchemy 数据库 URL。
    """
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return
    database_path = url.database
    if database_path is None or database_path == ":memory:":
        return

    path = Path(database_path)
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_async_engine() -> AsyncEngine:
    """创建并缓存异步数据库 Engine。"""

    ensure_sqlite_parent_dir(settings.database.url)
    return create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        pool_pre_ping=True,
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """创建并缓存异步 Session 工厂。"""

    return async_sessionmaker(get_async_engine(), expire_on_commit=False)


async def init_db() -> None:
    """初始化数据库表结构（create_all）。"""

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def dumps_metadata(metadata: dict[str, JsonValue]) -> str:
    """序列化 metadata 为 JSON 字符串。"""

    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def loads_metadata(metadata_json: str) -> dict[str, JsonValue]:
    """反序列化 metadata JSON 字符串为结构化字典。"""

    raw: object
    try:
        raw = json.loads(metadata_json)
    except json.JSONDecodeError:
        return {}
    try:
        return _METADATA_ADAPTER.validate_python(raw)
    except Exception:
        return {}
