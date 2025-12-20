"""时间相关工具函数。"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """获取当前 UTC 时间（带时区）。"""

    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | None) -> datetime | None:
    """确保 datetime 带 UTC 时区信息。

    SQLite 在读取 `DateTime(timezone=True)` 时可能返回 naive datetime；
    本函数将其视为 UTC 并补齐 tzinfo，以保证领域模型的时间语义一致。

    Args:
        value: 可能为 naive 或 aware 的 datetime。

    Returns:
        带 UTC tzinfo 的 datetime；若入参为 None 则返回 None。
    """

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
