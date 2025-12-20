"""Taskiq Broker 配置。

该 Broker 模块需要同时被：
- FastAPI 应用（作为 client 投递任务）
- Taskiq Worker 进程（作为 worker 消费任务）

因此应避免在 import 阶段做重量级初始化，仅做“可重复加载”的对象构建与事件注册。
"""

from taskiq import TaskiqEvents
from taskiq.state import TaskiqState
from taskiq_redis import RedisStreamBroker

from ...core.config import settings
from ..db import init_db


broker = RedisStreamBroker(
    url=settings.taskiq.broker_url,
    queue_name=settings.taskiq.queue_name,
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP, TaskiqEvents.CLIENT_STARTUP)
async def _init_resources(_: TaskiqState) -> None:
    """在 Worker/Client 启动阶段初始化必要资源。"""

    await init_db()

