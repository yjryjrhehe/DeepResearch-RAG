"""任务队列服务工厂方法。"""

from functools import lru_cache

from ...domain.task_queue import TaskQueueService
from .taskiq_task_queue_service import TaskiqTaskQueueService


@lru_cache
def get_task_queue_service() -> TaskQueueService:
    """创建并缓存任务队列服务实例。"""

    return TaskiqTaskQueueService()

