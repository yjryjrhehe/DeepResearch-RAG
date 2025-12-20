"""Taskiq 任务队列服务实现。"""

from __future__ import annotations

from ...domain.task_queue import TaskQueueService
from ...worker.tasks import process_document_ingestion, process_rag_query


class TaskiqTaskQueueService(TaskQueueService):
    """基于 Taskiq 的任务投递实现。"""

    async def enqueue_document_ingestion(self, document_id: str) -> str:
        task = await process_document_ingestion.kiq(document_id=document_id)
        return task.task_id

    async def enqueue_rag_query(self, task_id: str) -> str:
        task = await process_rag_query.kiq(task_id=task_id)
        return task.task_id

