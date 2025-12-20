"""任务队列抽象接口。

Domain 层以抽象接口形式定义“投递后台任务”的能力，避免 API/Service 直接依赖具体队列实现。基础设施层可用 Taskiq + Redis 等实现该接口。
"""

from abc import ABC, abstractmethod


class TaskQueueService(ABC):
    """任务队列服务抽象接口。"""

    @abstractmethod
    async def enqueue_document_ingestion(self, document_id: str) -> str:
        """投递“文档摄取”任务。

        Args:
            document_id: 文档 ID（SHA256 十六进制字符串）。

        Returns:
            任务 ID（由队列系统生成）。
        """

    @abstractmethod
    async def enqueue_rag_query(self, task_id: str) -> str:
        """投递“RAG 查询”任务（可选）。

        Args:
            task_id: 查询任务 ID（UUID）。

        Returns:
            任务 ID（由队列系统生成）。
        """
