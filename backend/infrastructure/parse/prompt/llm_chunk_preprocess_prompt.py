"""
解析阶段提示词兼容导出。

注意：提示词内容统一由 `backend.core.prompts` 管理，本模块仅做向后兼容导出，
避免外部代码大量改动。
"""

from backend.core.prompts import LLM_CHUNK_PREPROCESS_PROMPT as LLM_PREPROCESS_PROMPT
