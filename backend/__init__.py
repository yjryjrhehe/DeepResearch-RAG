"""
DeepResearch-RAG 后端主包。

本项目采用分层架构：
- domain：抽象接口与领域模型
- infrastructure：外部依赖适配（OpenSearch/Neo4j/Redis/LLM/解析等）
- services：业务编排（摄入/检索/问答）
- api：FastAPI 接口层
- core：配置、日志、提示词等通用能力
"""

