from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# 导入配置
from ...core.config import settings
# 导入自定义的 Reranker Client 类
from .reranker import TEIRerankerClient, TEIReranker

# ==========================================
#  通用构建辅助函数 (核心解耦逻辑)
# ==========================================
def _create_chat_llm(config_name: str, temperature: float = 0, max_retries: int = 3) -> ChatOpenAI:
    """
    私有辅助函数：根据配置名称动态创建 ChatOpenAI 实例。
    
    Args:
        config_name: 对应 config.py 中 get_llm_config_by_name 支持的名称 
                     (e.g., "rewrite", "research", "preprocess")
        temperature: 模型温度
        max_retries: 最大重试次数
    """
    # 1. 动态获取配置对象 (类型为 LLMProviderConfig)
    config = settings.get_llm_config_by_name(config_name)
    
    # 2. 统一实例化
    return ChatOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=temperature,
        max_retries=max_retries
    )

# ==========================================
# 1. 预处理 LLM (Preprocessing LLM)
# ==========================================
@lru_cache()
def get_preprocessing_llm() -> ChatOpenAI:
    """
    获取用于文本分块预处理的 LLM 客户端单例。
    """
    return _create_chat_llm("preprocess", temperature=0)

# ==========================================
# 2. Embedding 模型 (Embedding Model)
# ==========================================
@lru_cache()
def get_embedding_model() -> OllamaEmbeddings:
    """
    获取 Embedding 模型客户端单例。
    """
    # 1. 获取 Embedding 专用配置
    config = settings.get_llm_config_by_name("embedding")
    
    # 2. 实例化 OllamaEmbeddings
    return OllamaEmbeddings(
        base_url=config.base_url,
        model=config.model
    )

# ==========================================
# 3. 查询重写 LLM (Rewrite LLM)
# ==========================================
@lru_cache()
def get_rewrite_llm() -> ChatOpenAI:
    """
    获取用于 Query Rewrite 的 LLM 客户端单例。
    """
    return _create_chat_llm("rewrite", temperature=0)

# ==========================================
# 4. research LLM
# ==========================================
@lru_cache()
def get_research_llm() -> ChatOpenAI:
    """
    获取用于 research 的 LLM 客户端单例。
    """
    return _create_chat_llm("research", temperature=0)

# ==========================================
# 5. Answer LLM（回答生成）
# ==========================================
@lru_cache()
def get_answer_llm() -> ChatOpenAI:
    """
    获取用于回答生成的 LLM 客户端单例。

    说明：
    - 为了兼容既有配置，默认复用 `RESEARCH_LLM_*` 配置段；
    - 如需拆分，可在 `core/config.py` 中新增 answer_llm 配置并在此处替换。
    """
    return get_research_llm()


# ==========================================
# 6. Reranker（TEI）
# ==========================================
@lru_cache()
def get_rerank_client() -> TEIRerankerClient:
    """
    获取 TEI Reranker 客户端单例。
    注：Reranker 配置结构特殊（包含 timeout 等），且通常不视为标准 LLM，
    因此这里直接访问 settings.tei_rerank 。
    """
    return TEIRerankerClient(
        base_url=settings.tei_rerank.base_url,
        api_key=settings.tei_rerank.api_key,
        timeout=settings.tei_rerank.timeout,
        max_concurrency=settings.tei_rerank.max_concurrency
    )


@lru_cache()
def get_reranker() -> TEIReranker:
    """返回实现了 domain.Reranker 接口的重排器。"""
    return TEIReranker(client=get_rerank_client())
