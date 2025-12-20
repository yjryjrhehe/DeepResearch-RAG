"""
基于 LLM 的图谱相关抽取器实现。

包含：
- 关键词抽取（用于 Neo4j 全文检索的 query 构造）
- 文本块实体/关系抽取（用于知识图谱增量更新）
"""

import re
import logging
from typing import List, Tuple
import hashlib
import json

import json_repair
from langchain_core.language_models import BaseChatModel
from redis.asyncio import Redis

from ...core.prompts import (
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    KG_EXTRACT_PROMPT,
    KG_KEYWORDS_PROMPT,
)
from ...domain.interfaces import GraphExtractor, KeywordExtractor
from ...domain.models import DocumentChunk, GraphEntity, GraphRelation

log = logging.getLogger(__name__)

_RE_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
_RE_WHITESPACE = re.compile(r"\s+")


def _normalize_name(text: str) -> str:
    """
    规范化实体/关系端点名称（尽量提升实体对齐与关系落库成功率）。

    处理策略：
    - 去除首尾空白；
    - 去除“中文字符之间”的空白（例如：`板 载系统` -> `板载系统`）；
    - 其余连续空白折叠为单个空格（保留英文实体中的空格，例如：`D* Lite`）。
    """
    value = (text or "").strip()
    if not value:
        return ""
    value = _RE_CJK_SPACE.sub("", value)
    value = _RE_WHITESPACE.sub(" ", value)
    return value


def _clean_extraction_line(raw_line: str) -> str:
    """
    清理 LLM 输出行，尽量提升解析鲁棒性。

    说明：
    - 允许模型在行首输出项目符号（例如 `- `、`* `）；
    - 允许模型用反引号包裹整行；
    - 其余情况保持原样。
    """
    line = (raw_line or "").strip()
    if not line:
        return ""
    for prefix in ("- ", "* ", "• "):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
            break
    if line.startswith("`") and line.endswith("`") and DEFAULT_TUPLE_DELIMITER in line:
        line = line.strip("`").strip()
    return line


def _parse_tuple_delimited_extraction(
    content: str,
) -> tuple[list[GraphEntity], list[GraphRelation], bool]:
    """
    解析 LightRAG 风格的“行 + 分隔符”输出。

    支持的格式：
    - 实体行：entity<|#|>name<|#|>type<|#|>description
    - 关系行：relation<|#|>source<|#|>target<|#|>kw1,kw2<|#|>description
    - 完成行：<|COMPLETE|>
    """
    entities: list[GraphEntity] = []
    relations: list[GraphRelation] = []
    complete = False

    for raw_line in (content or "").splitlines():
        line = _clean_extraction_line(raw_line)
        if not line:
            continue

        if DEFAULT_COMPLETION_DELIMITER in line:
            complete = True
            if line.strip() == DEFAULT_COMPLETION_DELIMITER:
                break
            line = line.replace(DEFAULT_COMPLETION_DELIMITER, "").strip()
            if not line:
                break

        if DEFAULT_TUPLE_DELIMITER not in line:
            continue

        parts = [p.strip() for p in line.split(DEFAULT_TUPLE_DELIMITER)]
        if not parts:
            continue

        kind = parts[0].strip().lower()
        if kind == "entity":
            if len(parts) < 4:
                continue
            name = _normalize_name(parts[1])
            if not name:
                continue
            entity_type = (parts[2] or "Other").strip() or "Other"
            description = (parts[3] or "").strip()
            entities.append(GraphEntity(name=name, type=entity_type, description=description))
            continue

        if kind == "relation":
            if len(parts) < 5:
                continue
            source = _normalize_name(parts[1])
            target = _normalize_name(parts[2])
            if not source or not target or source == target:
                continue
            keywords_raw = (parts[3] or "").strip()
            keywords = [
                _normalize_name(k)
                for k in keywords_raw.split(",")
                if _normalize_name(k)
            ]
            description = (parts[4] or "").strip()
            relations.append(
                GraphRelation(
                    source=source,
                    target=target,
                    keywords=keywords,
                    description=description,
                    weight=1.0,
                )
            )
            continue

    return entities, relations, complete


class LLMKeywordExtractor(KeywordExtractor):
    """使用 LLM 从查询中抽取图谱检索关键词。"""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        redis: Redis | None = None,
        cache_ttl_seconds: int = 0,
        model_name: str = "",
    ):
        self._llm = llm
        self._redis = redis
        self._cache_ttl_seconds = cache_ttl_seconds
        self._model_name = model_name

    def _make_cache_key(self, query: str) -> str:
        digest = hashlib.sha256((query or "").strip().encode("utf-8")).hexdigest()
        return f"kgkw:{self._model_name}:{digest}"

    async def extract(self, query: str) -> Tuple[List[str], List[str]]:
        """
        关键词抽取，异常时返回空结果而不阻断流程。
        """
        # 先从缓存中查找是否存在已经对该query提取过关键词
        if self._redis and self._cache_ttl_seconds > 0 and query.strip():
            try:
                cache_key = self._make_cache_key(query)
                cached = await self._redis.get(cache_key)
                if cached:
                    if isinstance(cached, (bytes, bytearray)):
                        cached = cached.decode("utf-8")
                    data = json.loads(cached)
                    if isinstance(data, dict):
                        high = data.get("high", [])
                        low = data.get("low", [])
                        high = [
                            _normalize_name(str(x))
                            for x in high
                            if _normalize_name(str(x))
                        ]
                        low = [
                            _normalize_name(str(x))
                            for x in low
                            if _normalize_name(str(x))
                        ]
                        return high, low
            except Exception:
                pass

        prompt = KG_KEYWORDS_PROMPT.format(query=query)
        try:
            result = await self._llm.ainvoke(prompt)
            data = json_repair.loads(getattr(result, "content", "") or "")
        except Exception as exc:  # 保底兜底，防止任务失败
            log.error("关键词抽取失败: %s", exc, exc_info=True)
            return [], []

        high = data.get("high_level_keywords") if isinstance(data, dict) else None
        low = data.get("low_level_keywords") if isinstance(data, dict) else None

        high = [_normalize_name(str(x)) for x in (high or []) if _normalize_name(str(x))]
        low = [_normalize_name(str(x)) for x in (low or []) if _normalize_name(str(x))]

        if self._redis and self._cache_ttl_seconds > 0 and query and query.strip():
            try:
                cache_key = self._make_cache_key(query)
                payload = json.dumps(
                    {"high": high, "low": low},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                await self._redis.set(name=cache_key, value=payload, ex=self._cache_ttl_seconds)
            except Exception:
                pass
        return high, low


class LLMGraphExtractor(GraphExtractor):
    """使用 LLM 从文本块中抽取实体与关系。"""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    async def extract(self, chunk: DocumentChunk) -> tuple[List[GraphEntity], List[GraphRelation]]:
        parent_title = " > ".join(chunk.parent_headings) if chunk.parent_headings else ""
        prompt = KG_EXTRACT_PROMPT.format(
            document_name=chunk.document_name,
            parent_title=parent_title,
            chunk_text=chunk.content,
        )
        try:
            result = await self._llm.ainvoke(prompt)
        except Exception as exc:  # 保底兜底，避免任务失败导致整体流程中断
            log.error("图谱抽取失败 (chunk_id=%s): %s", chunk.chunk_id, exc, exc_info=True)
            return [], []

        content = getattr(result, "content", "") or ""

        # 1) 解析 LightRAG 风格输出
        entities, relations, _ = _parse_tuple_delimited_extraction(content)

        # 2) 兜底：如果关系端点没有被实体覆盖，补齐缺失实体，提升落库成功率
        entity_names = {e.name for e in entities if e.name}
        for r in relations:
            if r.source and r.source not in entity_names:
                entities.append(GraphEntity(name=r.source, type="Other", description=""))
                entity_names.add(r.source)
            if r.target and r.target not in entity_names:
                entities.append(GraphEntity(name=r.target, type="Other", description=""))
                entity_names.add(r.target)

        return entities, relations
