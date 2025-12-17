# 项目功能：

1. 实现上传文档，自动解析文档，存入向量库并构建知识图谱（支持动态更新），分层检索（同时使用知识图谱检索：参考lightrag实现；以及向量检索：参考backend实现），重排序，基于检索结果使用llm回答问题。
2. 具备缓存机制：采用Redis将query的embedding结果缓存（参考lightrag实现）
3. 最终实现多个api接口（fastapi）

# 项目架构：

1. 采用分层架构，区分抽象类抽象接口、infrastructure层、service层、core层和api层

## 文档解析：
1. 保留backend实现的引接文档、基于llm/vlm提取图表公式、文本分块、建立向量数据库；
2. 保留lightrag项目中：使用llm提取文档块信息、使用neo4j构建知识图谱

## 检索机制：

参考lightrag实现方式，将其中向量数据库的检索替换为backend中实现的方法。

## 重排序：

采用backend的实现方式（基于TEI引擎，本地部署qwen embedding模型）

## 回答生成：

参考lightrag实现，务必保留其对于“回答+参考引用”的实现方法。

## 缓存机制：

参考lightrag实现方式。

# 注意事项：

1. 使用标准python注释和docstring编写格式（使用中文）；
2. 所有提示词都使用中文，并建立单独模块管理提示词；
3. 建立统一日志管理
4. 所有方法、函数都应当是异步的；
5. 你需要为每一个模块编写markdown文档，说明你编写的目的和实现方法。
6. 你的编程风格应该是pythonic的，简洁优雅。

7. 删除原项目库中任何你觉得不需要的代码或文件或文件夹。
8. 给出具体部署方案和配置说明。



------

# 修改：

1. 通过llm抽取的文本块中的关系和实体应该分别在opensearch中建立索引。
   1. 对于关系：抽取出之后将其内容拼接为：“source_entity、target_entity、relationship_keywords、relationship_description” 格式存入opensearch。
   2. 对于实体：抽取出之后将其拼接为：“entity_name、entity_type、entity_description”格式存入opensearch。
   3. 同时也需要将实体和关系按照原先的方式存入neo4j
2. 在检索阶段，对于实体和关系的检索修改为先从opensearch中的实体、关系向量数据库分别检索解析出实体和关系（entity、source_entity、target_entity），再根据这些结果到neo4j中检索实际的实体、关系及其对应的chunk_id，通过neo4j进行节点度（degree）排序，筛选出top_k的实体和关系，从命中的实体出发取相连边/邻居节点，最后将结果拼接为结构化Context，并附上与实体相关的chunks。
3. 在缓存阶段，除了将query改写后的查询和Embedding向量进行缓存，还需要对以下内容进行缓存:
   1. 向量查询过程中产生的query和其改写结果缓存；
   2. 对query进行low/high level内容抽取结果。

---

# 当前实现进度（已落地）

- 文档上传与解析：Docling + VLM/LLM（公式/图片/表格增强）→ Markdown → 分块 → LLM 预处理（摘要/假设性问题）
- 向量库：OpenSearch（BM25 + 多路向量检索 + RRF 融合）
- 知识图谱：Neo4j（实体/关系抽取 + 动态增量更新 + 全文索引检索）
- 分层检索：并发执行「图谱检索」与「向量检索」，融合候选后用 TEI `/rerank` 重排
- 回答生成：中文提示词 + “回答 + References 引用段落”
- 缓存：Redis 缓存 query embedding（TTL 可配置）
- API：FastAPI 多接口（上传/问答/图谱/健康检查）

# 快速开始

## 1) 配置

复制 `.env.example` 为 `.env`，按你的服务地址/密钥填写。

## 2) 启动依赖（Docker Compose）

```bash
docker compose up -d
```

## 3) 启动 API

```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```

或：

```bash
python main.py
```

# 接口说明

- `GET /api/health`：健康检查
- `POST /api/documents/upload?stream=true`：上传并摄入（SSE 流式日志）
- `POST /api/query`：问答（返回回答 + 引用 + 证据块）
- `GET /api/graph/entity/{entity_name}`：实体子图

# 文档

- 部署与配置：`docs/deployment.md`
- 各层说明：`docs/modules/`
