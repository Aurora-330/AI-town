# 📚 知识/RAG系统使用指南

## 📖 概述

OC小镇当前已经具备独立的外部知识库检索链路。它和 NPC 记忆系统是**并行但分离**的两条能力：

- **Memory Retrieval**：回忆你和 NPC 之前聊过什么
- **Knowledge Retrieval**：检索 `knowledge_base/` 里的外部文档知识

这也是当前 chapter15 一个很关键的工程边界。

---

## ✨ 核心功能

### 1. **独立知识库目录**

当前默认知识目录为：

```text
code/chapter15/Helloagents-AI-Town/knowledge_base/
├── global/
│   ├── behavior_boundaries.md
│   ├── character_handbook.md
│   ├── interaction_playbook.md
│   └── knowledge_routing_examples.md
└── npc/
```

支持的文档类型包括：

- `.md`
- `.markdown`
- `.txt`

### 2. **独立 Qdrant collection**

知识库不会和记忆混成一个检索池，而是写入独立集合：

- 当前配置项：`KNOWLEDGE_COLLECTION`
- 默认值：`hello_agents_knowledge`

### 3. **Hybrid Search**

当前知识检索不是纯向量召回，而是组合了：

- 🔍 语义向量检索
- 🔎 轻量 BM25 lexical 检索
- 🧮 rerank 与低价值过滤

### 4. **范围与角色感知**

检索结果会带上：

- `source`
- `title`
- `scope`
- `tags`
- `chunk_index`

并支持：

- `global` 范围知识
- 角色相关知识过滤
- 跨 NPC 结果限制与筛选

---

## 🏗️ 当前架构

```text
knowledge_base/ 文档
   ↓
ingest_knowledge.py
   ↓
KnowledgeRetriever.ingest()
   ↓
Qdrant collection + lexical index
   ↓
search_with_debug()
   ↓
semantic candidates + lexical candidates
   ↓
rerank / dedupe / filter
   ↓
knowledge_context 注入 Prompt
```

---

## 🚀 初始化步骤

### 步骤1: 准备 Qdrant

保证 `QDRANT_URL` 指向可用的 Qdrant 服务。

### 步骤2: 配置 `.env`

```env
KNOWLEDGE_ENABLED=true
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
KNOWLEDGE_COLLECTION=hello_agents_knowledge
KNOWLEDGE_TOP_K=3
KNOWLEDGE_CHUNK_SIZE=420
KNOWLEDGE_CHUNK_OVERLAP=60
KNOWLEDGE_BASE_DIR=../knowledge_base
```

### 步骤3: 执行 ingest

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
python ingest_knowledge.py
```

执行后，知识文档会被切块、向量化并写入 Qdrant。

---

## 🔎 当前检索特点

### 1. 不是所有问题都会优先查知识库

系统会先通过 `RetrievalPlanner` 判断问题类型：

- `knowledge`：明显偏文档问答、规则解释、定义说明
- `routing`：需要推荐更适合的 NPC
- `mixed`：既需要历史记忆，也需要外部知识
- `recall`：优先回忆聊天历史，不查知识库

### 2. 知识上下文有独立预算

在 Prompt 组装时，`knowledge_context` 有单独的 token budget，不会无限塞入文档内容。

### 3. 更像“证据块注入”，不是整篇文档拼接

当前实现会尽量只注入命中的片段，而不是整段正文，避免外部知识把对话上下文挤爆。

---

## 📡 API接口

### 1. 调试知识检索

```http
GET /knowledge/search?q=character_handbook 里主要写了什么&limit=3
```

响应示例结构：

```json
{
  "query": "character_handbook 里主要写了什么",
  "hits": [
    {
      "id": "...",
      "content": "...",
      "source": "...",
      "title": "...",
      "scope": "global",
      "tags": ["..."],
      "chunk_index": 0,
      "score": 0.78,
      "semantic_score": 0.71,
      "lexical_score": 0.64,
      "rerank_score": 0.78
    }
  ],
  "total": 1
}
```

### 2. 通过 `/chat` 间接触发知识问答

```json
{
  "npc_name": "顾辰",
  "message": "interaction_playbook 里对协作分工是怎么写的？",
  "execution_mode": "auto"
}
```

---

## 🧪 测试方法

### 方法1: 运行检索测试

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
pytest test_knowledge_retriever.py
pytest test_knowledge_rerank.py
```

### 方法2: 运行 grounding 评测

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/evaluation
python eval_grounding.py
```

### 方法3: 结合日志观察

当你通过 `/chat` 发知识类问题时，可以在日志里观察：

- query 是否被判成 `knowledge` 或 `mixed`
- 命中了哪些知识块
- rerank 后留下了哪些结果

---

## 🔧 当前实现细节

### KnowledgeRetriever 主要职责

```text
KnowledgeRetriever
├── ingest()
├── search()
├── search_with_debug()
├── _semantic_search()
├── _lexical_search()
├── _merge_candidates()
└── _rerank_and_filter_chunks()
```

### 当前用到的关键策略

- 向量检索召回候选
- BM25 lexical 检索补专名词命中
- 候选融合
- rerank 打分
- 去重与低价值过滤
- 跨角色知识限制

---

## 📌 使用时要记住的边界

### 1. 记忆不是知识库

NPC 记得“你上次说过什么”，和 NPC 能回答“文档里怎么定义”，是两件事。

### 2. 知识库不是人格替代品

即使调用了知识块，最终回答仍然要保持 NPC 自己的口吻，而不是变成通用 FAQ 机器人。

### 3. 当前 chapter15 里 Neo4j 不是主链路

虽然历史配置里出现过 Neo4j，但当前实际知识主链路是：

- 文档
- embedding
- Qdrant
- rerank
- prompt 注入

---

## 🎯 适合拿来验证的问题

你可以优先测试这几类问题：

- “`character_handbook` 里主要写了什么？”
- “风泠更擅长处理哪类问题？”
- “如果让我选人帮我梳理路线图，文档里会更偏谁？”
- “`interaction_playbook` 里对多人协作有什么建议？”

这些问题更容易稳定命中当前知识库内容。
