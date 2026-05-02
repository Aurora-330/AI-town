# External Knowledge RAG Guide

## 新增内容

- `knowledge_base/`：外部知识文档目录
- `backend/knowledge_retriever.py`：独立知识检索模块
- `backend/ingest_knowledge.py`：文档入库脚本

## 边界说明

1. Memory retrieval 仍由现有 `MemoryManager` 负责。
2. Knowledge retrieval 使用独立的 Qdrant collection，默认是 `hello_agents_knowledge`。
3. `chat` API 的输入输出结构保持不变，只在内部 prompt 中新增 `knowledge_context`。
4. 第一阶段只使用 `global` scope，不把 memory 和 knowledge 混成统一检索池。

## 使用方式

在 `backend/` 目录执行：

```bash
/home/wjy/anaconda3/envs/hello_agents/bin/python ingest_knowledge.py
```

入库完成后，`/chat` 会在命中外部知识时自动把结果作为单独区块注入 prompt。

## 调试建议

优先用这类问题验证：

- “谁更适合帮我整理时间线？”
- “我现在只想被理解，不想立刻听建议，找谁更合适？”
- “只有两名开发、时间很紧时，应该找谁帮我拆计划？”

如果回复明显依赖知识库，日志中应能看到：

- `📚 外部知识命中N条`
- 每条命中的 `title / source / score`

## Prompt预算提醒

`max-model-len` 只是理论上限，真实吞吐更受“每次请求到底塞了多少上下文”影响。后续如果持续叠加 `memory + summary + knowledge + multi-agent context`，单次请求会更胖，并发能力和时延都会下降。

建议优先控制这些点：

- memory、summary、knowledge 分层各自限制 top-k
- 外部知识块保持短小，避免把整段长文直接塞进 prompt
- 多 Agent 场景优先传递摘要后的中间结论，而不是完整上下文转发
- 每次接新上下文时，都把“真实 prompt 长度”视为一等指标，而不只盯 `max-model-len`
