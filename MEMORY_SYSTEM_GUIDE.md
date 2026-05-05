# 🧠 NPC记忆系统使用指南

## 📚 概述

赛博小镇当前的记忆系统已经从“短期记忆 + 长期记忆”升级为**三层记忆架构**。系统会把最近对话、长期事件和压缩摘要分层存储，并根据查询类型动态决定优先注入哪些上下文。

---

## ✨ 核心功能

### 1. **Working Memory** - 短期记忆
- 📝 保存最近对话窗口
- ⚡ 用于当前轮次上下文延续
- 📦 当前容量：最近 `10` 条对话

### 2. **Episodic Memory** - 长期事件记忆
- 💾 保存较重要的历史对话
- 🔍 支持基于向量的语义检索
- 📚 当前容量：最多 `100` 条
- 🧹 启用遗忘机制，低重要度记忆可被淘汰

### 3. **Summary Memory** - 摘要记忆
- 🗂️ 把多轮对话压缩成高层摘要
- 🎯 用于长程召回，减少上下文堆积
- 🔄 支持再次压缩（recompress）
- 🧾 保留 `source_memory_ids` 等追踪字段，便于调试来源

### 4. **查询模式驱动的检索策略**

系统当前不是一套固定检索，而是会先判断问题类型：

- `recall`
- `knowledge`
- `mixed`
- `routing`
- `summary`
- `default`

不同模式下，对 summary / episodic / working 的使用优先级不同。

---

## 🏗️ 当前记忆架构

```text
NPCAgentManager
├── agents
├── memories
├── retrieval_planner
└── chat_with_debug()
    ├── 1. 分析 query_mode
    ├── 2. 检索 summary / episodic / working
    ├── 3. 按预算裁剪上下文
    ├── 4. 生成回复
    ├── 5. 写入新记忆
    └── 6. 视情况触发摘要生成/重压缩
```

---

## 📂 存储结构

当前每个 NPC 都有独立记忆目录：

```text
code/chapter15/Helloagents-AI-Town/backend/memory_data/
├── 风泠/
├── 郁米/
└── 顾辰/
```

每个 NPC 的记忆互相隔离，玩家相关信息也按 `player_id` 维度组织。

---

## 🔄 摘要记忆流程

当前代码中的关键阈值包括：

- `SUMMARY_TRIGGER_TURNS = 6`
- `SUMMARY_RETRIEVAL_LIMIT = 1`
- `EPISODIC_RETRIEVAL_LIMIT = 1`
- `WORKING_RETRIEVAL_LIMIT = 1`
- `SUMMARY_RECOMPRESS_TRIGGER = 5`

可以把当前流程理解为：

```text
1. 玩家与某个 NPC 连续对话
   ↓
2. 累积到一定轮数后触发 summary 生成
   ↓
3. 生成高层摘要记忆
   ↓
4. 记录 source_memory_ids / summary_level / is_compressed
   ↓
5. 低价值旧记忆归档，不立即粗暴删除
   ↓
6. 当摘要数量继续增长时，再做 merged summary 压缩
```

---

## 🎯 检索优先级

项目当前的主张非常明确：

- 记忆检索和知识检索是两条链路
- Summary 负责“长期高层回忆”
- Episodic 负责“具体历史事件”
- Working 负责“最近几轮上下文”

在回忆类问题中，通常会优先依赖：

1. Summary memory
2. Episodic memory
3. Working memory

---

## 🧾 Summary Debug 接口

为了方便观察摘要压缩状态，当前后端新增了：

```http
GET /npcs/{npc_name}/summary-debug
```

它会返回：

- 当前 NPC 的摘要记录
- 各 `player_id` 的 `summary_count`
- `pending_turn_count`
- `active_base_count`
- `merged_count`
- `compressed_count`
- `archived_memory_ids`

这个接口是当前项目里理解 summary memory 是否真的在工作的最佳入口。

---

## 🚀 API接口

### 1. 单角色对话（会自动走记忆链）

```http
POST /chat
Content-Type: application/json

{
  "npc_name": "风泠",
  "message": "你还记得我之前最怕哪种汇报方式吗？",
  "execution_mode": "auto"
}
```

### 2. 查看 NPC 记忆

```http
GET /npcs/风泠/memories?limit=10
```

### 3. 查看摘要调试信息

```http
GET /npcs/风泠/summary-debug?player_id=player
```

### 4. 清空记忆（调试用）

```http
DELETE /npcs/风泠/memories
DELETE /npcs/风泠/memories?memory_type=episodic
```

---

## 🧪 测试方法

### 方法1: 运行摘要与兼容性测试

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
pytest test_summary_memory.py
pytest test_agents_summary_compat.py
```

### 方法2: 运行记忆评测

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/evaluation
python eval_memory.py
python eval_summary.py
```

### 方法3: 手动做多轮对话

先连续聊 6 轮以上，再调用：

```http
GET /npcs/风泠/summary-debug
```

如果看到 `summary_count`、`summary_records` 增长，就说明摘要链路已经触发。

---

## 🔧 当前实现细节

### MemoryManager 配置

当前配置大致为：

```python
MemoryConfig(
    storage_path=memory_dir,
    working_memory_capacity=10,
    working_memory_tokens=2000,
    episodic_memory_capacity=100,
    enable_forgetting=True,
    forgetting_threshold=0.3
)
```

### Prompt 预算控制

记忆内容不是无上限注入的。当前代码会根据 `query_mode` 对：

- `summary_context`
- `episodic_context`
- `working_context`
- `recent_dialogue_context`

分别做预算控制，避免长对话把 Prompt 挤爆。

---

## 📌 当前系统和旧版的区别

如果和最早那版 chapter15 相比，当前记忆系统已经新增了：

- Summary memory 层
- 摘要自动触发
- 摘要重压缩
- 摘要调试接口
- 查询模式驱动的检索计划
- 更细的 Prompt 预算与裁剪机制

所以现在更适合把它看成“分层记忆系统”，而不是单纯“聊天历史保存器”。
