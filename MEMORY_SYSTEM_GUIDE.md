# 🧠 NPC记忆系统使用指南

## 📚 概述

赛博小镇的NPC现在拥有了**记忆系统**,能够记住与玩家的对话历史,并在后续对话中引用之前的内容,让NPC更加智能和真实!

---

## ✨ 核心功能

### 1. **工作记忆 (Working Memory)** - 短期记忆
- 📝 存储最近的10条对话
- ⏰ 2小时后自动过期
- 🚀 快速检索,用于当前对话上下文

### 2. **情景记忆 (Episodic Memory)** - 长期记忆
- 💾 持久化存储重要对话
- 🔍 支持语义检索 (基于Qdrant向量数据库)
- 📊 最多存储100条记忆
- 🧹 自动遗忘重要性低于0.3的记忆

### 3. **摘要记忆 (Summary Memory)** - 压缩记忆
- 🧩 每累计一定轮数对话后自动生成摘要
- 📝 摘要以长期记忆形式保存,用于长程上下文压缩
- 🎯 优先保留主要话题、稳定偏好、未完成事项和关系变化
- 📦 已被摘要覆盖的低价值原始对话会在检索阶段降权/归档

### 4. **记忆隔离**
- 🔒 每个NPC拥有独立的记忆系统
- 🚫 NPC之间的记忆不会互相干扰
- 👤 每个玩家的对话独立存储

---

## 🎯 使用示例

### 示例1: 基本对话记忆

```
第一次对话:
玩家: "你好,你是做什么的?"
风泠: "你好。我负责整理记录、线索和那些容易被忽略的细节。"

第二次对话 (5分钟后):
玩家: "还记得我刚才问你什么吗?"
风泠: "当然记得。你问我平时做什么,我说我主要负责整理记录和线索。"
```

### 示例2: 长期记忆

```
第一天:
玩家: "你平时最在意别人说话里的什么?"
风泠: "时间、顺序和那些前后对不上的细节。它们通常比表面的话更诚实。"

第二天:
玩家: "我们之前聊过你会记住什么吗?"
风泠: "聊过。我记得我说过,我最在意时间、顺序和前后不一致的地方。"
```

### 示例3: 记忆隔离

```
与风泠对话:
玩家: "我最近总是记不住会议上的重点"
风泠: "那就先别急着记全部,先抓时间线和决定点。剩下的细节我可以陪你慢慢理。"

与郁米对话:
玩家: "我刚才和风泠聊了什么?"
郁米: "抱歉,我不知道你和风泠聊了什么,我只负责产品方面的工作。"
```

---

## 🔧 技术实现

### 架构设计

```
NPCAgentManager
├── agents: Dict[str, SimpleAgent]          # NPC Agent
├── memories: Dict[str, MemoryManager]      # NPC记忆管理器
├── summary_state.json                      # 摘要状态与归档索引
└── chat(npc_name, message, player_id)      # 对话接口
    ├── 1. 分层检索相关记忆(summary / episodic / working)
    ├── 2. 构建增强提示词
    ├── 3. 调用Agent生成回复
    └── 4. 保存对话到记忆
        └── 满足阈值时生成摘要记忆
```

### 记忆存储结构

```
backend/memory_data/
├── 风泠/
│   ├── sqlite_store.db          # SQLite数据库 (权威存储)
│   ├── qdrant_collection/       # Qdrant向量索引 (语义检索)
│   └── summary_state.json       # 摘要状态、待摘要轮次、归档索引
├── 郁米/
│   ├── sqlite_store.db
│   ├── qdrant_collection/
│   └── summary_state.json
└── 顾辰/
    ├── sqlite_store.db
    ├── qdrant_collection/
    └── summary_state.json
```

### 记忆数据格式

```python
{
    "id": "memory_uuid",
    "content": "玩家说: 你好,你是做什么的?",
    "type": "working",  # working/episodic
    "importance": 0.5,  # 0-1之间
    "timestamp": "2024-01-15T10:30:00",
    "metadata": {
        "speaker": "player",
        "player_id": "player",
        "session_id": "player",
        "context": {
            "interaction_type": "dialogue",
            "npc_name": "风泠"
        }
    }
}
```

### 摘要记忆附加字段

```python
{
    "content": "摘要记忆: 最近主要围绕项目规划、用户偏好和未完成事项展开交流……",
    "type": "episodic",
    "importance": 0.85,
    "metadata": {
        "memory_tier": "summary",
        "summary_index": 1,
        "summary_source_count": 6,
        "source_memory_ids": ["uuid-1", "uuid-2"]
    }
}
```

### 当前边界说明

- `SummaryMemory` 在当前阶段以 `episodic + metadata.memory_tier=summary` 的方式接入。
- 不直接修改底层 `hello_agents` 的记忆类型定义,避免破坏现有库行为。
- 原始记忆默认不激进删除,而是在检索阶段根据摘要状态做降权/归档过滤。

---

## 🚀 API接口

### 1. 对话接口 (支持记忆)

```http
POST /chat
Content-Type: application/json

{
    "npc_name": "风泠",
    "message": "你好,你是做什么的?"
}
```

**响应:**
```json
{
    "npc_name": "风泠",
    "npc_title": "档案整理师",
    "message": "你好。我负责整理记录、线索和那些容易被忽略的细节。",
    "success": true
}
```

### 2. 获取NPC记忆

```http
GET /npcs/风泠/memories?limit=10
```

**响应:**
```json
{
    "npc_name": "风泠",
    "memories": [
        {
            "id": "uuid-1",
            "content": "玩家说: 你好,你是做什么的?",
            "type": "working",
            "importance": 0.5,
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {...}
        },
        ...
    ],
    "total": 10
}
```

### 3. 清空NPC记忆 (测试用)

```http
DELETE /npcs/风泠/memories?memory_type=working
```

**响应:**
```json
{
    "message": "已清空风泠的记忆",
    "npc_name": "风泠",
    "memory_type": "working"
}
```

---

## 🧪 测试方法

### 方法1: 使用测试脚本

```bash
cd backend
python test_memory.py
```

**测试内容:**
- ✅ 基本对话记忆
- ✅ 长期记忆检索
- ✅ 记忆隔离
- ✅ 相关性检索

### 方法2: 使用API测试

1. 启动后端服务:
```bash
cd backend
python main.py
```

2. 访问API文档: http://localhost:8000/docs

3. 测试对话接口:
   - 发送第一条消息: "你好,你是做什么的?"
   - 发送第二条消息: "还记得我刚才问你什么吗?"
   - 查看记忆列表: GET /npcs/风泠/memories

### 方法3: 在Godot中测试

1. 启动后端服务
2. 运行Godot游戏
3. 与NPC对话多次
4. 观察NPC是否能记住之前的对话

---

## 📊 记忆系统配置

### 配置参数 (agents.py)

```python
memory_config = MemoryConfig(
    storage_path=f"./memory_data/{npc_name}",  # 存储路径
    working_memory_capacity=10,                # 工作记忆容量
    working_memory_tokens=2000,                # 工作记忆token限制
    episodic_memory_capacity=100,              # 情景记忆容量
    enable_forgetting=True,                    # 启用遗忘机制
    forgetting_threshold=0.3                   # 遗忘阈值
)
```

### 调整建议

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| working_memory_capacity | 10 | 5-20 | 工作记忆容量,越大越占内存 |
| working_memory_tokens | 2000 | 1000-4000 | Token限制,影响上下文长度 |
| episodic_memory_capacity | 100 | 50-500 | 长期记忆容量,越大越占磁盘 |
| forgetting_threshold | 0.3 | 0.1-0.5 | 遗忘阈值,越低越容易遗忘 |

---

## 🎓 教学价值

### 学习要点

1. **MemoryManager的使用**
   - 如何初始化记忆管理器
   - 如何配置不同类型的记忆
   - 如何添加和检索记忆

2. **记忆检索策略**
   - 工作记忆: 快速检索最近对话
   - 情景记忆: 语义检索相关历史
   - 混合检索: 结合时间和相关性

3. **记忆存储机制**
   - SQLite: 权威数据存储
   - Qdrant: 向量语义检索
   - 双存储保证数据一致性

4. **记忆遗忘机制**
   - 基于重要性的自动遗忘
   - 基于时间的TTL过期
   - 容量限制的优先级淘汰

---

## 🔍 调试技巧

### 1. 查看记忆日志

```python
# 在agents.py的chat方法中
print(f"🧠 {npc_name}检索到{len(relevant_memories)}条相关记忆")
print(f"💾 对话已保存到{npc_name}的记忆中")
```

### 2. 检查记忆文件

```bash
# 查看SQLite数据库
cd backend/memory_data/风泠
sqlite3 sqlite_store.db
> SELECT * FROM memories;
```

### 3. 清空记忆重新测试

```python
# 使用API清空记忆
DELETE /npcs/风泠/memories

# 或者直接删除文件
rm -rf backend/memory_data/风泠
```

---

## ❓ 常见问题

### Q1: NPC为什么记不住对话?

**可能原因:**
- 记忆系统未正确初始化
- 存储路径权限问题
- 记忆被遗忘机制清除

**解决方法:**
- 检查日志中是否有"记忆系统已初始化"
- 检查memory_data目录是否存在
- 调高forgetting_threshold参数

### Q2: 记忆检索不准确?

**可能原因:**
- 查询语句与记忆内容相似度低
- 记忆重要性太低被过滤

**解决方法:**
- 降低min_importance参数
- 增加检索limit数量
- 使用更具体的查询语句

### Q3: 记忆占用空间太大?

**解决方法:**
- 降低episodic_memory_capacity
- 提高forgetting_threshold
- 定期清理旧记忆

---

## 🎉 下一步

现在记忆系统已经完成,接下来我们将实现:

1. ✅ **好感度系统** - NPC与玩家的关系管理
2. ✅ **情感分析** - 使用LLM分析对话情感
3. ✅ **关系等级** - 陌生、熟悉、友好、亲密、挚友

---

## 📝 总结

✅ NPC记忆系统已成功集成到赛博小镇!

**核心特性:**
- 🧠 短期记忆 (工作记忆)
- 💾 长期记忆 (情景记忆)
- 🔍 语义检索
- 🔒 记忆隔离
- 🧹 自动遗忘

**教学价值:**
- HelloAgents Memory系统的实战应用
- 多智能体记忆管理
- 向量数据库的使用
- 记忆检索策略

**下一步:**
- 实现好感度系统
- 集成情感分析
- 完善NPC交互体验

---
