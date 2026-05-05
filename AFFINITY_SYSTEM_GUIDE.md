# 💖 NPC好感度系统使用指南

## 📚 概述

OC小镇当前的好感度系统已经不只是“简单加减分”，而是把**结构化情感分析**、**关系档位**和**角色表达约束**接进了主对话链路。每次对话后，系统都会判断这轮互动是否应该改变 NPC 对玩家的态度，并把结果影响到后续回复风格。

---

## ✨ 核心功能

### 1. **结构化好感度分析**
- 🤖 使用独立的 `AffinityAnalyzer` 分析玩家消息与 NPC 回复
- 📋 固定输出 `should_change / change_amount / reason / sentiment`
- 🔁 首次解析失败时自动重试一次
- 🛡️ 对“表达焦虑、主动求助、暴露脆弱”的场景有护栏，避免误扣分

### 2. **动态分值更新**
- 📈 好感度范围固定为 `0-100`
- 🎯 初始值为 `50`
- 🔒 每轮变化量限制在 `-15` 到 `+10`
- 🧮 更新结果会返回旧值、新值、变化原因和情绪标签

### 3. **五档关系等级**
- 🥶 **陌生** `0-19`
- 😐 **熟悉** `20-39`
- 🙂 **友好** `40-59`
- 🤝 **亲密** `60-79`
- 💕 **挚友** `80-100`

### 4. **关系修饰词接入 Prompt**

当前代码中，好感度不仅生成“等级”，还会生成运行时修饰词：

- `low_affinity`
- `guarded_affinity`
- `neutral_affinity`
- `warm_affinity`
- `high_affinity`

这些修饰词会和角色专属规则一起注入 Prompt，用来影响：

- 距离感
- 主动性
- 语气冷暖
- 是否愿意分享更多
- 是否更明显地兜底、安抚或保持克制

---

## 🎭 当前角色差异

### 郁米
- 低好感时仍礼貌，但会更克制、更抽离
- 高好感时会更贴近、更有“被信任感”
- 明确避免用空泛鸡汤和反问句收尾

### 风泠
- 低好感时会收起俏皮感，偏公事公办
- 高好感时更愿意夸人、补上下文、用轻巧比喻替玩家卸压

### 顾辰
- 低好感时更短、更冷、更带刺，但不做人身攻击
- 高好感时会更主动兜底，给预案、给备选路径、给风险提醒

---

## 🎯 运行流程

```text
1. 玩家发送消息
   ↓
2. NPC 主链完成回复
   ↓
3. RelationshipManager 分析本轮互动
   ↓
4. 更新 affinity / level / modifier
   ↓
5. 好感度结果写入日志
   ↓
6. 相关信息可进入记忆 metadata
   ↓
7. 下一轮对话继续使用新的关系状态
```

---

## 📊 好感度变化规则

| 对话类型 | 典型变化 | 说明 |
|---------|----------|------|
| 赞美、感谢、请教 | `+3 ~ +8` | 会明显提升关系 |
| 友好问候、正常交流 | `+1 ~ +3` | 缓慢升温 |
| 普通闲聊、中性信息 | `0` | 不强制变化 |
| 批评、质疑、不耐烦 | `-3 ~ -8` | 只在明显指向 NPC 时扣分 |
| 侮辱、攻击、恶意 | `-8 ~ -15` | 强负向 |

当前代码还额外强调：

- 玩家在表达自己的压力、焦虑、害怕，不等于在攻击 NPC
- 求助、请教、坦露脆弱，默认更接近信任或中性，而不是负向

---

## 🚀 API接口

### 1. 获取单个NPC好感度

```http
GET /npcs/风泠/affinity?player_id=player
```

响应示例：

```json
{
  "npc_name": "风泠",
  "player_id": "player",
  "affinity": 62.0,
  "level": "亲密",
  "modifier": "warm_affinity"
}
```

### 2. 获取所有NPC好感度

```http
GET /affinities?player_id=player
```

### 3. 手动设置好感度（调试/测试）

```http
PUT /npcs/顾辰/affinity?affinity=85&player_id=player
```

---

## 🧪 测试方法

### 方法1: 直接通过 `/chat` 触发

先发送友好消息：

```json
{
  "npc_name": "郁米",
  "message": "谢谢你，刚才那段话真的帮到我了。"
}
```

再查询：

```http
GET /npcs/郁米/affinity?player_id=player
```

### 方法2: 运行评测脚本

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/evaluation
python eval_affinity.py
python eval_affinity_interactions.py
```

### 方法3: 结合日志观察

启动后端后查看日志：

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
python view_logs.py tail
```

日志中会看到：

- 当前好感度
- 变化前后数值
- 变化原因
- 情感标签
- 是否触发等级变化

---

## 🔧 技术实现

### 核心组件

```text
RelationshipManager
├── affinity_scores
├── analyzer_agent
├── get_affinity()
├── set_affinity()
├── analyze_and_update_affinity()
├── get_affinity_level()
└── get_affinity_modifier()
```

### 当前实现特点

- 使用 Pydantic 对分析结果做结构校验
- 首轮失败后自动重试一次
- 再失败时回退到规则兜底
- 对明显求助/脆弱表达做轻量护栏
- 把 modifier 真正接入后续对话 Prompt，而不是只停留在接口层

---

## 📌 适合理解成什么

当前 chapter15 的好感度系统，更适合理解为：

- 一套“关系状态机”
- 一套“对话风格调节器”
- 一套“可观测、可测试的情感分析链路”

而不是单纯的分数展示面板。
