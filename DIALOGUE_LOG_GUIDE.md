# 对话日志系统使用指南

## 📝 概述

赛博小镇当前的日志系统已经覆盖了**单角色对话主链**中的关键可观测节点，目的不是只留一份聊天记录，而是让你能看清：

- 这轮用了什么检索策略
- 检索到了什么记忆/知识
- 有没有触发 ReAct
- 好感度怎么变了
- 摘要有没有生成
- 安全编排有没有介入

---

## 🎯 当前日志会记录什么

### 1. 对话基础信息
- 💬 对话开始/结束
- 📝 玩家消息
- 🤖 NPC 回复

### 2. 好感度相关
- 💖 当前好感度与关系等级
- 📈 好感度变化前后数值
- 🎭 变化原因与情绪标签

### 3. 记忆相关
- 🧠 检索到的 summary / episodic / working 记忆
- 📦 各层记忆的调试信息
- 💾 本轮记忆写入确认
- 🗂️ 摘要触发 / 生成 / 跳过 / 重压缩日志

### 4. 知识检索相关
- 📚 知识检索 query
- 🔎 检索命中的知识块
- 🧮 rerank / filter 后的结果

### 5. 编排与执行相关
- 🧭 query analysis
- 🧱 retrieval plan
- ⚙️ coordinator 决策
- 🔁 react step / react finish

### 6. 安全相关
- 🛡️ 输入审核
- 🛡️ 输出审核
- 🛡️ 摘要前后安全决策
- 🛡️ 记忆写入策略分流

---

## 📂 日志目录

当前日志文件位于：

```text
code/chapter15/Helloagents-AI-Town/backend/logs/
├── dialogue_2026-05-02.log
├── dialogue_2026-05-03.log
├── dialogue_2026-05-04.log
└── dialogue_2026-05-05.log
```

日志按日期自动切分，文件名格式为：

```text
dialogue_YYYY-MM-DD.log
```

---

## 🚀 使用方法

### 方法1: 启动后端，自动记录

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
python main.py
```

启动时会输出日志文件位置。

### 方法2: 实时查看日志

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
python view_logs.py tail
```

### 方法3: 查看完整日志

```bash
python view_logs.py view
```

### 方法4: 列出日志文件

```bash
python view_logs.py list
```

---

## 📊 你会在日志里看到什么

### 单角色主链

典型流程包括：

```text
1. 对话开始
2. 输入安全审核
3. 当前好感度
4. query analysis
5. retrieval plan
6. coordinator / react 决策
7. 记忆检索 / 知识检索
8. prompt 组装与预算
9. NPC 回复
10. 输出安全审核
11. 好感度更新
12. 记忆写入
13. 摘要触发或跳过
14. 对话结束
```

### 关键观察点

如果你在调试“为什么回答不像预期”，最有用的是看这几段：

- `query analysis`
- `retrieval plan`
- `knowledge retrieval`
- `Prompt预算`
- `React激活`
- `summary created / skipped / recompressed`

---

## 🎓 教学价值

### 1. 看清记忆是否真的参与了回答

不是“系统说自己有记忆”，而是能看到：

- 检索到了几条
- 来自哪一层
- 最终注入了哪些上下文

### 2. 看清知识检索是否真的命中

可以观察：

- query 是什么
- 候选块有哪些
- rerank 后保留了哪些块

### 3. 看清执行模式为什么变化

当前 `/chat` 不是永远同一种执行方式。日志可以告诉你：

- 是 `static_coordinator`
- 还是 `controlled_react`
- 为什么激活或为什么回退

### 4. 看清安全层有没有介入

对于输入、输出、摘要和记忆写入，你都能看到安全决策结果。

---

## 🔧 当前日志覆盖范围

当前文档描述的是 **backend 主对话链路** 的日志系统，核心位置在：

- [logger.py](/home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend/logger.py)
- [agents.py](/home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend/agents.py)

它已经不只是早期版本中的“打印对话文本”，而是整合了：

- 对话日志
- 检索日志
- 安全日志
- 摘要日志
- ReAct日志
- Prompt预算日志

---

## 🧪 推荐调试场景

### 场景1: 验证回忆类问题

先多轮对话，再问：

```text
你还记得我之前最怕哪种汇报方式吗？
```

看日志中的：

- `query_mode=recall`
- memory retrieval 命中情况

### 场景2: 验证知识问答

提问：

```text
character_handbook 里主要写了什么？
```

看日志中的：

- knowledge retrieval
- rerank 结果

### 场景3: 验证结构说明型问题

提问：

```text
如果我要写一版路线图说明，通常应该包含哪些部分？
```

看日志中的：

- `React激活`
- `react_step_count`

---

