# 赛博小镇 - AI NPC对话系统

基于 HelloAgents 框架的 AI 小镇模拟项目，包含 Godot 前端、FastAPI 后端，以及围绕 NPC 对话构建的记忆、好感度、知识检索、多角色协作与评测能力。

## 🎮 功能特性

- ✅ 3个风格鲜明的AI NPC（风泠、郁米、顾辰）
- ✅ 单角色实时对话接口 `/chat`
- ✅ 单角色自动路由与受控 ReAct 执行模式
- ✅ 多层记忆系统（working / episodic / summary）
- ✅ 自动摘要压缩与 summary debug 能力
- ✅ 外部知识库 RAG（独立于记忆链路）
- ✅ 好感度分析与关系分层表达
- ✅ LangGraph 多角色协作接口 `/multi_chat`
- ✅ 输入/输出/摘要安全编排
- ✅ 批量 NPC 状态生成与完整日志系统
- ✅ 覆盖记忆、好感度、RAG、多角色、总结、安全的评测脚本

## 🛠️ 技术栈

- **游戏引擎:** Godot 4.x
- **后端框架:** FastAPI + Python 3.10+
- **AI框架:** HelloAgents
- **记忆系统:** HelloAgents MemoryManager
- **知识检索:** Qdrant + Sentence Transformers
- **编排能力:** PromptBuilder + Controlled ReAct + LangGraph(beta)

## 📦 项目结构

```text
code/chapter15/
├── Helloagents-AI-Town/
│   ├── backend/                 # FastAPI后端、Prompt、日志、测试
│   ├── helloagents-ai-town/     # Godot项目
│   ├── knowledge_base/          # 外部知识库文档
│   └── evaluation/              # 自动评测脚本与报告
├── README.md
├── SETUP_GUIDE.md
├── AFFINITY_SYSTEM_GUIDE.md
├── MEMORY_SYSTEM_GUIDE.md
├── DIALOGUE_LOG_GUIDE.md
└── KNOWLEDGE_RAG_GUIDE.md
```

## 👥 当前NPC设定

- **风泠** - 档案整理师  
  擅长信息归档、时间线梳理、事件回顾和长期记忆整理。

- **郁米** - 情绪顾问  
  擅长情绪支持、关系沟通、偏好记忆和陪伴式对话。

- **顾辰** - 策略设计师  
  擅长任务拆解、项目规划、知识整合和协作调度。

## 📡 当前核心接口

- `POST /chat`：单 NPC 对话
- `POST /multi_chat`：多角色协作对话
- `POST /orchestrate/delegate-preview`：LangGraph 委托调试入口
- `GET /npcs` / `GET /npcs/status`：NPC 信息与状态
- `GET /npcs/{npc_name}/memories`：查看记忆
- `GET /npcs/{npc_name}/summary-debug`：查看摘要压缩状态
- `GET /npcs/{npc_name}/affinity` / `GET /affinities`：查看好感度
- `GET /knowledge/search`：调试知识检索结果

## 📚 文档

- [安装配置指南](SETUP_GUIDE.md)
- [好感度系统](AFFINITY_SYSTEM_GUIDE.md)
- [记忆系统](MEMORY_SYSTEM_GUIDE.md)
- [对话日志系统](DIALOGUE_LOG_GUIDE.md)
- [知识/RAG系统](KNOWLEDGE_RAG_GUIDE.md)

## 📖 教程说明

本项目是《Hello-agents》教材第15章的配套案例。当前代码已经不只是最初的基础对话版本，而是在 chapter15 范围内继续演进出了摘要记忆、外部知识检索、多角色协作、受控 ReAct 和自动评测能力。

## 📄 许可证

CC BY-NC-SA 4.0
