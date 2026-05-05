# 赛博小镇 - 安装配置指南

## 📋 系统要求

- **操作系统:** Windows / macOS / Linux
- **Godot:** 4.2+（推荐 4.3 或更高）
- **Python:** 3.10+
- **可选组件:** Qdrant（启用知识库检索时需要）

---

## 📁 当前目录说明

本章项目的实际代码目录如下：

```text
code/chapter15/Helloagents-AI-Town/
├── backend/                 # FastAPI后端
├── helloagents-ai-town/     # Godot前端
├── knowledge_base/          # 外部知识库
└── evaluation/              # 评测脚本
```

后续命令默认都在这个目录下执行。

---

## 🚀 安装步骤

### 步骤1: 进入项目目录

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town
```

### 步骤2: 配置 Python 环境

```bash
cd backend
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 步骤3: 安装依赖

```bash
pip install -r requirements.txt
```

当前后端依赖中已经包含：
- FastAPI / Uvicorn
- HelloAgents
- qdrant-client
- sentence-transformers
- langgraph
- pytest / httpx

### 步骤4: 配置环境变量

后端会自动读取 `backend/.env`。

```bash
cp .env.example .env
```

建议优先配置以下字段：

```env
LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/

EMBED_MODEL_TYPE=local
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=

KNOWLEDGE_ENABLED=true
KNOWLEDGE_COLLECTION=hello_agents_knowledge
KNOWLEDGE_TOP_K=3
KNOWLEDGE_BASE_DIR=../knowledge_base
```

### 步骤5: 启动后端

**方式A：直接运行**
```bash
python main.py
```

**方式B：使用 uvicorn**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动后可访问：

- `http://localhost:8000/`
- `http://localhost:8000/docs`

---

## ⚠️ 前后端端口对齐

当前代码里有一个需要特别注意的现状：

- 后端默认监听 **8000**（`backend/config.py`）
- Godot 前端当前写的是 **8001**（`helloagents-ai-town/scripts/config.gd`）

也就是说，如果你要直接跑通 Godot 前端，需要二选一：

### 方案A: 保持后端 8000，修改 Godot 配置

把 [config.gd](/home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/helloagents-ai-town/scripts/config.gd) 里的：

```gdscript
const API_BASE_URL = "http://localhost:8001"
```

改成：

```gdscript
const API_BASE_URL = "http://localhost:8000"
```

### 方案B: 保持 Godot 8001，用 uvicorn 跑 8001

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

如果只是调试后端接口，直接使用 8000 即可。

---

## 🧠 知识库初始化

如果要启用外部知识检索，先保证 Qdrant 可用，然后执行：

```bash
cd /home/wjy/hello-agents/code/chapter15/Helloagents-AI-Town/backend
python ingest_knowledge.py
```

项目当前默认知识目录为：

```text
code/chapter15/Helloagents-AI-Town/knowledge_base/
├── global/
└── npc/   # 如存在，可存放角色定向知识
```

---

## 🎮 启动 Godot 前端

1. 打开 Godot
2. 导入项目目录：
   `code/chapter15/Helloagents-AI-Town/helloagents-ai-town`
3. 打开主场景 `scenes/main.tscn`
4. 运行项目

当前前端支持的基础交互包括：

- `WASD` 移动
- `E` 与 NPC 交互
- `Enter` 发送消息
- `ESC` 关闭对话框

---

## 🧪 推荐验证顺序

### 1. 先验证后端是否正常

访问：

- `GET /health`
- `GET /npcs`
- `GET /npcs/status`

### 2. 再验证单角色对话

测试 `POST /chat`：

```json
{
  "npc_name": "风泠",
  "message": "你最近在忙什么？",
  "execution_mode": "auto"
}
```

### 3. 再验证多角色协作

测试 `POST /multi_chat`：

```json
{
  "message": "我最近工作压力很大，但又想把项目推进下去，你们怎么看？",
  "mode": "auto",
  "player_id": "player",
  "return_intermediate": true
}
```

### 4. 再验证知识检索

先执行 `ingest_knowledge.py`，再访问：

```http
GET /knowledge/search?q=风泠 擅长什么&limit=3
```

---

## 📊 当前后端能力

后端当前已包含：

- 单角色对话与调试指标返回
- 好感度分析与关系档位
- Working / Episodic / Summary 记忆
- 摘要压缩治理与 `/summary-debug`
- 外部知识检索与轻量 rerank
- Controlled ReAct 执行模式
- LangGraph 多角色协作接口
- 输入/输出/摘要安全编排
- 日志记录与评测脚本

---

## ❓ 常见问题

### Q1: 后端能启动，但 Godot 无法对话？
**A:** 先检查 8000 / 8001 端口是否对齐，这是当前项目最容易踩到的点。

### Q2: `/chat` 能用，但知识问答没有命中文档？
**A:** 通常是还没执行 `python ingest_knowledge.py`，或 Qdrant 没有启动。

### Q3: 没有配置 `LLM_API_KEY` 能运行吗？
**A:** 可以启动，但会降级为模拟模式，无法体验真实的 LLM 对话链路。

### Q4: 为什么文档里不再写 Neo4j 启动步骤？
**A:** 当前 chapter15 实际主链路没有依赖 Neo4j，文档以“当前代码会生效的功能”为准。

---

## 🎉 开始体验

建议的最短路径是：

1. 启动 Qdrant（如果要体验知识检索）
2. 配好 `backend/.env`
3. 启动 FastAPI
4. 先用 `/docs` 验证接口
5. 再接入 Godot 前端
