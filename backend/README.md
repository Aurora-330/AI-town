# 赛博小镇 - FastAPI后端

基于HelloAgents框架的AI NPC对话系统后端服务。

## 🎯 功能特性

### 核心功能
- ✅ **单个NPC对话**: 玩家与NPC实时对话,使用独立Agent处理
- ✅ **批量对话生成**: 定时批量生成所有NPC的自主对话,降低API成本66%
- ✅ **状态管理**: 自动更新和缓存NPC状态
- ✅ **最小安全编排层**: 输入审核、输出审核、越狱检测、摘要前后安全检查
- ✅ **CORS支持**: 支持Godot HTML5导出跨域访问

### NPC角色
1. **风泠** - Python工程师 (工位区)
2. **郁米** - 产品经理 (会议室)
3. **顾辰** - UI设计师 (休息区)

## 📦 安装依赖

### 1. 安装Python依赖
```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量
创建`.env`文件或设置环境变量:

**注意**: 如果不配置API密钥,系统将使用预设对话模式运行。

## 🚀 启动服务

### 方法1: 直接运行
```bash
python main.py
```

### 方法2: 使用uvicorn
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问:
- **API文档**: http://localhost:8000/docs
- **根路径**: http://localhost:8000/

## 🧪 测试API

运行测试脚本:
```bash
python test_api.py
```

测试内容包括:
1. ✅ 根路径访问
2. ✅ 健康检查
3. ✅ 获取NPC列表
4. ✅ 获取NPC状态
5. ✅ 与NPC对话
6. ✅ 获取NPC详情
7. ✅ 强制刷新状态

## 📡 API接口

### 1. 获取NPC列表
```http
GET /npcs
```

响应示例:
```json
{
  "npcs": [
    {
      "name": "风泠",
      "title": "Python工程师",
      "location": "工位区",
      "activity": "写代码",
      "available": true
    }
  ],
  "total": 3
}
```

### 2. 与NPC对话
```http
POST /chat
Content-Type: application/json

{
  "npc_name": "风泠",
  "message": "你好,你在做什么?"
}
```

响应示例:
```json
{
  "npc_name": "风泠",
  "npc_title": "Python工程师",
  "message": "你好!我正在优化一个多智能体系统的性能,挺有意思的。",
  "success": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. 获取NPC状态
```http
GET /npcs/status
```

响应示例:
```json
{
  "dialogues": {
    "风泠": "终于把这个bug修复了,测试通过!",
    "郁米": "下周的产品评审会需要准备一下资料。",
    "顾辰": "这个配色方案看起来不错,再调整一下细节。"
  },
  "last_update": "2024-01-15T10:30:00",
  "next_update_in": 25
}
```

### 4. 强制刷新状态
```http
POST /npcs/status/refresh
```

## 🏗️ 项目结构

```
backend/
├── main.py              # FastAPI主程序
├── config.py            # 配置文件
├── models.py            # 数据模型(Pydantic)
├── agents.py            # NPC Agent系统
├── prompt_builder.py    # Prompt模板加载与渲染
├── prompts/             # system/runtime/summary prompt模板
├── batch_generator.py   # 批量对话生成器
├── safety.py            # 最小安全编排层
├── state_manager.py     # NPC状态管理器
├── test_api.py          # API测试脚本
├── requirements.txt     # Python依赖
└── README.md           # 本文件
```

### Prompt模块化说明

当前 `agents.py` 的主聊天链、摘要链仍保持原有行为与接口不变，但 prompt 文本已抽离到 `backend/prompts/` 下统一管理，并通过 `prompt_builder.py` 渲染。

- `system/`：NPC系统提示词与安全边界
- `runtime/`：好感度上下文与回答约束
- `summary/`：摘要记忆生成提示词

兼容性约束：

- 不修改 `/chat`、`/npcs`、`/memories`、`/affinity` 这些现有接口形状
- 模板文件缺失时会回退到内置 prompt，避免因模板问题直接中断聊天链路
- 现有安全、记忆、好感度流程继续沿用，只增加可维护性和日志可观察性

## 🎨 核心设计

### 批量对话生成
为了降低API成本和延迟,系统采用批量生成策略:

**传统方式**:
- 3个NPC × 每30秒 = 6次API调用/分钟
- 每小时: 360次调用

**批量方式**:
- 1次批量调用/30秒 = 2次API调用/分钟
- 每小时: 120次调用
- **成本降低66%!**

### 工作流程
```
1. 定时器触发(30秒)
   ↓
2. 批量生成器构建提示词
   ↓
3. 一次LLM调用生成所有NPC对话
   ↓
4. 解析JSON响应
   ↓
5. 更新状态管理器缓存
   ↓
6. Godot客户端定时获取状态
```

### 对话安全编排
`/chat` 当前在不改变协议的前提下,增加了一层最小侵入安全编排:

1. 用户输入先走规则扫描
2. 可疑输入再走一次轻量 LLM 审核
3. 组合态 prompt 只构造短摘要做审核,不把完整 prompt 再交给审核模型
4. 主模型输出后执行输出审核与泄露检测
5. 记忆写入前按 `allow_long_term / short_term_only / drop` 分流
6. 摘要生成前后分别执行脱敏与安全检查

目标是尽量不打扰普通对话,只对明显违规、越狱和隐私固化场景做拦截或改写。

记忆写入策略说明:

- `allow_long_term`: 正常写入工作记忆,并允许进入后续摘要链路。
- `short_term_only`: 写入脱敏后的工作记忆,但不进入摘要队列。
- `drop`: 不写入普通记忆,用于自残方法、诈骗流程、未成年人性内容等高风险细节。

## 🔧 配置说明

### config.py
```python
# NPC更新间隔(秒)
NPC_UPDATE_INTERVAL = 30

# LLM配置
OPENAI_MODEL = "gpt-4o-mini"  # 推荐使用mini版本降低成本
```

### 调整更新频率
修改`config.py`中的`NPC_UPDATE_INTERVAL`:
- 开发测试: 10秒
- 正式运行: 30-60秒
- 低成本模式: 120秒

## 🐛 故障排查

### 问题1: 启动失败
```
❌ LLM初始化失败
```
**解决**: 检查OPENAI_API_KEY环境变量是否设置

### 问题2: 对话无响应
```
⚠️ 将使用预设对话模式
```
**解决**: 系统自动降级到预设对话,不影响基本功能

### 问题3: CORS错误
**解决**: 检查`config.py`中的`CORS_ORIGINS`配置

## 📝 开发建议

### 添加新NPC
1. 在`agents.py`的`NPC_ROLES`中添加配置
2. 在`batch_generator.py`的`preset_dialogues`中添加预设对话
3. 重启服务

### 自定义对话风格
修改`agents.py`中的`create_system_prompt`函数

### 调整批量生成提示词
修改`batch_generator.py`中的`_build_batch_prompt`函数

## 📄 许可证

本项目遵循 HelloAgents 项目的开源协议。
