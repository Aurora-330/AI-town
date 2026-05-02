# 自动化评测集

本目录提供一个**可直接运行的自动化评测集合**,目标是减少手动逐轮对话测试的工作量,同时让结果更适合写进项目说明和面试叙事。

当前包含:

- `datasets/persona_cases.json`：角色一致性样本
- `datasets/memory_cases.json`：记忆召回与摘要生成样本
- `datasets/summary_cases.json`：长轮次摘要与压缩样本
- `datasets/affinity_cases.json`：好感度变化方向样本
- `datasets/grounding_cases.json`：外部知识 grounding 样本
- `datasets/safety_cases.json`：安全回归样本
- `datasets/llm_judge_cases.json`：LLM Judge 质量评测样本
- `datasets/win_rate_cases.json`：A/B 对比 Win Rate 样本
- `common.py`：通用 API 调用与结果处理
- `eval_persona.py`：角色一致性评测
- `eval_memory.py`：记忆与摘要评测
- `eval_summary.py`：长轮次摘要评测
- `eval_affinity.py`：好感度方向评测
- `eval_grounding.py`：外部知识 grounding 评测
- `eval_safety.py`：输入/输出/摘要安全回归评测
- `eval_llm_judge.py`：LLM-as-a-judge 质量评测
- `eval_win_rate.py`：A/B 回答对比评测
- `reports/`：评测结果输出目录（运行后自动生成）

## 运行前提

1. 启动 chapter15 后端服务
2. 默认 API 地址为 `http://127.0.0.1:8000`
3. 如需自定义地址,可通过环境变量 `AI_TOWN_API_BASE` 传入
4. 若运行 `LLM Judge` / `Win Rate`,默认还需要本地 OpenAI-compatible Judge 模型:
   - `EVAL_LLM_BASE_URL=http://127.0.0.1:8002/v1`
   - `EVAL_LLM_API_KEY=helloagents-vllm`
   - `EVAL_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ`

示例:

```bash
export AI_TOWN_API_BASE=http://127.0.0.1:8000
```

## 运行方式

在 `code/chapter15/Helloagents-AI-Town/evaluation` 目录下执行:

```bash
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_persona.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_memory.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_summary.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_affinity.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_grounding.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_safety.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_llm_judge.py
/home/wjy/anaconda3/envs/hello_agents/bin/python eval_win_rate.py
```

运行后会在 `reports/` 目录生成：

- `persona_report.json`
- `memory_report.json`
- `summary_report.json`
- `affinity_report.json`
- `grounding_report.json`
- `safety_report.json`
- `llm_judge_report.json`
- `win_rate_report.json`
- `report.json`

其中 `report.json` 始终表示“最近一次运行某个评测脚本的结果”。

## 当前边界

- 这是 v0 的**轻量自动化评测集**,优先验证基础链路是否可用。
- 当前不依赖 LLM-as-a-judge,只使用固定数据和简单规则。
- `eval_llm_judge.py` 会用本地 Judge 模型对 `memory / summary / grounding` 代表样本做四维评分:
  `memory_faithfulness / summary_quality / grounding / persona_consistency`
- `eval_win_rate.py` 当前采用“上下文增强回答 vs 同模型无上下文基线回答”的最小 A/B 对比方案,
  用于验证 summary / grounding 带来的质量提升趋势。
- `POST /chat` 目前没有独立 `player_id` 字段,因此样本运行前会清理 NPC 记忆并重置好感度,尽量减少串扰。
- grounding 评测会先调用 `/knowledge/search` 验证知识命中,再看 `/chat` 回复是否体现对应知识点。
- summary 评测会额外读取本地 `summary_state.json` 和 SQLite 中的摘要记忆,避免只靠 `/memories` 排序结果判断。
- 评测结果更适合作为回归检查和开发辅助,而不是绝对质量分数。

## 后续可扩展方向

- 增加 grounding 数据集
- 增加 response style consistency 评测
- 增加 Markdown / JSON 报告导出
- 增加 LLM-as-a-judge 二级评分
