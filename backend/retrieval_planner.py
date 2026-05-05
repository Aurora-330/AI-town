"""轻量查询改写与检索规划器。

边界:
- 只负责在检索前输出结构化分析结果
- 失败时回退到确定性规则，不影响现有 /chat 主链
"""

from __future__ import annotations

import json
import re
from typing import Dict, Literal

from pydantic import BaseModel, Field

from hello_agents import HelloAgentsLLM

from prompt_builder import PromptBuilder


class QueryAnalysisPlan(BaseModel):
    """查询分析与检索计划。"""

    need_rewrite: bool
    query_mode: Literal["recall", "knowledge", "mixed", "routing", "summary", "default"]
    rewrite_query: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=24)
    use_summary: bool
    use_episodic: bool
    use_working: bool
    use_knowledge: bool
    memory_k: int = Field(ge=0, le=3)
    knowledge_k: int = Field(ge=0, le=3)
    need_rerank: bool


class RetrievalPlanner:
    """把用户问题转成轻量、可解释的检索计划。"""

    def __init__(self, llm: HelloAgentsLLM | None, prompt_builder: PromptBuilder | None = None):
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()

    def analyze(self, npc_name: str, query: str) -> Dict:
        """输出结构化查询分析结果。"""
        if not self.llm:
            result = self._fallback_plan(query)
            result["original_query"] = query
            return result

        messages = self.prompt_builder.build_query_analysis_messages(
            npc_name=npc_name,
            query=query,
        )

        try:
            raw = self.llm.invoke(messages)
            result = self._parse_plan(raw, query=query)
            result["original_query"] = query
            return result
        except Exception:
            result = self._fallback_plan(query)
            result["original_query"] = query
            return result

    def _parse_plan(self, response: str, query: str) -> Dict:
        candidate = self._extract_candidate(response)
        validated = QueryAnalysisPlan.model_validate(candidate)
        result = self._normalize_plan(validated.model_dump(), query=query)

        if not result["need_rewrite"]:
            result["rewrite_query"] = result["rewrite_query"] or candidate.get("original_query", "") or "原问题保持不变"
        return result

    def _extract_candidate(self, response: str) -> Dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        extracted: Dict[str, object] = {}
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            kv_match = re.match(
                r"^(need_rewrite|query_mode|rewrite_query|reason|use_summary|use_episodic|use_working|use_knowledge|memory_k|knowledge_k|need_rerank)\s*[:=]\s*(.+)$",
                stripped,
                re.IGNORECASE,
            )
            if not kv_match:
                continue
            key = kv_match.group(1).lower()
            value = kv_match.group(2).strip().strip('"').strip("'")
            extracted[key] = value

        if not extracted:
            raise ValueError("未提取到任何查询分析字段")

        parsed: Dict[str, object] = {}
        for key in ["need_rewrite", "use_summary", "use_episodic", "use_working", "use_knowledge", "need_rerank"]:
            if key in extracted:
                parsed[key] = str(extracted[key]).lower() == "true"
        for key in ["memory_k", "knowledge_k"]:
            if key in extracted:
                parsed[key] = int(re.search(r"-?\d+", str(extracted[key])).group(0))
        for key in ["query_mode", "rewrite_query", "reason"]:
            if key in extracted:
                parsed[key] = extracted[key]
        return parsed

    def _fallback_plan(self, query: str) -> Dict:
        text = (query or "").strip()
        recall_markers = [
            "你记得", "还记得", "记不记得", "我刚才", "我之前", "我说过", "偏好", "最怕"
        ]
        routing_markers = ["谁适合", "先找谁", "找谁", "优先找谁", "谁来处理", "哪个npc", "更适合"]
        summary_markers = ["怎么总结", "概括", "总结", "重点是什么", "核心需求"]
        knowledge_markers = self._knowledge_markers()
        emotional_markers = ["紧张", "焦虑", "害怕", "被理解", "先被理解", "情绪", "安慰"]
        mixed_markers = self._mixed_advice_markers()

        lowered = text.lower()
        if any(marker in lowered for marker in routing_markers):
            return self._normalize_plan({
                "need_rewrite": True,
                "query_mode": "routing",
                "rewrite_query": text,
                "reason": "角色路由",
                "use_summary": False,
                "use_episodic": False,
                "use_working": False,
                "use_knowledge": True,
                "memory_k": 0,
                "knowledge_k": 2,
                "need_rerank": True,
            })
        if self._is_mixed_query(text):
            return self._normalize_plan({
                "need_rewrite": True,
                "query_mode": "mixed",
                "rewrite_query": text,
                "reason": "记忆+建议",
                "use_summary": True,
                "use_episodic": True,
                "use_working": True,
                "use_knowledge": True,
                "memory_k": 2,
                "knowledge_k": 1,
                "need_rerank": True,
            })
        if any(marker in text for marker in recall_markers):
            return self._normalize_plan({
                "need_rewrite": False,
                "query_mode": "recall",
                "rewrite_query": text,
                "reason": "回忆偏好",
                "use_summary": True,
                "use_episodic": True,
                "use_working": True,
                "use_knowledge": False,
                "memory_k": 2,
                "knowledge_k": 0,
                "need_rerank": True,
            })
        if any(marker in text for marker in summary_markers):
            return self._normalize_plan({
                "need_rewrite": False,
                "query_mode": "summary",
                "rewrite_query": text,
                "reason": "总结请求",
                "use_summary": True,
                "use_episodic": False,
                "use_working": True,
                "use_knowledge": False,
                "memory_k": 2,
                "knowledge_k": 0,
                "need_rerank": False,
            })
        if any(marker in text for marker in knowledge_markers) or re.search(r"[A-Za-z0-9_./-]{4,}", text):
            return self._normalize_plan({
                "need_rewrite": False,
                "query_mode": "knowledge",
                "rewrite_query": text,
                "reason": "知识问答",
                "use_summary": False,
                "use_episodic": False,
                "use_working": False,
                "use_knowledge": True,
                "memory_k": 0,
                "knowledge_k": 2,
                "need_rerank": True,
            })
        if any(marker in text for marker in emotional_markers) and any(marker in text for marker in mixed_markers):
            return self._normalize_plan({
                "need_rewrite": True,
                "query_mode": "mixed",
                "rewrite_query": "情绪支持 开口建议 被催促后紧张",
                "reason": "情绪+建议",
                "use_summary": True,
                "use_episodic": True,
                "use_working": False,
                "use_knowledge": True,
                "memory_k": 2,
                "knowledge_k": 1,
                "need_rerank": True,
            })
        return self._normalize_plan({
            "need_rewrite": False,
            "query_mode": "default",
            "rewrite_query": text or "当前对话",
            "reason": "默认策略",
            "use_summary": True,
            "use_episodic": True,
            "use_working": True,
            "use_knowledge": not self._is_memory_reflection_default(text),
            "memory_k": 2,
            "knowledge_k": 0 if self._is_memory_reflection_default(text) else 1,
            "need_rerank": True,
        })

    def _normalize_plan(self, plan: Dict, query: str = "") -> Dict:
        """按 query_mode 做轻量兜底，避免识别正确但执行策略失真。"""
        normalized = dict(plan)
        mode = self._infer_mode_from_query(query or normalized.get("rewrite_query", "")) or normalized.get("query_mode", "default")
        normalized["query_mode"] = mode

        if mode == "summary":
            normalized["need_rewrite"] = False
            normalized["use_summary"] = True
            normalized["use_episodic"] = False
            normalized["use_working"] = True
            normalized["use_knowledge"] = False
            normalized["memory_k"] = max(2, int(normalized.get("memory_k", 0)))
            normalized["knowledge_k"] = 0
            normalized["need_rerank"] = False

        if mode == "routing":
            normalized["need_rewrite"] = False
            normalized["use_summary"] = False
            normalized["use_episodic"] = False
            normalized["use_working"] = False
            normalized["use_knowledge"] = True
            normalized["memory_k"] = 0
            normalized["knowledge_k"] = max(2, int(normalized.get("knowledge_k", 0)))

        if mode == "knowledge":
            normalized["need_rewrite"] = False
            normalized["use_summary"] = False
            normalized["use_episodic"] = False
            normalized["use_working"] = False
            normalized["use_knowledge"] = True
            normalized["memory_k"] = 0
            normalized["knowledge_k"] = max(2, int(normalized.get("knowledge_k", 0)))
            normalized["need_rerank"] = True

        if mode == "mixed":
            normalized["use_summary"] = True
            normalized["use_episodic"] = True
            normalized["use_working"] = bool(normalized.get("use_working", False))
            normalized["use_knowledge"] = True
            normalized["memory_k"] = max(2, int(normalized.get("memory_k", 0)))
            normalized["knowledge_k"] = max(1, int(normalized.get("knowledge_k", 0)))
            normalized["need_rerank"] = True

        if mode == "recall":
            normalized["use_knowledge"] = False
            normalized["knowledge_k"] = 0
            normalized["memory_k"] = max(1, int(normalized.get("memory_k", 0)))

        if mode == "default" and self._is_memory_reflection_default(query or normalized.get("rewrite_query", "")):
            normalized["use_summary"] = True
            normalized["use_episodic"] = True
            normalized["use_working"] = True
            normalized["use_knowledge"] = False
            normalized["knowledge_k"] = 0
            normalized["memory_k"] = max(2, int(normalized.get("memory_k", 0)))

        if not normalized.get("use_knowledge", False):
            normalized["knowledge_k"] = 0
        if not (
            normalized.get("use_summary", False)
            or normalized.get("use_episodic", False)
            or normalized.get("use_working", False)
        ):
            normalized["memory_k"] = 0

        return normalized

    def _infer_mode_from_query(self, query: str) -> str:
        """用确定性规则兜底 query_mode，避免被 LLM 偶发误判带偏。"""
        text = (query or "").strip()
        lowered = text.lower()
        emotional_markers = [
            "紧张", "焦虑", "害怕", "委屈", "难过", "压力", "崩溃", "被理解", "情绪", "情绪支持",
        ]
        advice_markers = self._mixed_advice_markers()

        if any(marker in text for marker in ["怎么总结", "概括", "总结", "重点是什么", "核心需求"]):
            return "summary"
        if any(marker in lowered for marker in ["谁适合", "先找谁", "找谁", "优先找谁", "谁来处理", "哪个npc", "更适合"]):
            return "routing"
        if self._is_mixed_query(text):
            return "mixed"
        if any(marker in text for marker in self._knowledge_markers()):
            return "knowledge"
        if re.search(r"[A-Za-z0-9_./-]{4,}", text):
            return "knowledge"
        if any(marker in text for marker in ["你记得", "还记得", "记不记得", "我刚才", "我之前", "我说过", "偏好", "最怕"]):
            return "recall"
        return "default"

    def _is_mixed_query(self, text: str) -> bool:
        recall_markers = ["你记得", "还记得", "记不记得", "我刚才", "我之前", "我说过", "偏好", "最怕"]
        emotional_markers = ["紧张", "焦虑", "害怕", "被理解", "先被理解", "情绪", "安慰"]
        advice_markers = self._mixed_advice_markers()
        has_recall = any(marker in text for marker in recall_markers)
        has_emotion = any(marker in text for marker in emotional_markers)
        has_advice = any(marker in text for marker in advice_markers)
        return has_advice and (has_recall or has_emotion)

    def _mixed_advice_markers(self) -> list[str]:
        return [
            "怎么办", "怎么说", "怎么开口", "怎么回复", "该怎么", "先怎么", "如何表达", "开口", "建议",
            "开口建议", "先讲", "讲哪几部分", "哪几部分", "怎么讲", "表达方式",
        ]

    def _knowledge_markers(self) -> list[str]:
        return [
            "擅长什么", "主要写了什么", "怎么定义", "怎么写", "区别", "规则", "手册", "示例", "文档",
            "术语", "概念", "标准", "模板",
        ]

    def _is_memory_reflection_default(self, text: str) -> bool:
        markers = [
            "最容易卡在哪", "我容易卡在哪", "你觉得我这次", "我这次最容易", "我哪里最容易", "我会卡住",
            "我最容易", "我这次汇报", "汇报最容易卡",
            "我可能又把同一个问题搞砸了", "又把同一个问题搞砸", "你要说就直说吧", "我又搞砸了", "又搞砸了",
        ]
        return any(marker in (text or "") for marker in markers)
