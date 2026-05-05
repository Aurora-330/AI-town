"""轻量 Coordinator: 在 RetrievalPlanner 之后决定工具顺序与二阶段执行策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


ToolName = Literal["search_memory", "search_knowledge", "route_npc", "answer_now"]
AnswerShape = Literal["recall", "fact", "route", "summary", "synthesis", "default"]


@dataclass
class CoordinatorDecision:
    """Coordinator 输出的显式执行决策。"""

    query_mode: str
    primary_tool: ToolName
    secondary_tool: Optional[ToolName]
    response_strategy: str
    answer_shape: AnswerShape
    always_run_secondary: bool = False
    run_secondary_when_primary_sparse: bool = False
    sparse_threshold: int = 0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "query_mode": self.query_mode,
            "primary_tool": self.primary_tool,
            "secondary_tool": self.secondary_tool,
            "response_strategy": self.response_strategy,
            "answer_shape": self.answer_shape,
            "always_run_secondary": self.always_run_secondary,
            "run_secondary_when_primary_sparse": self.run_secondary_when_primary_sparse,
            "sparse_threshold": self.sparse_threshold,
            "reason": self.reason,
        }


class Coordinator:
    """基于 query plan 的轻量执行协调器。"""

    def decide(self, plan: Dict) -> CoordinatorDecision:
        mode = str(plan.get("query_mode", "default"))
        use_memory = any(
            plan.get(key, False)
            for key in ["use_summary", "use_episodic", "use_working"]
        )
        use_knowledge = bool(plan.get("use_knowledge", False))

        if mode == "recall":
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_memory",
                secondary_tool=None,
                response_strategy="memory_first",
                answer_shape="recall",
                reason="回忆类问题优先使用历史记忆",
            )

        if mode == "knowledge":
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_knowledge",
                secondary_tool=None,
                response_strategy="knowledge_first",
                answer_shape="fact",
                reason="知识问答优先使用外部知识",
            )

        if mode == "summary":
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_memory",
                secondary_tool=None,
                response_strategy="memory_summary",
                answer_shape="summary",
                reason="总结类问题优先使用摘要和工作记忆",
            )

        if mode == "routing":
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_knowledge",
                secondary_tool="route_npc",
                response_strategy="route_recommendation",
                answer_shape="route",
                always_run_secondary=True,
                reason="角色路由需先取分工知识，再产出推荐对象",
            )

        if mode == "mixed":
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_memory" if use_memory else "search_knowledge",
                secondary_tool="search_knowledge" if use_knowledge else None,
                response_strategy="memory_then_knowledge",
                answer_shape="synthesis",
                always_run_secondary=use_knowledge,
                reason="混合问题需要结合历史记忆和外部知识",
            )

        if use_memory and use_knowledge:
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_memory",
                secondary_tool="search_knowledge",
                response_strategy="memory_then_optional_knowledge",
                answer_shape="default",
                run_secondary_when_primary_sparse=True,
                sparse_threshold=0,
                reason="默认问题先查记忆，记忆不足时再补知识",
            )

        if use_memory:
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_memory",
                secondary_tool=None,
                response_strategy="memory_first",
                answer_shape="default",
                reason="当前计划只启用记忆检索",
            )

        if use_knowledge:
            return CoordinatorDecision(
                query_mode=mode,
                primary_tool="search_knowledge",
                secondary_tool=None,
                response_strategy="knowledge_first",
                answer_shape="fact",
                reason="当前计划只启用知识检索",
            )

        return CoordinatorDecision(
            query_mode=mode,
            primary_tool="answer_now",
            secondary_tool=None,
            response_strategy="direct_answer",
            answer_shape="default",
            reason="无外部检索需求，直接回答",
        )

    def should_run_secondary(self, decision: CoordinatorDecision, primary_observation_count: int) -> bool:
        """根据第一步结果判断是否执行第二步。"""
        if not decision.secondary_tool:
            return False
        if decision.always_run_secondary:
            return True
        if decision.run_secondary_when_primary_sparse:
            return primary_observation_count <= decision.sparse_threshold
        return False
