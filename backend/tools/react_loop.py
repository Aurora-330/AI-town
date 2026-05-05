"""受控 3-step ReAct loop：只覆盖 routing / mixed / 一部分 default。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ReactActivationDecision:
    should_activate: bool
    rule: str
    reason: str

    def to_dict(self) -> Dict:
        return {
            "should_activate": self.should_activate,
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass
class ReactTraceStep:
    step_index: int
    thought: str
    action: str
    observation_count: int
    observation_summary: str
    input_tokens_est: int

    def to_dict(self) -> Dict:
        return {
            "step_index": self.step_index,
            "thought": self.thought,
            "action": self.action,
            "observation_count": self.observation_count,
            "observation_summary": self.observation_summary,
            "input_tokens_est": self.input_tokens_est,
        }


class ControlledReactLoop:
    """Coordinator 驱动的受控 react loop，最多 3 步。"""

    MAX_STEPS = 3
    OBSERVATION_MAX_TOKENS = 120
    STEP_INPUT_MAX_TOKENS = 220

    def __init__(self, coordinator, dialogue_tools, token_counter):
        self.coordinator = coordinator
        self.dialogue_tools = dialogue_tools
        self.token_counter = token_counter

    def should_activate(self, query_analysis: Dict) -> bool:
        return self.analyze_activation(query_analysis).should_activate

    def analyze_activation(self, query_analysis: Dict) -> ReactActivationDecision:
        mode = str(query_analysis.get("query_mode", "default"))
        if mode in {"routing", "mixed"}:
            return ReactActivationDecision(
                should_activate=True,
                rule=f"{mode}_always",
                reason=f"{mode} 模式默认进入 controlled react loop",
            )
        if mode == "default" and self._should_activate_default(query_analysis):
            query = str(query_analysis.get("original_query") or query_analysis.get("rewrite_query") or "").strip()
            return ReactActivationDecision(
                should_activate=True,
                rule="default_structural",
                reason=f"default 结构说明型问题进入 loop: {query[:40]}",
            )
        return ReactActivationDecision(
            should_activate=False,
            rule=f"{mode}_fallback_static",
            reason=f"{mode} 模式当前不满足 react 激活条件，回退静态 coordinator",
        )

    def run(
        self,
        npc_name: str,
        player_id: str,
        query: str,
        query_analysis: Dict,
        memory_manager,
    ) -> Dict:
        decision = self.coordinator.decide(query_analysis)
        trace: List[ReactTraceStep] = []
        summary_memories = []
        episodic_memories = []
        working_memories = []
        knowledge_chunks = []
        memory_debug = {"query": query, "memory_budget": 0, "layers": []}
        knowledge_debug = None
        routing_recommended_npc = ""

        current_tool = decision.primary_tool
        step_index = 1
        primary_observation_count = 0
        secondary_used = False

        while current_tool and current_tool != "answer_now" and step_index <= self.MAX_STEPS:
            result = self.dialogue_tools.execute(
                current_tool,
                **self._build_tool_kwargs(
                    tool_name=current_tool,
                    npc_name=npc_name,
                    player_id=player_id,
                    query=query,
                    query_analysis=query_analysis,
                    memory_manager=memory_manager,
                    knowledge_chunks=knowledge_chunks,
                ),
            )
            observation_count = int(result.get("observation_count", 0))
            observation_summary = self.dialogue_tools.summarize_observation(
                current_tool,
                result,
                max_tokens=self.OBSERVATION_MAX_TOKENS,
            )
            input_tokens_est = min(
                self.STEP_INPUT_MAX_TOKENS,
                self.token_counter.count_text_tokens(f"{query}\n{observation_summary}"),
            )
            trace.append(
                ReactTraceStep(
                    step_index=step_index,
                    thought=self._build_step_thought(
                        tool_name=current_tool,
                        query_mode=query_analysis.get("query_mode", "default"),
                        secondary_used=secondary_used,
                    ),
                    action=current_tool,
                    observation_count=observation_count,
                    observation_summary=observation_summary,
                    input_tokens_est=input_tokens_est,
                )
            )

            if current_tool == "search_memory":
                summary_memories = result["summary_memories"]
                episodic_memories = result["episodic_memories"]
                working_memories = result["working_memories"]
                memory_debug = result["memory_debug"]
            elif current_tool == "search_knowledge":
                knowledge_chunks = result["knowledge_chunks"]
                knowledge_debug = result["knowledge_debug"]
            elif current_tool == "route_npc":
                routing_recommended_npc = result.get("recommended_npc", "")

            if step_index == 1:
                primary_observation_count = observation_count

            next_tool = None
            if not secondary_used and self.coordinator.should_run_secondary(decision, primary_observation_count):
                next_tool = decision.secondary_tool
                secondary_used = True
            current_tool = next_tool
            step_index += 1

        return {
            "summary_memories": summary_memories,
            "episodic_memories": episodic_memories,
            "working_memories": working_memories,
            "knowledge_chunks": knowledge_chunks,
            "memory_debug": memory_debug,
            "knowledge_debug": knowledge_debug,
            "routing_recommended_npc": routing_recommended_npc,
            "trace": [item.to_dict() for item in trace],
        }

    def _build_tool_kwargs(
        self,
        tool_name: str,
        npc_name: str,
        player_id: str,
        query: str,
        query_analysis: Dict,
        memory_manager,
        knowledge_chunks,
    ) -> Dict:
        if tool_name == "search_memory":
            return {
                "memory_manager": memory_manager,
                "npc_name": npc_name,
                "query": query,
                "player_id": player_id,
                "retrieval_plan": query_analysis,
            }
        if tool_name == "search_knowledge":
            return {
                "npc_name": npc_name,
                "query": query,
                "player_id": player_id,
                "query_mode": query_analysis.get("query_mode", "default"),
                "knowledge_k": int(query_analysis.get("knowledge_k", 1)),
            }
        if tool_name == "route_npc":
            return {"knowledge_chunks": knowledge_chunks}
        if tool_name == "get_summary_state":
            return {"npc_name": npc_name, "player_id": player_id}
        return {}

    def _build_step_thought(self, tool_name: str, query_mode: str, secondary_used: bool) -> str:
        if tool_name == "search_memory":
            return "先查历史记忆，确认用户偏好、事实和未完成事项。"
        if tool_name == "search_knowledge":
            return "需要外部知识来补充分工、规则或事实定义。"
        if tool_name == "route_npc":
            return "已有知识证据，下一步产出首选角色推荐。"
        if secondary_used:
            return f"{query_mode} 模式下继续补证据。"
        return f"{query_mode} 模式下开始执行。"

    def _should_activate_default(self, query_analysis: Dict) -> bool:
        if not query_analysis.get("use_summary", False):
            return False
        if not query_analysis.get("use_knowledge", False):
            return False

        query = str(query_analysis.get("original_query") or query_analysis.get("rewrite_query") or "").strip()
        if not query:
            return False

        structural_markers = [
            "通常应该", "一般应该", "包含哪些部分", "包含什么部分", "怎么写", "怎么组织", "怎么展开", "模板",
            "应该包含", "需要包含", "分成哪几部分", "先写什么",
        ]
        memory_first_markers = [
            "最容易卡在哪", "我容易卡在哪", "你觉得我这次", "我这次最容易", "我哪里最容易", "我会卡住",
        ]
        if any(marker in query for marker in memory_first_markers):
            return False
        return any(marker in query for marker in structural_markers)
