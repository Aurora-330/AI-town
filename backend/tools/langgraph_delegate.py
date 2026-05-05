"""LangGraph V1 最小骨架：只实现 Delegate Lookup 这条路径。

设计目标：
- 默认前台单角色输出
- 需要时把“查事实/查策略”委托给后台 worker
- 节点之间只传 observation card，不传全文
- 环境未安装 langgraph 时，仍可用同一套节点契约走 fallback runner
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

try:  # pragma: no cover - 当前环境可能还未安装 langgraph
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    StateGraph = None


class DelegateLookupState(TypedDict, total=False):
    user_message: str
    active_speaker: str
    player_id: str
    intent_type: str
    query_mode: str
    orchestration_mode: Literal["direct", "delegate"]
    needs_delegate: bool
    delegate_to: str
    delegate_task: str
    observation_cards: List[Dict[str, object]]
    final_style_owner: str
    final_answer: str
    node_trace: List[Dict[str, object]]


@dataclass
class ObservationCard:
    worker: str
    card_type: str
    summary: str
    confidence: str = "medium"
    tokens_est: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "worker": self.worker,
            "card_type": self.card_type,
            "summary": self.summary,
            "confidence": self.confidence,
            "tokens_est": self.tokens_est,
        }


class LangGraphDelegateOrchestrator:
    """Sequence8 / LangGraph 最小编排器。

    当前只覆盖：
    IntentRouter -> DelegatePlanner -> (FenglingArchiveWorker | GuchenStrategyWorker) -> BackToActiveSpeaker
    """

    NODE_TOKEN_BUDGETS = {
        "intent_router": 120,
        "direct_answer": 900,
        "delegate_planner": 120,
        "fengling_archive_worker": 260,
        "guchen_strategy_worker": 280,
        "back_to_active_speaker": 550,
    }

    def __init__(self, manager):
        self.manager = manager
        self._compiled_graph = None

    @property
    def available(self) -> bool:
        return StateGraph is not None

    def invoke(self, user_message: str, active_speaker: str, player_id: str = "player") -> Dict[str, object]:
        initial_state: DelegateLookupState = {
            "user_message": user_message,
            "active_speaker": active_speaker,
            "player_id": player_id,
            "observation_cards": [],
            "node_trace": [],
        }
        if self.available:
            app = self._get_or_compile_graph()
            return app.invoke(initial_state)
        return self._run_fallback(initial_state)

    def build_graph(self):
        if not self.available:
            raise RuntimeError("langgraph 未安装，当前只能使用 fallback runner")

        graph = StateGraph(DelegateLookupState)
        graph.add_node("intent_router", self.intent_router_node)
        graph.add_node("direct_answer", self.direct_answer_node)
        graph.add_node("delegate_planner", self.delegate_planner_node)
        graph.add_node("fengling_archive_worker", self.fengling_archive_worker_node)
        graph.add_node("guchen_strategy_worker", self.guchen_strategy_worker_node)
        graph.add_node("back_to_active_speaker", self.back_to_active_speaker_node)

        graph.add_conditional_edges(
            "intent_router",
            self.route_after_intent_router,
            {
                "direct": "direct_answer",
                "delegate": "delegate_planner",
            },
        )
        graph.add_conditional_edges(
            "delegate_planner",
            self.route_after_delegate_planner,
            {
                "fengling": "fengling_archive_worker",
                "guchen": "guchen_strategy_worker",
                "direct": "direct_answer",
            },
        )
        graph.add_edge("fengling_archive_worker", "back_to_active_speaker")
        graph.add_edge("guchen_strategy_worker", "back_to_active_speaker")
        graph.add_edge("back_to_active_speaker", END)
        graph.add_edge("direct_answer", END)
        graph.set_entry_point("intent_router")
        return graph.compile()

    def _get_or_compile_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def _run_fallback(self, state: DelegateLookupState) -> Dict[str, object]:
        state = self.intent_router_node(state)
        if self.route_after_intent_router(state) == "direct":
            return self.direct_answer_node(state)
        state = self.delegate_planner_node(state)
        worker_route = self.route_after_delegate_planner(state)
        if worker_route == "fengling":
            state = self.fengling_archive_worker_node(state)
        elif worker_route == "guchen":
            state = self.guchen_strategy_worker_node(state)
        else:
            return self.direct_answer_node(state)
        return self.back_to_active_speaker_node(state)

    def route_after_intent_router(self, state: DelegateLookupState) -> str:
        return "delegate" if state.get("orchestration_mode") == "delegate" else "direct"

    def route_after_delegate_planner(self, state: DelegateLookupState) -> str:
        delegate_to = str(state.get("delegate_to", "")).strip()
        if delegate_to == "风泠":
            return "fengling"
        if delegate_to == "顾辰":
            return "guchen"
        return "direct"

    def intent_router_node(self, state: DelegateLookupState) -> DelegateLookupState:
        user_message = str(state.get("user_message", "")).strip()
        active_speaker = str(state.get("active_speaker", "")).strip()
        query_analysis = self.manager._analyze_query(active_speaker, user_message)
        intent_type = self._classify_delegate_intent(user_message)
        delegate_to = self._match_delegate_target(active_speaker, user_message, query_analysis.get("query_mode", "default"))
        orchestration_mode: Literal["direct", "delegate"] = "delegate" if delegate_to else "direct"
        next_state: DelegateLookupState = dict(state)
        next_state.update(
            {
                "intent_type": intent_type,
                "query_mode": str(query_analysis.get("query_mode", "default")),
                "orchestration_mode": orchestration_mode,
                "needs_delegate": bool(delegate_to),
                "delegate_to": delegate_to,
                "final_style_owner": active_speaker,
            }
        )
        self._append_node_trace(next_state, "intent_router", f"mode={orchestration_mode}, delegate_to={delegate_to or 'none'}")
        return next_state

    def direct_answer_node(self, state: DelegateLookupState) -> DelegateLookupState:
        active_speaker = str(state.get("active_speaker", "")).strip()
        final_answer = self._build_direct_stub(active_speaker, str(state.get("user_message", "")))
        next_state: DelegateLookupState = dict(state)
        next_state["final_answer"] = final_answer
        self._append_node_trace(next_state, "direct_answer", "single-speaker direct reply")
        return next_state

    def delegate_planner_node(self, state: DelegateLookupState) -> DelegateLookupState:
        active_speaker = str(state.get("active_speaker", "")).strip()
        user_message = str(state.get("user_message", "")).strip()
        query_mode = str(state.get("query_mode", "default"))
        delegate_to = str(state.get("delegate_to", "")).strip() or self._match_delegate_target(active_speaker, user_message, query_mode)
        delegate_task = self._build_delegate_task(delegate_to, user_message, query_mode)
        next_state: DelegateLookupState = dict(state)
        next_state.update(
            {
                "delegate_to": delegate_to,
                "delegate_task": delegate_task,
                "needs_delegate": bool(delegate_to),
            }
        )
        self._append_node_trace(next_state, "delegate_planner", f"delegate_to={delegate_to or 'none'}")
        return next_state

    def fengling_archive_worker_node(self, state: DelegateLookupState) -> DelegateLookupState:
        player_id = str(state.get("player_id", "player"))
        query = str(state.get("delegate_task") or state.get("user_message") or "").strip()
        memory_manager = self.manager.memories.get("风泠")
        result = self.manager.dialogue_tools.execute(
            "search_memory",
            memory_manager=memory_manager,
            npc_name="风泠",
            query=query,
            player_id=player_id,
            retrieval_plan={
                "query_mode": "recall",
                "use_summary": True,
                "use_episodic": True,
                "use_working": False,
                "use_knowledge": False,
                "memory_k": 2,
                "knowledge_k": 0,
            },
        )
        summary = self.manager.dialogue_tools.summarize_observation("search_memory", result, max_tokens=60)
        card = ObservationCard(
            worker="风泠",
            card_type="memory_fact",
            summary=summary or "风泠暂时没翻到明确档案结论。",
            confidence="high" if result.get("observation_count", 0) > 0 else "low",
            tokens_est=self.manager.token_counter.count_text_tokens(summary or "风泠暂时没翻到明确档案结论。"),
        )
        next_state: DelegateLookupState = dict(state)
        cards = list(next_state.get("observation_cards", []))
        cards.append(card.to_dict())
        next_state["observation_cards"] = cards
        self._append_node_trace(next_state, "fengling_archive_worker", card.summary)
        return next_state

    def guchen_strategy_worker_node(self, state: DelegateLookupState) -> DelegateLookupState:
        player_id = str(state.get("player_id", "player"))
        query = str(state.get("delegate_task") or state.get("user_message") or "").strip()
        result = self.manager.dialogue_tools.execute(
            "search_knowledge",
            npc_name="顾辰",
            query=query,
            player_id=player_id,
            query_mode="knowledge",
            knowledge_k=1,
        )
        summary = self.manager.dialogue_tools.summarize_observation("search_knowledge", result, max_tokens=70)
        if result.get("observation_count", 0) == 0:
            summary = "顾辰建议先按目标、约束、风险、下一步四段来拆解。"
        card = ObservationCard(
            worker="顾辰",
            card_type="strategy_path",
            summary=summary,
            confidence="medium_high",
            tokens_est=self.manager.token_counter.count_text_tokens(summary),
        )
        next_state: DelegateLookupState = dict(state)
        cards = list(next_state.get("observation_cards", []))
        cards.append(card.to_dict())
        next_state["observation_cards"] = cards
        self._append_node_trace(next_state, "guchen_strategy_worker", card.summary)
        return next_state

    def back_to_active_speaker_node(self, state: DelegateLookupState) -> DelegateLookupState:
        active_speaker = str(state.get("active_speaker", "")).strip()
        delegate_to = str(state.get("delegate_to", "")).strip()
        cards = list(state.get("observation_cards", []))
        summary = cards[-1]["summary"] if cards else "我先帮你理了一下。"
        final_answer = self._compose_back_to_speaker(active_speaker, delegate_to, str(summary))
        next_state: DelegateLookupState = dict(state)
        next_state["final_answer"] = final_answer
        self._append_node_trace(next_state, "back_to_active_speaker", final_answer)
        return next_state

    def _classify_delegate_intent(self, user_message: str) -> str:
        if any(marker in user_message for marker in ["档案", "指标", "上个月", "去年", "记录", "历史", "翻一下", "查一下"]):
            return "archive_lookup"
        if any(marker in user_message for marker in ["路线图", "优先级", "方案", "拆解", "转岗", "离职", "利弊", "选哪个好"]):
            return "strategy_support"
        return "direct"

    def _match_delegate_target(self, active_speaker: str, user_message: str, query_mode: str) -> str:
        hard_history_markers = ["档案", "指标", "上个月", "去年", "记录", "历史", "翻一下", "查一下", "事实"]
        strategy_markers = ["路线图", "优先级", "方案", "拆解", "转岗", "离职", "利弊", "选哪个好", "计划"]

        if any(marker in user_message for marker in hard_history_markers):
            return "风泠"
        if any(marker in user_message for marker in strategy_markers) or query_mode in {"knowledge", "routing"}:
            return "顾辰"
        if active_speaker == "郁米" and query_mode == "recall" and any(marker in user_message for marker in ["项目", "指标", "版本", "时间线"]):
            return "风泠"
        return ""

    def _build_delegate_task(self, delegate_to: str, user_message: str, query_mode: str) -> str:
        if delegate_to == "风泠":
            return f"请帮忙查历史档案/指标事实：{user_message}"
        if delegate_to == "顾辰":
            return f"请帮忙给出结构化方案/拆解建议：{user_message} (mode={query_mode})"
        return ""

    def _build_direct_stub(self, active_speaker: str, user_message: str) -> str:
        if active_speaker == "郁米":
            return "我在呢，你可以继续和我说说你现在最想先处理哪一部分。"
        if active_speaker == "风泠":
            return "我先帮你把事实和线索理清，再继续往下看。"
        if active_speaker == "顾辰":
            return "先把目标和限制说清楚，我再帮你拆路径。"
        return f"{active_speaker} 正在整理你的问题：{user_message[:40]}"

    def _compose_back_to_speaker(self, active_speaker: str, delegate_to: str, summary: str) -> str:
        if active_speaker == "郁米":
            if delegate_to == "风泠":
                return f"风泠刚刚帮我查到啦，{summary}"
            if delegate_to == "顾辰":
                return f"我刚刚让顾辰帮我理了一下，{summary}"
            return f"我帮你理了一下，{summary}"
        if active_speaker == "风泠":
            return f"我补查过了，结论是：{summary}"
        if active_speaker == "顾辰":
            return f"我已经补齐信息了，结论直接给你：{summary}"
        return summary

    def _append_node_trace(self, state: DelegateLookupState, node_name: str, detail: str):
        trace = list(state.get("node_trace", []))
        trace.append(
            {
                "node": node_name,
                "budget": self.NODE_TOKEN_BUDGETS.get(node_name, 0),
                "detail": detail,
            }
        )
        state["node_trace"] = trace


class MultiAgentState(TypedDict, total=False):
    message: str
    player_id: str
    requested_mode: Literal["auto", "reactive_duo", "parallel_b", "serial_a"]
    script_id: Literal["reactive_duo", "parallel_b"]
    duo_pattern: str
    primary_agent: str
    secondary_agent: str
    selected_agents: List[str]
    execution_order: List[str]
    aggregation_strategy: str
    final_answer: str
    return_intermediate: bool
    langgraph_runtime: bool
    agent_outputs: Annotated[List[Dict[str, object]], operator.add]
    node_trace: Annotated[List[Dict[str, object]], operator.add]


class LangGraphMultiAgentOrchestrator:
    """Sequence8 正式多角色编排器。

    轻双人互动:
    Router -> primary -> secondary(final)

    圆桌 B:
    Router -> (Guchen || Fengling) -> Yumi(aggregate/final)
    """

    NODE_TOKEN_BUDGETS = {
        "router": 120,
        "reactive_duo_primary": 360,
        "reactive_duo_secondary": 440,
        "parallel_b_fanout": 80,
        "parallel_b_guchen": 360,
        "parallel_b_fengling": 360,
        "parallel_b_yumi": 520,
    }

    def __init__(self, manager):
        self.manager = manager
        self._compiled_graph = None

    @property
    def available(self) -> bool:
        return StateGraph is not None

    def invoke(
        self,
        user_message: str,
        player_id: str = "player",
        mode: Literal["auto", "reactive_duo", "parallel_b", "serial_a"] = "auto",
        selected_agents: Optional[List[str]] = None,
        return_intermediate: bool = True,
    ) -> Dict[str, object]:
        initial_state: MultiAgentState = {
            "message": user_message,
            "player_id": player_id,
            "requested_mode": mode,
            "selected_agents": list(selected_agents or []),
            "return_intermediate": return_intermediate,
            "agent_outputs": [],
            "node_trace": [],
            "langgraph_runtime": self.available,
        }
        if self.available:
            app = self._get_or_compile_graph()
            return app.invoke(initial_state)
        return self._run_fallback(initial_state)

    def build_graph(self):
        if not self.available:
            raise RuntimeError("langgraph 未安装，当前只能使用 fallback runner")

        graph = StateGraph(MultiAgentState)
        graph.add_node("router", self.router_node)
        graph.add_node("reactive_duo_primary", self.reactive_duo_primary_node)
        graph.add_node("reactive_duo_secondary", self.reactive_duo_secondary_node)
        graph.add_node("parallel_b_fanout", self.parallel_b_fanout_node)
        graph.add_node("parallel_b_guchen", self.parallel_b_guchen_node)
        graph.add_node("parallel_b_fengling", self.parallel_b_fengling_node)
        graph.add_node("parallel_b_yumi", self.parallel_b_yumi_node)

        graph.add_conditional_edges(
            "router",
            self.route_after_router,
            {
                "reactive_duo": "reactive_duo_primary",
                "parallel_b": "parallel_b_fanout",
            },
        )
        graph.add_edge("reactive_duo_primary", "reactive_duo_secondary")
        graph.add_edge("reactive_duo_secondary", END)

        graph.add_edge("parallel_b_fanout", "parallel_b_guchen")
        graph.add_edge("parallel_b_fanout", "parallel_b_fengling")
        graph.add_edge("parallel_b_guchen", "parallel_b_yumi")
        graph.add_edge("parallel_b_fengling", "parallel_b_yumi")
        graph.add_edge("parallel_b_yumi", END)
        graph.set_entry_point("router")
        return graph.compile()

    def _get_or_compile_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def _run_fallback(self, state: MultiAgentState) -> Dict[str, object]:
        state = self._merge_state(state, self.router_node(state))
        script_id = self.route_after_router(state)
        if script_id == "reactive_duo":
            state = self._merge_state(state, self.reactive_duo_primary_node(state))
            state = self._merge_state(state, self.reactive_duo_secondary_node(state))
            return state

        state = self._merge_state(state, self.parallel_b_fanout_node(state))
        state = self._merge_state(state, self.parallel_b_guchen_node(state))
        state = self._merge_state(state, self.parallel_b_fengling_node(state))
        state = self._merge_state(state, self.parallel_b_yumi_node(state))
        return state

    def route_after_router(self, state: MultiAgentState) -> str:
        return str(state.get("script_id", "reactive_duo"))

    def router_node(self, state: MultiAgentState) -> Dict[str, object]:
        message = str(state.get("message", "")).strip()
        requested_mode = str(state.get("requested_mode", "auto"))
        requested_agents = list(state.get("selected_agents", []))
        script_id = self._select_script(message, requested_mode)
        duo_pattern = self._select_duo_pattern(message, requested_agents) if script_id == "reactive_duo" else ""
        defaults = {
            "reactive_duo": self._duo_default_order(duo_pattern),
            "parallel_b": ["顾辰", "风泠", "郁米"],
        }
        strategies = {
            "reactive_duo": self._duo_strategy(duo_pattern),
            "parallel_b": "parallel_roundtable_to_yumi",
        }
        execution_order = defaults[script_id]
        selected_agents = self._merge_requested_agents(requested_agents, execution_order)
        trace_detail = f"script={script_id}, selected={','.join(selected_agents)}"
        if duo_pattern:
            trace_detail += f", duo_pattern={duo_pattern}"
        return {
            "script_id": script_id,
            "duo_pattern": duo_pattern,
            "primary_agent": execution_order[0],
            "secondary_agent": execution_order[1] if len(execution_order) > 1 else execution_order[0],
            "selected_agents": selected_agents,
            "execution_order": execution_order,
            "aggregation_strategy": strategies[script_id],
            "node_trace": [self._build_trace("router", trace_detail)],
        }

    def reactive_duo_primary_node(self, state: MultiAgentState) -> Dict[str, object]:
        primary_agent = str(state.get("primary_agent", "郁米"))
        duo_pattern = str(state.get("duo_pattern", "support_anchor"))
        extra_context = self._build_reactive_duo_primary_context(duo_pattern, primary_agent)
        stage_name = f"reactive_duo_primary_{primary_agent}"
        return self._run_npc_stage(state, primary_agent, stage_name, extra_context)

    def reactive_duo_secondary_node(self, state: MultiAgentState) -> Dict[str, object]:
        secondary_agent = str(state.get("secondary_agent", "顾辰"))
        duo_pattern = str(state.get("duo_pattern", "support_anchor"))
        prior = self._format_prior_outputs(state, [str(state.get("primary_agent", ""))])
        extra_context = self._build_reactive_duo_secondary_context(duo_pattern, secondary_agent, prior)
        stage_name = f"reactive_duo_secondary_{secondary_agent}"
        result = self._run_npc_stage(state, secondary_agent, stage_name, extra_context)
        message = self._extract_stage_message(result)
        message = self._polish_reactive_duo_secondary_message(message, state)
        if result.get("agent_outputs"):
            result["agent_outputs"][-1]["message"] = message
        result["final_answer"] = message
        result["node_trace"] = result.get("node_trace", []) + [
            self._build_trace("reactive_duo_secondary", f"finalized_by={secondary_agent}")
        ]
        return result

    def parallel_b_fanout_node(self, state: MultiAgentState) -> Dict[str, object]:
        return {
            "node_trace": [
                self._build_trace("parallel_b_fanout", "fanout=顾辰,风泠 -> 郁米")
            ]
        }

    def parallel_b_guchen_node(self, state: MultiAgentState) -> Dict[str, object]:
        extra_context = (
            "你现在处于多角色协作剧本B的并行讨论节点之一。\n"
            "任务：从ROI、成本、成长性、风险控制的角度分析玩家面前的两条路或当前处境。\n"
            "请一定覆盖三点：1. 留岗的收益或代价；2. 换岗的收益或代价；3. 你更倾向哪一边以及为什么。\n"
            "口吻要求：理性、直接、靠谱，不做情绪安抚，不要替别的角色发言。\n"
            "输出要求：3到4句。最后一句必须是明确建议，不能用反问句结尾，不能把决定再丢回给玩家。"
        )
        return self._run_npc_stage(state, "顾辰", "parallel_b_guchen", extra_context)

    def parallel_b_fengling_node(self, state: MultiAgentState) -> Dict[str, object]:
        extra_context = (
            "你现在处于多角色协作剧本B的并行讨论节点之一。\n"
            "任务：从历史行为偏好、过往模式、擅长点和容易误判的地方补充客观线索。\n"
            "请一定覆盖三点：1. 玩家此刻的判断可能受什么状态影响；2. 玩家过往更适合什么决策方式；3. 需要先校正的误判是什么。\n"
            "口吻要求：像在归档、校正和补事实，不要替代顾辰做成本收益分析，也不要做安抚。\n"
            "输出要求：3到4句。最后一句必须给出一个“先确认什么再决定”的提醒，不能用反问句结尾。"
        )
        return self._run_npc_stage(state, "风泠", "parallel_b_fengling", extra_context)

    def parallel_b_yumi_node(self, state: MultiAgentState) -> Dict[str, object]:
        prior = self._format_prior_outputs(state, ["顾辰", "风泠"])
        extra_context = (
            "你现在处于多角色协作剧本B的最终汇总节点。\n"
            "任务：收集顾辰和风泠的意见，把它们整理成玩家最容易接受的表达。\n"
            "要求：保留他们各自的视角差异，可以有一点群聊接力感，但最终必须由你完成收束，给出一个清晰、温柔、可执行的结论。\n"
            "请按这个顺序组织：1. 先用1句接住玩家；2. 用1到2句转述顾辰和风泠各自重点；3. 明确说出你综合后的倾向；4. 给出一个现在就能做的下一步。\n"
            "输出要求：4到6句，最终直接对玩家说话。绝对不要用“你觉得”“你认为”“要不要”这种反问句收尾。\n"
            f"{prior}"
        )
        result = self._run_npc_stage(state, "郁米", "parallel_b_yumi", extra_context)
        message = self._extract_stage_message(result)
        message = self._polish_parallel_b_yumi_message(message, state)
        result["final_answer"] = message
        if result.get("agent_outputs"):
            result["agent_outputs"][-1]["message"] = message
        result["node_trace"] = result.get("node_trace", []) + [
            self._build_trace("parallel_b_yumi", "finalized_by=郁米")
        ]
        return result

    def _run_npc_stage(
        self,
        state: MultiAgentState,
        npc_name: str,
        stage_name: str,
        extra_context: str,
    ) -> Dict[str, object]:
        result = self.manager.chat_with_debug(
            npc_name=npc_name,
            message=str(state.get("message", "")),
            player_id=str(state.get("player_id", "player")),
            execution_mode="auto",
            extra_context=extra_context,
            persist_side_effects=False,
        )
        message = str(result.get("message", ""))
        if self._looks_like_runtime_failure(message):
            message = self._build_stage_fallback(
                npc_name=npc_name,
                stage_name=stage_name,
                user_message=str(state.get("message", "")),
                state=state,
            )
        output = {
            "npc_name": npc_name,
            "stage": stage_name,
            "message": message,
            "query_mode": str(result.get("query_mode", "default")),
            "tool_call_count": int(result.get("tool_call_count", 0)),
            "latency_ms": int(result.get("latency_ms", 0)),
        }
        detail = f"{npc_name} output_len={len(output['message'])} tools={output['tool_call_count']}"
        return {
            "agent_outputs": [output],
            "node_trace": [self._build_trace(stage_name, detail)],
        }

    def _select_script(self, user_message: str, requested_mode: str) -> Literal["reactive_duo", "parallel_b"]:
        if requested_mode == "parallel_b":
            return "parallel_b"
        if requested_mode in {"reactive_duo", "serial_a"}:
            return "reactive_duo"

        parallel_markers = [
            "哪个好", "哪条路", "怎么选", "选哪个", "分别怎么看", "利弊", "比较", "两个方案",
        ]
        if any(marker in user_message for marker in parallel_markers):
            return "parallel_b"
        return "reactive_duo"

    def _select_duo_pattern(self, user_message: str, requested_agents: List[str]) -> str:
        text = (user_message or "").strip()
        hostile_yumi_markers = ["郁米", "你懂我就该", "别装温柔", "发泄", "滚开", "闭嘴", "少来"]
        quiet_fengling_markers = ["风泠", "怎么突然这么安静", "你今天怎么不说话", "你还好吗", "没电", "安静"]
        guchen_tease_markers = ["顾辰", "你嘴怎么这么毒", "说话太难听", "别这么刻薄", "能不能好好说话"]

        requested_pair = [name for name in requested_agents if name in {"郁米", "风泠", "顾辰"}]
        if requested_pair[:2] == ["顾辰", "风泠"] or requested_pair[:2] == ["风泠", "顾辰"]:
            return "tease_guchen"
        if requested_pair[:2] == ["郁米", "风泠"] or requested_pair[:2] == ["风泠", "郁米"]:
            return "comfort_fengling"

        if any(marker in text for marker in hostile_yumi_markers) and any(marker in text for marker in ["郁米", "懂我", "温柔", "发泄"]):
            return "protect_yumi"
        if any(marker in text for marker in quiet_fengling_markers):
            return "comfort_fengling"
        if any(marker in text for marker in guchen_tease_markers):
            return "tease_guchen"
        return "support_anchor"

    def _duo_default_order(self, duo_pattern: str) -> List[str]:
        mapping = {
            "support_anchor": ["郁米", "顾辰"],
            "protect_yumi": ["郁米", "顾辰"],
            "tease_guchen": ["顾辰", "风泠"],
            "comfort_fengling": ["郁米", "风泠"],
        }
        return mapping.get(duo_pattern, ["郁米", "顾辰"])

    def _duo_strategy(self, duo_pattern: str) -> str:
        mapping = {
            "support_anchor": "support_handoff_to_guchen",
            "protect_yumi": "protective_handoff_to_guchen",
            "tease_guchen": "tease_handoff_to_fengling",
            "comfort_fengling": "comfort_handoff_to_fengling",
        }
        return mapping.get(duo_pattern, "support_handoff_to_guchen")

    def _merge_requested_agents(self, requested_agents: List[str], execution_order: List[str]) -> List[str]:
        normalized = [name for name in requested_agents if name in execution_order]
        merged = list(normalized)
        for name in execution_order:
            if name not in merged:
                merged.append(name)
        return merged

    def _format_prior_outputs(self, state: MultiAgentState, npc_names: List[str]) -> str:
        outputs = []
        for item in state.get("agent_outputs", []):
            name = str(item.get("npc_name", ""))
            if name in npc_names:
                outputs.append(f"[{name}] {str(item.get('message', '')).strip()}")
        if not outputs:
            return ""
        return "【前序角色发言】\n" + "\n".join(outputs)

    def _extract_stage_message(self, node_result: Dict[str, object]) -> str:
        outputs = node_result.get("agent_outputs", [])
        if outputs:
            return str(outputs[-1].get("message", ""))
        return ""

    def _polish_reactive_duo_secondary_message(self, message: str, state: MultiAgentState) -> str:
        text = (message or "").strip()
        duo_pattern = str(state.get("duo_pattern", "support_anchor"))
        secondary_agent = str(state.get("secondary_agent", "顾辰"))
        if not text:
            return self._build_stage_fallback(secondary_agent, f"reactive_duo_secondary_{duo_pattern}", str(state.get("message", "")), state)

        if duo_pattern in {"support_anchor", "protect_yumi"}:
            required_markers = ["先", "建议", "别急着", "语气", "边界", "下一步"]
            if not any(marker in text for marker in required_markers):
                return self._build_stage_fallback(secondary_agent, f"reactive_duo_secondary_{duo_pattern}", str(state.get("message", "")), state)

        if duo_pattern == "tease_guchen":
            if "顾辰" not in text and "老顾" not in text and "嘴" not in text:
                return self._build_stage_fallback(secondary_agent, f"reactive_duo_secondary_{duo_pattern}", str(state.get("message", "")), state)

        if duo_pattern == "comfort_fengling":
            if "风泠" not in text and "先" not in text and "不用" not in text:
                return self._build_stage_fallback(secondary_agent, f"reactive_duo_secondary_{duo_pattern}", str(state.get("message", "")), state)

        generic_markers = ["感觉如何", "你觉得", "你认为", "要不要"]
        if any(marker in text for marker in generic_markers) or "？" in text or "?" in text:
            return self._build_stage_fallback(secondary_agent, f"reactive_duo_secondary_{duo_pattern}", str(state.get("message", "")), state)

        return text

    def _polish_parallel_b_yumi_message(self, message: str, state: MultiAgentState) -> str:
        text = (message or "").strip()
        if not text:
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        weak_endings = ["你觉得哪个更重要？", "你认为哪个更重要？", "你怎么选？", "要不要再想想？"]
        if any(text.endswith(item) for item in weak_endings):
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        if "顾辰建议继续留岗，风泠建议换岗" in text:
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        generic_markers = [
            "感觉如何",
            "你现在的感受如何",
            "你现在既想先被理解",
            "又需要知道怎么开口",
            "我在这里陪你",
            "你可以慢慢说",
        ]
        if any(marker in text for marker in generic_markers):
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        # B剧本的最终汇总至少应当体现“圆桌感”和“结论感”
        if "顾辰" not in text and "风泠" not in text:
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        recommendation_markers = ["建议", "倾向", "先", "下一步", "可以", "更适合"]
        if not any(marker in text for marker in recommendation_markers):
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        if "？" in text or "?" in text:
            return self._build_stage_fallback("郁米", "parallel_b_yumi", str(state.get("message", "")), state)

        return text

    def _looks_like_runtime_failure(self, message: str) -> bool:
        failure_markers = [
            "抱歉,我现在有点忙",
            "LLM调用失败",
            "Error code: 502",
            "错误:",
        ]
        return any(marker in message for marker in failure_markers)

    def _build_stage_fallback(
        self,
        npc_name: str,
        stage_name: str,
        user_message: str,
        state: MultiAgentState,
    ) -> str:
        if stage_name == "serial_a_yumi":
            return "你先别急着把自己判死刑。我听得出来你已经被这段时间的压力磨得很累了，先让我陪你把这口气缓下来。"
        if stage_name == "serial_a_fengling":
            return "先别急着把结论直接归到自己能力不行上，这里面很可能有判断偏差。你现在面对的更像是阶段性项目问题、评价环境和连续消耗叠在一起，而不只是你个人出了问题。下一步先确认到底是任务结构失衡，还是外部压力把你的判断拉偏了。"
        if stage_name == "serial_a_guchen":
            return "先别急着把结论一步跳到离职。你现在更需要把“情绪已经撑不住”和“这份工作长期不适合我”拆开判断，这两件事不是一回事。我的建议是先给自己留出两到三天缓冲，再把最近最消耗你的任务、关系和节奏问题列出来。下一步就看一件事：如果休整后你仍然持续失衡，再正式准备转岗或离职，而不是今天就拍板。"
        if stage_name == "parallel_b_guchen":
            return "从成本收益看，继续留岗的好处是现金流和熟悉度还在，代价是你会继续暴露在当前压力源里。换岗的好处是能更快止损，代价是需要重新适应和承担试错成本。我的建议是先别冲动离开，先用一到两周边保留岗位边准备换岗筹码。"
        if stage_name == "parallel_b_fengling":
            return "你现在的判断里有明显的高压放大效应，所以短期崩溃感不能直接等同于长期不适合。按你这种状态，更适合先把触发压力的具体事件拆出来，而不是在最累的时候给整条职业路径下判决。先确认到底是岗位本身不合适，还是最近这段环境把你拖垮了。"
        if stage_name == "parallel_b_yumi":
            prior = self._format_prior_outputs(state, ["顾辰", "风泠"])
            if prior:
                return "我帮你收一下。顾辰在看现实代价和筹码，意思是别在没有准备的时候猛跳；风泠在提醒你，现在的疲惫感会放大判断偏差。综合下来，我更倾向你先稳住自己，再用短时间观察和准备去决定是不是换岗。你现在先做的一步，是把最近最压垮你的三件事写下来，区分哪些是环境问题，哪些才是岗位不匹配。"
            return "我会先把你接住，再陪你把这件事看清楚。我的倾向不是立刻逼你做决定，而是先把情绪和判断分开。你现在先停一下，记下最消耗你的几个压力源，我们再一起判断这到底是暂时太累，还是这条路真的不适合你。"
        if stage_name == "reactive_duo_secondary_support_anchor":
            return "先别急着把结论一步跳到离职。你现在更需要把“压力已经把人压空了”和“这份工作长期不适合我”拆开看，这两件事不是一回事。我的建议是先缓一到两天，再把最近最消耗你的任务和关系问题列出来。下一步只看一个观察点：如果休整后你还是持续失衡，再认真准备转岗或离开。"
        if stage_name == "reactive_duo_secondary_protect_yumi":
            return "收一下你的语气。她不是给你发泄情绪用的，你心情差也不是拿最温柔的人试边界的许可。要说事就把事说明白，别把迁怒包装成“你要是真懂我”。"
        if stage_name == "reactive_duo_secondary_tease_guchen":
            return "老顾这人嘴是硬了点，但他的结论通常不是乱开的。你要是真嫌他的话太刮人，我帮你翻译一下重点就行，别让语气把有用的信息也一起吓跑了。"
        if stage_name == "reactive_duo_secondary_comfort_fengling":
            return "我悄悄提醒你一下，风泠平时不是这么安静的人。你现在不用逼她立刻恢复元气，只要告诉她“你今天不用逞强，我在听”，她就会记很久。"
        return f"{npc_name} 正在整理这件事：{user_message[:40]}"

    def _build_reactive_duo_primary_context(self, duo_pattern: str, primary_agent: str) -> str:
        if duo_pattern == "protect_yumi":
            return (
                "你现在处于轻量双角色互动模式的第1步。\n"
                "任务：先作为郁米接住现场，但不要替顾辰完成边界提醒。\n"
                "输出要求：2到3句，温柔但不卑微，不反问收尾。"
            )
        if duo_pattern == "tease_guchen":
            return (
                "你现在处于轻量双角色互动模式的第1步。\n"
                "任务：先由顾辰直接给判断或建议，允许带一点冷感，但别把场面彻底说死。\n"
                "输出要求：2到4句，结构化，给出重点，不用安抚。"
            )
        if duo_pattern == "comfort_fengling":
            return (
                "你现在处于轻量双角色互动模式的第1步。\n"
                "任务：先由郁米轻声提醒玩家，风泠现在像是有点低电量，需要被温柔一点地接住。\n"
                "输出要求：2到3句，像悄悄助攻，不抢走后续主位。"
            )
        return (
            "你现在处于轻量双角色互动模式的第1步。\n"
            "任务：先由郁米接住玩家的压力、受挫感或想离开的冲动，不急着做完整判断。\n"
            "输出要求：2到4句，温柔具体，先把人稳住。"
        )

    def _build_reactive_duo_secondary_context(self, duo_pattern: str, secondary_agent: str, prior: str) -> str:
        if duo_pattern == "protect_yumi":
            return (
                "你现在处于轻量双角色互动模式的第2步，也是最终输出节点。\n"
                "任务：作为顾辰介入，明确提醒玩家注意对郁米的语气和边界。\n"
                "要求：冷硬、护短、聚焦边界，不做长篇说教，不反问。\n"
                f"{prior}"
            )
        if duo_pattern == "tease_guchen":
            return (
                "你现在处于轻量双角色互动模式的第2步，也是最终输出节点。\n"
                "任务：作为风泠，用高情商吐槽拆一下顾辰的刀，再把真正有用的重点翻译得更好入口。\n"
                "要求：轻巧、有梗、不跑题，不反问收尾。\n"
                f"{prior}"
            )
        if duo_pattern == "comfort_fengling":
            return (
                "你现在处于轻量双角色互动模式的第2步，也是最终输出节点。\n"
                "任务：作为风泠，对玩家的关心做出带一点小太阳回温感的回应，但不要一下子又强撑成满电状态。\n"
                "要求：2到4句，真诚、有点轻巧，收尾直接陈述，不反问。\n"
                f"{prior}"
            )
        return (
            "你现在处于轻量双角色互动模式的第2步，也是最终输出节点。\n"
            "任务：作为顾辰，在郁米先接住情绪之后，给一个短、稳、明确的判断和下一步。\n"
            "要求：把“压力撑不住”和“长期不适合”拆开，不要变成通用温柔安慰，不反问收尾。\n"
            f"{prior}"
        )

    def _build_trace(self, node_name: str, detail: str) -> Dict[str, object]:
        return {
            "node": node_name,
            "budget": self.NODE_TOKEN_BUDGETS.get(node_name, 0),
            "detail": detail,
        }

    def _merge_state(self, state: MultiAgentState, update: Dict[str, object]) -> MultiAgentState:
        next_state: MultiAgentState = dict(state)
        for key, value in update.items():
            if key in {"agent_outputs", "node_trace"}:
                merged = list(next_state.get(key, []))
                merged.extend(list(value))
                next_state[key] = merged
            else:
                next_state[key] = value
        return next_state
