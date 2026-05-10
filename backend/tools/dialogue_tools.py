"""显式 dialogue tools，复用现有 manager 能力而不改变主链接口。"""

from __future__ import annotations

from typing import Dict, List, Optional

from hello_agents.memory import MemoryManager

from knowledge import KnowledgeChunk


class DialogueTools:
    """包装 memory / knowledge / routing 工具，减少 agents.py 里的执行细节。"""

    TOOL_REGISTRY = {
        "search_memory": {
            "description": "检索 summary / episodic / working 三层记忆",
            "args": ["memory_manager", "npc_name", "query", "player_id", "retrieval_plan"],
        },
        "search_knowledge": {
            "description": "检索外部知识库并做单轮去重与跨轮降权",
            "args": ["npc_name", "query", "player_id", "query_mode", "knowledge_k"],
        },
        "route_npc": {
            "description": "根据知识命中结果提炼首选 NPC 推荐",
            "args": ["knowledge_chunks"],
        },
        "get_summary_state": {
            "description": "读取 sequence6 摘要治理状态，用于轻量决策",
            "args": ["npc_name", "player_id"],
        },
    }

    def __init__(self, manager):
        self.manager = manager

    def list_tools(self) -> Dict[str, Dict]:
        """返回可调用工具定义。"""
        return dict(self.TOOL_REGISTRY)

    def execute(self, tool_name: str, **kwargs) -> Dict:
        """统一的 tool registry 执行入口。"""
        if tool_name == "search_memory":
            return self.run_memory_tool(**kwargs)
        if tool_name == "search_knowledge":
            return self.run_knowledge_tool(**kwargs)
        if tool_name == "route_npc":
            return self.run_route_tool(**kwargs)
        if tool_name == "get_summary_state":
            return self.run_summary_state_tool(**kwargs)
        raise ValueError(f"unknown tool: {tool_name}")

    def run_memory_tool(
        self,
        memory_manager: Optional[MemoryManager],
        npc_name: str,
        query: str,
        player_id: str,
        retrieval_plan: Dict,
    ) -> Dict:
        if not memory_manager:
            return {
                "summary_memories": [],
                "episodic_memories": [],
                "working_memories": [],
                "memory_debug": {"query": query, "memory_budget": 0, "layers": []},
                "observation_count": 0,
                "retrieval_metrics": {
                    "tool_name": "search_memory",
                    "memory_hit_count": 0,
                    "summary_hit_count": 0,
                    "episodic_hit_count": 0,
                    "working_hit_count": 0,
                },
            }

        summary_memories, episodic_memories, working_memories, memory_debug = self.manager._retrieve_memory_layers(
            memory_manager=memory_manager,
            npc_name=npc_name,
            query=query,
            player_id=player_id,
            retrieval_plan=retrieval_plan,
        )
        return {
            "summary_memories": summary_memories,
            "episodic_memories": episodic_memories,
            "working_memories": working_memories,
            "memory_debug": memory_debug,
            "observation_count": len(summary_memories) + len(episodic_memories) + len(working_memories),
            "retrieval_metrics": {
                "tool_name": "search_memory",
                "memory_hit_count": len(summary_memories) + len(episodic_memories) + len(working_memories),
                "summary_hit_count": len(summary_memories),
                "episodic_hit_count": len(episodic_memories),
                "working_hit_count": len(working_memories),
            },
        }

    def run_knowledge_tool(
        self,
        npc_name: str,
        query: str,
        player_id: str,
        query_mode: str,
        knowledge_k: int,
    ) -> Dict:
        if not self.manager.knowledge_retriever:
            return {
                "knowledge_chunks": [],
                "knowledge_debug": {
                    "scope": "global",
                    "scopes": [],
                    "limit": knowledge_k,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "selected_or_filtered_reason": "knowledge_tool_unavailable",
                },
                "observation_count": 0,
                "retrieval_metrics": {
                    "tool_name": "search_knowledge",
                    "knowledge_hit_count": 0,
                    "knowledge_source_count": 0,
                    "knowledge_sources": [],
                    "knowledge_chunk_keys": [],
                    "repeated_knowledge_chunk_count": 0,
                    "repeated_knowledge_source_count": 0,
                    "repeated_knowledge_sources": [],
                },
            }

        knowledge_scopes = self.manager._select_knowledge_scopes(npc_name)
        knowledge_chunks, knowledge_debug = self.manager.knowledge_retriever.search_with_debug(
            query=query,
            limit=max(1, int(knowledge_k or self.manager.KNOWLEDGE_RETRIEVAL_LIMIT)),
            scope=knowledge_scopes[0],
            npc_name=npc_name,
            allow_cross_npc=(query_mode == "routing"),
            scopes=knowledge_scopes,
        )
        knowledge_chunks, knowledge_dedupe = self.manager._dedupe_knowledge_chunks(knowledge_chunks)
        previous_knowledge_keys = set(
            self.manager._load_summary_state(npc_name).get("players", {}).get(player_id, {}).get("last_injected_knowledge_keys", [])
        )
        knowledge_chunks, knowledge_cross_turn = self.manager._downweight_repeated_knowledge_chunks(
            knowledge_chunks,
            previous_keys=previous_knowledge_keys,
            enabled=(query_mode in {"default", "knowledge", "routing", "mixed"}),
        )
        knowledge_debug["dedupe"] = knowledge_dedupe
        knowledge_debug["cross_turn"] = knowledge_cross_turn
        knowledge_debug["selected_count"] = len(knowledge_chunks)
        if knowledge_dedupe.get("removed_count", 0) > 0:
            knowledge_debug["selected_or_filtered_reason"] = (
                f"{knowledge_debug.get('selected_or_filtered_reason', 'selected')}|single_turn_dedup"
            )
        if knowledge_cross_turn.get("downweighted_count", 0) > 0:
            knowledge_debug["selected_or_filtered_reason"] = (
                f"{knowledge_debug.get('selected_or_filtered_reason', 'selected')}|cross_turn_downweight"
            )
        chunk_keys = [self.manager._build_knowledge_chunk_key(chunk) for chunk in knowledge_chunks]
        repeated_chunk_keys = [key for key in chunk_keys if key in previous_knowledge_keys]
        knowledge_sources = []
        repeated_knowledge_sources = []
        seen_sources = set()
        seen_repeated_sources = set()
        for chunk, chunk_key in zip(knowledge_chunks, chunk_keys):
            source = chunk.source or chunk.title or chunk.point_id
            if source and source not in seen_sources:
                knowledge_sources.append(source)
                seen_sources.add(source)
            if chunk_key in previous_knowledge_keys and source and source not in seen_repeated_sources:
                repeated_knowledge_sources.append(source)
                seen_repeated_sources.add(source)
        return {
            "knowledge_chunks": knowledge_chunks,
            "knowledge_debug": knowledge_debug,
            "observation_count": len(knowledge_chunks),
            "retrieval_metrics": {
                "tool_name": "search_knowledge",
                "knowledge_hit_count": len(knowledge_chunks),
                "knowledge_source_count": len(knowledge_sources),
                "knowledge_sources": knowledge_sources,
                "knowledge_chunk_keys": chunk_keys,
                "repeated_knowledge_chunk_count": len(repeated_chunk_keys),
                "repeated_knowledge_source_count": len(repeated_knowledge_sources),
                "repeated_knowledge_sources": repeated_knowledge_sources,
            },
        }

    def run_route_tool(self, knowledge_chunks: List[KnowledgeChunk]) -> Dict:
        recommended_npc = self.manager._infer_routing_recommendation(knowledge_chunks)
        return {
            "recommended_npc": recommended_npc,
            "observation_count": 1 if recommended_npc else 0,
        }

    def run_summary_state_tool(self, npc_name: str, player_id: str) -> Dict:
        debug = self.manager.get_summary_debug_info(npc_name, player_id=player_id)
        player_info = debug.get("players", {}).get(player_id, {})
        return {
            "summary_state": player_info,
            "observation_count": 1 if player_info else 0,
        }

    def summarize_observation(self, tool_name: str, result: Dict, max_tokens: int) -> str:
        """把原始 tool 结果压成轻量 observation，供 react loop 使用。"""
        if tool_name == "search_memory":
            summary_count = len(result.get("summary_memories", []))
            episodic_count = len(result.get("episodic_memories", []))
            working_count = len(result.get("working_memories", []))
            top_bits = []
            for memory in (
                result.get("summary_memories", [])[:1]
                + result.get("episodic_memories", [])[:1]
                + result.get("working_memories", [])[:1]
            ):
                snippet = self.manager._clip_text(getattr(memory, "content", "") or "", 36)
                if snippet:
                    top_bits.append(snippet)
            text = (
                f"memory命中 summary={summary_count} episodic={episodic_count} working={working_count}。"
                + (" 关键内容: " + " / ".join(top_bits) if top_bits else "")
            )
            return self._clip_summary_to_token_budget(text, max_tokens)

        if tool_name == "search_knowledge":
            chunks = result.get("knowledge_chunks", [])
            titles = []
            for chunk in chunks[:2]:
                label = chunk.title or chunk.source or chunk.point_id
                if label:
                    titles.append(label)
            text = (
                f"knowledge命中 {len(chunks)} 条。"
                + (" 主要来源: " + " / ".join(titles) if titles else "")
            )
            return self._clip_summary_to_token_budget(text, max_tokens)

        if tool_name == "route_npc":
            recommended_npc = result.get("recommended_npc", "")
            text = f"route结果: 首选 {recommended_npc}。" if recommended_npc else "route结果: 暂无明确推荐对象。"
            return self._clip_summary_to_token_budget(text, max_tokens)

        if tool_name == "get_summary_state":
            state = result.get("summary_state", {}) or {}
            text = (
                "summary_state: summary_count=%s merged_count=%s compressed_count=%s active_base_count=%s"
                % (
                    state.get("summary_count", 0),
                    state.get("merged_count", 0),
                    state.get("compressed_count", 0),
                    state.get("active_base_count", 0),
                )
            )
            return self._clip_summary_to_token_budget(text, max_tokens)

        return self._clip_summary_to_token_budget("无可用 observation。", max_tokens)

    def _clip_summary_to_token_budget(self, text: str, max_tokens: int) -> str:
        """按 token 预算裁 observation summary。"""
        if not text or max_tokens <= 0:
            return ""

        clipped = text
        current_tokens = self.manager.token_counter.count_text_tokens(clipped)
        while clipped and current_tokens > max_tokens:
            approx_chars = max(24, int(len(clipped) * max_tokens / max(current_tokens, 1)))
            new_text = self.manager._clip_text(clipped, approx_chars)
            if not new_text or new_text == clipped:
                clipped = ""
            else:
                clipped = new_text
            current_tokens = self.manager.token_counter.count_text_tokens(clipped)
        return clipped
