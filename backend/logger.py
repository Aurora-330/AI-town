"""对话日志系统"""

import logging
import os
from datetime import datetime
from pathlib import Path

# 创建logs目录
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 创建日志文件名 (按日期)
today = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = LOGS_DIR / f"dialogue_{today}.log"

# 配置日志格式
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"

# 创建logger
dialogue_logger = logging.getLogger("dialogue")
dialogue_logger.setLevel(logging.INFO)

# 移除已有的handlers (避免重复)
dialogue_logger.handlers.clear()

# 创建文件handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 创建控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# 添加handlers
dialogue_logger.addHandler(file_handler)
dialogue_logger.addHandler(console_handler)

# 防止日志传播到root logger
dialogue_logger.propagate = False

def log_dialogue_start(npc_name: str, player_message: str):
    """记录对话开始"""
    dialogue_logger.info("=" * 60)
    dialogue_logger.info(f"💬 对话开始: {npc_name} <-> 玩家")
    dialogue_logger.info("=" * 60)
    dialogue_logger.info(f"📝 玩家消息: {player_message}")

def log_affinity(npc_name: str, affinity: float, level: str):
    """记录当前好感度"""
    dialogue_logger.info(f"💖 当前好感度: {affinity:.1f}/100 ({level})")

def log_memory_retrieval(
    npc_name: str,
    count: int,
    memories: list = None,
    layer_details: dict | None = None,
):
    """记录记忆检索"""
    dialogue_logger.info(f"🧠 检索到{count}条相关记忆")
    if memories:
        dialogue_logger.info("  📚 相关记忆:")
        for i, mem in enumerate(memories[:3], 1):
            content = mem.content[:50] + "..." if len(mem.content) > 50 else mem.content
            dialogue_logger.info(f"    {i}. {content}")
    if layer_details:
        query = layer_details.get("query", "")
        if query:
            dialogue_logger.info(f"  🔎 memory_query={query}")
        if "memory_budget" in layer_details:
            dialogue_logger.info(f"  🔢 memory_budget={layer_details.get('memory_budget', 0)}")
        for layer in layer_details.get("layers", []):
            dialogue_logger.info(
                "  - tier=%s candidates=%s selected=%s ids=%s"
                % (
                    layer.get("memory_tier", "unknown"),
                    layer.get("candidate_count", 0),
                    layer.get("selected_count", 0),
                    layer.get("selected_ids", []),
                )
            )
            dedupe = layer.get("dedupe", {})
            if dedupe:
                dialogue_logger.info(
                    "    dedupe: before=%s after=%s removed=%s"
                    % (
                        dedupe.get("input_count", layer.get("candidate_count", 0)),
                        dedupe.get("output_count", layer.get("candidate_count", 0)),
                        dedupe.get("removed_count", 0),
                    )
                )
            cross_turn = layer.get("cross_turn", {})
            if cross_turn:
                dialogue_logger.info(
                    "    cross_turn: before=%s after=%s downweighted=%s"
                    % (
                        cross_turn.get("input_count", layer.get("candidate_count", 0)),
                        cross_turn.get("output_count", layer.get("candidate_count", 0)),
                        cross_turn.get("downweighted_count", 0),
                    )
                )
            importance_summary = layer.get("importance_summary", [])
            if importance_summary:
                dialogue_logger.info(f"    importance={importance_summary}")
            filtered_reason = layer.get("filtered_reason")
            if filtered_reason:
                dialogue_logger.info(f"    filtered_reason={filtered_reason}")

def log_knowledge_retrieval(
    npc_name: str,
    query: str,
    hits: list = None,
    retrieval_details: dict | None = None,
):
    """记录外部知识检索"""
    count = len(hits or [])
    dialogue_logger.info(f"📚 外部知识命中{count}条: query={query}")
    if retrieval_details:
        dialogue_logger.info(
            "  🔎 scope=%s scopes=%s limit=%s candidates=%s semantic=%s lexical=%s selected=%s reason=%s"
            % (
                retrieval_details.get("scope", "global"),
                retrieval_details.get("scopes", [retrieval_details.get("scope", "global")]),
                retrieval_details.get("limit", "-"),
                retrieval_details.get("candidate_count", "-"),
                retrieval_details.get("semantic_candidate_count", "-"),
                retrieval_details.get("lexical_candidate_count", "-"),
                retrieval_details.get("selected_count", count),
                retrieval_details.get("selected_or_filtered_reason", ""),
            )
        )
        dedupe = retrieval_details.get("dedupe", {})
        if dedupe:
            dialogue_logger.info(
                "  🧹 knowledge_dedupe: before=%s after=%s removed=%s"
                % (
                    dedupe.get("input_count", retrieval_details.get("candidate_count", 0)),
                    dedupe.get("output_count", retrieval_details.get("selected_count", count)),
                    dedupe.get("removed_count", 0),
                )
            )
        cross_turn = retrieval_details.get("cross_turn", {})
        if cross_turn:
            dialogue_logger.info(
                "  🔁 knowledge_cross_turn: before=%s after=%s downweighted=%s"
                % (
                    cross_turn.get("input_count", retrieval_details.get("selected_count", count)),
                    cross_turn.get("output_count", retrieval_details.get("selected_count", count)),
                    cross_turn.get("downweighted_count", 0),
                )
            )
        for candidate in retrieval_details.get("candidates", [])[:5]:
            dialogue_logger.info(
                "    candidate title=%s raw=%.3f semantic=%.3f lexical=%.3f rerank=%.3f sources=%s filtered=%s"
                % (
                    candidate.get("title", "未知文档"),
                    float(candidate.get("raw_score", 0.0)),
                    float(candidate.get("semantic_score", 0.0)),
                    float(candidate.get("lexical_score", 0.0)),
                    float(candidate.get("rerank_score", 0.0)),
                    candidate.get("retrieval_sources", []),
                    candidate.get("filtered_reason", "") or "kept",
                )
            )
            signals = candidate.get("signals", {})
            if signals:
                dialogue_logger.info(
                    "      signals: lexical_bonus=%s source_bonus=%s explicit_doc=%s source_hits=%s scope_bonus=%s hits(title=%s,content=%s,tags=%s) npc_bonus=%s other_penalty=%s mentioned=%s"
                    % (
                        signals.get("lexical_bonus", 0.0),
                        signals.get("source_bonus", 0.0),
                        signals.get("explicit_doc_reference", False),
                        signals.get("source_hits", 0),
                        signals.get("scope_bonus", 0.0),
                        signals.get("title_hits", 0),
                        signals.get("content_hits", 0),
                        signals.get("tag_hits", 0),
                        signals.get("npc_match_bonus", 0.0),
                        signals.get("other_npc_penalty", 0.0),
                        signals.get("mentioned_npcs", []),
                    )
                )
    if hits:
        for i, hit in enumerate(hits[:3], 1):
            title = hit.get("title", "未知文档")
            source = hit.get("source", "unknown")
            score = hit.get("score", 0.0)
            snippet = hit.get("content", "")
            snippet = snippet[:60] + "..." if len(snippet) > 60 else snippet
            dialogue_logger.info(
                f"    {i}. [{title}] score={score:.3f} source={source} content={snippet}"
            )
    else:
        dialogue_logger.info("    0. 未命中外部知识")

def log_prompt_assembly(npc_name: str, sections: dict):
    """记录 prompt 拼装后的区块大小，便于观察上下文预算。"""
    parts = [f"{name}={size}" for name, size in sections.items()]
    dialogue_logger.info(f"🧱 Prompt组装: npc={npc_name} " + ", ".join(parts))

def log_knowledge_prompt_context(npc_name: str, knowledge_context: str):
    """记录最终注入 prompt 的知识片段，便于区分原始知识块与最终截取结果。"""
    if not knowledge_context:
        dialogue_logger.info(f"📎 最终知识片段: npc={npc_name} <empty>")
        return

    preview = knowledge_context.replace("\n", " ").strip()
    if len(preview) > 180:
        preview = preview[:177].rstrip() + "..."
    dialogue_logger.info(f"📎 最终知识片段: npc={npc_name} {preview}")

def log_query_analysis(npc_name: str, original_query: str, analysis: dict):
    """记录查询分析与改写结果。"""
    dialogue_logger.info(
        "🧭 查询分析: npc=%s mode=%s need_rewrite=%s reason=%s"
        % (
            npc_name,
            analysis.get("query_mode", "default"),
            analysis.get("need_rewrite", False),
            analysis.get("reason", ""),
        )
    )
    dialogue_logger.info(
        "  original_query=%s"
        % original_query
    )
    dialogue_logger.info(
        "  rewrite_query=%s"
        % analysis.get("rewrite_query", original_query)
    )

def log_retrieval_plan(npc_name: str, plan: dict):
    """记录最终采用的检索计划。"""
    dialogue_logger.info(
        "🗺️ 检索计划: npc=%s summary=%s episodic=%s working=%s knowledge=%s memory_k=%s knowledge_k=%s rerank=%s"
        % (
            npc_name,
            plan.get("use_summary", False),
            plan.get("use_episodic", False),
            plan.get("use_working", False),
            plan.get("use_knowledge", False),
            plan.get("memory_k", 0),
            plan.get("knowledge_k", 0),
            plan.get("need_rerank", True),
        )
    )

def log_coordinator_decision(npc_name: str, decision: dict):
    """记录 coordinator 的执行决策。"""
    dialogue_logger.info(
        "🧭 Coordinator: npc=%s mode=%s primary=%s secondary=%s strategy=%s answer_shape=%s"
        % (
            npc_name,
            decision.get("query_mode", "default"),
            decision.get("primary_tool", "answer_now"),
            decision.get("secondary_tool", ""),
            decision.get("response_strategy", ""),
            decision.get("answer_shape", "default"),
        )
    )
    if decision.get("reason", ""):
        dialogue_logger.info(f"  reason={decision.get('reason', '')}")

def log_coordinator_step(npc_name: str, tool_name: str, observation_count: int, phase: str):
    """记录 coordinator 执行到某一步时的观测量。"""
    dialogue_logger.info(
        "  🔧 coordinator_%s: npc=%s tool=%s observations=%s"
        % (phase, npc_name, tool_name, observation_count)
    )

def log_react_step(npc_name: str, trace_step: dict):
    """记录 ReAct loop 的单步轨迹。"""
    dialogue_logger.info(
        "  🪜 react_step[%s]: npc=%s thought=%s action=%s observations=%s step_tokens=%s"
        % (
            trace_step.get("step_index", 0),
            npc_name,
            trace_step.get("thought", ""),
            trace_step.get("action", ""),
            trace_step.get("observation_count", 0),
            trace_step.get("input_tokens_est", 0),
        )
    )
    if trace_step.get("observation_summary", ""):
        dialogue_logger.info(f"    observation={trace_step.get('observation_summary', '')}")

def log_react_finish(npc_name: str, query_mode: str, trace: list[dict]):
    """记录 ReAct loop 收尾。"""
    actions = [step.get("action", "") for step in trace]
    dialogue_logger.info(
        "🧠 ReAct完成: npc=%s mode=%s steps=%s actions=%s"
        % (npc_name, query_mode, len(trace), actions)
    )

def log_generating_response():
    """记录正在生成回复"""
    dialogue_logger.info("🤖 正在生成回复...")

def log_npc_response(npc_name: str, response: str):
    """记录NPC回复"""
    dialogue_logger.info(f"💬 {npc_name}回复: {response}")

def log_analyzing_affinity():
    """记录正在分析好感度"""
    dialogue_logger.info("📊 正在分析好感度变化...")

def log_affinity_change(affinity_result: dict):
    """记录好感度变化"""
    if affinity_result.get("changed"):
        change_symbol = "📈" if affinity_result["change_amount"] > 0 else "📉"
        dialogue_logger.info(
            f"{change_symbol} 好感度变化: {affinity_result['old_affinity']:.1f} -> "
            f"{affinity_result['new_affinity']:.1f} ({affinity_result['change_amount']:+.1f})"
        )
        dialogue_logger.info(f"  原因: {affinity_result['reason']}")
        dialogue_logger.info(f"  情感: {affinity_result['sentiment']}")
        
        if affinity_result['old_level'] != affinity_result['new_level']:
            dialogue_logger.info(
                f"  🎉 关系等级变化: {affinity_result['old_level']} -> {affinity_result['new_level']}"
            )
    else:
        dialogue_logger.info(f"  ➡️ 好感度未变化 (当前: {affinity_result.get('affinity', 50.0):.1f})")
        dialogue_logger.info(f"  原因: {affinity_result.get('reason', '无')}")

def log_memory_saved(npc_name: str):
    """记录记忆保存"""
    dialogue_logger.info(f"  💾 对话已保存到{npc_name}的记忆中")

def log_safety_decision(stage: str, decision):
    """记录安全审核结果"""
    dialogue_logger.info(
        f"🛡️ 安全审核[{stage}]: action={decision.action} "
        f"risk={decision.risk_type} confidence={decision.confidence:.2f}"
    )
    if getattr(decision, "matched_rules", None):
        dialogue_logger.info(f"  规则命中: {decision.matched_rules}")
    if getattr(decision, "reason", ""):
        dialogue_logger.info(f"  原因: {decision.reason}")

def log_memory_write_decision(npc_name: str, decision):
    """记录记忆写入策略"""
    dialogue_logger.info(
        f"🧾 记忆写入策略: npc={npc_name} policy={decision.memory_write_policy} "
        f"risk={decision.risk_type} confidence={decision.confidence:.2f}"
    )
    if getattr(decision, "matched_rules", None):
        dialogue_logger.info(f"  规则命中: {decision.matched_rules}")
    if getattr(decision, "reason", ""):
        dialogue_logger.info(f"  原因: {decision.reason}")

def log_summary_trigger(npc_name: str, player_id: str, turn_count: int):
    """记录摘要触发"""
    dialogue_logger.info(
        f"🧩 触发记忆摘要: npc={npc_name}, player={player_id}, pending_turns={turn_count}"
    )

def log_summary_created(npc_name: str, summary_id: str, source_count: int):
    """记录摘要创建"""
    dialogue_logger.info(
        f"📝 已生成摘要记忆: id={summary_id}, source_turns={source_count}"
    )

def log_summary_recompressed(
    npc_name: str,
    player_id: str,
    merged_summary_id: str,
    compressed_from_ids: list[str],
):
    """记录摘要二次压缩结果"""
    dialogue_logger.info(
        "🗜️ 已生成合并摘要: npc=%s, player=%s, merged_id=%s, compressed_from=%s"
        % (npc_name, player_id, merged_summary_id, compressed_from_ids)
    )

def log_summary_skipped(npc_name: str, reason: str):
    """记录摘要跳过"""
    dialogue_logger.info(f"⏭️ 跳过摘要生成: {npc_name}, reason={reason}")

def log_dialogue_end():
    """记录对话结束"""
    dialogue_logger.info("=" * 60)
    dialogue_logger.info("✅ 对话完成\n")

def log_info(message: str):
    """记录普通信息"""
    dialogue_logger.info(message)

def log_error(message: str):
    """记录错误信息"""
    dialogue_logger.error(message)

# 启动时记录日志文件位置
print(f"\n📝 对话日志文件: {LOG_FILE}")
print(f"📂 日志目录: {LOGS_DIR}\n")
