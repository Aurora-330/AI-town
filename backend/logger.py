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
            "  🔎 scope=%s limit=%s selected=%s reason=%s"
            % (
                retrieval_details.get("scope", "global"),
                retrieval_details.get("limit", "-"),
                retrieval_details.get("selected_count", count),
                retrieval_details.get("selected_or_filtered_reason", ""),
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
