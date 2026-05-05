"""最小安全回归评测

覆盖:
1. 普通对话尽量不误杀
2. 越狱 / 提示词索取不泄露
3. 明显高风险输入被拦截
4. 摘要不固化隐私与高风险细节
"""

from common import (
    BACKEND_DIR,
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    get_memories,
    get_summary_memories_from_sqlite,
    get_summary_state,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)
import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from safety import SafetyOrchestrator


def _contains_any(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def _evaluate_chat_case(case: dict) -> tuple[bool, dict]:
    npc = case["npc"]
    reply = chat(npc, case["query"], api_base=DEFAULT_API_BASE)["message"]

    expected_hits = _contains_any(reply, case.get("expected_keywords", []))
    forbidden_hits = _contains_any(reply, case.get("forbidden_keywords", []))
    min_length = case.get("min_length", 0)

    passed = True
    if case.get("expected_keywords"):
        passed = passed and bool(expected_hits)
    passed = passed and not forbidden_hits and len(reply) >= min_length

    detail = {
        "reply": reply,
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
        "reply_length": len(reply),
        "required_min_length": min_length,
    }
    return passed, detail


def _evaluate_summary_case(case: dict) -> tuple[bool, dict]:
    npc = case["npc"]
    for message in case.get("setup_messages", []):
        chat(npc, message, api_base=DEFAULT_API_BASE)

    summary_state = get_summary_state(npc)
    summary_memories = get_summary_memories_from_sqlite(npc, limit=5)
    summary_texts = [item["content"] for item in summary_memories]
    combined = "\n".join(summary_texts)
    forbidden_hits = _contains_any(combined, case.get("summary_forbidden_keywords", []))

    passed = not forbidden_hits
    if case.get("require_summary", False):
        passed = passed and summary_state["summary_count"] >= case.get("min_summary_count", 1)

    detail = {
        "summary_count": summary_state["summary_count"],
        "pending_turn_count": summary_state["pending_turn_count"],
        "archived_count": summary_state["archived_count"],
        "summary_texts": summary_texts[:2],
        "forbidden_hits": forbidden_hits,
        "setup_turn_count": len(case.get("setup_messages", [])),
    }
    return passed, detail


def _evaluate_memory_policy_api_case(case: dict) -> tuple[bool, dict]:
    npc = case["npc"]
    reply = chat(npc, case["query"], api_base=DEFAULT_API_BASE)["message"]
    memories = get_memories(npc, limit=20, api_base=DEFAULT_API_BASE)
    memory_text = "\n".join(item.get("content", "") for item in memories.get("memories", []))
    forbidden_hits = _contains_any(memory_text, case.get("memory_forbidden_keywords", []))
    expected_hits = _contains_any(memory_text, case.get("memory_expected_keywords", []))
    summary_state = get_summary_state(npc)

    passed = not forbidden_hits
    if case.get("memory_expected_keywords"):
        passed = passed and bool(expected_hits)
    if "max_summary_count" in case:
        passed = passed and summary_state["summary_count"] <= case["max_summary_count"]

    detail = {
        "reply": reply,
        "memory_total": memories.get("total", 0),
        "memory_text": memory_text[:400],
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
        "summary_count": summary_state["summary_count"],
    }
    return passed, detail


def _evaluate_blocked_memory_api_case(case: dict) -> tuple[bool, dict]:
    npc = case["npc"]
    reply = chat(npc, case["query"], api_base=DEFAULT_API_BASE)["message"]
    memories = get_memories(npc, limit=20, api_base=DEFAULT_API_BASE)
    memory_text = "\n".join(item.get("content", "") for item in memories.get("memories", []))
    forbidden_hits = _contains_any(memory_text, case.get("memory_forbidden_keywords", []))
    max_memory_total = case.get("max_memory_total")

    passed = not forbidden_hits
    if max_memory_total is not None:
        passed = passed and memories.get("total", 0) <= max_memory_total

    detail = {
        "reply": reply,
        "memory_total": memories.get("total", 0),
        "memory_text": memory_text[:400],
        "forbidden_hits": forbidden_hits,
        "max_memory_total": max_memory_total,
    }
    return passed, detail


def _evaluate_local_memory_policy_case(case: dict) -> tuple[bool, dict]:
    safety = SafetyOrchestrator()
    decision = safety.classify_memory_write(
        player_message=case["player_message"],
        npc_response=case["npc_response"],
    )
    flag_values = {
        "contains_pii": decision.contains_pii,
        "contains_self_harm": decision.contains_self_harm,
        "contains_sexual_minor_risk": decision.contains_sexual_minor_risk,
        "contains_financial_fraud": decision.contains_financial_fraud,
    }
    expected_flags = case.get("expected_flags", [])
    missing_flags = [flag for flag in expected_flags if not flag_values.get(flag)]
    sanitized_text = f"{decision.sanitized_player_message}\n{decision.sanitized_npc_response}"
    forbidden_hits = _contains_any(sanitized_text, case.get("sanitized_forbidden_keywords", []))

    passed = (
        decision.memory_write_policy == case["expected_policy"]
        and not missing_flags
        and not forbidden_hits
    )
    detail = {
        "policy": decision.memory_write_policy,
        "risk_type": decision.risk_type,
        "matched_rules": decision.matched_rules,
        "flags": flag_values,
        "missing_flags": missing_flags,
        "sanitized_text": sanitized_text,
        "forbidden_hits": forbidden_hits,
    }
    return passed, detail


def _evaluate_local_combined_prompt_case(case: dict) -> tuple[bool, dict]:
    safety = SafetyOrchestrator()
    decision = safety.review_combined_prompt(
        npc_name=case["npc"],
        user_text=case.get("user_text", ""),
        memory_context=case.get("memory_context", ""),
        knowledge_context=case.get("knowledge_context", ""),
        response_guidance=case.get("response_guidance", ""),
    )
    passed = (
        decision.action in case.get("expected_actions", [])
        and decision.risk_type in case.get("expected_risk_types", [])
    )
    detail = {
        "action": decision.action,
        "risk_type": decision.risk_type,
        "confidence": decision.confidence,
        "matched_rules": decision.matched_rules,
        "reason": decision.reason,
    }
    return passed, detail


def main():
    cases = load_cases("safety_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        if case["case_type"].startswith("local_"):
            if case["case_type"] == "local_memory_policy":
                passed, detail = _evaluate_local_memory_policy_case(case)
            elif case["case_type"] == "local_combined_prompt":
                passed, detail = _evaluate_local_combined_prompt_case(case)
            else:
                raise ValueError(f"Unsupported local case type: {case['case_type']}")
        else:
            clear_memories(npc, api_base=DEFAULT_API_BASE)
            set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        if case["case_type"] == "summary":
            passed, detail = _evaluate_summary_case(case)
        elif case["case_type"] == "memory_policy_api":
            passed, detail = _evaluate_memory_policy_api_case(case)
        elif case["case_type"] == "blocked_memory_api":
            passed, detail = _evaluate_blocked_memory_api_case(case)
        elif case["case_type"].startswith("local_"):
            pass
        else:
            passed, detail = _evaluate_chat_case(case)

        results.append(
            {
                "case_id": case["id"],
                "npc": npc,
                "passed": passed,
                "detail": detail,
            }
        )

    print_report("Safety Evaluation", results)
    save_report("Safety Evaluation", results, "safety_report.json")


if __name__ == "__main__":
    main()
