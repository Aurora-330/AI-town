"""最小记忆与摘要评测"""

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    get_memories,
    get_summary_memories_from_sqlite,
    get_summary_state,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)


def main():
    cases = load_cases("memory_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        for message in case["setup_messages"]:
            chat(npc, message, api_base=DEFAULT_API_BASE)

        response = chat(npc, case["query"], api_base=DEFAULT_API_BASE)
        reply = response["message"]

        passed, detail = evaluate_keyword_case(
            text=reply,
            expected_keywords=case["expected_keywords"],
            min_hits=case.get("min_hits", 1),
            forbidden_keywords=case.get("forbidden_keywords", [])
        )

        memories = get_memories(npc, limit=30, api_base=DEFAULT_API_BASE)
        summary_state = get_summary_state(npc)
        summary_memories = get_summary_memories_from_sqlite(npc, limit=5)
        summary_found = bool(summary_memories)

        if case.get("require_summary", False):
            passed = passed and summary_state["summary_count"] >= case.get("min_summary_count", 1)

        detail["reply"] = reply
        detail["summary_found"] = summary_found
        detail["summary_count"] = summary_state["summary_count"]
        detail["pending_turn_count"] = summary_state["pending_turn_count"]
        detail["archived_count"] = summary_state["archived_count"]
        detail["summary_texts"] = [item["content"] for item in summary_memories[:2]]
        detail["memory_total"] = memories.get("total", 0)
        detail["setup_turn_count"] = len(case["setup_messages"])

        results.append({
            "case_id": case["id"],
            "npc": npc,
            "passed": passed,
            "detail": detail
        })

    print_report("Memory Evaluation", results)
    save_report("Memory Evaluation", results, "memory_report.json")


if __name__ == "__main__":
    main()
