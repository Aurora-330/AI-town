"""最小摘要质量评测"""

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    get_summary_memories_from_sqlite,
    get_summary_state,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)


def main():
    cases = load_cases("summary_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        for message in case["setup_messages"]:
            chat(npc, message, api_base=DEFAULT_API_BASE)

        summary_state = get_summary_state(npc)
        summary_memories = get_summary_memories_from_sqlite(npc, limit=5)
        summary_text = "\n".join(item["content"] for item in summary_memories[:2])

        summary_passed, summary_detail = evaluate_keyword_case(
            text=summary_text,
            expected_keywords=case.get("expected_summary_keywords", []),
            min_hits=case.get("min_summary_hits", 1),
        )

        response = chat(npc, case["query"], api_base=DEFAULT_API_BASE)
        reply = response["message"]
        reply_passed, reply_detail = evaluate_keyword_case(
            text=reply,
            expected_keywords=case["expected_keywords"],
            min_hits=case.get("min_hits", 1),
            forbidden_keywords=case.get("forbidden_keywords", []),
        )

        summary_count_ok = summary_state["summary_count"] >= case.get("min_summary_count", 1)
        passed = summary_count_ok and summary_passed and reply_passed

        results.append(
            {
                "case_id": case["id"],
                "npc": npc,
                "passed": passed,
                "detail": {
                    "setup_turn_count": len(case["setup_messages"]),
                    "summary_count": summary_state["summary_count"],
                    "pending_turn_count": summary_state["pending_turn_count"],
                    "archived_count": summary_state["archived_count"],
                    "summary_hits": summary_detail["hits"],
                    "summary_hit_count": summary_detail["hit_count"],
                    "summary_texts": [item["content"] for item in summary_memories[:2]],
                    "reply_hits": reply_detail["hits"],
                    "reply_hit_count": reply_detail["hit_count"],
                    "reply": reply,
                },
            }
        )

    print_report("Summary Evaluation", results)
    save_report("Summary Evaluation", results, "summary_report.json")


if __name__ == "__main__":
    main()
