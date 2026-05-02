"""最小外部知识 grounding 评测"""

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    load_cases,
    print_report,
    save_report,
    search_knowledge,
    set_affinity,
)


def main():
    cases = load_cases("grounding_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        knowledge = search_knowledge(
            case["query"],
            limit=3,
            api_base=DEFAULT_API_BASE
        )
        hits = knowledge.get("hits", [])
        hit_titles = [item.get("title", "") for item in hits]
        combined_knowledge_text = "\n".join(
            f"{item.get('title', '')}\n{item.get('content', '')}" for item in hits
        )

        knowledge_passed, knowledge_detail = evaluate_keyword_case(
            text=combined_knowledge_text,
            expected_keywords=case.get("knowledge_expected_keywords", []),
            min_hits=case.get("knowledge_min_hits", 1),
        )
        expected_hit_titles = case.get("expected_hit_titles", [])
        title_hits = [title for title in expected_hit_titles if title in hit_titles]
        titles_ok = len(title_hits) >= case.get("min_title_hits", 1)

        response = chat(npc, case["query"], api_base=DEFAULT_API_BASE)
        reply = response["message"]
        reply_passed, reply_detail = evaluate_keyword_case(
            text=reply,
            expected_keywords=case["expected_keywords"],
            min_hits=case.get("min_hits", 1),
            forbidden_keywords=case.get("forbidden_keywords", [])
        )

        passed = knowledge_passed and titles_ok and reply_passed
        results.append({
            "case_id": case["id"],
            "npc": npc,
            "passed": passed,
            "detail": {
                "knowledge_hits": knowledge_detail["hits"],
                "knowledge_hit_count": knowledge_detail["hit_count"],
                "knowledge_titles": hit_titles,
                "expected_title_hits": title_hits,
                "reply_hits": reply_detail["hits"],
                "reply_forbidden_hits": reply_detail["forbidden_hits"],
                "reply_hit_count": reply_detail["hit_count"],
                "reply": reply
            }
        })

    print_report("Grounding Evaluation", results)
    save_report("Grounding Evaluation", results, "grounding_report.json")


if __name__ == "__main__":
    main()
