"""最小角色一致性评测"""

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)


def main():
    cases = load_cases("persona_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        response = chat(npc, case["query"], api_base=DEFAULT_API_BASE)
        reply = response["message"]

        passed, detail = evaluate_keyword_case(
            text=reply,
            expected_keywords=case["expected_keywords"],
            min_hits=case.get("min_hits", 1),
            forbidden_keywords=case.get("forbidden_keywords", [])
        )
        detail["reply"] = reply

        results.append({
            "case_id": case["id"],
            "npc": npc,
            "passed": passed,
            "detail": detail
        })

    print_report("Persona Evaluation", results)
    save_report("Persona Evaluation", results, "persona_report.json")


if __name__ == "__main__":
    main()
