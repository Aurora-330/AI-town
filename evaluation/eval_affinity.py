"""最小好感度方向评测"""

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    get_affinity,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)


def main():
    cases = load_cases("affinity_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, case["start_affinity"], api_base=DEFAULT_API_BASE)

        before = get_affinity(npc, api_base=DEFAULT_API_BASE)
        response = chat(npc, case["message"], api_base=DEFAULT_API_BASE)
        after = get_affinity(npc, api_base=DEFAULT_API_BASE)

        delta = after["affinity"] - before["affinity"]
        expected_direction = case["expected_direction"]

        if expected_direction == "up":
            passed = delta > 0
        elif expected_direction == "down":
            passed = delta < 0
        else:
            passed = delta == 0

        results.append({
            "case_id": case["id"],
            "npc": npc,
            "passed": passed,
            "detail": {
                "expected_direction": expected_direction,
                "delta": delta,
                "before": before["affinity"],
                "after": after["affinity"],
                "reply": response["message"]
            }
        })

    print_report("Affinity Evaluation", results)
    save_report("Affinity Evaluation", results, "affinity_report.json")


if __name__ == "__main__":
    main()
