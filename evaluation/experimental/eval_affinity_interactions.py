"""好感度与互动效果观察脚本。

目标：
1. 批量设置不同 NPC 的好感度。
2. 运行单角色 /chat 与多角色 /multi_chat 场景。
3. 输出结构正确性，并把回复正文存进报告，方便人工对比低好感 / 高好感时的语气变化。

注意：
- 当前实现里，好感度主要影响“语气和距离感”，不会稳定直接决定第二个角色是否出场。
- reactive_duo / parallel_b 的出场方式仍主要由内容和模式触发。
"""

from __future__ import annotations

from typing import Any, Dict, List

from common import (
    DEFAULT_API_BASE,
    chat,
    get_affinity,
    load_cases,
    multi_chat,
    print_report,
    save_report,
    set_affinity,
)


def _apply_affinities(case: Dict[str, Any], api_base: str) -> Dict[str, Dict[str, Any]]:
    applied: Dict[str, Dict[str, Any]] = {}
    for npc_name, value in case.get("affinities", {}).items():
        set_affinity(npc_name, float(value), api_base=api_base)
        applied[npc_name] = get_affinity(npc_name, api_base=api_base)
    return applied


def _evaluate_single_chat(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    npc_name = str(case["npc"])
    response = chat(
        npc_name=npc_name,
        message=str(case["query"]),
        api_base=api_base,
    )
    passed = bool(response.get("success", True)) and bool(response.get("message", "").strip())
    return {
        "npc": npc_name,
        "passed": passed,
        "detail": {
            "type": "single_chat",
            "query": case["query"],
            "reply": response.get("message", ""),
            "query_mode": response.get("query_mode", ""),
            "execution_mode": response.get("execution_mode", ""),
            "tool_call_count": response.get("tool_call_count", 0),
            "latency_ms": response.get("latency_ms", 0),
        },
    }


def _evaluate_multi_chat(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    response = multi_chat(
        message=str(case["query"]),
        api_base=api_base,
        mode=str(case.get("mode", "auto")),
        player_id="player",
        return_intermediate=True,
    )
    expected_mode = case.get("expected_mode")
    expected_selected_agents = case.get("expected_selected_agents", [])
    selected_agents = list(response.get("selected_agents", []))
    structure_ok = (
        response.get("mode") == expected_mode
        and selected_agents == expected_selected_agents
        and bool(response.get("final_answer", "").strip())
    )
    return {
        "npc": " / ".join(expected_selected_agents) if expected_selected_agents else "multi",
        "passed": structure_ok,
        "detail": {
            "type": "multi_chat",
            "query": case["query"],
            "request_mode": case.get("mode", "auto"),
            "actual_mode": response.get("mode"),
            "selected_agents": selected_agents,
            "execution_order": response.get("execution_order", []),
            "aggregation_strategy": response.get("aggregation_strategy", ""),
            "final_answer": response.get("final_answer", ""),
            "intermediate_outputs": response.get("intermediate_outputs", []),
            "node_trace": response.get("node_trace", []),
        },
    }


def main():
    cases = load_cases("affinity_interaction_cases.json")
    results: List[Dict[str, Any]] = []

    for case in cases:
        applied_affinities = _apply_affinities(case, api_base=DEFAULT_API_BASE)
        if case["category"] == "single_chat":
            result = _evaluate_single_chat(case, api_base=DEFAULT_API_BASE)
        else:
            result = _evaluate_multi_chat(case, api_base=DEFAULT_API_BASE)

        result["case_id"] = case["id"]
        result["detail"]["applied_affinities"] = applied_affinities
        results.append(result)

    print_report("Affinity Interaction Evaluation", results)
    save_report("Affinity Interaction Evaluation", results, "affinity_interaction_report.json")


if __name__ == "__main__":
    main()
