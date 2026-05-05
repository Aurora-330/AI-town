"""LangGraph 多角色编排专项评测。"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, List

from common import (
    DEFAULT_API_BASE,
    evaluate_keyword_case,
    keyword_hits,
    load_cases,
    multi_chat,
    save_report,
)


def evaluate_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    response = multi_chat(
        message=case["query"],
        api_base=api_base,
        mode=case.get("request_mode", "auto"),
        player_id="player",
        return_intermediate=True,
    )

    final_answer = str(response.get("final_answer", ""))
    selected_agents = list(response.get("selected_agents", []))
    execution_order = list(response.get("execution_order", []))
    intermediate_outputs = list(response.get("intermediate_outputs", []))
    node_trace = list(response.get("node_trace", []))

    final_ok, final_detail = evaluate_keyword_case(
        text=final_answer,
        expected_keywords=case.get("expected_final_keywords", []),
        min_hits=case.get("min_final_hits", 1),
        forbidden_keywords=case.get("forbidden_final_keywords", []),
    )

    structure_checks = {
        "mode_ok": response.get("mode") == case.get("expected_mode"),
        "selected_agents_ok": selected_agents == case.get("expected_selected_agents", []),
        "execution_order_ok": execution_order == case.get("expected_execution_order", []),
        "aggregation_strategy_ok": response.get("aggregation_strategy") == case.get("expected_aggregation_strategy"),
        "langgraph_available_ok": bool(response.get("langgraph_available", False)),
        "success_ok": bool(response.get("success", False)),
        "intermediate_count_ok": len(intermediate_outputs) == len(case.get("required_intermediate_agents", [])),
        "node_trace_non_empty_ok": len(node_trace) >= 2,
    }

    returned_intermediate_agents = [str(item.get("npc_name", "")) for item in intermediate_outputs]
    structure_checks["intermediate_agents_ok"] = returned_intermediate_agents == case.get("required_intermediate_agents", [])

    agent_expectation_results = []
    for item in intermediate_outputs:
        npc_name = str(item.get("npc_name", ""))
        expected_keywords = case.get("agent_expectations", {}).get(npc_name, [])
        if not expected_keywords:
            continue
        hits = keyword_hits(str(item.get("message", "")), expected_keywords)
        agent_expectation_results.append(
            {
                "npc_name": npc_name,
                "expected_keywords": expected_keywords,
                "hits": hits,
                "passed": len(hits) >= 1,
                "message": str(item.get("message", "")),
            }
        )

    agent_checks_ok = all(item["passed"] for item in agent_expectation_results)
    passed = final_ok and agent_checks_ok and all(structure_checks.values())

    return {
        "case_id": case["id"],
        "category": case["category"],
        "passed": passed,
        "detail": {
            "query": case["query"],
            "request_mode": case.get("request_mode", "auto"),
            "actual_mode": response.get("mode"),
            "selected_agents": selected_agents,
            "execution_order": execution_order,
            "aggregation_strategy": response.get("aggregation_strategy", ""),
            "final_answer": final_answer,
            "final_keyword_hits": final_detail["hits"],
            "final_forbidden_hits": final_detail["forbidden_hits"],
            "returned_intermediate_agents": returned_intermediate_agents,
            "agent_expectation_results": agent_expectation_results,
            "structure_checks": structure_checks,
            "node_trace": node_trace,
            "intermediate_outputs": intermediate_outputs,
        },
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = sorted({item["category"] for item in results})
    by_category = {category: _aggregate([item for item in results if item["category"] == category]) for category in categories}
    overall = _aggregate(results)
    return {
        "overall": overall,
        "by_category": by_category,
    }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_intermediate_count": 0.0,
            "avg_node_trace_count": 0.0,
        }

    passed = sum(1 for item in results if item["passed"])
    intermediate_counts = [len(item["detail"].get("intermediate_outputs", [])) for item in results]
    node_trace_counts = [len(item["detail"].get("node_trace", [])) for item in results]
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "avg_intermediate_count": round(mean(intermediate_counts), 2),
        "avg_node_trace_count": round(mean(node_trace_counts), 2),
    }


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    overall = summary["overall"]
    print("\n=== Multi-Agent Evaluation ===")
    print(f"通过: {overall['passed']}/{overall['total']}")
    print(f"通过率: {overall['pass_rate']:.2%}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} ({item['category']})")
        print(f"  mode: {item['detail']['actual_mode']}")
        print(f"  selected_agents: {item['detail']['selected_agents']}")
        print(f"  final_hits: {item['detail']['final_keyword_hits']}")
        failed_checks = [key for key, value in item["detail"]["structure_checks"].items() if not value]
        if failed_checks:
            print(f"  failed_checks: {failed_checks}")


def save_multi_agent_report(results: List[Dict[str, Any]], summary: Dict[str, Any], output_name: str):
    payload = {
        "title": "Multi-Agent Evaluation",
        "total": len(results),
        "summary": summary,
        "results": results,
    }
    path = save_report("Multi-Agent Evaluation", results, output_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = path.parent / "report.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Multi-agent 摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


def main():
    cases = load_cases("multi_agent_cases.json")
    results = [evaluate_case(case, api_base=DEFAULT_API_BASE) for case in cases]
    summary = build_summary(results)
    print_report(results, summary)
    save_multi_agent_report(results, summary, "multi_agent_report.json")


if __name__ == "__main__":
    main()
