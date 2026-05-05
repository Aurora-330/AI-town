"""controlled react vs static coordinator 的 A/B 评测。"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    load_cases,
    print_report,
    save_report,
)


EXECUTION_MODES = ["static_coordinator", "controlled_react"]


def seed_memory(npc_name: str, seed_messages: List[str], api_base: str):
    """用最小对话把重要上下文写入记忆。"""
    for index, message in enumerate(seed_messages, start=1):
        chat(
            npc_name,
            f"记忆种子{index}: {message}",
            api_base=api_base,
            execution_mode="static_coordinator",
        )


def evaluate_case(case: Dict[str, Any], execution_mode: str, api_base: str) -> Dict[str, Any]:
    npc = case["npc"]
    clear_memories(npc, api_base=api_base)
    seed_memory(npc, case.get("memory_seed", []), api_base=api_base)

    response = chat(
        npc,
        case["query"],
        api_base=api_base,
        execution_mode=execution_mode,
    )
    reply = response["message"]
    passed, detail = evaluate_keyword_case(
        text=reply,
        expected_keywords=case["expected_keywords"],
        min_hits=case.get("min_hits", 1),
        forbidden_keywords=case.get("forbidden_keywords", []),
    )

    preferred_hits = [
        keyword for keyword in case.get("preferred_npc_keywords", [])
        if keyword in reply
    ]

    expected_mode = case.get("expected_query_mode")
    mode_ok = not expected_mode or response.get("query_mode") == expected_mode
    if case["category"] == "default_memory_enough":
        tool_policy_ok = response.get("tool_call_count", 0) <= 1
    elif case["category"] == "default_memory_sparse":
        tool_policy_ok = response.get("tool_call_count", 0) >= 2 if execution_mode == "controlled_react" else True
    elif case["category"] == "routing":
        tool_policy_ok = response.get("tool_call_count", 0) >= 2 if execution_mode == "controlled_react" else True
    else:
        tool_policy_ok = True

    passed = passed and mode_ok and tool_policy_ok
    return {
        "case_id": case["id"],
        "npc": npc,
        "category": case["category"],
        "execution_mode": execution_mode,
        "passed": passed,
        "detail": {
            "expected_query_mode": expected_mode,
            "actual_query_mode": response.get("query_mode"),
            "keyword_hits": detail["hits"],
            "forbidden_hits": detail["forbidden_hits"],
            "preferred_npc_hits": preferred_hits,
            "reply": reply,
            "react_activated": response.get("react_activated", False),
            "react_activation_rule": response.get("react_activation_rule", ""),
            "react_activation_reason": response.get("react_activation_reason", ""),
            "react_step_count": response.get("react_step_count", 0),
            "tool_call_count": response.get("tool_call_count", 0),
            "input_tokens_est": response.get("input_tokens_est", 0),
            "latency_ms": response.get("latency_ms", 0),
            "execution_mode": response.get("execution_mode", execution_mode),
        },
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 execution_mode / category 聚合统计。"""
    by_mode: Dict[str, Dict[str, Any]] = {}
    by_mode_category: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for execution_mode in EXECUTION_MODES:
        mode_results = [item for item in results if item["execution_mode"] == execution_mode]
        by_mode[execution_mode] = _aggregate_bucket(mode_results)

        categories = sorted({item["category"] for item in mode_results})
        for category in categories:
            bucket = [item for item in mode_results if item["category"] == category]
            by_mode_category[execution_mode][category] = _aggregate_bucket(bucket)

    comparison = {}
    if all(mode in by_mode for mode in EXECUTION_MODES):
        static_metrics = by_mode["static_coordinator"]
        react_metrics = by_mode["controlled_react"]
        comparison = {
            "pass_rate_delta": round(react_metrics["pass_rate"] - static_metrics["pass_rate"], 4),
            "avg_tokens_delta": round(react_metrics["avg_input_tokens_est"] - static_metrics["avg_input_tokens_est"], 2),
            "avg_latency_delta_ms": round(react_metrics["avg_latency_ms"] - static_metrics["avg_latency_ms"], 2),
            "avg_tool_calls_delta": round(react_metrics["avg_tool_call_count"] - static_metrics["avg_tool_call_count"], 2),
            "avg_react_steps_delta": round(react_metrics["avg_react_step_count"] - static_metrics["avg_react_step_count"], 2),
        }

    return {
        "by_execution_mode": by_mode,
        "by_execution_mode_and_category": by_mode_category,
        "comparison": comparison,
    }


def _aggregate_bucket(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_input_tokens_est": 0.0,
            "avg_latency_ms": 0.0,
            "avg_tool_call_count": 0.0,
            "avg_react_step_count": 0.0,
            "react_activation_rate": 0.0,
            "react_activation_rule_counts": {},
            "fallback_to_static_count": 0,
            "misfire_second_or_third_step_count": 0,
        }

    passed = sum(1 for item in results if item["passed"])
    input_tokens = [item["detail"]["input_tokens_est"] for item in results]
    latencies = [item["detail"]["latency_ms"] for item in results]
    tool_calls = [item["detail"]["tool_call_count"] for item in results]
    react_steps = [item["detail"]["react_step_count"] for item in results]
    activated = [1 for item in results if item["detail"]["react_activated"]]
    activation_rule_counts = defaultdict(int)
    fallback_to_static_count = 0
    for item in results:
        rule = item["detail"].get("react_activation_rule", "")
        if rule:
            activation_rule_counts[rule] += 1
        if (
            item["execution_mode"] == "controlled_react"
            and item["detail"].get("execution_mode") == "static_coordinator"
        ):
            fallback_to_static_count += 1
    misfires = [
        item for item in results
        if item["category"] == "default_memory_enough" and item["detail"]["tool_call_count"] >= 2
    ]

    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "avg_input_tokens_est": round(mean(input_tokens), 2),
        "avg_latency_ms": round(mean(latencies), 2),
        "avg_tool_call_count": round(mean(tool_calls), 2),
        "avg_react_step_count": round(mean(react_steps), 2),
        "react_activation_rate": round(len(activated) / len(results), 4),
        "react_activation_rule_counts": dict(sorted(activation_rule_counts.items())),
        "fallback_to_static_count": fallback_to_static_count,
        "misfire_second_or_third_step_count": len(misfires),
    }


def save_ab_report(title: str, results: List[Dict[str, Any]], summary: Dict[str, Any], output_name: str):
    """保存扩展版 A/B 报告。"""
    payload = {
        "title": title,
        "total": len(results),
        "summary": summary,
        "results": results,
    }
    path = save_report(title, results, output_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = path.parent / "report.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"A/B 摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


def main():
    cases = load_cases("react_cases.json")
    results: List[Dict[str, Any]] = []

    for execution_mode in EXECUTION_MODES:
        for case in cases:
            results.append(evaluate_case(case, execution_mode=execution_mode, api_base=DEFAULT_API_BASE))

    print_report("React A/B Evaluation", results)
    summary = build_summary(results)
    save_ab_report("React A/B Evaluation", results, summary, "react_ab_report.json")


if __name__ == "__main__":
    main()
