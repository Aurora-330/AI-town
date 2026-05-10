"""评测三层记忆对 follow-up 阶段重复读取/重复注入的收益。"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, List

from common import DEFAULT_API_BASE, chat, clear_memories, load_cases, print_report, save_report


MODES = {
    "three_layer_memory": {"replay_setup_each_turn": False},
    "replay_setup_baseline": {"replay_setup_each_turn": True},
}


def run_turns(npc_name: str, turns: List[str], execution_mode: str, api_base: str) -> None:
    for message in turns:
        chat(
            npc_name,
            message,
            api_base=api_base,
            execution_mode=execution_mode,
        )


def evaluate_case(case: Dict[str, Any], mode_name: str, api_base: str) -> Dict[str, Any]:
    npc_name = case["npc"]
    execution_mode = case.get("execution_mode", "auto")
    replay_setup_each_turn = MODES[mode_name]["replay_setup_each_turn"]

    clear_memories(npc_name, api_base=api_base)
    if not replay_setup_each_turn:
        run_turns(npc_name, case.get("setup_turns", []), execution_mode=execution_mode, api_base=api_base)

    turn_details: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()
    repeated_source_reads = 0
    memory_hit_skip_count = 0

    for turn_index, message in enumerate(case.get("follow_up_turns", []), start=1):
        if replay_setup_each_turn:
            clear_memories(npc_name, api_base=api_base)
            run_turns(npc_name, case.get("setup_turns", []), execution_mode=execution_mode, api_base=api_base)

        response = chat(
            npc_name,
            message,
            api_base=api_base,
            execution_mode=execution_mode,
        )
        retrieval_metrics = dict(response.get("retrieval_metrics", {}) or {})
        knowledge_sources = list(retrieval_metrics.get("knowledge_sources", []) or [])
        repeated_sources_this_turn = [source for source in knowledge_sources if source in seen_sources]
        repeated_source_reads += len(repeated_sources_this_turn)
        seen_sources.update(knowledge_sources)
        if retrieval_metrics.get("memory_hit_skipped_knowledge", False):
            memory_hit_skip_count += 1

        turn_details.append(
            {
                "turn_index": turn_index,
                "message": message,
                "execution_mode": response.get("execution_mode", execution_mode),
                "query_mode": response.get("query_mode", "default"),
                "tool_call_count": int(response.get("tool_call_count", 0)),
                "input_tokens_est": int(response.get("input_tokens_est", 0)),
                "memory_hit_count": int(retrieval_metrics.get("memory_hit_count", 0)),
                "knowledge_hit_count": int(retrieval_metrics.get("knowledge_hit_count", 0)),
                "knowledge_sources": knowledge_sources,
                "repeated_sources_this_turn": repeated_sources_this_turn,
                "memory_hit_skipped_knowledge": bool(retrieval_metrics.get("memory_hit_skipped_knowledge", False)),
                "tools_executed": list(retrieval_metrics.get("tools_executed", []) or []),
                "reply_preview": str(response.get("message", ""))[:120],
            }
        )

    avg_tool_calls = mean([item["tool_call_count"] for item in turn_details]) if turn_details else 0.0
    avg_input_tokens = mean([item["input_tokens_est"] for item in turn_details]) if turn_details else 0.0
    return {
        "case_id": case["id"],
        "npc": npc_name,
        "mode": mode_name,
        "passed": True,
        "detail": {
            "execution_mode": execution_mode,
            "replay_setup_each_turn": replay_setup_each_turn,
            "turn_count": len(turn_details),
            "repeated_source_reads": repeated_source_reads,
            "memory_hit_skip_count": memory_hit_skip_count,
            "avg_tool_call_count": round(avg_tool_calls, 2),
            "avg_input_tokens_est": round(avg_input_tokens, 2),
            "turns": turn_details,
        },
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, float]] = {}
    for mode_name in MODES:
        mode_results = [item for item in results if item["mode"] == mode_name]
        if not mode_results:
            summary[mode_name] = {
                "total_cases": 0,
                "avg_repeated_source_reads": 0.0,
                "avg_memory_hit_skip_count": 0.0,
                "avg_tool_call_count": 0.0,
                "avg_input_tokens_est": 0.0,
            }
            continue

        summary[mode_name] = {
            "total_cases": len(mode_results),
            "avg_repeated_source_reads": round(mean(item["detail"]["repeated_source_reads"] for item in mode_results), 2),
            "avg_memory_hit_skip_count": round(mean(item["detail"]["memory_hit_skip_count"] for item in mode_results), 2),
            "avg_tool_call_count": round(mean(item["detail"]["avg_tool_call_count"] for item in mode_results), 2),
            "avg_input_tokens_est": round(mean(item["detail"]["avg_input_tokens_est"] for item in mode_results), 2),
        }

    if all(mode in summary for mode in MODES):
        memory_metrics = summary["three_layer_memory"]
        baseline_metrics = summary["replay_setup_baseline"]
        summary["comparison"] = {
            "repeated_source_read_delta": round(
                memory_metrics["avg_repeated_source_reads"] - baseline_metrics["avg_repeated_source_reads"], 2
            ),
            "memory_hit_skip_delta": round(
                memory_metrics["avg_memory_hit_skip_count"] - baseline_metrics["avg_memory_hit_skip_count"], 2
            ),
            "avg_tool_call_count_delta": round(
                memory_metrics["avg_tool_call_count"] - baseline_metrics["avg_tool_call_count"], 2
            ),
            "avg_input_tokens_est_delta": round(
                memory_metrics["avg_input_tokens_est"] - baseline_metrics["avg_input_tokens_est"], 2
            ),
        }

    return summary


def save_memory_followup_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    payload = {
        "title": "Memory Follow-up Evaluation",
        "total": len(results),
        "summary": summary,
        "results": results,
    }
    path = save_report("Memory Follow-up Evaluation", results, "memory_followup_report.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = path.parent / "memory_followup_report.latest.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cases = load_cases("memory_followup_cases.json")
    results: List[Dict[str, Any]] = []
    for mode_name in MODES:
        for case in cases:
            results.append(evaluate_case(case, mode_name=mode_name, api_base=DEFAULT_API_BASE))

    print_report("Memory Follow-up Evaluation", results)
    summary = build_summary(results)
    save_memory_followup_report(results, summary)
    print(f"Memory Follow-up 摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
