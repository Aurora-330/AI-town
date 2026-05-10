"""评测上下文治理收益。

统计 `_apply_prompt_budget()` 前后的 token 变化，并汇总压缩收益与超窗异常。
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, List

from common import DEFAULT_API_BASE, chat, clear_memories, load_cases, print_report, save_report


def seed_memory(npc_name: str, seed_messages: List[str], api_base: str, execution_mode: str):
    for index, message in enumerate(seed_messages, start=1):
        chat(
            npc_name,
            f"预算种子{index}: {message}",
            api_base=api_base,
            execution_mode=execution_mode,
        )


def evaluate_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    npc_name = case["npc"]
    execution_mode = case.get("execution_mode", "auto")
    clear_memories(npc_name, api_base=api_base)
    seed_memory(npc_name, case.get("memory_seed", []), api_base=api_base, execution_mode=execution_mode)

    response = chat(
        npc_name,
        case["query"],
        api_base=api_base,
        execution_mode=execution_mode,
    )
    budget = dict(response.get("prompt_budget_debug", {}) or {})
    before_total = int(budget.get("before_total_tokens_est", 0) or 0)
    after_total = int(budget.get("after_total_tokens_est", response.get("input_tokens_est", 0)) or 0)
    compression_ratio = ((before_total - after_total) / before_total) if before_total > 0 else 0.0
    overflow = response.get("error_type") == "context_window_exceeded"

    return {
        "case_id": case["id"],
        "npc": npc_name,
        "passed": not overflow,
        "detail": {
            "execution_mode": response.get("execution_mode", execution_mode),
            "query_mode": response.get("query_mode", "default"),
            "before_section_tokens": budget.get("before_section_tokens", {}),
            "after_section_tokens": budget.get("after_section_tokens", {}),
            "before_total_tokens_est": before_total,
            "after_total_tokens_est": after_total,
            "total_input_tokens_est": int(response.get("input_tokens_est", 0)),
            "trimmed_sections": budget.get("trimmed_sections", []),
            "input_limit_tokens": int(budget.get("input_limit_tokens", 0) or 0),
            "compression_ratio": round(compression_ratio, 4),
            "overflow_error": overflow,
            "error_type": response.get("error_type", ""),
            "reply_preview": str(response.get("message", ""))[:120],
        },
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "avg_before_total_tokens_est": 0.0,
            "avg_after_total_tokens_est": 0.0,
            "avg_compression_ratio": 0.0,
            "max_compression_ratio": 0.0,
            "overflow_error_count": 0,
            "trimmed_case_count": 0,
        }

    before_totals = [int(item["detail"]["before_total_tokens_est"]) for item in results]
    after_totals = [int(item["detail"]["after_total_tokens_est"]) for item in results]
    compression_ratios = [float(item["detail"]["compression_ratio"]) for item in results]
    overflow_count = sum(1 for item in results if item["detail"]["overflow_error"])
    trimmed_case_count = sum(1 for item in results if item["detail"].get("trimmed_sections"))

    return {
        "total": len(results),
        "avg_before_total_tokens_est": round(mean(before_totals), 2),
        "avg_after_total_tokens_est": round(mean(after_totals), 2),
        "avg_compression_ratio": round(mean(compression_ratios), 4),
        "max_compression_ratio": round(max(compression_ratios), 4),
        "overflow_error_count": overflow_count,
        "trimmed_case_count": trimmed_case_count,
    }


def save_budget_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    payload = {
        "title": "Prompt Budget Evaluation",
        "total": len(results),
        "summary": summary,
        "results": results,
    }
    path = save_report("Prompt Budget Evaluation", results, "prompt_budget_report.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = path.parent / "prompt_budget_report.latest.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cases = load_cases("prompt_budget_cases.json")
    results = [evaluate_case(case, api_base=DEFAULT_API_BASE) for case in cases]
    print_report("Prompt Budget Evaluation", results)
    summary = build_summary(results)
    save_budget_report(results, summary)
    print(f"Prompt Budget 摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
