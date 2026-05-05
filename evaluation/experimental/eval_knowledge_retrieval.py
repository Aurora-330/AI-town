"""Knowledge retrieval 专项评测。

目标：
1. 验证 lexical/BM25 是否真正参与中文 query 检索。
2. 验证 rerank 是否能把精确专名词/标题块推到前列。
3. 验证当前 chunk 结构是否支持多事实单块命中。
4. 验证长上下文 prompt budgeting 是否影响 grounded reply。
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, List, Optional

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    evaluate_keyword_case,
    keyword_hits,
    load_cases,
    request_json,
    save_report,
    set_affinity,
)


def search_knowledge_debug(query: str, limit: int = 5, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """显式使用 params，避免中文 query 拼接到 URL 时出问题。"""
    return request_json(
        "GET",
        "/knowledge/search",
        api_base=api_base,
        params={"q": query, "limit": limit},
    )


def _find_rank(
    hits: List[Dict[str, Any]],
    source: str,
    chunk_index: Optional[int] = None,
) -> Optional[int]:
    for index, hit in enumerate(hits, start=1):
        if hit.get("source") != source:
            continue
        if chunk_index is not None and int(hit.get("chunk_index", -1)) != int(chunk_index):
            continue
        return index
    return None


def _contains_keywords(text: str, keywords: List[str]) -> Dict[str, Any]:
    hits = keyword_hits(text or "", keywords)
    return {
        "hits": hits,
        "hit_count": len(hits),
        "all_hit": len(hits) == len(keywords),
    }


def _evaluate_topk_presence(hits: List[Dict[str, Any]], expectations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for item in expectations:
        rank = _find_rank(
            hits,
            source=str(item["source"]),
            chunk_index=item.get("chunk_index"),
        )
        max_rank = int(item.get("max_rank", len(hits)))
        results.append(
            {
                "source": item["source"],
                "chunk_index": item.get("chunk_index"),
                "rank": rank,
                "max_rank": max_rank,
                "passed": rank is not None and rank <= max_rank,
            }
        )
    return results


def _collect_retrieval_signals(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    lexical_ranks = []
    semantic_ranks = []
    hybrid_ranks = []
    for index, hit in enumerate(hits, start=1):
        sources = list(hit.get("retrieval_sources", []))
        if "lexical" in sources:
            lexical_ranks.append(index)
        if "semantic" in sources:
            semantic_ranks.append(index)
        if "lexical" in sources and "semantic" in sources:
            hybrid_ranks.append(index)
    return {
        "lexical_ranks": lexical_ranks,
        "semantic_ranks": semantic_ranks,
        "hybrid_ranks": hybrid_ranks,
        "lexical_hit_in_topk": bool(lexical_ranks),
        "hybrid_hit_in_topk": bool(hybrid_ranks),
    }


def _evaluate_search_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    response = search_knowledge_debug(
        query=case["query"],
        limit=int(case.get("limit", 5)),
        api_base=api_base,
    )
    hits = list(response.get("hits", []))
    top1 = hits[0] if hits else {}
    top1_text = str(top1.get("content", ""))
    signals = _collect_retrieval_signals(hits)

    top1_source_ok = not case.get("expected_top1_source") or top1.get("source") == case.get("expected_top1_source")
    top1_chunk_ok = "expected_top1_chunk_index" not in case or int(top1.get("chunk_index", -1)) == int(case["expected_top1_chunk_index"])
    top1_keyword_detail = _contains_keywords(top1_text, case.get("expected_top1_keywords", []))

    any_hit_keywords = case.get("expected_any_hit_keywords", [])
    aggregated_hit_text = "\n".join(str(item.get("content", "")) for item in hits)
    any_hit_keyword_detail = _contains_keywords(aggregated_hit_text, any_hit_keywords)

    topk_presence = _evaluate_topk_presence(hits, case.get("expected_topk_contains", []))
    topk_presence_ok = all(item["passed"] for item in topk_presence) if topk_presence else True

    lexical_expectation = case.get("expect_lexical_hit_in_topk")
    lexical_ok = True if lexical_expectation is None else signals["lexical_hit_in_topk"] == bool(lexical_expectation)

    passed = all(
        [
            bool(hits),
            top1_source_ok,
            top1_chunk_ok,
            top1_keyword_detail["all_hit"] if case.get("expected_top1_keywords") else True,
            any_hit_keyword_detail["all_hit"] if any_hit_keywords else True,
            topk_presence_ok,
            lexical_ok,
        ]
    )

    return {
        "case_id": case["id"],
        "hypothesis": case["hypothesis"],
        "mode": "search",
        "npc": case.get("npc", ""),
        "passed": passed,
        "detail": {
            "goal": case.get("goal", ""),
            "query": case["query"],
            "total_hits": len(hits),
            "top1_source": top1.get("source", ""),
            "top1_chunk_index": top1.get("chunk_index", -1),
            "top1_keywords": top1_keyword_detail,
            "any_hit_keywords": any_hit_keyword_detail,
            "topk_presence": topk_presence,
            "retrieval_signals": signals,
            "top3_preview": [
                {
                    "rank": index + 1,
                    "source": hit.get("source", ""),
                    "chunk_index": hit.get("chunk_index", -1),
                    "retrieval_sources": hit.get("retrieval_sources", []),
                    "semantic_score": hit.get("semantic_score", 0.0),
                    "lexical_score": hit.get("lexical_score", 0.0),
                    "rerank_score": hit.get("rerank_score", 0.0),
                    "content_preview": str(hit.get("content", ""))[:180],
                }
                for index, hit in enumerate(hits[:3])
            ],
        },
    }


def _seed_memory(npc_name: str, setup_messages: List[str], api_base: str, execution_mode: str = "static_coordinator"):
    for message in setup_messages:
        chat(
            npc_name=npc_name,
            message=message,
            api_base=api_base,
            execution_mode=execution_mode,
        )


def _evaluate_chat_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    npc_name = str(case["npc"])
    execution_mode = str(case.get("execution_mode", "auto"))

    clear_memories(npc_name, api_base=api_base)
    set_affinity(npc_name, 50, api_base=api_base)
    _seed_memory(
        npc_name=npc_name,
        setup_messages=list(case.get("setup_messages", [])),
        api_base=api_base,
    )

    search_debug = search_knowledge_debug(
        query=case["query"],
        limit=int(case.get("limit", 5)),
        api_base=api_base,
    )
    search_hits = list(search_debug.get("hits", []))
    search_topk_presence = _evaluate_topk_presence(search_hits, case.get("expected_search_topk_contains", []))

    response = chat(
        npc_name=npc_name,
        message=case["query"],
        api_base=api_base,
        execution_mode=execution_mode,
    )
    reply = str(response.get("message", ""))
    reply_ok, reply_detail = evaluate_keyword_case(
        text=reply,
        expected_keywords=case.get("expected_reply_keywords", []),
        min_hits=len(case.get("expected_reply_keywords", [])) if case.get("expected_reply_keywords") else 1,
        forbidden_keywords=case.get("forbidden_reply_keywords", []),
    )

    query_mode_ok = not case.get("expected_query_mode") or response.get("query_mode") == case.get("expected_query_mode")
    search_grounding_ok = all(item["passed"] for item in search_topk_presence) if search_topk_presence else True

    passed = reply_ok and query_mode_ok and search_grounding_ok and bool(response.get("success", False))

    return {
        "case_id": case["id"],
        "hypothesis": case["hypothesis"],
        "mode": "chat",
        "npc": npc_name,
        "passed": passed,
        "detail": {
            "goal": case.get("goal", ""),
            "query": case["query"],
            "reply": reply,
            "reply_keyword_hits": reply_detail["hits"],
            "reply_forbidden_hits": reply_detail["forbidden_hits"],
            "query_mode": response.get("query_mode", ""),
            "expected_query_mode": case.get("expected_query_mode", ""),
            "tool_call_count": response.get("tool_call_count", 0),
            "react_activated": response.get("react_activated", False),
            "input_tokens_est": response.get("input_tokens_est", 0),
            "latency_ms": response.get("latency_ms", 0),
            "search_topk_presence": search_topk_presence,
            "search_top3_preview": [
                {
                    "rank": index + 1,
                    "source": hit.get("source", ""),
                    "chunk_index": hit.get("chunk_index", -1),
                    "retrieval_sources": hit.get("retrieval_sources", []),
                    "semantic_score": hit.get("semantic_score", 0.0),
                    "lexical_score": hit.get("lexical_score", 0.0),
                    "rerank_score": hit.get("rerank_score", 0.0),
                    "content_preview": str(hit.get("content", ""))[:180],
                }
                for index, hit in enumerate(search_hits[:3])
            ],
        },
    }


def evaluate_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    if case.get("mode") == "chat":
        return _evaluate_chat_case(case, api_base=api_base)
    return _evaluate_search_case(case, api_base=api_base)


def _aggregate_bucket(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_input_tokens_est": 0.0,
        }

    passed = sum(1 for item in results if item["passed"])
    latencies = [
        int(item["detail"].get("latency_ms", 0))
        for item in results
        if item["mode"] == "chat"
    ]
    input_tokens = [
        int(item["detail"].get("input_tokens_est", 0))
        for item in results
        if item["mode"] == "chat"
    ]
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
        "avg_input_tokens_est": round(mean(input_tokens), 2) if input_tokens else 0.0,
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    hypotheses = sorted({item["hypothesis"] for item in results})
    modes = sorted({item["mode"] for item in results})

    return {
        "overall": _aggregate_bucket(results),
        "by_hypothesis": {
            hypothesis: _aggregate_bucket([item for item in results if item["hypothesis"] == hypothesis])
            for hypothesis in hypotheses
        },
        "by_mode": {
            mode: _aggregate_bucket([item for item in results if item["mode"] == mode])
            for mode in modes
        },
    }


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    overall = summary["overall"]
    print("\n=== Knowledge Retrieval Evaluation ===")
    print(f"通过: {overall['passed']}/{overall['total']}")
    print(f"通过率: {overall['pass_rate']:.2%}")

    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} ({item['hypothesis']}/{item['mode']})")
        print(f"  query: {item['detail']['query']}")
        if item["mode"] == "search":
            print(f"  top1: {item['detail']['top1_source']}#{item['detail']['top1_chunk_index']}")
            print(f"  retrieval_signals: {item['detail']['retrieval_signals']}")
        else:
            print(f"  query_mode: {item['detail']['query_mode']}")
            print(f"  reply_hits: {item['detail']['reply_keyword_hits']}")


def save_knowledge_report(results: List[Dict[str, Any]], summary: Dict[str, Any], output_name: str):
    payload = {
        "title": "Knowledge Retrieval Evaluation",
        "total": len(results),
        "summary": summary,
        "results": results,
    }
    path = save_report("Knowledge Retrieval Evaluation", results, output_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = path.parent / "report.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Knowledge retrieval 摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")


def main():
    cases = load_cases("knowledge_retrieval_cases.json")
    results = [evaluate_case(case, api_base=DEFAULT_API_BASE) for case in cases]
    summary = build_summary(results)
    print_report(results, summary)
    save_knowledge_report(results, summary, "knowledge_retrieval_report.json")


if __name__ == "__main__":
    main()
