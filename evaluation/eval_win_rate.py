"""最小 Win Rate 对比评测"""

from __future__ import annotations

from common import (
    DEFAULT_API_BASE,
    build_stateless_baseline_reply,
    chat,
    clear_memories,
    get_npc_info,
    llm_chat_json,
    load_cases,
    print_report,
    save_report,
    set_affinity,
)


WIN_RATE_SYSTEM_PROMPT = """你是一个 A/B 回答对比裁判。
请比较两个候选回答，判断哪个更好。

只输出一个 JSON 对象，不要输出 Markdown，不要解释。
JSON 字段必须包含：
- winner: "A" / "B" / "Tie"
- reason: 不超过80字的中文理由

判断标准：
- 是否更贴合用户问题
- 是否更准确利用长期上下文或知识线索
- 是否更符合对应 NPC 的角色定位
- 是否更具体、更自然、信息量更高

不要因为句子更长就默认更好。"""


def _judge_pair(case: dict, answer_a: str, answer_b: str) -> dict:
    prompt = f"""请比较两个候选回答。

实验类型: {case["experiment"]}
NPC: {case["npc"]}
用户问题: {case["query"]}
评测目标: {case["evaluation_goal"]}

候选回答 A:
{answer_a}

候选回答 B:
{answer_b}
"""
    return llm_chat_json(
        [
            {"role": "system", "content": WIN_RATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=220,
    )


def main():
    cases = load_cases("win_rate_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        for message in case.get("setup_messages", []):
            chat(npc, message, api_base=DEFAULT_API_BASE)

        context_reply = chat(npc, case["query"], api_base=DEFAULT_API_BASE)["message"]
        npc_info = get_npc_info(npc, api_base=DEFAULT_API_BASE)
        baseline_reply = build_stateless_baseline_reply(
            npc_name=npc,
            npc_title=npc_info.get("title", ""),
            query=case["query"],
        )

        judge_result = _judge_pair(case, context_reply, baseline_reply)
        winner = judge_result["winner"]
        passed = winner == "A"

        results.append(
            {
                "case_id": case["id"],
                "npc": npc,
                "passed": passed,
                "detail": {
                    "experiment": case["experiment"],
                    "winner": winner,
                    "reason": judge_result["reason"],
                    "context_aware_reply": context_reply,
                    "stateless_baseline_reply": baseline_reply,
                },
            }
        )

    print_report("Win Rate Evaluation", results)
    save_report("Win Rate Evaluation", results, "win_rate_report.json")


if __name__ == "__main__":
    main()
