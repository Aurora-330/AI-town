"""基于 LLM-as-a-judge 的最小质量评测"""

from __future__ import annotations

from statistics import mean

from common import (
    DEFAULT_API_BASE,
    chat,
    clear_memories,
    get_summary_memories_from_sqlite,
    keyword_hits,
    llm_chat_json,
    load_cases,
    print_report,
    save_report,
    search_knowledge,
    set_affinity,
)


JUDGE_SYSTEM_PROMPT = """你是一个严格但公正的 AI 应用评测裁判。
你将评估多角色 Agent 的单轮回复质量。

请只输出一个 JSON 对象，不要输出 Markdown，不要解释。
JSON 字段必须包含：
- memory_faithfulness: 1-5 整数
- summary_quality: 1-5 整数
- grounding: 1-5 整数
- persona_consistency: 1-5 整数
- pass: true/false
- reason: 不超过60字的中文说明

评分原则：
- memory_faithfulness：是否正确利用已有互动记忆，不编造历史
- summary_quality：总结类问题时，是否抓住主要话题、偏好、未完成事项、关键事实
- grounding：是否正确利用知识，不脱离给定知识点乱答
- persona_consistency：是否符合对应 NPC 的角色定位与说话风格

如果某个维度和当前样本关系较弱，也仍需给分：
- 5 = 表现优秀，明显满足目标
- 4 = 基本到位，有轻微缺点
- 3 = 部分满足，质量一般
- 2 = 明显不足
- 1 = 严重不满足

pass 规则：整体质量达到可接受水平时为 true，否则为 false。"""


def _build_user_prompt(case: dict, reply: str, summary_texts: list[str], knowledge_hits: list[dict]) -> str:
    knowledge_brief = []
    for item in knowledge_hits[:2]:
        knowledge_brief.append(
            f"- {item.get('title', '')}: {item.get('content', '')[:180]}"
        )

    summary_brief = "\n".join(f"- {text[:180]}" for text in summary_texts[:2]) or "- 无"
    knowledge_text = "\n".join(knowledge_brief) or "- 无"
    expected_focus = "\n".join(f"- {item}" for item in case.get("expected_focus", []))

    return f"""请评估下面这条 Agent 回复。

样本类型: {case["case_type"]}
NPC: {case["npc"]}
用户问题: {case["query"]}
期望关注点:
{expected_focus}

评测目标:
{case["evaluation_goal"]}

可用摘要记忆:
{summary_brief}

可用知识命中:
{knowledge_text}

模型回复:
{reply}
"""


def main():
    cases = load_cases("llm_judge_cases.json")
    results = []

    for case in cases:
        npc = case["npc"]
        clear_memories(npc, api_base=DEFAULT_API_BASE)
        set_affinity(npc, 50, api_base=DEFAULT_API_BASE)

        for message in case.get("setup_messages", []):
            chat(npc, message, api_base=DEFAULT_API_BASE)

        response = chat(npc, case["query"], api_base=DEFAULT_API_BASE)
        reply = response["message"]
        summary_memories = get_summary_memories_from_sqlite(npc, limit=3)
        knowledge = search_knowledge(case["query"], limit=2, api_base=DEFAULT_API_BASE)

        judge_result = llm_chat_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        case=case,
                        reply=reply,
                        summary_texts=[item["content"] for item in summary_memories],
                        knowledge_hits=knowledge.get("hits", []),
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=500,
        )

        scores = {
            "memory_faithfulness": int(judge_result["memory_faithfulness"]),
            "summary_quality": int(judge_result["summary_quality"]),
            "grounding": int(judge_result["grounding"]),
            "persona_consistency": int(judge_result["persona_consistency"]),
        }
        average_score = round(mean(scores.values()), 2)
        expected_hits = keyword_hits(reply, case.get("expected_focus", []))

        results.append(
            {
                "case_id": case["id"],
                "npc": npc,
                "passed": bool(judge_result["pass"]),
                "detail": {
                    "case_type": case["case_type"],
                    "scores": scores,
                    "average_score": average_score,
                    "reason": judge_result["reason"],
                    "expected_focus_hits": expected_hits,
                    "reply": reply,
                },
            }
        )

    print_report("LLM Judge Evaluation", results)
    save_report("LLM Judge Evaluation", results, "llm_judge_report.json")


if __name__ == "__main__":
    main()
