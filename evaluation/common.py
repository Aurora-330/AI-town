"""最小自动化评测通用工具"""

from __future__ import annotations

import json
import os
import sqlite3
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_API_BASE = os.getenv("AI_TOWN_API_BASE", "http://127.0.0.1:8000")
DEFAULT_JUDGE_BASE = os.getenv("EVAL_LLM_BASE_URL", "http://127.0.0.1:8002/v1")
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
DEFAULT_JUDGE_API_KEY = os.getenv("EVAL_LLM_API_KEY", "helloagents-vllm")
DATASET_DIR = Path(__file__).parent / "datasets"
REPORT_DIR = Path(__file__).parent / "reports"
BACKEND_DIR = Path(__file__).parent.parent / "backend"
MEMORY_DATA_DIR = BACKEND_DIR / "memory_data"
REQUEST_TIMEOUT = 30
_SESSION = requests.Session()
_SESSION.trust_env = False


def load_cases(filename: str) -> List[Dict[str, Any]]:
    """加载数据集"""
    path = DATASET_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(method: str, path: str, api_base: str = DEFAULT_API_BASE, **kwargs) -> Dict[str, Any]:
    """发起 HTTP 请求并返回 JSON"""
    url = api_base.rstrip("/") + path
    response = _SESSION.request(method=method, url=url, timeout=REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def chat(npc_name: str, message: str, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """发送对话请求"""
    return request_json(
        "POST",
        "/chat",
        api_base=api_base,
        json={"npc_name": npc_name, "message": message}
    )


def clear_memories(npc_name: str, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """清空 NPC 记忆"""
    return request_json(
        "DELETE",
        f"/npcs/{npc_name}/memories",
        api_base=api_base
    )


def get_memories(npc_name: str, limit: int = 20, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """获取 NPC 记忆"""
    return request_json(
        "GET",
        f"/npcs/{npc_name}/memories?limit={limit}",
        api_base=api_base
    )


def get_npc_info(npc_name: str, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """获取 NPC 详情"""
    return request_json(
        "GET",
        f"/npcs/{npc_name}",
        api_base=api_base,
    )


def set_affinity(npc_name: str, affinity: float, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """设置 NPC 好感度"""
    return request_json(
        "PUT",
        f"/npcs/{npc_name}/affinity?affinity={affinity}&player_id=player",
        api_base=api_base
    )


def get_affinity(npc_name: str, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """获取 NPC 好感度"""
    return request_json(
        "GET",
        f"/npcs/{npc_name}/affinity?player_id=player",
        api_base=api_base
    )


def search_knowledge(query: str, limit: int = 3, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """查询外部知识库命中结果"""
    return request_json(
        "GET",
        f"/knowledge/search?q={query}&limit={limit}",
        api_base=api_base
    )


def get_summary_state(npc_name: str, player_id: str = "player") -> Dict[str, Any]:
    """读取本地 summary_state.json"""
    path = MEMORY_DATA_DIR / npc_name / "summary_state.json"
    if not path.exists():
        return {
            "summary_count": 0,
            "pending_turn_count": 0,
            "archived_count": 0,
            "raw_state": {"players": {}, "archived_memory_ids": []},
        }

    state = json.loads(path.read_text(encoding="utf-8"))
    player_state = state.get("players", {}).get(player_id, {})
    pending_turns = player_state.get("pending_turns", [])
    archived_memory_ids = state.get("archived_memory_ids", [])
    return {
        "summary_count": player_state.get("summary_count", 0),
        "pending_turn_count": len(pending_turns),
        "archived_count": len(archived_memory_ids),
        "raw_state": state,
    }


def get_summary_memories_from_sqlite(npc_name: str, player_id: str = "player", limit: int = 5) -> List[Dict[str, Any]]:
    """直接从本地 SQLite 读取摘要记忆

    评测场景运行在同一台机器上，因此这里允许把 SQLite 作为更稳定的验收信号，
    避免过度依赖 `/memories` 的检索排序结果。
    """
    db_path = MEMORY_DATA_DIR / npc_name / "memory.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, content, memory_type, properties
            FROM memories
            WHERE user_id = ? AND memory_type = 'episodic'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (npc_name, limit * 10),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    summaries: List[Dict[str, Any]] = []
    for row in rows:
        properties = json.loads(row[3] or "{}")
        context = properties.get("context", {})
        session_id = properties.get("session_id")

        # HelloAgents 当前实际落盘的摘要信号:
        # 1. memory_type=episodic
        # 2. content 以“摘要记忆:”开头
        # 3. properties.context.interaction_type == "summary"
        # 4. properties.session_id 对应 player_id
        if context.get("interaction_type") != "summary":
            continue
        if session_id != player_id:
            continue
        if not (row[1] or "").startswith("摘要记忆:"):
            continue
        summaries.append(
            {
                "id": row[0],
                "content": row[1],
                "memory_type": row[2],
                "metadata": {
                    "session_id": session_id,
                    "context": context,
                },
            }
        )
        if len(summaries) >= limit:
            break

    return summaries


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    """返回命中的关键词列表"""
    return [keyword for keyword in keywords if keyword in text]


def evaluate_keyword_case(
    text: str,
    expected_keywords: List[str],
    min_hits: int = 1,
    forbidden_keywords: List[str] | None = None
) -> Tuple[bool, Dict[str, Any]]:
    """评估关键词样本"""
    hits = keyword_hits(text, expected_keywords)
    forbidden_hits = keyword_hits(text, forbidden_keywords or [])
    passed = len(hits) >= min_hits and not forbidden_hits
    return passed, {
        "hits": hits,
        "forbidden_hits": forbidden_hits,
        "hit_count": len(hits),
        "required_min_hits": min_hits
    }


def print_report(title: str, results: List[Dict[str, Any]]):
    """打印简单报告"""
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    print(f"\n=== {title} ===")
    print(f"通过: {passed}/{total}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']}")
        print(f"  npc: {item['npc']}")
        print(f"  detail: {item['detail']}")


def save_report(title: str, results: List[Dict[str, Any]], output_name: str) -> Path:
    """保存 JSON 报告到 reports 目录"""
    REPORT_DIR.mkdir(exist_ok=True)
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    payload = {
        "title": title,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
        "results": results
    }

    output_path = REPORT_DIR / output_name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    latest_path = REPORT_DIR / "report.json"
    latest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n报告已保存: {output_path}")
    print(f"最新报告已更新: {latest_path}")
    return output_path


def _extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON 对象"""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty_response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"json_not_found: {cleaned[:200]}")

    return json.loads(match.group(0))


def llm_chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE,
    api_key: str = DEFAULT_JUDGE_API_KEY,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> str:
    """调用本地 OpenAI-compatible LLM"""
    url = base_url.rstrip("/") + "/chat/completions"
    response = _SESSION.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def llm_chat_json(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE,
    api_key: str = DEFAULT_JUDGE_API_KEY,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """调用 Judge 并解析 JSON"""
    content = llm_chat(
        messages=messages,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_json_object(content)


def build_stateless_baseline_reply(
    npc_name: str,
    npc_title: str,
    query: str,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE,
    api_key: str = DEFAULT_JUDGE_API_KEY,
) -> str:
    """生成无记忆、无知识注入的同模型基线回答"""
    system_prompt = (
        f"你是{npc_name}，身份是{npc_title}。"
        "请仅根据用户当前这句输入作答，不要假装记得历史对话，"
        "不要假装读过外部知识库，也不要编造未提供的背景。"
        "回复保持自然、简洁、符合角色身份。"
    )
    return llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=220,
    ).strip()
