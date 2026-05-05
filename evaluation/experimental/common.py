"""实验评测的本地包装层。

复用 evaluation/common.py 里的 HTTP 与评测工具，
只把数据集与报告目录切到 experimental/ 自己的子目录。
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent.parent
BASE_COMMON_PATH = EVAL_ROOT / "common.py"
_spec = importlib.util.spec_from_file_location("evaluation_root_common", BASE_COMMON_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载共享 common.py: {BASE_COMMON_PATH}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


DEFAULT_API_BASE = _base.DEFAULT_API_BASE
DEFAULT_JUDGE_BASE = _base.DEFAULT_JUDGE_BASE
DEFAULT_JUDGE_MODEL = _base.DEFAULT_JUDGE_MODEL
DEFAULT_JUDGE_API_KEY = _base.DEFAULT_JUDGE_API_KEY
BACKEND_DIR = _base.BACKEND_DIR
MEMORY_DATA_DIR = _base.MEMORY_DATA_DIR
REQUEST_TIMEOUT = _base.REQUEST_TIMEOUT

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

request_json = _base.request_json
chat = _base.chat
multi_chat = _base.multi_chat
clear_memories = _base.clear_memories
get_memories = _base.get_memories
get_npc_info = _base.get_npc_info
set_affinity = _base.set_affinity
get_affinity = _base.get_affinity
search_knowledge = _base.search_knowledge
get_summary_state = _base.get_summary_state
get_summary_memories_from_sqlite = _base.get_summary_memories_from_sqlite
keyword_hits = _base.keyword_hits
evaluate_keyword_case = _base.evaluate_keyword_case
print_report = _base.print_report
llm_chat_json = _base.llm_chat_json
build_stateless_baseline_reply = _base.build_stateless_baseline_reply


def load_cases(filename: str):
    path = DATASET_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def save_report(title: str, results, output_name: str):
    REPORT_DIR.mkdir(exist_ok=True)
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    payload = {
        "title": title,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "results": results,
    }
    output_path = REPORT_DIR / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
