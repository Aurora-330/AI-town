"""Token counting utilities for prompt budgeting.

当前阶段的目标:
- 引入 Hugging Face tokenizer 作为 token 计数基础设施
- 不直接改变现有聊天行为
- tokenizer 不可用时，降级到轻量估算器，避免服务启动失败
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass
class TokenCountBreakdown:
    """结构化 token 计数结果。"""

    total_tokens: int
    text_tokens: int
    overhead_tokens: int
    tokenizer_backend: str


class BaseTokenCounter:
    """Token 计数器抽象基类。"""

    backend_name = "base"

    def count_text_tokens(self, text: str) -> int:
        raise NotImplementedError

    def count_messages_tokens(self, messages: list[dict]) -> TokenCountBreakdown:
        text_tokens = 0
        overhead_tokens = 0
        for message in messages:
            text_tokens += self.count_text_tokens(str(message.get("content", "") or ""))
            # role/content 包装以及消息边界的粗略固定开销。
            overhead_tokens += 4

        return TokenCountBreakdown(
            total_tokens=text_tokens + overhead_tokens,
            text_tokens=text_tokens,
            overhead_tokens=overhead_tokens,
            tokenizer_backend=self.backend_name,
        )

    def count_sections_tokens(self, sections: dict[str, str]) -> dict[str, int]:
        return {
            name: self.count_text_tokens(text)
            for name, text in sections.items()
        }


class HeuristicTokenCounter(BaseTokenCounter):
    """当 HF tokenizer 不可用时的轻量降级估算器。"""

    backend_name = "heuristic"

    def count_text_tokens(self, text: str) -> int:
        cleaned = str(text or "").strip()
        if not cleaned:
            return 0
        # 对中英混合文本做保守估算，宁可稍微高估。
        return max(1, ceil(len(cleaned) / 1.6))


class HFTokenCounter(BaseTokenCounter):
    """基于 Hugging Face tokenizer 的 token 计数器。"""

    backend_name = "huggingface"

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

    def count_text_tokens(self, text: str) -> int:
        cleaned = str(text or "")
        if not cleaned:
            return 0
        return len(self.tokenizer.encode(cleaned, add_special_tokens=False))


def build_token_counter(
    model_name: str,
    trust_remote_code: bool = True,
) -> BaseTokenCounter:
    """优先构建 HF tokenizer，失败时降级为 heuristic。"""
    try:
        return HFTokenCounter(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(f"⚠️ Tokenizer初始化失败，已降级为heuristic估算: {exc}")
        return HeuristicTokenCounter()
