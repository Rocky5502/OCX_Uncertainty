"""Phase I, Step 2: Generate NL Descriptions for each knowledge item."""
from __future__ import annotations

from typing import List

from ..llm.client import LLMClient
from .extract import RepoFunction

_SYSTEM = (
    "You are a senior code reviewer. Given a Python function or class, write a single "
    "concise paragraph (3-5 sentences) describing WHAT it does and WHEN you would call it. "
    "Do not restate the code. Focus on intent and side effects."
)


def describe_functions(items: List[RepoFunction], llm: LLMClient, batch_log: bool = False) -> List[RepoFunction]:
    for i, it in enumerate(items):
        prompt = f"File: {it.file_path}\nSignature: {it.signature}\n\nCode:\n```python\n{it.body[:4000]}\n```"
        try:
            resp = llm.complete_one(prompt, system=_SYSTEM, max_tokens=200, return_logprobs=True)
            it.description = resp.text.strip()
            it.metadata["describe_logprobs"] = resp.logprobs
        except Exception as e:
            it.description = it.docstring or it.signature
            it.metadata["describe_error"] = str(e)
        if batch_log and (i + 1) % 25 == 0:
            print(f"  described {i + 1}/{len(items)}")
    return items
