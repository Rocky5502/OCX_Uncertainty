"""Phase V, Step 13: Repair / Refine.

Sends the failed code + diagnostics back to the LLM for a focused
repair pass. The feedback also flows to Phase III to trigger another
retrieval round if requested (see pipeline.run() loop).
"""
from __future__ import annotations

import re

from ..llm.client import LLMClient

_SYSTEM = (
    "You are repairing Python code that failed verification. Read the diagnostics, "
    "produce a corrected version. Return ONLY a single Python code block. If the "
    "original task is a missing-region completion, return only the repaired missing "
    "region, not the surrounding file."
)

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
MAX_REPAIR_TASK_CHARS = 24_000
MAX_REPAIR_CODE_CHARS = 32_000
MAX_REPAIR_DIAGNOSTIC_CHARS = 32_000


def _bounded_text(value: str, limit: int, *, tail_fraction: float = 0.75) -> tuple[str, bool]:
    """Keep a deterministic head/tail view within a repair-prompt budget."""
    value = str(value or "")
    if len(value) <= limit:
        return value, False
    marker = "\n... [TRUNCATED TO FROZEN REPAIR PROMPT BUDGET] ...\n"
    available = max(0, limit - len(marker))
    tail = int(available * tail_fraction)
    head = available - tail
    return value[:head] + marker + value[-tail:], True


def build_repair_prompt(
    code: str,
    diagnostics: str,
    *,
    task: str | None = None,
    completion_mode: bool = False,
    expected_indent: str = "",
) -> tuple[str, dict[str, object]]:
    """Build the provider-safe repair prompt and return its budget audit."""
    bounded_task, task_truncated = _bounded_text(
        task or "", MAX_REPAIR_TASK_CHARS, tail_fraction=0.5
    )
    bounded_code, code_truncated = _bounded_text(
        code, MAX_REPAIR_CODE_CHARS, tail_fraction=0.5
    )
    bounded_diagnostics, diagnostics_truncated = _bounded_text(
        diagnostics, MAX_REPAIR_DIAGNOSTIC_CHARS, tail_fraction=0.8
    )
    prompt_parts = []
    if task:
        prompt_parts.append(f"# Original Task\n{bounded_task}")
    prompt_parts.append(f"# Failed Code\n```python\n{bounded_code}\n```")
    prompt_parts.append(
        f"# Diagnostics\n{bounded_diagnostics}\n\n"
        f"# Instruction\nReturn a corrected version that addresses the diagnostics."
    )
    if completion_mode:
        indent_desc = f"{len(expected_indent)} spaces" if expected_indent else "the prefix-implied indentation"
        prompt_parts.append(
            "# Completion Constraint\n"
            "Return only the missing middle region. Do not include the prefix code, "
            "suffix code, markdown commentary, tests, or a replacement full file. "
            f"Top-level statements in the missing region must use exactly {indent_desc}; "
            "nested blocks must add four spaces. If diagnostics mention indentation, "
            "fix indentation before changing behavior."
        )
    prompt = "\n\n".join(prompt_parts)
    return prompt, {
        "policy": "frozen_head_tail_character_budget_v1",
        "task_original_chars": len(task or ""),
        "task_used_chars": len(bounded_task),
        "task_truncated": task_truncated,
        "code_original_chars": len(code),
        "code_used_chars": len(bounded_code),
        "code_truncated": code_truncated,
        "diagnostics_original_chars": len(diagnostics),
        "diagnostics_used_chars": len(bounded_diagnostics),
        "diagnostics_truncated": diagnostics_truncated,
        "prompt_chars": len(prompt),
        "limits": {
            "task_chars": MAX_REPAIR_TASK_CHARS,
            "code_chars": MAX_REPAIR_CODE_CHARS,
            "diagnostics_chars": MAX_REPAIR_DIAGNOSTIC_CHARS,
        },
    }


def repair_code(
    code: str,
    diagnostics: str,
    llm: LLMClient,
    *,
    task: str | None = None,
    completion_mode: bool = False,
    expected_indent: str = "",
    prompt_audit: dict[str, object] | None = None,
) -> str:
    prompt, audit = build_repair_prompt(
        code,
        diagnostics,
        task=task,
        completion_mode=completion_mode,
        expected_indent=expected_indent,
    )
    if prompt_audit is not None:
        prompt_audit.update(audit)
    resp = llm.complete_one(prompt, system=_SYSTEM, max_tokens=1200, temperature=0.2, return_logprobs=False)
    m = _FENCE.search(resp.text)
    return (m.group(1) if m else resp.text).strip()
