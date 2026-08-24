from opencoder.phase5_verify.repair import (
    MAX_REPAIR_CODE_CHARS,
    MAX_REPAIR_DIAGNOSTIC_CHARS,
    MAX_REPAIR_TASK_CHARS,
    build_repair_prompt,
)


def test_repair_prompt_applies_frozen_head_tail_budgets() -> None:
    task = "TASK-START\n" + "t" * 100_000 + "\nTASK-END"
    code = "CODE-START\n" + "c" * 100_000 + "\nCODE-END"
    diagnostics = "DIAG-START\n" + "d" * 500_000 + "\nDIAG-END"

    prompt, audit = build_repair_prompt(
        code,
        diagnostics,
        task=task,
        completion_mode=True,
        expected_indent="    ",
    )

    assert audit["task_used_chars"] == MAX_REPAIR_TASK_CHARS
    assert audit["code_used_chars"] == MAX_REPAIR_CODE_CHARS
    assert audit["diagnostics_used_chars"] == MAX_REPAIR_DIAGNOSTIC_CHARS
    assert audit["task_truncated"] is True
    assert audit["code_truncated"] is True
    assert audit["diagnostics_truncated"] is True
    assert "TASK-START" in prompt and "TASK-END" in prompt
    assert "CODE-START" in prompt and "CODE-END" in prompt
    assert "DIAG-START" in prompt and "DIAG-END" in prompt
    assert len(prompt) < 90_000


def test_repair_prompt_leaves_short_inputs_unchanged() -> None:
    prompt, audit = build_repair_prompt("code", "diagnostic", task="task")

    assert "# Original Task\ntask" in prompt
    assert "# Failed Code\n```python\ncode\n```" in prompt
    assert "# Diagnostics\ndiagnostic" in prompt
    assert audit["task_truncated"] is False
    assert audit["code_truncated"] is False
    assert audit["diagnostics_truncated"] is False
