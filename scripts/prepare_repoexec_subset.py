"""Prepare a local execution-backed RepoExec subset.

The public RepoExec rows include a full ``check`` script plus ``test_list``.
Some tests read precomputed files from the original /output mount, which is
not available in this workspace. This script keeps only self-contained tests
from ``test_list`` and stores enough repository prefix context to execute the
generated function with the local validator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset


def _normalize_prefix(prefix: str) -> str:
    text = (
        str(prefix)
        .replace("from ._regex import *", "from string_utils._regex import *")
        .replace("from .errors import", "from string_utils.errors import")
        .replace("from .validation import", "from string_utils.validation import")
    )
    if "from typing import Any" not in text:
        text = text.replace("from typing import Union", "from typing import Any, Union")
    return text


def _self_contained_tests(tests: Iterable[str]) -> List[str]:
    out = []
    for test in tests:
        if "/output/" in test or "open(" in test:
            continue
        if not str(test).strip():
            continue
        out.append(str(test).rstrip())
    return out


def _call_tests(tests: List[str]) -> str:
    blocks: List[str] = ["import pytest"]
    for idx, test in enumerate(tests):
        blocks.append(test)
        name = None
        first = test.strip().splitlines()[0] if test.strip() else ""
        if first.startswith("def ") and "(" in first:
            name = first.split("def ", 1)[1].split("(", 1)[0].strip()
        if name:
            blocks.append(f"{name}()")
        else:
            blocks.append(f"# skipped auto-call for test block {idx}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full_context")
    ap.add_argument("--project", default="test-apps/python-string-utils")
    ap.add_argument("--limit", type=int, default=7)
    ap.add_argument("--min-tests", type=int, default=5)
    ap.add_argument("--out", default="input/repoexec_python_string_utils_inline7.jsonl")
    args = ap.parse_args()

    ds = load_dataset("Fsoft-AIC/RepoExec", split=args.split)
    rows = []
    for item in ds:
        if args.project and item.get("project") != args.project:
            continue
        tests = _self_contained_tests(item.get("test_list") or [])
        if len(tests) < args.min_tests:
            continue
        prefix = _normalize_prefix(item.get("prompt") or "")
        rows.append({
            "task_id": f"{item.get('project')}/{item.get('id')}",
            "project": item.get("project"),
            "module": item.get("module"),
            "entry_point": item.get("entry_point"),
            "instruction": item.get("target_function_prompt"),
            "target_function_prompt": item.get("target_function_prompt"),
            "current_file": prefix,
            "execution_prefix_code": prefix,
            "extra_pythonpath": "input",
            "solution": item.get("solution"),
            "test": _call_tests(tests),
            "n_inline_tests": len(tests),
            "n_original_tests": len(item.get("test_list") or []),
            "repoexec_split": args.split,
            "repoexec_project": args.project,
        })
        if args.limit and len(rows) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
