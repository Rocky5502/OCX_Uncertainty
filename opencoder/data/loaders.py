"""Dataset loaders for ExecRepoBench, CoderEval, RepoExec, and samples.

Each loader normalizes records into the Example dataclass below. Only
the fields needed by the pipeline are required; the rest are kept in
`raw` for downstream analysis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class Example:
    id: str
    query: str                # natural-language task description
    repo_root: Optional[str]  # path to a checked-out repo, if any
    reference_code: Optional[str] = None
    test_code: Optional[str] = None
    raw: dict = field(default_factory=dict)


def _read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _clip(text: str, *, head: int = 3000, tail: int = 3000) -> str:
    if len(text) <= head + tail:
        return text
    head_part = text[:head] if head > 0 else ""
    tail_part = text[-tail:] if tail > 0 else ""
    return head_part + "\n\n# ... clipped ...\n\n" + tail_part


def _repo_root_for_jsonl(path: str) -> str:
    return os.path.dirname(os.path.abspath(path)) or "."


def _completion_query(rec: dict) -> str:
    prefix = rec.get("prefix_code") or ""
    suffix = rec.get("suffix_code") or ""
    file_name = rec.get("file_name") or rec.get("path") or "unknown.py"
    fill_type = rec.get("fill_type")
    label = f" ({fill_type})" if fill_type else ""
    return (
        f"Complete the missing Python code{label} in {file_name}. "
        "Return only the code that belongs in the missing region.\n\n"
        "# Prefix Code\n"
        f"```python\n{_clip(prefix, head=0, tail=4000)}\n```\n\n"
        "# Suffix Code\n"
        f"```python\n{_clip(suffix, head=4000, tail=0)}\n```"
    )


def _is_opencoderx_execrepobench(rec: dict) -> bool:
    return str(rec.get("manifest_version") or "").startswith(
        "execrepobench_opencoderx_"
    )


def _opencoderx_execrepobench_query(rec: dict) -> str:
    stub = str(rec.get("target_function_prompt") or rec.get("prompt") or "")
    prefix = str(rec.get("execution_prefix_code") or "")
    suffix = str(rec.get("execution_suffix_code") or "")
    file_name = str(rec.get("file_name") or "unknown.py")
    local_context = (
        _clip(prefix, head=0, tail=3500)
        + "\n"
        + stub
        + "\n"
        + _clip(suffix, head=3500, tail=0)
    )
    return (
        "Implement the complete target Python function in this repository. "
        "Return exactly one complete function definition, preserving the target "
        "signature, indentation, decorators, and docstring.\n\n"
        "# Target Function Stub\n"
        f"```python\n{stub}\n```\n\n"
        f"# Target File: {file_name}\n"
        f"```python\n{local_context}\n```"
    )


def _opencoderx_repo_root(path: str, rec: dict) -> Optional[str]:
    configured = os.environ.get("OPENCODER_EXECREPOBENCH_REPOS")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    manifest = Path(path).resolve()
    candidates.extend(
        parent / ".benchmarks" / "execrepobench" / "repos"
        for parent in manifest.parents
    )
    repo_name = str(rec.get("repo_name") or "")
    for repos_root in candidates:
        repo = repos_root / repo_name
        if repo.is_dir():
            return os.fspath(repo.resolve())
    return None


def load_execrepobench(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        raw = dict(rec)
        if _is_opencoderx_execrepobench(rec):
            query = _opencoderx_execrepobench_query(rec)
            reference = rec.get("solution")
            test_code = None
            repo_root = _opencoderx_repo_root(path, rec)
            if repo_root:
                raw["_runtime_repo_root"] = repo_root
        elif rec.get("prefix_code") is not None or rec.get("suffix_code") is not None:
            query = _completion_query(rec)
            reference = rec.get("middle_code") or rec.get("reference") or rec.get("canonical_solution")
            test_code = rec.get("test") or rec.get("test_code")
            repo_root = rec.get("repo_root")
        else:
            query = str(rec.get("prompt") or rec.get("instruction") or rec.get("query") or "")
            reference = rec.get("reference") or rec.get("canonical_solution")
            test_code = rec.get("test") or rec.get("test_code")
            repo_root = rec.get("repo_root")
        yield Example(
            id=str(rec.get("task_id") or rec.get("id") or i),
            query=query,
            repo_root=repo_root,
            reference_code=reference,
            test_code=test_code,
            raw=raw,
        )


def load_sample(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    """Load the bundled small function-completion JSONL in ``input/input.jsonl``."""
    root = _repo_root_for_jsonl(path)
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        meta = rec.get("metadata") or {}
        prompt = str(rec.get("prompt") or "")
        current_file = str(rec.get("current_file") or "")
        query = (
            "Complete the Python function from this repository. "
            "Return only the completed function code.\n\n"
            "# Function Stub\n"
            f"```python\n{prompt}\n```\n\n"
            "# Current File Context\n"
            f"```python\n{_clip(current_file, head=0, tail=5000)}\n```"
        )
        yield Example(
            id=str(meta.get("task_id") or rec.get("id") or i),
            query=query,
            repo_root=root,
            reference_code=meta.get("ground_truth"),
            test_code=rec.get("test") or rec.get("test_code"),
            raw=rec,
        )


def load_codereval(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        meta = rec.get("metadata") or {}
        prompt = str(rec.get("docstring") or rec.get("nl") or rec.get("prompt") or "")
        query = prompt
        if rec.get("current_file"):
            query = (
                "Complete the Python function from this repository. "
                "Return the complete target function definition, including the "
                "original signature and docstring from the stub.\n\n"
                "# Function Stub\n"
                f"```python\n{query}\n```\n\n"
                "# Current File Context\n"
                f"```python\n{_clip(str(rec.get('current_file')), head=0, tail=5000)}\n```"
            )
        raw = dict(rec)
        if prompt:
            raw.setdefault("target_function_prompt", prompt)
        yield Example(
            id=str(meta.get("task_id") or rec.get("_id") or rec.get("id") or i),
            query=query,
            repo_root=rec.get("project_path"),
            reference_code=rec.get("code") or meta.get("ground_truth"),
            test_code=rec.get("tests"),
            raw=raw,
        )


def load_repoexec(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        query = str(
            rec.get("instruction")
            or rec.get("target_function_prompt")
            or rec.get("prompt")
            or ""
        )
        current_file = rec.get("current_file") or rec.get("execution_prefix_code")
        if current_file:
            query = (
                "Complete the Python function from this repository. "
                "Return only the completed function code.\n\n"
                "# Function Stub\n"
                f"```python\n{query}\n```\n\n"
                "# Current File Context\n"
                f"```python\n{_clip(str(current_file), head=0, tail=5000)}\n```"
            )
        yield Example(
            id=str(rec.get("task_id") or i),
            query=query,
            repo_root=rec.get("repo_path"),
            reference_code=rec.get("solution"),
            test_code=rec.get("test"),
            raw=rec,
        )


_REGISTRY = {
    "sample": load_sample,
    "execrepobench": load_execrepobench,
    "codereval": load_codereval,
    "repoexec": load_repoexec,
}

_DEFAULT_PATHS = {
    "sample": "input/input.jsonl",
    "execrepobench": "input/execrepobench_data.jsonl",
}


def load_dataset(name: str, path: str | None = None, limit: Optional[int] = None) -> Iterator[Example]:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(_REGISTRY)}")
    path = path or _DEFAULT_PATHS.get(name)
    if not path:
        raise ValueError(f"No default path registered for dataset: {name}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return _REGISTRY[name](path, limit=limit)
