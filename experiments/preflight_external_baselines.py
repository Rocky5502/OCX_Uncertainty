"""Offline manifest and repository-asset preflight for external baselines."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import Example, load_dataset  # noqa: E402
from opencoder.phase1_repo_knowledge.extract import (  # noqa: E402
    extract_repo_knowledge,
    extract_sources_knowledge,
)


DEFAULT_MANIFESTS = {
    "repoexec": "input/repoexec_python_string_utils_inline14.jsonl",
    "codereval": "input/codereval_neo4j_executable19.jsonl",
    "execrepobench": "input/execrepobench_testbacked.jsonl",
}


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{normalized.strip('/')}/"


def _record_sources(example: Example) -> list[tuple[str, str]]:
    raw = example.raw or {}
    sources: list[tuple[str, str]] = []
    current = raw.get("current_file")
    if isinstance(current, str) and current.strip():
        meta = raw.get("metadata") or {}
        path = "/".join(meta.get("fpath_tuple") or []) or "current_file.py"
        sources.append((path, current))
    for index, entry in enumerate(raw.get("context_code") or []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            path, text = str(entry[0]).lstrip("/"), str(entry[1])
        elif isinstance(entry, dict):
            path = str(entry.get("file_path") or entry.get("path") or f"context_{index}.py")
            text = str(entry.get("code") or entry.get("text") or "")
        else:
            path, text = f"context_{index}.py", str(entry)
        if text.strip() and not _is_test_path(path):
            sources.append((path, text))
    prefix, suffix = raw.get("prefix_code"), raw.get("suffix_code")
    if isinstance(prefix, str) or isinstance(suffix, str):
        path = str(raw.get("file_name") or "target.py").lstrip("/")
        prefix = prefix or ""
        trailing = prefix.rsplit("\n", 1)[-1]
        if trailing and not trailing.strip():
            indent = trailing
        else:
            previous = next((line for line in reversed(prefix.splitlines()) if line.strip()), "")
            indent = previous[: len(previous) - len(previous.lstrip(" "))]
            if previous.rstrip().endswith(":"):
                indent += "    "
        sources.append((path, f"{prefix}{indent}pass\n{suffix or ''}"))
    return sources


def _target_name(example: Example) -> str | None:
    raw = example.raw or {}
    meta = raw.get("metadata") or {}
    explicit = raw.get("entry_point") or meta.get("function_name") or raw.get("target_function_name")
    if explicit:
        return str(explicit)
    prefix = str(raw.get("prefix_code") or "")
    definitions = re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", prefix, re.MULTILINE)
    return definitions[-1] if definitions else None


def _explicit_repo_root(benchmark: str, example: Example) -> str | None:
    if example.repo_root and os.path.isdir(example.repo_root):
        return os.path.abspath(example.repo_root)
    if benchmark == "repoexec":
        candidate = ROOT / "input" / "string_utils"
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def inspect_manifest(benchmark: str, path: str) -> dict:
    examples = list(load_dataset(benchmark, path))
    repo_cache: dict[str, tuple[int, int]] = {}
    tasks = []
    for example in examples:
        repo_root = _explicit_repo_root(benchmark, example)
        sources = _record_sources(example)
        if repo_root:
            if repo_root not in repo_cache:
                items = extract_repo_knowledge(repo_root)
                repo_cache[repo_root] = (
                    len({item.file_path for item in items}),
                    len(items),
                )
            file_count, api_count = repo_cache[repo_root]
            asset_mode = "repository_snapshot"
        else:
            items = extract_sources_knowledge(sources)
            file_count, api_count = len({path for path, _ in sources}), len(items)
            asset_mode = "manifest_context"
        tasks.append(
            {
                "id": example.id,
                "target_name": _target_name(example),
                "asset_mode": asset_mode,
                "repo_root": repo_root,
                "source_files": file_count,
                "extractable_apis": api_count,
                "has_executable_protocol": bool(
                    example.test_code
                    or (example.raw or {}).get("codereval_project_tests")
                    or (example.raw or {}).get("context_code")
                ),
                "ready_for_alliancecoder_smoke": bool(api_count and _target_name(example)),
            }
        )
    return {
        "benchmark": benchmark,
        "manifest": path,
        "n_tasks": len(tasks),
        "asset_modes": dict(Counter(task["asset_mode"] for task in tasks)),
        "n_smoke_ready": sum(task["ready_for_alliancecoder_smoke"] for task in tasks),
        "min_extractable_apis": min((task["extractable_apis"] for task in tasks), default=0),
        "max_extractable_apis": max((task["extractable_apis"] for task in tasks), default=0),
        "tasks": tasks,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/external_baseline/preflight.json")
    args = parser.parse_args(argv)
    reports = [inspect_manifest(name, path) for name, path in DEFAULT_MANIFESTS.items()]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0,
        "purpose": "external baseline asset preflight; not an empirical result",
        "benchmarks": reports,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({report["benchmark"]: {k: report[k] for k in ("n_tasks", "asset_modes", "n_smoke_ready", "min_extractable_apis", "max_extractable_apis")} for report in reports}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
