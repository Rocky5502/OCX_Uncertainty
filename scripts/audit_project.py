"""Preflight audit for OpenCoder code, configuration, and active datasets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.pipeline import PipelineConfig  # noqa: E402


DATASETS = {
    "sample": ROOT / "input" / "input.jsonl",
    "execrepobench": ROOT / "input" / "execrepobench_data.jsonl",
}


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in f"/{normalized.strip('/')}/"
    )


def _has_repository_tests(raw: dict) -> bool:
    for entry in raw.get("context_code") or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and _is_test_path(str(entry[0])):
            return True
        if isinstance(entry, dict) and _is_test_path(str(entry.get("file_path") or entry.get("path") or "")):
            return True
    return False


def audit_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: expected JSON object")
                continue
            rows.append(row)
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-tests", action="store_true",
                        help="Fail when an active dataset has no executable tests.")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    for name, path in DATASETS.items():
        if not path.exists():
            failures.append(f"Missing dataset: {path}")
            continue
        raw_rows, errors = audit_jsonl(path)
        failures.extend(errors)
        examples = list(load_dataset(name, str(path)))
        ids = [example.id for example in examples]
        duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
        if duplicate_ids:
            failures.append(f"{name}: duplicate IDs: {duplicate_ids[:5]}")
        if len(raw_rows) != len(examples):
            failures.append(
                f"{name}: raw/loaded row mismatch ({len(raw_rows)} != {len(examples)})"
            )
        empty_queries = sum(not example.query.strip() for example in examples)
        if empty_queries:
            failures.append(f"{name}: {empty_queries} empty normalized queries")
        inline_tests = sum(bool(example.test_code) for example in examples)
        repo_tests = sum(_has_repository_tests(example.raw or {}) for example in examples)
        tests = inline_tests + repo_tests
        references = sum(example.reference_code is not None for example in examples)
        if tests == 0:
            msg = (
                f"{name}: no executable test fields; correctness falls back to "
                "reference exact match"
            )
            (failures if args.strict_tests else warnings).append(msg)
        print(
            f"{name}: rows={len(examples)} unique_ids={len(set(ids))} "
            f"references={references} executable_tests={tests} "
            f"(inline={inline_tests}, repository={repo_tests})"
        )

    config_paths = [ROOT / "configs" / "default.yaml", ROOT / "opencoder" / "configs" / "default.yaml"]
    configs = [PipelineConfig.from_yaml(str(path)) for path in config_paths]
    if configs[0] != configs[1]:
        failures.append("Default config files do not normalize to the same PipelineConfig")
    print(f"config: backend={configs[0].llm_backend} model={configs[0].llm_model} "
          f"samples={configs[0].n_samples_for_uncertainty}")

    stale_scripts = ROOT / "opencoder" / "scripts"
    if stale_scripts.exists() and any(stale_scripts.glob("*.py")):
        failures.append("Stale duplicate scripts remain under opencoder/scripts/")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")

    if failures:
        print(f"AUDIT FAILED: {len(failures)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"AUDIT PASSED: {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
