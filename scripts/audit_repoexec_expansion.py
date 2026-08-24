"""Audit and freeze additional RepoExec-inline tasks before model execution."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset as load_hf_dataset  # noqa: E402

from opencoder.phase5_verify.static_checks import static_check  # noqa: E402
from opencoder.phase5_verify.test_validate import run_tests  # noqa: E402


AUDIT_COLUMNS = (
    "task_id",
    "repository",
    "included",
    "exclusion_reason",
    "reference_tests_pass",
    "dependency_complete",
    "artifact_hash",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_prefix(prefix: str) -> str:
    text = re.sub(
        r"from \.([A-Za-z_][A-Za-z0-9_]*) import",
        r"from string_utils.\1 import",
        str(prefix),
    )
    if "from typing import Any" not in text:
        text = text.replace(
            "from typing import Union",
            "from typing import Any, Union",
        )
    return text


def _dependency_complete_execution_prefix(
    source: str,
    target_name: str,
) -> str:
    """Keep local dependencies while removing the target implementation."""
    tree = ast.parse(source)
    stripped = False

    class TargetStripper(ast.NodeTransformer):
        def visit_FunctionDef(self, node):  # noqa: N802
            nonlocal stripped
            if node.name == target_name:
                node.body = [ast.copy_location(ast.Pass(), node)]
                stripped = True
                return node
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            nonlocal stripped
            if node.name == target_name:
                node.body = [ast.copy_location(ast.Pass(), node)]
                stripped = True
                return node
            return self.generic_visit(node)

    sanitized = TargetStripper().visit(tree)
    if not stripped:
        raise ValueError(f"target {target_name!r} not found in local module")
    ast.fix_missing_locations(sanitized)
    return _normalize_prefix(ast.unparse(sanitized)) + "\n"


def _self_contained_tests(tests: Iterable[str]) -> List[str]:
    return [
        str(test).rstrip()
        for test in tests
        if str(test).strip()
        and "/output/" not in str(test)
        and "open(" not in str(test)
    ]


def _call_tests(tests: List[str]) -> str:
    blocks: List[str] = ["import pytest"]
    for index, test in enumerate(tests):
        blocks.append(test)
        first = test.strip().splitlines()[0] if test.strip() else ""
        name = None
        if first.startswith("def ") and "(" in first:
            name = first.split("def ", 1)[1].split("(", 1)[0].strip()
        blocks.append(f"{name}()" if name else f"# no auto-call for block {index}")
    return "\n\n".join(blocks) + "\n"


def _artifact_hash(record: Dict[str, Any]) -> str:
    frozen = {
        key: record.get(key)
        for key in (
            "task_id",
            "project",
            "module",
            "entry_point",
            "instruction",
            "current_file",
            "execution_prefix_code",
            "solution",
            "test",
            "n_inline_tests",
            "n_original_tests",
            "repoexec_split",
        )
    }
    payload = json.dumps(
        frozen,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _prompt_isolated(record: Dict[str, Any], tests: List[str]) -> bool:
    prompt = "\n".join([
        str(record.get("instruction") or ""),
        str(record.get("current_file") or ""),
    ])
    solution = str(record.get("solution") or "").strip()
    if solution and solution in prompt:
        return False
    return not any(test.strip() and test.strip() in prompt for test in tests)


def _record_from_source(
    item: Dict[str, Any],
    tests: List[str],
    *,
    split: str,
    execution_prefix: str | None = None,
) -> Dict[str, Any]:
    prompt_prefix = _normalize_prefix(item.get("prompt") or "")
    return {
        "task_id": f"{item.get('project')}/{item.get('id')}",
        "project": item.get("project"),
        "module": item.get("module"),
        "entry_point": item.get("entry_point"),
        "instruction": item.get("target_function_prompt"),
        "target_function_prompt": item.get("target_function_prompt"),
        "current_file": prompt_prefix,
        "execution_prefix_code": execution_prefix or prompt_prefix,
        "extra_pythonpath": "input",
        "solution": item.get("solution"),
        "test": _call_tests(tests),
        "n_inline_tests": len(tests),
        "n_original_tests": len(item.get("test_list") or []),
        "repoexec_split": split,
        "repoexec_project": item.get("project"),
    }


def _failure_category(stderr: str, stdout: str) -> str:
    text = f"{stderr}\n{stdout}".lower()
    if "timeout" in text:
        return "test_timeout"
    if "modulenotfounderror" in text or "importerror" in text:
        return "missing_dependency"
    if "filenotfounderror" in text or "/output/" in text:
        return "unavailable_artifact"
    return "reference_tests_failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="full_context")
    parser.add_argument("--project", default="test-apps/python-string-utils")
    parser.add_argument(
        "--original-manifest",
        default="input/repoexec_python_string_utils_inline14.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="results/rq3_expanded_repoexec",
    )
    parser.add_argument("--min-tests", type=int, default=5)
    parser.add_argument("--test-timeout", type=int, default=30)
    parser.add_argument("--repository-root", default="input")
    parser.add_argument("--gpt-config", default="configs/rq3/gpt4o_mini.yaml")
    parser.add_argument(
        "--gemini-config",
        default="configs/rq3/gemini_2_5_flash.yaml",
    )
    parser.add_argument(
        "--validated-gpt-run",
        default="results/rq3/runs/matched_external/gpt_repoexec_full/rq3.json",
    )
    parser.add_argument(
        "--validated-gemini-run",
        default="results/rq3/runs/matched_external/gemini_repoexec_full/rq3.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_ids = {
        str(row["task_id"])
        for row in _read_jsonl(Path(args.original_manifest))
    }

    source = load_hf_dataset(
        "Fsoft-AIC/RepoExec",
        split=args.split,
    )
    remaining = [
        item
        for item in source
        if item.get("project") == args.project
        and f"{item.get('project')}/{item.get('id')}" not in original_ids
    ]

    audit_rows: List[Dict[str, Any]] = []
    included_records: List[Dict[str, Any]] = []
    for item in remaining:
        tests = _self_contained_tests(item.get("test_list") or [])
        module_path = (
            Path(args.repository_root)
            / (str(item.get("module") or "").replace(".", "/") + ".py")
        )
        dependency_error = ""
        execution_prefix = None
        if not module_path.is_file():
            dependency_error = "local_repository_module_missing"
        else:
            try:
                execution_prefix = _dependency_complete_execution_prefix(
                    module_path.read_text(encoding="utf-8"),
                    str(item.get("entry_point") or ""),
                )
            except (SyntaxError, ValueError) as exc:
                dependency_error = (
                    "cannot_construct_target_stripped_execution_prefix:"
                    + type(exc).__name__
                )
        record = _record_from_source(
            item,
            tests,
            split=args.split,
            execution_prefix=execution_prefix,
        )
        task_id = str(record["task_id"])
        artifact_hash = _artifact_hash(record)
        reason = ""
        reference_pass = False
        dependency_complete = False

        if dependency_error:
            reason = dependency_error
        elif len(tests) < args.min_tests:
            reason = "fewer_than_five_self_contained_public_tests"
        elif not _prompt_isolated(record, tests):
            reason = "reference_or_test_content_in_generation_prompt"
        else:
            target = (
                f"{str(record['execution_prefix_code']).rstrip()}\n\n"
                f"{str(record['solution']).lstrip()}"
            )
            static_report = static_check(target)
            if not static_report.ok:
                reason = "reference_static_failure"
            else:
                report = run_tests(
                    target,
                    record["test"],
                    timeout=args.test_timeout,
                    extra_pythonpath="input",
                )
                reference_pass = report.passed is True
                dependency_complete = reference_pass
                if not reference_pass:
                    reason = _failure_category(report.stderr, report.stdout)

        included = (
            not reason
            and reference_pass
            and dependency_complete
        )
        if included:
            included_records.append(record)
        audit_rows.append({
            "task_id": task_id,
            "repository": str(record.get("project") or ""),
            "included": str(included).lower(),
            "exclusion_reason": reason,
            "reference_tests_pass": str(reference_pass).lower(),
            "dependency_complete": str(dependency_complete).lower(),
            "artifact_hash": artifact_hash,
        })
        print(
            f"{task_id}: {'INCLUDE' if included else 'EXCLUDE'}"
            + (f" ({reason})" if reason else ""),
            flush=True,
        )

    audit_path = out_dir / "task_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest_path = out_dir / "new_tasks_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in included_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    validated_metadata = {}
    for backend, run_path in (
        ("GPT", Path(args.validated_gpt_run)),
        ("Gemini", Path(args.validated_gemini_run)),
    ):
        validated_metadata[backend] = json.loads(
            run_path.read_text(encoding="utf-8")
        ).get("metadata") or {}
    protocol_fields = (
        "temperature",
        "n_samples_for_uncertainty",
        "max_repair_rounds",
        "retrieval_budget",
        "max_generation_tokens_per_candidate",
        "seed",
        "condition_feature_flags",
    )
    protocol = {
        backend: {
            key: metadata.get(key)
            for key in protocol_fields
        }
        for backend, metadata in validated_metadata.items()
    }
    if protocol["GPT"] != protocol["Gemini"]:
        differing = [
            key
            for key in protocol_fields
            if protocol["GPT"].get(key) != protocol["Gemini"].get(key)
        ]
        allowed_backend_specific = set()
        if set(differing) - allowed_backend_specific:
            raise ValueError(
                "validated backend protocols differ: "
                + ", ".join(differing)
            )

    freeze = {
        "source": "locally cached Fsoft-AIC/RepoExec",
        "split": args.split,
        "project": args.project,
        "original_manifest": Path(args.original_manifest).name,
        "selection_rule": {
            "minimum_self_contained_public_tests": args.min_tests,
            "reference_must_pass": True,
            "dependencies_must_be_complete": True,
            "prompt_must_exclude_reference_and_tests": True,
            "artifact_dependent_tests_excluded": True,
        },
        "remaining_tasks_audited": len(audit_rows),
        "new_tasks_included": len(included_records),
        "included_task_ids": [row["task_id"] for row in included_records],
        "new_manifest_hash": _file_sha256(manifest_path),
        "config_hashes": {
            "GPT": _file_sha256(Path(args.gpt_config)),
            "Gemini": _file_sha256(Path(args.gemini_config)),
        },
        "validated_protocol": protocol["GPT"],
        "models": {
            backend: metadata.get("model")
            for backend, metadata in validated_metadata.items()
        },
        "task_selection_completed_before_new_method_outputs": not any(
            (out_dir / "runs" / name / "rq3.json").exists()
            for name in ("gpt_new", "gemini_new")
        ),
        "artifact_hashes": {
            row["task_id"]: row["artifact_hash"]
            for row in audit_rows
        },
    }
    (out_dir / "protocol_freeze.json").write_text(
        json.dumps(freeze, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {audit_path} and froze {len(included_records)} new tasks.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
