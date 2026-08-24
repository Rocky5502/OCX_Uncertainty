#!/usr/bin/env python3
"""Build leakage-safe retrieval indexes and freeze ExecRepoBench-120."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SKIP_DIRECTORIES = {".git", ".tox", ".venv", "__pycache__", "build", "dist", "htmlcov"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    parts = {part for part in normalized.strip("/").split("/") if part}
    return name.startswith("test_") or name.endswith("_test.py") or bool(parts & {"test", "tests"})


def _function_dump(source: str) -> str | None:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ast.dump(node, annotate_fields=False, include_attributes=False)
    return None


def _function_signature(source: str) -> str:
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    node = next(
        item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    result = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result + ":"


def _call_symbols(source: str) -> list[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    def name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    return sorted({symbol for call in ast.walk(tree) if isinstance(call, ast.Call) if (symbol := name(call.func))})


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.stack: list[str] = []
        self.items: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join([*self.stack, node.name])
        segment = ast.get_source_segment(self.source, node)
        if segment:
            self.items.append((qualified, node, segment))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _source_documents(
    repo_name: str,
    repo_path: Path,
    target_dumps: set[str],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for root, directories, files in os.walk(repo_path):
        directories[:] = sorted(directory for directory in directories if directory not in SKIP_DIRECTORIES)
        for filename in sorted(files):
            path = Path(root) / filename
            relative = path.relative_to(repo_path).as_posix()
            if path.suffix != ".py" or _is_test_path(relative):
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            collector = _FunctionCollector(source)
            collector.visit(tree)
            for qualified, node, segment in collector.items:
                dump = ast.dump(node, annotate_fields=False, include_attributes=False)
                contained_function_dumps = {
                    ast.dump(item, annotate_fields=False, include_attributes=False)
                    for item in ast.walk(node)
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if contained_function_dumps & target_dumps:
                    continue
                identity = f"{repo_name}\0{relative}\0{qualified}\0{node.lineno}\0{dump}".encode("utf-8")
                base_id = _sha256_bytes(identity)[:20]
                documents.append({
                    "document_id": f"similar-{base_id}",
                    "repository": repo_name,
                    "path": relative,
                    "qualified_name": qualified,
                    "line": node.lineno,
                    "source_type": "similar_code",
                    "content": segment,
                })
                docstring = ast.get_docstring(node, clean=False) or ""
                api_content = _function_signature(segment)
                if docstring:
                    api_content += f"\n{docstring}"
                documents.append({
                    "document_id": f"api-{base_id}",
                    "repository": repo_name,
                    "path": relative,
                    "qualified_name": qualified,
                    "line": node.lineno,
                    "source_type": "api",
                    "content": api_content,
                })
    return documents


def _context_documents(
    selected: Iterable[dict[str, Any]],
    target_paths: dict[str, set[str]],
    target_dumps: dict[str, set[str]],
) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for task in selected:
        repo = str(task["repo_name"])
        for item in task.get("context_code") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_path, content = str(item[0]), str(item[1])
            elif isinstance(item, dict):
                raw_path = str(item.get("path") or item.get("file_path") or "")
                content = str(item.get("content") or item.get("code") or item.get("text") or "")
            else:
                continue
            relative = raw_path.replace("\\", "/").lstrip("/")
            if relative.startswith(f"{repo}/"):
                relative = relative[len(repo) + 1 :]
            if _is_test_path(relative) or relative in target_paths[repo]:
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                tree = None
            if tree is not None:
                dumps = {
                    ast.dump(node, annotate_fields=False, include_attributes=False)
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if dumps & target_dumps[repo]:
                    continue
            identity = f"{repo}\0{relative}\0{content}".encode("utf-8")
            document_id = f"context-{_sha256_bytes(identity)[:20]}"
            documents[document_id] = {
                "document_id": document_id,
                "repository": repo,
                "path": relative,
                "source_type": "context",
                "content": content,
            }
    return list(documents.values())


def _document_contains_target(document: dict[str, Any], target_dump: str) -> bool:
    try:
        tree = ast.parse(str(document.get("content") or ""))
    except SyntaxError:
        return False
    return any(
        ast.dump(node, annotate_fields=False, include_attributes=False) == target_dump
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/execrepobench_opencoderx_120_v1.jsonl")
    parser.add_argument("--repos-root", default=".benchmarks/execrepobench/repos")
    parser.add_argument("--index-out", default="data/indexes/execrepobench_120_repository_knowledge_v1.jsonl")
    parser.add_argument("--leakage-out", default="results/data_quality/retrieval_leakage_report.csv")
    parser.add_argument("--freeze-report", default="results/data_quality/execrepobench_120_freeze.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    selected = _read_jsonl(manifest_path)
    if len(selected) != 120 or any(not row.get("reference_tests_pass") for row in selected):
        raise SystemExit("manifest must contain exactly 120 reference-validated tasks")

    target_dumps: dict[str, set[str]] = defaultdict(set)
    target_paths: dict[str, set[str]] = defaultdict(set)
    task_target_dump: dict[str, str] = {}
    for task in selected:
        repo = str(task["repo_name"])
        dump = _function_dump(str(task["solution"]))
        if dump is None:
            raise SystemExit(f"cannot parse selected target: {task['task_id']}")
        target_dumps[repo].add(dump)
        task_target_dump[str(task["task_id"])] = dump
        relative = str(task["file_name"]).replace("\\", "/").lstrip("/")
        if relative.startswith(f"{repo}/"):
            relative = relative[len(repo) + 1 :]
        target_paths[repo].add(relative)

    repos_root = Path(args.repos_root).resolve()
    documents: list[dict[str, Any]] = []
    for repo in sorted(target_dumps):
        documents.extend(_source_documents(repo, repos_root / repo, target_dumps[repo]))
    documents.extend(_context_documents(selected, target_paths, target_dumps))
    unique_documents = {str(document["document_id"]): document for document in documents}
    documents = [unique_documents[key] for key in sorted(unique_documents)]

    by_repo_type: dict[tuple[str, str], list[str]] = defaultdict(list)
    for document in documents:
        by_repo_type[(str(document["repository"]), str(document["source_type"]))].append(
            str(document["document_id"])
        )

    leakage_rows = []
    for task in selected:
        repo = str(task["repo_name"])
        dump = task_target_dump[str(task["task_id"])]
        repo_documents = [document for document in documents if document["repository"] == repo]
        leaking = [document["document_id"] for document in repo_documents if _document_contains_target(document, dump)]
        leakage_rows.append({
            "task_id": task["task_id"],
            "repository": repo,
            "checked_documents": len(repo_documents),
            "leaking_documents": len(leaking),
            "leaking_document_ids": ";".join(map(str, leaking)),
            "status": "pass" if not leaking else "fail",
        })
    if any(row["status"] == "fail" for row in leakage_rows):
        raise SystemExit("retrieval leakage detected; frozen manifest was not modified")

    enriched = []
    for task in sorted(selected, key=lambda row: str(row["artifact_hash"])):
        repo = str(task["repo_name"])
        context_ids = by_repo_type[(repo, "context")]
        similar_ids = by_repo_type[(repo, "similar_code")]
        api_ids = by_repo_type[(repo, "api")]
        row = {key: value for key, value in task.items() if key != "official_input"}
        row.update({
            "manifest_version": "execrepobench_opencoderx_120_v1",
            "language": "Python",
            "commit": task.get("upstream_commit"),
            "target_function": task.get("function_name"),
            "signature": _function_signature(str(task["solution"])),
            "prompt": task.get("instruction"),
            "repository_context": {
                "index": "execrepobench_120_repository_knowledge_v1",
                "source_type": "context",
                "document_ids": context_ids,
                "document_count": len(context_ids),
            },
            "applicable_api_information": {
                "source": "reference-derived evaluation label",
                "prompt_visible": False,
                "invoked_symbols": _call_symbols(str(task["solution"])),
            },
            "api_candidate_pool": {
                "index": "execrepobench_120_repository_knowledge_v1",
                "source_type": "api",
                "document_count": len(api_ids),
            },
            "similar_code_candidate_pool": {
                "index": "execrepobench_120_repository_knowledge_v1",
                "source_type": "similar_code",
                "document_count": len(similar_ids),
            },
            "expected_reference_test_status": "pass",
            "inclusion_rationale": (
                "Grammar-block mask maps uniquely to a complete function; the archived source "
                "matches the reference; official repository tests pass; retrieval leakage audit passes."
            ),
            "adaptation_metadata": {
                "upstream_row_index": task.get("upstream_row_index"),
                "original_fill_type": task.get("fill_type"),
                "original_mask": task.get("official_middle_code"),
                "selection_order": "ascending pre-output artifact_hash",
                "maximum_tasks_per_repository": 4,
            },
            "provenance": {
                "dataset": task.get("source_dataset"),
                "dataset_sha256": task.get("source_dataset_sha256"),
                "repository_archive_sha256": task.get("repository_archive_sha256"),
                "artifact_hash": task.get("artifact_hash"),
            },
        })
        enriched.append(row)

    index_path = Path(args.index_out)
    leakage_path = Path(args.leakage_out)
    report_path = Path(args.freeze_report)
    for path in (index_path, leakage_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".freeze")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in enriched:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    with leakage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leakage_rows[0]))
        writer.writeheader()
        writer.writerows(leakage_rows)
    temporary.replace(manifest_path)
    report = {
        "status": "FROZEN",
        "tasks": len(enriched),
        "repositories": len({row["repo_name"] for row in enriched}),
        "index_documents": len(documents),
        "context_documents": sum(document["source_type"] == "context" for document in documents),
        "api_documents": sum(document["source_type"] == "api" for document in documents),
        "similar_code_documents": sum(document["source_type"] == "similar_code" for document in documents),
        "leakage_failures": 0,
        "manifest_sha256": _sha256_file(manifest_path),
        "index_sha256": _sha256_file(index_path),
        "leakage_report_sha256": _sha256_file(leakage_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
