#!/usr/bin/env python3
"""Audit and adapt official ExecRepoBench blocks to function generation.

This script is deliberately pre-generation. It never reads model outputs and
does not call an LLM. Structurally valid records are written to a candidate
manifest; they are not considered frozen until the separate repository and
reference-test audit succeeds.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_DATASET = "CSJianYang/ExecRepoBench"
UPSTREAM_ROWS_SHA256 = "2fe757830b53ae61f0a06d18bda3cf9e9b7b092805e7be02943a6c81e14a7086"


@dataclass(frozen=True)
class FunctionUnit:
    name: str
    start: int
    end: int
    body_start: int
    start_line: int
    end_line: int
    prompt_stub: str
    reference: str


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _position_to_offset(source: str, starts: list[int], line: int, byte_column: int) -> int:
    """Convert CPython AST's UTF-8 byte column into a character offset."""
    line_start = starts[line - 1]
    line_end = source.find("\n", line_start)
    if line_end < 0:
        line_end = len(source)
    line_text = source[line_start:line_end]
    prefix = line_text.encode("utf-8")[:byte_column].decode("utf-8")
    return line_start + len(prefix)


def _node_offsets(source: str, node: ast.AST, starts: list[int]) -> tuple[int, int]:
    start = _position_to_offset(source, starts, int(node.lineno), int(node.col_offset))
    end = _position_to_offset(source, starts, int(node.end_lineno), int(node.end_col_offset))
    return start, end


def _decorated_start(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef, starts: list[int]) -> int:
    first_line = node.lineno
    if node.decorator_list:
        first_line = min(item.lineno for item in node.decorator_list)
    # Include source indentation and the decorator's ``@`` so the extracted
    # function remains parseable after textwrap.dedent.
    return starts[first_line - 1]


def _function_stub(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    start: int,
    body_start: int,
    starts: list[int],
) -> str:
    body_line_start = starts[node.body[0].lineno - 1]
    header = source[start:body_line_start]
    indent = " " * (node.col_offset + 4)
    body = ""
    if node.body and isinstance(node.body[0], ast.Expr):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            _, doc_end = _node_offsets(source, node.body[0], starts)
            doc_line_start = starts[node.body[0].lineno - 1]
            body = source[doc_line_start:doc_end].rstrip() + "\n"
    return header + body + indent + "pass\n"


def locate_function_unit(source: str, mask_start: int, mask_end: int) -> tuple[FunctionUnit | None, str]:
    """Return the smallest function enclosing the original masked span."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, "target_file_syntax_error"
    starts = _line_starts(source)
    candidates: list[tuple[int, ast.FunctionDef | ast.AsyncFunctionDef, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node_start, node_end = _node_offsets(source, node, starts)
        if node_start <= mask_start and mask_end <= node_end:
            candidates.append((node_end - node_start, node, node_start, node_end))
    if not candidates:
        return None, "mask_not_inside_function"
    _, node, node_start, node_end = min(candidates, key=lambda item: item[0])
    if not node.body:
        return None, "function_has_no_body"
    body_start = _position_to_offset(source, starts, node.body[0].lineno, node.body[0].col_offset)
    if mask_start < body_start:
        return None, "mask_overlaps_signature_or_decorator"
    decorated_start = _decorated_start(source, node, starts)
    reference = source[decorated_start:node_end]
    stub = _function_stub(source, node, decorated_start, body_start, starts)
    return (
        FunctionUnit(
            name=node.name,
            start=decorated_start,
            end=node_end,
            body_start=body_start,
            start_line=source.count("\n", 0, decorated_start) + 1,
            end_line=int(node.end_lineno),
            prompt_stub=stub,
            reference=reference,
        ),
        "structurally_adaptable",
    )


def _is_test_path(raw_path: str) -> bool:
    path = raw_path.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{path.strip('/')}/"


def _context_path(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, dict):
        return str(item.get("file_path") or item.get("path") or "")
    return ""


def _context_text(item: Any) -> str:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[1])
    if isinstance(item, dict):
        return str(item.get("code") or item.get("text") or "")
    return ""


def _context_has_tests(context_code: Any) -> bool:
    if not isinstance(context_code, list):
        return False
    for item in context_code:
        if _is_test_path(_context_path(item)):
            return True
    return False


def _filter_test_context(context_code: Any) -> tuple[list[Any], int]:
    if not isinstance(context_code, list):
        return [], 0
    retained = [item for item in context_code if not _is_test_path(_context_path(item))]
    return retained, len(context_code) - len(retained)


def _context_contains_reference(context_code: list[Any], reference: str) -> bool:
    try:
        reference_tree = ast.parse(textwrap.dedent(reference))
        target_nodes = [
            ast.dump(node, annotate_fields=False, include_attributes=False)
            for node in reference_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except SyntaxError:
        target_nodes = []
    for item in context_code:
        text = _context_text(item)
        if reference.strip() and reference.strip() in text:
            return True
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        document_nodes = {
            ast.dump(node, annotate_fields=False, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if any(target in document_nodes for target in target_nodes):
            return True
    return False


def _record_hash(row_index: int, record: dict[str, Any]) -> str:
    identity = {
        "dataset_sha256": UPSTREAM_ROWS_SHA256,
        "row_index": row_index,
        "repo_name": record.get("repo_name"),
        "file_name": record.get("file_name"),
        "prefix_code": record.get("prefix_code"),
        "middle_code": record.get("middle_code"),
        "suffix_code": record.get("suffix_code"),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adapt_record(row_index: int, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prefix = record.get("prefix_code")
    middle = record.get("middle_code")
    suffix = record.get("suffix_code")
    digest = _record_hash(row_index, record)
    base = {
        "upstream_row_index": row_index,
        "task_id": f"execrepobench-{row_index:04d}-{digest[:10]}",
        "repo_name": str(record.get("repo_name") or ""),
        "file_name": str(record.get("file_name") or ""),
        "fill_type": str(record.get("fill_type") or ""),
        "artifact_hash": digest,
        "source_dataset": UPSTREAM_DATASET,
        "source_dataset_sha256": UPSTREAM_ROWS_SHA256,
        "has_embedded_tests": _context_has_tests(record.get("context_code")),
        "repository_available": False,
        "dependency_complete": False,
        "reference_tests_pass": False,
        "selected": False,
    }
    if str(record.get("fill_type") or "").lower() != "grammar-based: block":
        return {**base, "structural_status": "excluded", "exclusion_reason": "not_grammar_block"}, None
    if not all(isinstance(value, str) for value in (prefix, middle, suffix)):
        return {**base, "structural_status": "excluded", "exclusion_reason": "missing_code_fields"}, None
    source = prefix + middle + suffix
    unit, reason = locate_function_unit(source, len(prefix), len(prefix) + len(middle))
    if unit is None:
        return {**base, "structural_status": "excluded", "exclusion_reason": reason}, None

    current_file = source[: unit.start] + unit.prompt_stub + source[unit.end :]
    filtered_context, removed_tests = _filter_test_context(record.get("context_code"))
    # The target reference must not survive in prompt-visible fields.
    leakage = (
        unit.reference.strip() in current_file
        or unit.reference.strip() in unit.prompt_stub
        or _context_contains_reference(filtered_context, unit.reference)
    )
    audit = {
        **base,
        "structural_status": "adaptable",
        "exclusion_reason": "pending_repository_and_reference_test_audit",
        "function_name": unit.name,
        "function_start_line": unit.start_line,
        "function_end_line": unit.end_line,
        "reference_leakage": leakage,
        "context_test_files_removed": removed_tests,
    }
    if leakage:
        audit.update(structural_status="excluded", exclusion_reason="reference_leakage_in_prompt")
        return audit, None

    candidate = {
        **base,
        "function_name": unit.name,
        "target_function_prompt": unit.prompt_stub,
        "instruction": unit.prompt_stub,
        "current_file": current_file,
        "execution_prefix_code": source[: unit.start],
        "execution_suffix_code": source[unit.end :],
        "solution": unit.reference,
        "context_code": filtered_context,
        "official_middle_code": middle,
        "official_prefix_code": prefix,
        "official_suffix_code": suffix,
        "official_input": record.get("input"),
    }
    return audit, candidate


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".benchmarks/execrepobench/exec_repo_bench_full.jsonl")
    parser.add_argument("--audit-out", default="results/data_quality/execrepobench_120_candidate_audit.csv")
    parser.add_argument("--candidates-out", default="data/manifests/execrepobench_opencoderx_candidates_v1.jsonl")
    args = parser.parse_args()

    audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row_index, record in enumerate(_read_jsonl(Path(args.input))):
        audit, candidate = adapt_record(row_index, record)
        audits.append(audit)
        if candidate is not None:
            candidates.append(candidate)

    audit_path = Path(args.audit_out)
    candidate_path = Path(args.candidates_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in audits for key in row})
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audits)
    with candidate_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for row in audits:
        key = str(row["exclusion_reason"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"rows": len(audits), "candidates": len(candidates), "outcomes": counts}, indent=2))
    print(f"Wrote {audit_path}")
    print(f"Wrote {candidate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
