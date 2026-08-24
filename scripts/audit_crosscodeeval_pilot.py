#!/usr/bin/env python3
"""Audit and summarize the frozen multilingual CrossCodeEval pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/manifests/crosscodeeval_opencoderx_pilot8_v1.jsonl"
DEFAULT_RESULTS = ROOT / "results/tosem/crosscodeeval_pilot"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = {
    "direct": "Direct Generation",
    "context_rag": "Cross-file Context RAG",
}
METRICS = ("exact_match", "edit_similarity", "identifier_f1")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pricing() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(
        (ROOT / "configs/tosem/campaign.yaml").read_text(encoding="utf-8")
    )
    return data["cost_controls"]["gateway_pricing"]


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "requests",
        "retries",
        "failed_attempts",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    return {
        field: sum(int((row.get("llm_usage") or {}).get(field) or 0) for row in rows)
        for field in fields
    }


def _cost(model: str, usage: dict[str, int]) -> tuple[float, str]:
    pricing = _pricing()[model]
    amount = (
        usage["prompt_tokens"] * float(pricing["input_per_million"])
        + usage["completion_tokens"] * float(pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(pricing["currency"])


def _numeric_metrics(record: dict[str, Any]) -> bool:
    return all(isinstance(record.get(metric), (int, float)) for metric in METRICS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    manifest = _read_jsonl(args.manifest)
    expected_ids = [str(row["task_id"]) for row in manifest]
    expected_set = set(expected_ids)
    manifest_by_id = {str(row["task_id"]): row for row in manifest}
    issues: list[str] = []
    summaries: list[dict[str, Any]] = []
    language_summaries: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    result_files: dict[str, Any] = {}
    response_ids: set[str] = set()

    for model, directory in MODELS.items():
        path = args.results_dir / directory / "results.json"
        if not path.is_file():
            issues.append(f"missing result: {path.relative_to(ROOT)}")
            continue
        data = _read(path)
        metadata = data.get("metadata") or {}
        rows = list(data.get("rows") or [])
        result_files[model] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "created_at": metadata.get("created_at"),
        }
        if metadata.get("model") != model:
            issues.append(f"{model}: metadata model mismatch")
        if int(metadata.get("candidate_count") or 0) != 5:
            issues.append(f"{model}: candidate count is not five")
        if metadata.get("functional_execution") is not False:
            issues.append(f"{model}: functional execution must be false")
        if metadata.get("paper_eligible") is not False:
            issues.append(f"{model}: pilot is incorrectly marked paper-eligible")
        if set(metadata.get("native_metrics") or []) != set(METRICS):
            issues.append(f"{model}: native metric declaration mismatch")

        expected_cells = {(task_id, method) for task_id in expected_ids for method in METHODS}
        actual_cells = {(str(row.get("task_id")), str(row.get("method"))) for row in rows}
        if len(actual_cells) != len(rows):
            issues.append(f"{model}: duplicate task-method cells")
        if actual_cells != expected_cells:
            issues.append(f"{model}: task-method cells differ from frozen pilot")

        by_method: dict[str, list[dict[str, Any]]] = {}
        for method, label in METHODS.items():
            method_rows = [row for row in rows if row.get("method") == method]
            by_method[method] = method_rows
            usage = _usage(method_rows)
            amount, currency = _cost(model, usage)
            resources.append({
                "model": model,
                "method": label,
                **usage,
                "estimated_cost": amount,
                "currency": currency,
                "mean_latency_seconds": _mean(
                    float(row.get("latency_seconds") or 0.0) for row in method_rows
                ),
            })
            candidate_metrics = [
                metric
                for row in method_rows
                for metric in (row.get("candidate_metrics") or [])
            ]
            summaries.append({
                "model": model,
                "method": label,
                "tasks": len(method_rows),
                "selected_exact_match": _mean(
                    float((row.get("selected_metrics") or {}).get("exact_match") or 0.0)
                    for row in method_rows
                ),
                "selected_edit_similarity": _mean(
                    float((row.get("selected_metrics") or {}).get("edit_similarity") or 0.0)
                    for row in method_rows
                ),
                "selected_identifier_f1": _mean(
                    float((row.get("selected_metrics") or {}).get("identifier_f1") or 0.0)
                    for row in method_rows
                ),
                "candidate_exact_match": _mean(
                    float(metric.get("exact_match") or 0.0) for metric in candidate_metrics
                ),
                "candidate_edit_similarity": _mean(
                    float(metric.get("edit_similarity") or 0.0) for metric in candidate_metrics
                ),
                "candidate_identifier_f1": _mean(
                    float(metric.get("identifier_f1") or 0.0) for metric in candidate_metrics
                ),
                "mean_aggregate_risk": _mean(
                    float((row.get("uncertainty") or {}).get("aggregate_risk") or 0.0)
                    for row in method_rows
                ),
                "paper_eligible": False,
            })
            for language in sorted({str(row.get("language")) for row in method_rows}):
                subset = [row for row in method_rows if row.get("language") == language]
                language_summaries.append({
                    "model": model,
                    "method": label,
                    "language": language,
                    "tasks": len(subset),
                    "selected_exact_match": _mean(
                        float((row.get("selected_metrics") or {}).get("exact_match") or 0.0)
                        for row in subset
                    ),
                    "selected_edit_similarity": _mean(
                        float((row.get("selected_metrics") or {}).get("edit_similarity") or 0.0)
                        for row in subset
                    ),
                    "selected_identifier_f1": _mean(
                        float((row.get("selected_metrics") or {}).get("identifier_f1") or 0.0)
                        for row in subset
                    ),
                })

        for row in rows:
            task_id = str(row.get("task_id"))
            method = str(row.get("method"))
            prefix = f"{model}/{method}/{task_id}"
            if row.get("status") != "COMPLETED_NATIVE_METRICS" or row.get("error"):
                issues.append(f"{prefix}: incomplete or failed cell")
            if row.get("functional_correctness") is not None:
                issues.append(f"{prefix}: functional correctness was assigned")
            candidates = list(row.get("candidates") or [])
            metrics = list(row.get("candidate_metrics") or [])
            if len(candidates) != 5 or len(metrics) != 5:
                issues.append(f"{prefix}: expected five candidates and metric records")
            if not all(_numeric_metrics(metric) for metric in metrics):
                issues.append(f"{prefix}: missing native candidate metrics")
            if not _numeric_metrics(row.get("selected_metrics") or {}):
                issues.append(f"{prefix}: missing native selected metrics")
            if (row.get("generation_integrity") or {}).get("valid") is not True:
                issues.append(f"{prefix}: generation integrity failed")
            if int(row.get("selected_candidate_index", -1)) != 0:
                issues.append(f"{prefix}: pilot selection must use the first candidate")
            evidence = list(row.get("retrieval_evidence") or [])
            if method == "direct" and evidence:
                issues.append(f"{prefix}: Direct contains retrieved evidence")
            if method == "context_rag" and not evidence:
                issues.append(f"{prefix}: Context RAG has no frozen evidence")
            source = manifest_by_id.get(task_id) or {}
            future = str(source.get("right_context") or "")
            if future and future in str(row.get("selected_output") or ""):
                # This is diagnostic only; a generated continuation can legitimately equal
                # the target. Prompt construction is tested separately for suffix exclusion.
                pass
            audit = list(row.get("provider_response_audit") or [])
            requests = int((row.get("llm_usage") or {}).get("requests") or 0)
            if len(audit) != requests:
                issues.append(f"{prefix}: all-call audit count differs from request count")
            for item in audit:
                response_id = str(item.get("response_id") or "")
                if not response_id:
                    issues.append(f"{prefix}: missing provider response ID")
                elif response_id in response_ids:
                    issues.append(f"{prefix}: duplicate provider response ID {response_id}")
                else:
                    response_ids.add(response_id)
                if item.get("requested_model") != model or item.get("served_model") != model:
                    issues.append(f"{prefix}: requested/served model mismatch")

    output_dir = args.results_dir
    if summaries:
        _write_csv(output_dir / "summary.csv", summaries)
        _write_csv(output_dir / "language_summary.csv", language_summaries)
        _write_csv(output_dir / "resource_summary.csv", resources)
    integrity = {
        "valid": not issues,
        "paper_eligible": False,
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": _sha256(args.manifest),
        "expected_tasks": len(expected_ids),
        "expected_models": len(MODELS),
        "expected_methods": len(METHODS),
        "expected_cells": len(expected_ids) * len(MODELS) * len(METHODS),
        "unique_provider_responses": len(response_ids),
        "result_files": result_files,
        "issues": issues,
    }
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    memo = [
        "# CrossCodeEval Multilingual Pilot",
        "",
        "This artifact is a protocol and feasibility pilot, not a confirmatory paper result.",
        "CrossCodeEval is evaluated with its native exact-match, edit-similarity, and identifier metrics.",
        "These scores must not be described as executable functional correctness.",
        "",
        f"- Frozen tasks: {len(expected_ids)} (two per language)",
        f"- Models expected: {len(MODELS)}",
        f"- Methods: {', '.join(METHODS.values())}",
        f"- Integrity: {'PASS' if not issues else 'FAIL'}",
        f"- Unique audited provider responses: {len(response_ids)}",
        "",
    ]
    if issues:
        memo.extend(["## Integrity Issues", "", *[f"- {item}" for item in issues]])
    else:
        memo.extend([
            "## Interpretation Boundary",
            "",
            "The pilot validates multilingual prompt construction, frozen cross-file evidence, provider provenance, and native metric computation. It does not evaluate executable verification/repair or establish method superiority.",
        ])
    (output_dir / "results_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
