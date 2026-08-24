#!/usr/bin/env python3
"""Build source-code-free benchmark indexes and redacted reviewer records.

The private campaign workspace may contain third-party repository source,
provider response identifiers, generated code, and local paths. This utility
retains the provenance and outcome fields needed to reproduce the released
statistics while excluding those payloads from the public artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXEC_FIELDS = (
    "task_id",
    "source_dataset",
    "upstream_row_index",
    "repo_name",
    "language",
    "upstream_commit",
    "artifact_hash",
    "source_dataset_sha256",
    "repository_archive_sha256",
    "validation_status",
    "dependency_complete",
    "reference_tests_pass",
)

CROSS_FIELDS = (
    "task_id",
    "language",
    "repository",
    "upstream_row_index",
    "native_metrics",
    "functional_execution",
    "selection_rule",
    "provenance",
)

AGENT_FIELDS = (
    "record_type",
    "study_mode",
    "protocol",
    "agent_id",
    "family",
    "requested_model",
    "served_model",
    "assignment_group",
    "episode_index",
    "task_id",
    "condition",
    "signal_category",
    "initial_correct",
    "starting_judgment_correct",
    "starting_confidence",
    "final_confidence",
    "failure_detection_accurate",
    "final_code_sha256",
    "code_changed",
    "final_correct",
    "repair_opportunity",
    "repair_success",
    "unnecessary_edit",
    "parse_mode",
    "parse_errors",
    "evaluator_status",
    "evaluator_returncode",
    "finish_reason",
    "raw_response_sha256",
    "prompt_sha256",
    "usage",
    "latency_seconds",
    "started_at_utc",
    "completed_at_utc",
    "error",
)

RQ2_CONDITIONS = (
    "without",
    "decomposition",
    "filtering",
    "guidance",
    "selection",
    "with",
)

RQ2_METRIC_FIELDS = (
    "id",
    "passed",
    "correctness_mode",
    "sample_correctness",
    "pass_at_k",
    "pass_rate_variance",
    "u",
    "uncertainty_components",
    "source_diagnostics",
    "repair_rounds",
    "index_latency_s",
    "run_latency_s",
    "verified_selection_applied",
)

RQ1_METRIC_FIELDS = (
    "example_id",
    "condition",
    "enabled",
    "u",
    "passed",
    "correctness_mode",
    "source_diagnostics",
    "sample_correctness",
    "pass_at_k",
    "pass_rate_variance",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def selected(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_rq2(source: Path, destination: Path) -> int:
    """Retain task metrics while replacing source-bearing plans with hashes."""
    data = json.loads(source.read_text(encoding="utf-8"))
    public: dict[str, Any] = {
        "metadata": data.get("metadata") or {},
        "summary": data.get("summary") or {},
        "public_record": {
            "source_payloads_released": False,
            "generated_code_released": False,
            "test_logs_released": False,
            "phase2_plans_released": False,
        },
    }
    count = 0
    for condition in RQ2_CONDITIONS:
        rows = []
        for row in data.get(condition) or []:
            clean = selected(row, RQ2_METRIC_FIELDS)
            clean["candidate_count"] = len(row.get("generated_samples") or [])
            plan = [
                {
                    "step": step.get("step"),
                    "step_uncertainty": step.get("step_uncertainty"),
                    "intent": step.get("intent"),
                    "queries": {
                        name: stats.get("query")
                        for name, stats in (step.get("sources") or {}).items()
                    },
                }
                for step in (row.get("per_step") or [])
            ]
            clean["phase2_plan_sha256"] = stable_hash(plan)
            rows.append(clean)
            count += 1
        public[condition] = rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return count


def sanitize_rq1(source: Path, destination: Path) -> int:
    """Retain the complete factorial metric matrix without query traces."""
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = [selected(row, RQ1_METRIC_FIELDS) for row in (data.get("rows") or [])]
    public = {
        "metadata": data.get("metadata") or {},
        "rows": rows,
        "summary": data.get("summary") or {},
        "public_record": {
            "source_payloads_released": False,
            "generated_code_released": False,
            "query_traces_released": False,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(rows)


def write_task_catalog(path: Path, exec_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]]) -> None:
    fields = ("benchmark", "task_id", "repository", "language", "upstream_row_index", "artifact_hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in exec_rows:
            writer.writerow(
                {
                    "benchmark": "ExecRepoBench-120",
                    "task_id": row["task_id"],
                    "repository": row.get("repo_name"),
                    "language": row.get("language") or "python",
                    "upstream_row_index": row.get("upstream_row_index"),
                    "artifact_hash": row.get("artifact_hash"),
                }
            )
        for row in cross_rows:
            writer.writerow(
                {
                    "benchmark": "CrossCodeEval-100",
                    "task_id": row["task_id"],
                    "repository": row.get("repository"),
                    "language": row.get("language"),
                    "upstream_row_index": row.get("upstream_row_index"),
                    "artifact_hash": (row.get("provenance") or {}).get("artifact_hash"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    private = args.private_root.resolve()
    public = args.public_root.resolve()
    exec_source = private / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
    cross_source = private / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl"
    agent_source = private / "results/agent_gateway_v1/raw_results.jsonl"

    exec_rows = [selected(row, EXEC_FIELDS) for row in read_jsonl(exec_source)]
    cross_rows = [selected(row, CROSS_FIELDS) for row in read_jsonl(cross_source)]
    agent_rows = [selected(row, AGENT_FIELDS) for row in read_jsonl(agent_source)]

    counts = {
        "execrepobench_tasks": write_jsonl(
            public / "data/manifests/execrepobench_120_public.jsonl", exec_rows
        ),
        "crosscodeeval_tasks": write_jsonl(
            public / "data/manifests/crosscodeeval_100_public.jsonl", cross_rows
        ),
        "automated_reviewer_episodes": write_jsonl(
            public / "results/agent_gateway_v1/raw_results_public.jsonl", agent_rows
        ),
    }
    for backend in ("gpt", "gemini"):
        counts[f"rq1_{backend}_metric_records"] = sanitize_rq1(
            private / f"results/rq12_corrected_10/{backend}/rq1.json",
            public / f"results/rq12_corrected_10/{backend}/rq1.json",
        )
        counts[f"rq2_{backend}_metric_records"] = sanitize_rq2(
            private / f"results/rq12_corrected_10/{backend}/rq2.json",
            public / f"results/rq12_corrected_10/{backend}/rq2.json",
        )
    write_task_catalog(public / "data/manifests/task_catalog.csv", exec_rows, cross_rows)
    outputs = {
        str(path.relative_to(public)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            public / "data/manifests/execrepobench_120_public.jsonl",
            public / "data/manifests/crosscodeeval_100_public.jsonl",
            public / "data/manifests/task_catalog.csv",
            public / "results/agent_gateway_v1/raw_results_public.jsonl",
            public / "results/rq12_corrected_10/gpt/rq1.json",
            public / "results/rq12_corrected_10/gemini/rq1.json",
            public / "results/rq12_corrected_10/gpt/rq2.json",
            public / "results/rq12_corrected_10/gemini/rq2.json",
        )
    }
    report = {
        "status": "PUBLIC_RECORDS_CREATED",
        "counts": counts,
        "sha256": outputs,
        "redactions": [
            "third-party repository source and reference implementations",
            "generated code and raw provider responses",
            "provider response identifiers and evaluator logs",
            "local filesystem paths and credentials",
        ],
    }
    report_path = public / "results/public_record_integrity.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
