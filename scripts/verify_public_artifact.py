#!/usr/bin/env python3
"""Fail closed when the public OpenCoderX release is incomplete or unsafe."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISSUES: list[str] = []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expect(condition: bool, message: str) -> None:
    if not condition:
        ISSUES.append(message)


def main() -> int:
    exec_manifest = read_jsonl(ROOT / "data/manifests/execrepobench_120_public.jsonl")
    cross_manifest = read_jsonl(ROOT / "data/manifests/crosscodeeval_100_public.jsonl")
    agent_rows = read_jsonl(ROOT / "results/agent_gateway_v1/raw_results_public.jsonl")
    exec_rows = read_csv(ROOT / "results/tosem/confirmatory_analysis/task_level.csv")
    cross_rows = read_csv(ROOT / "results/tosem/crosscodeeval_confirmatory/task_level.csv")
    rq2_records = {
        backend: json.loads(
            (ROOT / f"results/rq12_corrected_10/{backend}/rq2.json").read_text(encoding="utf-8")
        )
        for backend in ("gpt", "gemini")
    }
    rq1_records = {
        backend: json.loads(
            (ROOT / f"results/rq12_corrected_10/{backend}/rq1.json").read_text(encoding="utf-8")
        )
        for backend in ("gpt", "gemini")
    }

    expect(len(exec_manifest) == 120, "ExecRepoBench public manifest must contain 120 tasks")
    expect(len({row["task_id"] for row in exec_manifest}) == 120, "ExecRepoBench task IDs must be unique")
    expect(len(cross_manifest) == 100, "CrossCodeEval public manifest must contain 100 tasks")
    expect(len({row["task_id"] for row in cross_manifest}) == 100, "CrossCodeEval task IDs must be unique")

    expect(len(exec_rows) == 1920, "ExecRepoBench analysis must contain 1,920 cells")
    expect(
        len({(row["task_id"], row["model"], row["method"]) for row in exec_rows}) == 1920,
        "ExecRepoBench cells must be unique",
    )
    expect(len(cross_rows) == 800, "CrossCodeEval analysis must contain 800 cells")
    expect(
        len({(row["task_id"], row["model"], row["method"]) for row in cross_rows}) == 800,
        "CrossCodeEval cells must be unique",
    )

    expect(len(agent_rows) == 144, "automated-reviewer release must contain 144 episodes")
    expect(
        len({(row["agent_id"], row["episode_index"]) for row in agent_rows}) == 144,
        "automated-reviewer episode keys must be unique",
    )
    evaluable = sum(row.get("evaluator_status") == "ok" for row in agent_rows)
    missing = len(agent_rows) - evaluable
    infrastructure_errors = sum(row.get("evaluator_status") == "error" for row in agent_rows)
    mismatches = sum(
        bool(row.get("served_model")) and row.get("served_model") != row.get("requested_model")
        for row in agent_rows
    )
    expect(evaluable == 134, "automated-reviewer evaluable count must be 134")
    expect(missing == 10, "automated-reviewer missing count must be 10")
    expect(infrastructure_errors == 0, "automated-reviewer evaluator errors must be zero")
    expect(mismatches == 0, "automated-reviewer served-model mismatches must be zero")

    forbidden_agent_fields = {
        "final_code",
        "raw_response",
        "response_id",
        "evaluator_stdout",
        "evaluator_stderr",
        "prompt_path",
    }
    for index, row in enumerate(agent_rows):
        overlap = forbidden_agent_fields.intersection(row)
        expect(not overlap, f"agent record {index} contains redacted fields: {sorted(overlap)}")

    rq2_conditions = ("without", "decomposition", "filtering", "guidance", "selection", "with")
    forbidden_rq2_fields = {
        "code",
        "generated_samples",
        "fused_evidence",
        "per_step",
        "static_report",
        "test_report",
        "initial_test_report",
        "post_selection_test_report",
    }
    for backend, data in rq2_records.items():
        rows = [row for condition in rq2_conditions for row in (data.get(condition) or [])]
        expect(len(rows) == 60, f"RQ2 {backend} release must contain 60 metric-only records")
        expect(
            all(int(row.get("candidate_count", 0)) == 3 for row in rows),
            f"RQ2 {backend} records must preserve the three-candidate audit",
        )
        for index, row in enumerate(rows):
            overlap = forbidden_rq2_fields.intersection(row)
            expect(not overlap, f"RQ2 {backend} record {index} contains source-bearing fields: {sorted(overlap)}")

    for backend, data in rq1_records.items():
        rows = data.get("rows") or []
        expect(len(rows) == 80, f"RQ1 {backend} release must contain the complete 10x8 metric matrix")
        for index, row in enumerate(rows):
            overlap = forbidden_rq2_fields.intersection(row)
            expect(not overlap, f"RQ1 {backend} record {index} contains source-bearing fields: {sorted(overlap)}")

    secret_patterns = (
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    )
    private_path = re.compile(r"/(?:Users|home)/[^/\s]+/")
    text_suffixes = {
        ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".csv",
        ".tex", ".txt", ".cff", ".example",
    }
    detector_files = {
        "verify_public_artifact.py", "audit_tosem_final_artifacts.py", "analyze_results.py",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        expect(
            path.stat().st_size < 100 * 1024 * 1024,
            f"file exceeds GitHub's 100 MiB limit: {path.relative_to(ROOT)}",
        )
        if path.suffix.lower() not in text_suffixes or path.stat().st_size > 8 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name not in detector_files:
            expect(not private_path.search(text), f"personal filesystem path in {path.relative_to(ROOT)}")
        for pattern in secret_patterns:
            expect(not pattern.search(text), f"credential-like value in {path.relative_to(ROOT)}")

    manuscript_candidates = [
        path for path in ROOT.rglob("*.pdf")
        if any(
            token in path.name.casefold()
            for token in ("paper", "manuscript", "tosem_project", "technical_track")
        )
    ]
    expect(not manuscript_candidates, "a manuscript-like PDF is present")

    report = {
        "passed": not ISSUES,
        "issues": ISSUES,
        "counts": {
            "execrepobench_tasks": len(exec_manifest),
            "crosscodeeval_tasks": len(cross_manifest),
            "technical_cells": len(exec_rows) + len(cross_rows),
            "automated_reviewer_episodes": len(agent_rows),
            "automated_reviewer_evaluable": evaluable,
            "automated_reviewer_missing": missing,
            "rq2_metric_records": sum(
                len(data.get(condition) or [])
                for data in rq2_records.values()
                for condition in rq2_conditions
            ),
            "rq1_metric_records": sum(len(data.get("rows") or []) for data in rq1_records.values()),
        },
    }
    output = ROOT / "results/release_integrity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not ISSUES else 1


if __name__ == "__main__":
    raise SystemExit(main())
