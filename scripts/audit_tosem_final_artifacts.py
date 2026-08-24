#!/usr/bin/env python3
"""Fail-closed audit for the completed OpenCoderX TOSEM artifact package."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOSEM = ROOT / "tosem"
RESULTS = ROOT / "results/tosem"
MODELS = {
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "claude-sonnet-5",
    "qwen3-coder-plus",
}
EXEC_METHODS = {
    "Direct Generation",
    "Standard RAG",
    "RAG + Verify/Repair",
    "OpenCoderX",
}
CROSS_METHODS = {"Direct Generation", "Cross-file Context RAG"}
SELECTED_ANCHORS = {
    "gpt-4o-mini": (0.16666666666666666, 0.25833333333333336, 0.325, 0.30833333333333335),
    "gemini-2.5-flash": (0.225, 0.25833333333333336, 0.35, 0.36666666666666664),
    "claude-sonnet-5": (0.5166666666666667, 0.5833333333333334, 0.7333333333333333, 0.7416666666666667),
    "qwen3-coder-plus": (0.48333333333333334, 0.5333333333333333, 0.6083333333333333, 0.6),
}
AGGREGATE_AUROC_ANCHORS = {
    "gpt-4o-mini": 0.6320416802344513,
    "gemini-2.5-flash": 0.6016746411483253,
    "claude-sonnet-5": 0.6922798115259152,
    "qwen3-coder-plus": 0.5960648148148148,
}
CROSS_EM_ANCHORS = {
    "gpt-4o-mini": (0.06, 0.15, 0.015625),
    "gemini-2.5-flash": (0.05, 0.10, 0.2265625),
    "claude-sonnet-5": (0.16, 0.27, 0.02545166015625),
    "qwen3-coder-plus": (0.07, 0.17, 0.01904296875),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.issues: list[str] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        record = {"name": name, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        self.checks.append(record)
        if not condition:
            self.issues.append(f"{name}: {detail}")


def audit_integrity_records(audit: Audit) -> None:
    expected = {
        "confirmatory_analysis/integrity.json": {
            "task_count": 120,
            "model_count": 4,
            "method_count": 4,
            "task_method_model_cells": 1920,
            "duplicate_provider_response_ids": 0,
            "retrieval_leakage_failures": 0,
        },
        "crosscodeeval_confirmatory/integrity.json": {
            "tasks": 100,
            "models": 4,
            "methods": 2,
            "task_method_model_cells": 800,
            "functional_execution": False,
        },
        "collaboration_analysis/integrity.json": {
            "opencoder_records": 480,
            "decision_records": 960,
            "review_simulation_rows": 5040,
            "observed_intervention_rows": 28,
            "not_run_intervention_rows": 3,
        },
        "rq1_rq2_analysis/integrity.json": {
            "factorial_backends": 2,
            "factorial_tasks_per_backend": 10,
            "factorial_conditions": 8,
            "exec_uncertainty_model_task_records": 480,
            "crosscodeeval_tasks": 100,
            "crosscodeeval_languages": 4,
            "crosscodeeval_functional_execution": False,
        },
    }
    for relative, fields in expected.items():
        path = RESULTS / relative
        audit.check(f"integrity file exists: {relative}", path.is_file())
        if not path.is_file():
            continue
        record = read_json(path)
        audit.check(f"integrity passed: {relative}", record.get("passed") is True)
        audit.check(f"paper eligible: {relative}", record.get("paper_eligible") is True)
        for field, value in fields.items():
            audit.check(
                f"integrity field: {relative}:{field}",
                record.get(field) == value,
                {"actual": record.get(field), "expected": value},
            )


def audit_freezes(audit: Audit) -> None:
    for relative in ("protocol_freeze.json", "crosscodeeval_confirmatory/protocol_freeze.json"):
        freeze_path = RESULTS / relative
        freeze = read_json(freeze_path)
        audit.check(f"freeze approved: {relative}", freeze.get("status") == "FROZEN_APPROVED")
        audit.check(f"freeze paper eligible: {relative}", freeze.get("paper_eligible") is True)
        for path_text, expected_hash in freeze.get("file_sha256", {}).items():
            path = ROOT / path_text
            actual_hash = sha256(path) if path.is_file() else None
            audit.check(
                f"frozen hash: {path_text}",
                actual_hash == expected_hash,
                {"actual": actual_hash, "expected": expected_hash},
            )
    cross = read_json(RESULTS / "crosscodeeval_confirmatory/protocol_freeze.json")
    manifest = ROOT / cross["manifest"]
    audit.check("CrossCodeEval manifest hash", sha256(manifest) == cross["manifest_sha256"])


def audit_exec_results(audit: Audit) -> None:
    rows = read_csv(RESULTS / "confirmatory_analysis/task_level.csv")
    audit.check("Exec task-level row count", len(rows) == 1920, len(rows))
    audit.check("Exec models", {row["model"] for row in rows} == MODELS)
    audit.check("Exec methods", {row["method"] for row in rows} == EXEC_METHODS)
    keys = {(row["task_id"], row["model"], row["method"]) for row in rows}
    audit.check("Exec task-model-method keys unique", len(keys) == len(rows), len(keys))
    task_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        task_sets[(row["model"], row["method"])].add(row["task_id"])
        count = int(row["candidate_correct_count"])
        audit.check(f"candidate count range: {row['task_id']}:{row['model']}:{row['method']}", 0 <= count <= 5)
    unique_sets = {frozenset(tasks) for tasks in task_sets.values()}
    audit.check("Exec identical matched task sets", len(unique_sets) == 1 and len(next(iter(unique_sets))) == 120)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["method"])].append(row)
    summary = {(row["model"], row["method"]): row for row in read_csv(RESULTS / "confirmatory_analysis/summary.csv")}
    for key, group in grouped.items():
        selected = sum(row["selected_output_correct"] == "True" for row in group) / len(group)
        audit.check(f"Exec summary selected recomputes: {key}", close(selected, float(summary[key]["selected_output_correctness"])))

    method_order = ("Direct Generation", "Standard RAG", "RAG + Verify/Repair", "OpenCoderX")
    for model, expected_values in SELECTED_ANCHORS.items():
        actual = tuple(float(summary[(model, method)]["selected_output_correctness"]) for method in method_order)
        audit.check(f"selected-output anchors: {model}", all(close(a, e) for a, e in zip(actual, expected_values)), {"actual": actual, "expected": expected_values})

    selected_stats = read_csv(RESULTS / "confirmatory_analysis/selected_output_statistics.csv")
    matched_control = [row for row in selected_stats if row["comparator"] == "RAG + Verify/Repair"]
    audit.check("matched-control selected statistics cover four models", len(matched_control) == 4)
    audit.check("matched-control selected statistics use 120 tasks", all(row["matched_tasks"] == "120" for row in matched_control))


def audit_cross_results(audit: Audit) -> None:
    rows = read_csv(RESULTS / "crosscodeeval_confirmatory/task_level.csv")
    audit.check("CrossCodeEval task-level row count", len(rows) == 800, len(rows))
    audit.check("CrossCodeEval models", {row["model"] for row in rows} == MODELS)
    audit.check("CrossCodeEval methods", {row["method"] for row in rows} == CROSS_METHODS)
    audit.check("CrossCodeEval languages", {row["language"] for row in rows} == {"python", "java", "typescript", "csharp"})
    keys = {(row["task_id"], row["model"], row["method"]) for row in rows}
    audit.check("CrossCodeEval task-model-method keys unique", len(keys) == len(rows), len(keys))
    task_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task_sets[(row["model"], row["method"])].add(row["task_id"])
        grouped[(row["model"], row["method"])].append(row)
    unique_sets = {frozenset(tasks) for tasks in task_sets.values()}
    audit.check("CrossCodeEval identical matched task sets", len(unique_sets) == 1 and len(next(iter(unique_sets))) == 100)
    summary = {(row["model"], row["method"]): row for row in read_csv(RESULTS / "crosscodeeval_confirmatory/summary.csv")}
    for key, group in grouped.items():
        exact = sum(float(row["selected_exact_match"]) for row in group) / len(group)
        audit.check(f"CrossCodeEval exact match recomputes: {key}", close(exact, float(summary[key]["selected_exact_match"])))
    paired = {
        row["model"]: row
        for row in read_csv(RESULTS / "crosscodeeval_confirmatory/paired_statistics.csv")
        if row["metric"] == "exact_match"
    }
    for model, expected in CROSS_EM_ANCHORS.items():
        row = paired[model]
        actual = (float(row["direct_value"]), float(row["context_rag_value"]), float(row["holm_p"]))
        audit.check(f"CrossCodeEval exact-match anchors: {model}", all(close(a, e) for a, e in zip(actual, expected)), {"actual": actual, "expected": expected})


def audit_uncertainty_and_collaboration(audit: Audit) -> None:
    discrimination = read_csv(RESULTS / "rq1_rq2_analysis/uncertainty_signal_discrimination.csv")
    aggregate = {row["model"]: float(row["auroc_failure"]) for row in discrimination if row["signal"] == "aggregate"}
    for model, expected in AGGREGATE_AUROC_ANCHORS.items():
        audit.check(f"aggregate failure AUROC anchor: {model}", close(aggregate[model], expected), aggregate[model])
    decisions = read_csv(RESULTS / "collaboration_analysis/decision_summary.csv")
    primary = {
        row["model"]: row
        for row in decisions
        if row["decision_policy"] == "uncertainty_only" and row["decision"] == "ALL_POLICY_METRICS"
    }
    audit.check("uncertainty-only primary policy covers four models", set(primary) == MODELS)
    for model, row in primary.items():
        audit.check(f"uncertainty-only policy uses 120 tasks: {model}", row["tasks"] == "120")
        audit.check(f"uncertainty AUROC agrees with policy: {model}", close(float(row["failure_auroc"]), AGGREGATE_AUROC_ANCHORS[model]))
    review = read_csv(RESULTS / "collaboration_analysis/review_budget_summary.csv")
    audit.check("all review-budget outcomes are labeled simulations", all(row["simulation"] == "True" for row in review))


def tex_closure(audit: Audit) -> tuple[list[Path], str]:
    visited: list[Path] = []
    chunks: list[str] = []

    def resolve(reference: str, current: Path, extension: str) -> Path | None:
        raw = Path(reference)
        options = []
        if raw.suffix:
            options.extend((TOSEM / raw, current.parent / raw))
        else:
            options.extend((TOSEM / f"{reference}{extension}", current.parent / f"{reference}{extension}"))
        return next((path.resolve() for path in options if path.is_file()), None)

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.append(path)
        text = path.read_text(encoding="utf-8")
        chunks.append(text)
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            child = resolve(name, path, ".tex")
            audit.check(f"TeX input exists: {name}", child is not None)
            if child is not None:
                visit(child)
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            graphic = resolve(name, path, ".pdf")
            audit.check(f"graphic exists: {name}", graphic is not None)

    visit(TOSEM / "main.tex")
    combined = "\n".join(chunks)
    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    references = re.findall(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\{([^}]+)\}", combined)
    duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
    unresolved = sorted(set(references) - set(labels))
    audit.check("compiled TeX labels are unique", not duplicates, duplicates)
    audit.check("compiled TeX references resolve", not unresolved, unresolved)
    placeholder_patterns = (r"\bTBD\b", r"PLACEHOLDER", r"resulttodo", r"todoresult", r"XX\.X")
    placeholders = [pattern for pattern in placeholder_patterns if re.search(pattern, combined, flags=re.IGNORECASE)]
    audit.check("compiled manuscript contains no result placeholders", not placeholders, placeholders)
    return visited, combined


def audit_publication_artifacts(audit: Audit) -> list[Path]:
    manifest = read_json(RESULTS / "publication_tables/manifest.json")
    names = manifest.get("latex_files", [])
    audit.check("publication table manifest status", manifest.get("status") == "GENERATED_FROM_AUDITED_ARTIFACTS")
    audit.check("publication table count", manifest.get("tables") == 13 and len(names) == 13)
    for name in names:
        source = RESULTS / "publication_tables/latex" / name
        target = TOSEM / "generated/tables" / name
        audit.check(f"publication table exists: {name}", source.is_file() and target.is_file())
        if source.is_file() and target.is_file():
            audit.check(f"compiled publication table is current: {name}", sha256(source) == sha256(target))
    tex_files, _ = tex_closure(audit)
    log_path = TOSEM / "main.log"
    pdf_path = TOSEM / "main.pdf"
    audit.check("compiled manuscript PDF exists", pdf_path.is_file() and pdf_path.stat().st_size > 0)
    audit.check("compiled manuscript log exists", log_path.is_file())
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_patterns = (
            "There were undefined references",
            "LaTeX Error",
            "Emergency stop",
            "Fatal error",
            "Citation `",
        )
        found = [pattern for pattern in fatal_patterns if pattern in log]
        audit.check("LaTeX log has no unresolved or fatal errors", not found, found)
        overfull = re.findall(r"Overfull \\hbox \(([^)]+)\)", log)
        audit.check("LaTeX log has no overfull boxes", not overfull, overfull)
    return tex_files


def audit_hygiene(audit: Audit) -> None:
    roots = [TOSEM, ROOT / "docs/tosem", RESULTS / "publication_tables", RESULTS / "confirmatory_analysis", RESULTS / "crosscodeeval_confirmatory", RESULTS / "collaboration_analysis", RESULTS / "rq1_rq2_analysis"]
    suffixes = {".tex", ".md", ".csv", ".json", ".yaml", ".yml"}
    secret = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
    personal = re.compile(r"/(?:Users|home)/[^/\s]+(?:/|\b)")
    findings: list[str] = []
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if secret.search(text):
                findings.append(f"credential-like token in {path.relative_to(ROOT)}")
            if personal.search(text):
                findings.append(f"identifying absolute path in {path.relative_to(ROOT)}")
    audit.check("processed and manuscript artifacts contain no credentials or identifying paths", not findings, findings)


def write_manifest(tex_files: list[Path]) -> None:
    paths: set[Path] = {
        ROOT / "Makefile",
        ROOT / "scripts/audit_tosem_final_artifacts.py",
        ROOT / "docs/tosem/FINAL_STATUS.md",
        ROOT / "docs/tosem/PROGRESS.md",
        RESULTS / "final_results_memo.md",
        RESULTS / "final_integrity.json",
        TOSEM / "main.pdf",
        RESULTS / "protocol_freeze.json",
        RESULTS / "crosscodeeval_confirmatory/protocol_freeze.json",
        RESULTS / "confirmatory_analysis/integrity.json",
        RESULTS / "crosscodeeval_confirmatory/integrity.json",
        RESULTS / "collaboration_analysis/integrity.json",
        RESULTS / "rq1_rq2_analysis/integrity.json",
        ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl",
        ROOT / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl",
    }
    paths.update(tex_files)
    paths.update((RESULTS / "publication_tables/latex").glob("*.tex"))
    paths.update((RESULTS / "publication_tables").glob("*.csv"))
    paths.update((TOSEM / "Figures").glob("*.pdf"))
    records = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(paths)
        if path.is_file()
    ]
    output = {"schema_version": 1, "artifact_count": len(records), "artifacts": records}
    (RESULTS / "final_artifact_manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    audit = Audit()
    audit_integrity_records(audit)
    audit_freezes(audit)
    audit_exec_results(audit)
    audit_cross_results(audit)
    audit_uncertainty_and_collaboration(audit)
    tex_files = audit_publication_artifacts(audit)
    audit_hygiene(audit)
    campaign = read_json(RESULTS / "crosscodeeval_confirmatory/campaign_status.json")
    spend = campaign["campaign_spend"]
    audit.check("campaign USD spend below cap", float(spend["USD"]) <= 80.0, spend["USD"])
    audit.check("campaign CNY spend below cap", float(spend["CNY"]) <= 65.0, spend["CNY"])

    output = {
        "schema_version": 1,
        "passed": not audit.issues,
        "paper_eligible": not audit.issues,
        "checks": len(audit.checks),
        "failed_checks": len(audit.issues),
        "issues": audit.issues,
        "campaign_spend": spend,
        "details": audit.checks,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_integrity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if not audit.issues:
        write_manifest(tex_files)
    print(json.dumps({key: output[key] for key in ("passed", "paper_eligible", "checks", "failed_checks", "issues", "campaign_spend")}, indent=2))
    return 0 if not audit.issues else 1


if __name__ == "__main__":
    sys.exit(main())
