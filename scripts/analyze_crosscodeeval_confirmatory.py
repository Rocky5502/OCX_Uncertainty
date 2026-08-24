#!/usr/bin/env python3
"""Audit and analyze the frozen CrossCodeEval-100 native-metric campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/tosem/crosscodeeval_confirmatory"
PROTOCOL = RESULT_ROOT / "protocol_freeze.json"
MANIFEST = ROOT / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl"
CONFIG = ROOT / "configs/tosem/campaign.yaml"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = {"direct": "Direct Generation", "context_rag": "Cross-file Context RAG"}
METRICS = ("exact_match", "edit_similarity", "identifier_f1")
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260809


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(left: Sequence[float], right: Sequence[float], seed_offset: int) -> tuple[float, float]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if not len(a) or a.shape != b.shape:
        raise ValueError("paired bootstrap inputs must be non-empty and matched")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    deltas = np.empty(BOOTSTRAP_ITERATIONS)
    for start in range(0, BOOTSTRAP_ITERATIONS, 1_000):
        stop = min(start + 1_000, BOOTSTRAP_ITERATIONS)
        indices = rng.integers(0, len(a), size=(stop - start, len(a)))
        deltas[start:stop] = np.mean(b[indices] - a[indices], axis=1)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def _ece(risk: Sequence[float], failures: Sequence[int], bins: int = 10) -> float:
    prediction = np.clip(np.asarray(risk, dtype=float), 0.0, 1.0)
    target = np.asarray(failures, dtype=float)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (prediction >= low) & (prediction < high if index < bins - 1 else prediction <= high)
        if np.any(mask):
            result += float(np.mean(mask)) * abs(float(np.mean(prediction[mask])) - float(np.mean(target[mask])))
    return result


def _aurc(risk: Sequence[float], failures: Sequence[int]) -> float:
    order = np.argsort(np.asarray(risk), kind="stable")
    target = np.asarray(failures, dtype=float)[order]
    return float(np.mean(np.cumsum(target) / np.arange(1, len(target) + 1)))


def _calibration(risk: Sequence[float], failures: Sequence[int]) -> tuple[float | None, float | None]:
    target = np.asarray(failures, dtype=float)
    if len(np.unique(target)) < 2:
        return None, None
    clipped = np.clip(np.asarray(risk), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * logit
        return float(np.sum(np.logaddexp(0.0, linear) - target * linear))

    result = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not np.all(np.isfinite(result.x)):
        return None, None
    return float(result.x[0]), float(result.x[1])


def _price(model: str, prompt: int, completion: int) -> tuple[float, str]:
    pricing = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["cost_controls"]["gateway_pricing"][model]
    amount = (
        prompt * float(pricing["input_per_million"])
        + completion * float(pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(pricing["currency"])


def _latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def _table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{CrossCodeEval-100 native completion results. EM denotes exact match, Edit denotes normalized edit similarity, and Id-F1 denotes identifier F1. These are native completion metrics, not executable functional correctness.}",
        r"\label{tab:crosscodeeval100}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Method & Sel. EM & Sel. Edit & Sel. Id-F1 & Cand. EM & Cand. Edit & Cand. Id-F1 \\",
        r"\midrule",
    ]
    for model_index, model in enumerate(MODELS):
        for row in [item for item in rows if item["model"] == model]:
            lines.append(
                f"{_latex_escape(model)} & {_latex_escape(row['method'])} & "
                f"{100*row['selected_exact_match']:.1f} & {100*row['selected_edit_similarity']:.1f} & "
                f"{100*row['selected_identifier_f1']:.1f} & {100*row['candidate_exact_match']:.1f} & "
                f"{100*row['candidate_edit_similarity']:.1f} & {100*row['candidate_identifier_f1']:.1f} " + r"\\"
            )
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    issues: list[str] = []
    protocol = _read(PROTOCOL)
    if protocol.get("status") != "FROZEN_APPROVED" or protocol.get("paper_eligible") is not True:
        issues.append("protocol is not frozen and paper eligible")
    for relative, expected in (protocol.get("file_sha256") or {}).items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            issues.append(f"frozen file hash mismatch: {relative}")
    if protocol.get("manifest_sha256") != _sha256(MANIFEST):
        issues.append("manifest hash mismatch")
    status = _read(RESULT_ROOT / "campaign_status.json")
    if status.get("status") != "COMPLETED" or status.get("failures"):
        issues.append("campaign status is not cleanly completed")

    manifest = _read_jsonl(MANIFEST)
    manifest_by_id = {str(row["task_id"]): row for row in manifest}
    expected_ids = [str(row["task_id"]) for row in manifest]
    expected_set = set(expected_ids)
    if len(expected_ids) != 100 or len(expected_set) != 100:
        issues.append("manifest does not contain 100 unique tasks")

    response_ids: set[str] = set()
    raw_files = []
    task_level: list[dict[str, Any]] = []
    for model, directory in MODELS.items():
        model_rows = []
        files = sorted((RESULT_ROOT / directory).glob("batch_*.json"))
        if len(files) != 4:
            issues.append(f"{model}: expected four result batches, found {len(files)}")
        for path in files:
            data = _read(path)
            metadata = data.get("metadata") or {}
            if metadata.get("model") != model:
                issues.append(f"{path.relative_to(ROOT)}: model mismatch")
            if metadata.get("dataset_version") != "crosscodeeval_opencoderx_100_v1":
                issues.append(f"{path.relative_to(ROOT)}: dataset version mismatch")
            if metadata.get("experiment_version") != "crosscodeeval_native_confirmatory_v1":
                issues.append(f"{path.relative_to(ROOT)}: experiment version mismatch")
            if metadata.get("paper_eligible") is not True or metadata.get("functional_execution") is not False:
                issues.append(f"{path.relative_to(ROOT)}: eligibility or execution declaration mismatch")
            if int(metadata.get("candidate_count") or 0) != 5 or int(metadata.get("max_output_tokens") or 0) != 128:
                issues.append(f"{path.relative_to(ROOT)}: generation budget mismatch")
            rows = list(data.get("rows") or [])
            model_rows.extend(rows)
            raw_files.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "model": model,
                "created_at": metadata.get("created_at"),
                "cells": len(rows),
            })
        expected_cells = {(task_id, method) for task_id in expected_ids for method in METHODS}
        actual_cells = {(str(row.get("task_id")), str(row.get("method"))) for row in model_rows}
        if actual_cells != expected_cells or len(model_rows) != 200:
            issues.append(f"{model}: task-method cells differ from the frozen design")

        for row in model_rows:
            task_id = str(row.get("task_id"))
            method = str(row.get("method"))
            prefix = f"{model}/{method}/{task_id}"
            if row.get("status") != "COMPLETED_NATIVE_METRICS" or row.get("error"):
                issues.append(f"{prefix}: incomplete cell")
            if row.get("functional_correctness") is not None:
                issues.append(f"{prefix}: functional correctness must remain null")
            candidates = list(row.get("candidates") or [])
            candidate_metrics = list(row.get("candidate_metrics") or [])
            if len(candidates) != 5 or len(candidate_metrics) != 5:
                issues.append(f"{prefix}: candidate count mismatch")
            if (row.get("generation_integrity") or {}).get("valid") is not True:
                issues.append(f"{prefix}: generation integrity failed")
            if int(row.get("selected_candidate_index", -1)) != 0:
                issues.append(f"{prefix}: selected candidate is not the frozen first candidate")
            selected = row.get("selected_metrics") or {}
            for record in [selected, *candidate_metrics]:
                if any(not isinstance(record.get(metric), (int, float)) for metric in METRICS):
                    issues.append(f"{prefix}: native metric is missing")
                    break
            evidence = list(row.get("retrieval_evidence") or [])
            if method == "direct" and evidence:
                issues.append(f"{prefix}: direct method contains retrieval evidence")
            if method == "context_rag" and not evidence:
                issues.append(f"{prefix}: context RAG has no retrieval evidence")
            audit = list(row.get("provider_response_audit") or [])
            requests = int((row.get("llm_usage") or {}).get("requests") or 0)
            if len(audit) != requests or not audit:
                issues.append(f"{prefix}: provider audit count mismatch")
            for call in audit:
                response_id = str(call.get("response_id") or "")
                if not response_id:
                    issues.append(f"{prefix}: missing response ID")
                elif response_id in response_ids:
                    issues.append(f"{prefix}: duplicate response ID")
                else:
                    response_ids.add(response_id)
                if call.get("requested_model") != model or call.get("served_model") != model:
                    issues.append(f"{prefix}: requested/served model mismatch")
            source = manifest_by_id.get(task_id) or {}
            candidate_means = {
                metric: _mean(float(item[metric]) for item in candidate_metrics)
                for metric in METRICS
            }
            task_level.append({
                "task_id": task_id,
                "repository": source.get("repository"),
                "language": source.get("language"),
                "model": model,
                "method": METHODS.get(method, method),
                "selected_exact_match": float(selected.get("exact_match") or 0.0),
                "selected_edit_similarity": float(selected.get("edit_similarity") or 0.0),
                "selected_identifier_f1": float(selected.get("identifier_f1") or 0.0),
                "candidate_exact_match": candidate_means["exact_match"],
                "candidate_edit_similarity": candidate_means["edit_similarity"],
                "candidate_identifier_f1": candidate_means["identifier_f1"],
                "aggregate_risk": float((row.get("uncertainty") or {}).get("aggregate_risk") or 0.0),
                "candidate_disagreement": float((row.get("uncertainty") or {}).get("candidate_disagreement") or 0.0),
                "retrieval_score_dispersion": float((row.get("uncertainty") or {}).get("retrieval_score_dispersion") or 0.0),
                "evidence_count": len(evidence),
                "prompt_tokens": int((row.get("llm_usage") or {}).get("prompt_tokens") or 0),
                "completion_tokens": int((row.get("llm_usage") or {}).get("completion_tokens") or 0),
                "total_tokens": int((row.get("llm_usage") or {}).get("total_tokens") or 0),
                "provider_requests": requests,
                "latency_seconds": float(row.get("latency_seconds") or 0.0),
            })

    if len(task_level) != 800:
        issues.append(f"expected 800 task-method-model cells, found {len(task_level)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_level:
        grouped[(str(row["model"]), str(row["method"]))].append(row)

    summary = []
    resources = []
    calibration = []
    language_summary = []
    for model in MODELS:
        for method in METHODS.values():
            rows = grouped[(model, method)]
            summary.append({
                "model": model,
                "method": method,
                "tasks": len(rows),
                **{f"selected_{metric}": _mean(float(row[f"selected_{metric}"]) for row in rows) for metric in METRICS},
                **{f"candidate_{metric}": _mean(float(row[f"candidate_{metric}"]) for row in rows) for metric in METRICS},
                "mean_aggregate_risk": _mean(float(row["aggregate_risk"]) for row in rows),
            })
            prompt = sum(int(row["prompt_tokens"]) for row in rows)
            completion = sum(int(row["completion_tokens"]) for row in rows)
            cost, currency = _price(model, prompt, completion)
            resources.append({
                "model": model,
                "method": method,
                "tasks": len(rows),
                "provider_requests": sum(int(row["provider_requests"]) for row in rows),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "mean_tokens": _mean(float(row["total_tokens"]) for row in rows),
                "mean_latency_seconds": _mean(float(row["latency_seconds"]) for row in rows),
                "estimated_cost": cost,
                "currency": currency,
            })
            risk = [float(row["aggregate_risk"]) for row in rows]
            failures = [int(float(row["selected_exact_match"]) < 1.0) for row in rows]
            intercept, slope = _calibration(risk, failures)
            calibration.append({
                "model": model,
                "method": method,
                "tasks": len(rows),
                "native_failure_definition": "selected exact-match failure",
                "failure_rate": _mean(failures),
                "auroc_failure": float(roc_auc_score(failures, risk)) if len(set(failures)) > 1 else None,
                "auprc_failure": float(average_precision_score(failures, risk)) if len(set(failures)) > 1 else None,
                "brier_failure": _mean((prediction - target) ** 2 for prediction, target in zip(risk, failures)),
                "ece_failure_10_bins": _ece(risk, failures),
                "aurc": _aurc(risk, failures),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            })
            for language in ("python", "java", "typescript", "csharp"):
                subset = [row for row in rows if row["language"] == language]
                language_summary.append({
                    "model": model,
                    "method": method,
                    "language": language,
                    "tasks": len(subset),
                    **{f"selected_{metric}": _mean(float(row[f"selected_{metric}"]) for row in subset) for metric in METRICS},
                })

    paired = []
    for model_index, model in enumerate(MODELS):
        direct = {str(row["task_id"]): row for row in grouped[(model, "Direct Generation")]}
        context = {str(row["task_id"]): row for row in grouped[(model, "Cross-file Context RAG")]}
        for metric_index, metric in enumerate(METRICS):
            field = f"selected_{metric}"
            left = [float(direct[task_id][field]) for task_id in expected_ids]
            right = [float(context[task_id][field]) for task_id in expected_ids]
            low, high = _bootstrap_ci(left, right, 100 * model_index + metric_index)
            wins = sum(b > a for a, b in zip(left, right))
            losses = sum(b < a for a, b in zip(left, right))
            paired.append({
                "model": model,
                "metric": metric,
                "direct_value": _mean(left),
                "context_rag_value": _mean(right),
                "absolute_difference": _mean(right) - _mean(left),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "context_rag_wins": wins,
                "context_rag_losses": losses,
                "ties": len(left) - wins - losses,
                "exact_mcnemar_p": _mcnemar_exact(wins, losses) if metric == "exact_match" else None,
                "holm_p": None,
                "matched_tasks": len(left),
            })
    exact_family = [row for row in paired if row["metric"] == "exact_match"]
    adjusted = _holm([float(row["exact_mcnemar_p"]) for row in exact_family])
    for row, value in zip(exact_family, adjusted):
        row["holm_p"] = value

    _write_csv(RESULT_ROOT / "task_level.csv", task_level)
    _write_csv(RESULT_ROOT / "summary.csv", summary)
    _write_csv(RESULT_ROOT / "language_summary.csv", language_summary)
    _write_csv(RESULT_ROOT / "paired_statistics.csv", paired)
    _write_csv(RESULT_ROOT / "uncertainty_calibration.csv", calibration)
    _write_csv(RESULT_ROOT / "resource_summary.csv", resources)
    latex = RESULT_ROOT / "latex"
    latex.mkdir(exist_ok=True)
    _table(latex / "table_crosscodeeval100.tex", summary)

    provenance = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "raw_result_files": raw_files,
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "metric_scope": "CrossCodeEval native completion metrics; no executable functional correctness.",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "unit": "task"},
    }
    (RESULT_ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    integrity = {
        "passed": not issues,
        "paper_eligible": not issues,
        "tasks": len(expected_ids),
        "models": len(MODELS),
        "methods": len(METHODS),
        "task_method_model_cells": len(task_level),
        "raw_result_files": len(raw_files),
        "unique_provider_responses": len(response_ids),
        "functional_execution": False,
        "issues": issues,
    }
    (RESULT_ROOT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    memo = [
        "# CrossCodeEval-100 Confirmatory Results",
        "",
        f"Integrity: **{'PASSED' if not issues else 'FAILED'}**.",
        "",
        "All values are native completion metrics. They are not executable functional correctness.",
        "",
        "| Model | Method | EM | Edit | Identifier F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        memo.append(
            f"| {row['model']} | {row['method']} | {100*row['selected_exact_match']:.1f} | "
            f"{100*row['selected_edit_similarity']:.1f} | {100*row['selected_identifier_f1']:.1f} |"
        )
    memo.extend(["", "## Paired Context Effects", ""])
    for row in exact_family:
        memo.append(
            f"- {row['model']}: exact-match difference {100*row['absolute_difference']:+.1f} points, "
            f"95% CI [{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}], "
            f"W/L/T={row['context_rag_wins']}/{row['context_rag_losses']}/{row['ties']}, "
            f"McNemar p={row['exact_mcnemar_p']:.3f}, Holm p={row['holm_p']:.3f}."
        )
    memo.extend([
        "",
        "Cross-file context effects are reported as backend- and language-dependent. A non-significant difference is not interpreted as equivalence.",
        "",
    ])
    (RESULT_ROOT / "results_memo.md").write_text("\n".join(memo), encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
