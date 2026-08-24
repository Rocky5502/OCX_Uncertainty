#!/usr/bin/env python3
"""Audit and analyze the frozen ExecRepoBench-120 confirmatory campaign.

Candidate Pass@k is computed only from the five original generated samples.
Selected-output correctness is reported separately and may reflect verified
selection and up to two repair rounds. This separation prevents repaired
outputs from being relabeled as ordinary sampling gains.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from opencoder.evaluation.metrics import estimate_pass_at_k


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/tosem/confirmatory"
OUTPUT_ROOT = ROOT / "results/tosem/confirmatory_analysis"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
PROTOCOL = ROOT / "results/tosem/protocol_freeze.json"
CAMPAIGN_CONFIG = ROOT / "configs/tosem/campaign.yaml"
LEAKAGE_REPORT = ROOT / "results/data_quality/execrepobench_120_freeze.json"

MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = {
    "direct": ("direct", "Direct Generation"),
    "baseline_rag": ("without", "Standard RAG"),
    "rag_verify_repair": ("rag_repair", "RAG + Verify/Repair"),
    "opencoder": ("with", "OpenCoderX"),
}
COMPARATORS = ("Direct Generation", "Standard RAG", "RAG + Verify/Repair")
KS = (1, 3, 5)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260809
REPAIR_LIMITS = {"task_chars": 24_000, "code_chars": 32_000, "diagnostics_chars": 32_000}
REPAIR_POLICY = "frozen_head_tail_character_budget_v1"


def _read_json(path: Path) -> dict[str, Any]:
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


def _optional_mean(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return _mean(items) if items else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"field names are required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pass_at_k(outcomes: Sequence[bool], k: int) -> float:
    return estimate_pass_at_k(len(outcomes), sum(bool(value) for value in outcomes), k)


def _paired_bootstrap_ci(
    comparator: Sequence[float],
    opencoder: Sequence[float],
    *,
    seed_offset: int,
) -> tuple[float, float]:
    comp = np.asarray(comparator, dtype=float)
    op = np.asarray(opencoder, dtype=float)
    if not len(comp) or comp.shape != op.shape:
        raise ValueError("paired bootstrap inputs must be non-empty and matched")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    deltas = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    chunk_size = 1_000
    for start in range(0, BOOTSTRAP_ITERATIONS, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_ITERATIONS)
        indices = rng.integers(0, len(comp), size=(stop - start, len(comp)))
        deltas[start:stop] = np.mean(op[indices] - comp[indices], axis=1)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm_adjust(values: Sequence[float]) -> list[float]:
    count = len(values)
    ordered = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _ece(risk: Sequence[float], failures: Sequence[int], bins: int = 10) -> float:
    predicted = np.clip(np.asarray(risk, dtype=float), 0.0, 1.0)
    observed = np.asarray(failures, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (predicted >= low) & (predicted < high if index < bins - 1 else predicted <= high)
        if np.any(mask):
            value += float(np.mean(mask)) * abs(float(np.mean(predicted[mask])) - float(np.mean(observed[mask])))
    return value


def _aurc(risk: Sequence[float], failures: Sequence[int]) -> float:
    order = np.argsort(np.asarray(risk, dtype=float), kind="stable")
    ordered_failures = np.asarray(failures, dtype=float)[order]
    selective_risk = np.cumsum(ordered_failures) / np.arange(1, len(order) + 1)
    return float(np.mean(selective_risk))


def _calibration_fit(risk: Sequence[float], failures: Sequence[int]) -> tuple[float | None, float | None]:
    target = np.asarray(failures, dtype=float)
    if len(np.unique(target)) < 2:
        return None, None
    clipped = np.clip(np.asarray(risk, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * logit
        return float(np.sum(np.logaddexp(0.0, linear) - target * linear))

    result = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not result.success and not np.all(np.isfinite(result.x)):
        return None, None
    return float(result.x[0]), float(result.x[1])


def _usage(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    fields = ("requests", "retries", "failed_attempts", "prompt_tokens", "completion_tokens", "total_tokens")
    return {
        field: sum(int((row.get("llm_usage") or {}).get(field) or 0) for row in rows)
        for field in fields
    }


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, str]:
    pricing = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))["cost_controls"]["gateway_pricing"][model]
    amount = (
        prompt_tokens * float(pricing["input_per_million"])
        + completion_tokens * float(pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(pricing["currency"])


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _fmt_metric(value: float | None, decimals: int = 3) -> str:
    return "--" if value is None else f"{value:.{decimals}f}"


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def _build_main_table(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{ExecRepoBench-120 effectiveness under the frozen five-candidate protocol. Pass@$k$ is computed from the original candidate set; Selected is executable correctness after each method's declared selection and repair policy. Values are percentages.}",
        r"\label{tab:exec120_effectiveness}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Method & Pass@1 & Pass@3 & Pass@5 & Selected & Static \\",
        r"\midrule",
    ]
    for model_index, model in enumerate(MODELS):
        model_rows = [row for row in summary if row["model"] == model]
        for row in model_rows:
            lines.append(
                f"{_latex_escape(model)} & {_latex_escape(row['method'])} & "
                f"{_fmt_percent(row['pass_at_1'])} & {_fmt_percent(row['pass_at_3'])} & "
                f"{_fmt_percent(row['pass_at_5'])} & {_fmt_percent(row['selected_output_correctness'])} & "
                f"{_fmt_percent(row['static_success'])} " + r"\\"
            )
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_matched_table(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["comparator"] == "RAG + Verify/Repair"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Matched selected-output comparison between OpenCoderX and the ordinary RAG verification/repair control on 120 tasks. $\Delta$ and W/L/T are from OpenCoderX's perspective; confidence intervals are paired task-bootstrap intervals. $p$ is the two-sided exact McNemar test and $p_{\mathrm{Holm}}$ adjusts across the four model backends.}",
        r"\label{tab:exec120_matched}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & Control & OpenCoderX & $\Delta$ & 95\% CI & W & L & T & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for row in selected:
        ci = f"[{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}]"
        lines.append(
            f"{_latex_escape(row['model'])} & {_fmt_percent(row['comparator_correctness'])} & "
            f"{_fmt_percent(row['opencoder_correctness'])} & {100*row['absolute_difference']:.1f} & "
            f"{ci} & {row['opencoder_wins']} & {row['opencoder_losses']} & {row['ties']} & "
            f"{row['mcnemar_holm_p']:.3f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "latex").mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    warnings: list[str] = []
    protocol = _read_json(PROTOCOL)
    if protocol.get("status") != "FROZEN_APPROVED" or protocol.get("paper_eligible") is not True:
        issues.append("scientific protocol is not FROZEN_APPROVED and paper eligible")
    frozen_hashes: dict[str, dict[str, Any]] = {}
    for relative, expected in (protocol.get("file_sha256") or {}).items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        frozen_hashes[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            issues.append(f"frozen file hash mismatch: {relative}")

    leakage = _read_json(LEAKAGE_REPORT)
    if leakage.get("status") != "FROZEN" or int(leakage.get("leakage_failures") or 0) != 0:
        issues.append("retrieval leakage report is not clean")
    if leakage.get("manifest_sha256") != _sha256(MANIFEST):
        issues.append("leakage report manifest hash does not match the campaign manifest")

    manifest_rows = _read_jsonl(MANIFEST)
    expected_ids = [str(row["task_id"]) for row in manifest_rows]
    expected_set = set(expected_ids)
    if len(expected_ids) != 120 or len(expected_set) != 120:
        issues.append("campaign manifest does not contain 120 unique tasks")
    task_metadata = {str(row["task_id"]): row for row in manifest_rows}

    campaign_status = _read_json(RESULT_ROOT / "campaign_status.json")
    if campaign_status.get("status") != "COMPLETED" or campaign_status.get("failures"):
        issues.append("campaign status is not cleanly completed")

    all_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_files: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    duplicate_response_ids: set[str] = set()
    task_level: list[dict[str, Any]] = []
    repair_prompt_calls = 0
    length_limited_candidates = 0

    for model, model_directory in MODELS.items():
        for method_directory, (result_key, method_label) in METHODS.items():
            method_rows: list[dict[str, Any]] = []
            files = sorted((RESULT_ROOT / model_directory / method_directory).glob("batch_*.json"))
            if len(files) != 12:
                issues.append(f"{model}/{method_label}: expected 12 batch files, found {len(files)}")
            for batch_index, path in enumerate(files):
                data = _read_json(path)
                metadata = data.get("metadata") or {}
                batch_manifest = ROOT / str(metadata.get("dataset_path") or "")
                expected_batch = ROOT / f"data/manifests/execrepobench_opencoderx_120_batches_v1/batch_{batch_index:02d}.jsonl"
                if batch_manifest != expected_batch:
                    issues.append(f"{path.relative_to(ROOT)}: unexpected batch manifest")
                if metadata.get("model") != model:
                    issues.append(f"{path.relative_to(ROOT)}: model metadata mismatch")
                expected_temperature = None if model == "claude-sonnet-5" else 0.7
                if metadata.get("temperature") != expected_temperature:
                    issues.append(f"{path.relative_to(ROOT)}: temperature mismatch")
                if metadata.get("initial_candidate_budget") != 5 or metadata.get("n_samples_for_uncertainty") != 5:
                    issues.append(f"{path.relative_to(ROOT)}: candidate budget mismatch")
                if metadata.get("max_repair_rounds") != 2:
                    issues.append(f"{path.relative_to(ROOT)}: repair-round budget mismatch")
                if metadata.get("retrieval_budget") != {
                    "api_top_k": 8, "context_top_k": 8, "similar_code_top_k": 8, "fused_top_k": 10
                }:
                    issues.append(f"{path.relative_to(ROOT)}: retrieval budget mismatch")
                rows = list(data.get(result_key) or [])
                other_count = sum(len(data.get(key) or []) for key in {"direct", "without", "rag_repair", "with"} - {result_key})
                if other_count:
                    issues.append(f"{path.relative_to(ROOT)}: contains rows for an undeclared method")
                method_rows.extend(rows)
                raw_files.append({
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                    "created_at": metadata.get("created_at"),
                    "model": model,
                    "method": method_label,
                    "rows": len(rows),
                })

            ids = [str(row.get("id")) for row in method_rows]
            if len(ids) != len(set(ids)):
                issues.append(f"{model}/{method_label}: duplicate task IDs")
            if set(ids) != expected_set or len(ids) != 120:
                issues.append(f"{model}/{method_label}: task set differs from the frozen manifest")

            for row in method_rows:
                task_id = str(row.get("id"))
                prefix = f"{model}/{method_label}/{task_id}"
                all_rows[model][method_label, task_id] = row
                if row.get("error"):
                    issues.append(f"{prefix}: run error")
                samples = list(row.get("generated_samples") or [])
                outcomes = list(row.get("sample_correctness") or [])
                effective = list(row.get("effective_sample_correctness") or [])
                if len(samples) != 5 or len(outcomes) != 5 or len(effective) != 5:
                    issues.append(f"{prefix}: expected five samples and five outcomes")
                if (row.get("generation_integrity") or {}).get("valid") is not True:
                    issues.append(f"{prefix}: generation-integrity flag failed")
                length_limited_candidates += int((row.get("generation_integrity") or {}).get("n_length_limited_candidates") or 0)
                if row.get("correctness_mode") != "repository_tests":
                    issues.append(f"{prefix}: non-executable correctness mode")
                final_passed = (row.get("test_report") or {}).get("passed")
                if not isinstance(final_passed, bool) or bool(row.get("passed")) != final_passed:
                    issues.append(f"{prefix}: missing or inconsistent selected-output test result")
                if row.get("passed") is True and not (row.get("static_report") or {}).get("ok"):
                    issues.append(f"{prefix}: passing output failed static validation")
                stored_passk = row.get("pass_at_k") or {}
                for k in KS:
                    recomputed = _pass_at_k(outcomes, k) if len(outcomes) == 5 else None
                    if recomputed is not None and not math.isclose(float(stored_passk.get(f"pass@{k}", -1)), recomputed, abs_tol=1e-12):
                        issues.append(f"{prefix}: stored pass@{k} does not match raw candidates")
                rounds = int(row.get("repair_rounds") or 0)
                history = list(row.get("repair_history") or [])
                if rounds != len(history) or rounds > 2:
                    issues.append(f"{prefix}: repair history does not match the frozen limit")
                if method_label in {"Direct Generation", "Standard RAG"} and rounds:
                    issues.append(f"{prefix}: non-repair method used repair")
                evidence = list(row.get("fused_evidence_ids") or [])
                if method_label == "Direct Generation" and evidence:
                    issues.append(f"{prefix}: direct generation contains retrieved evidence")
                if method_label != "Direct Generation" and not evidence:
                    issues.append(f"{prefix}: retrieval method has no evidence")
                for repair in history:
                    repair_prompt_calls += 1
                    audit = repair.get("prompt_budget") or {}
                    if audit.get("policy") != REPAIR_POLICY:
                        issues.append(f"{prefix}: repair prompt policy mismatch")
                    limits = audit.get("limits") or {}
                    if limits != REPAIR_LIMITS:
                        issues.append(f"{prefix}: repair prompt limits mismatch")
                    for name, limit_name in (("task", "task_chars"), ("code", "code_chars"), ("diagnostics", "diagnostics_chars")):
                        used = int(audit.get(f"{name}_used_chars") or 0)
                        if used > REPAIR_LIMITS[limit_name]:
                            issues.append(f"{prefix}: repair {name} exceeds the frozen character limit")

                audit_calls = list(row.get("provider_response_audit") or [])
                requests = int((row.get("llm_usage") or {}).get("requests") or 0)
                if len(audit_calls) != requests or not audit_calls:
                    issues.append(f"{prefix}: provider response audit does not match request count")
                for call in audit_calls:
                    response_id = str(call.get("response_id") or "")
                    if not response_id:
                        issues.append(f"{prefix}: missing provider response ID")
                    elif response_id in response_ids:
                        duplicate_response_ids.add(response_id)
                    else:
                        response_ids.add(response_id)
                    if call.get("requested_model") != model or call.get("served_model") != model:
                        issues.append(f"{prefix}: requested/served model mismatch")

                manifest_item = task_metadata.get(task_id) or {}
                initial_passed = bool((row.get("initial_test_report") or {}).get("passed"))
                post_selection_passed = bool((row.get("post_selection_test_report") or {}).get("passed"))
                task_level.append({
                    "task_id": task_id,
                    "repository": manifest_item.get("repo_name"),
                    "model": model,
                    "method": method_label,
                    "candidate_correct_count": sum(bool(value) for value in outcomes),
                    "pass_at_1": _pass_at_k(outcomes, 1) if len(outcomes) == 5 else None,
                    "pass_at_3": _pass_at_k(outcomes, 3) if len(outcomes) == 5 else None,
                    "pass_at_5": _pass_at_k(outcomes, 5) if len(outcomes) == 5 else None,
                    "selected_output_correct": bool(row.get("passed")),
                    "static_success": bool((row.get("static_report") or {}).get("ok")),
                    "initial_output_correct": initial_passed,
                    "post_selection_correct": post_selection_passed,
                    "verified_selection_applied": bool(row.get("verified_selection_applied")),
                    "repair_attempted": rounds > 0,
                    "repair_success": rounds > 0 and not post_selection_passed and bool(row.get("passed")),
                    "repair_rounds": rounds,
                    "uncertainty": float((row.get("u") or {}).get("aggregate")),
                    "token_entropy": float((row.get("u") or {}).get("token_entropy")),
                    "self_consistency_uncertainty": float((row.get("u") or {}).get("self_consistency")),
                    "semantic_variance": float((row.get("u") or {}).get("semantic_variance")),
                    "evidence_count": len(evidence),
                    "prompt_tokens": int((row.get("llm_usage") or {}).get("prompt_tokens") or 0),
                    "completion_tokens": int((row.get("llm_usage") or {}).get("completion_tokens") or 0),
                    "total_tokens": int((row.get("llm_usage") or {}).get("total_tokens") or 0),
                    "provider_requests": requests,
                    "latency_seconds": float(row.get("run_latency_s") or 0.0),
                    "length_limited_candidates": int((row.get("generation_integrity") or {}).get("n_length_limited_candidates") or 0),
                })

    if duplicate_response_ids:
        issues.append(f"duplicate provider response IDs: {len(duplicate_response_ids)}")
    if len(task_level) != 1_920:
        issues.append(f"expected 1,920 task-method-model cells, found {len(task_level)}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_level:
        grouped[(str(row["model"]), str(row["method"]))].append(row)

    summary: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for model in MODELS:
        for _, (_, method_label) in METHODS.items():
            rows = grouped[(model, method_label)]
            attempts = [row for row in rows if row["repair_attempted"]]
            summary.append({
                "model": model,
                "method": method_label,
                "tasks": len(rows),
                "pass_at_1": _mean(float(row["pass_at_1"]) for row in rows),
                "pass_at_3": _mean(float(row["pass_at_3"]) for row in rows),
                "pass_at_5": _mean(float(row["pass_at_5"]) for row in rows),
                "selected_output_correctness": _mean(float(row["selected_output_correct"]) for row in rows),
                "static_success": _mean(float(row["static_success"]) for row in rows),
                "mean_uncertainty": _mean(float(row["uncertainty"]) for row in rows),
                "mean_tokens": _mean(float(row["total_tokens"]) for row in rows),
                "mean_latency_seconds": _mean(float(row["latency_seconds"]) for row in rows),
                "mean_repair_rounds": _mean(float(row["repair_rounds"]) for row in rows),
            })
            usage = _usage([all_rows[model][method_label, task_id] for task_id in expected_ids])
            estimated_cost, currency = _price(model, usage["prompt_tokens"], usage["completion_tokens"])
            resources.append({
                "model": model,
                "method": method_label,
                "tasks": len(rows),
                **usage,
                "mean_tokens": _mean(float(row["total_tokens"]) for row in rows),
                "mean_latency_seconds": _mean(float(row["latency_seconds"]) for row in rows),
                "mean_repair_rounds": _mean(float(row["repair_rounds"]) for row in rows),
                "estimated_cost": estimated_cost,
                "currency": currency,
            })
            risk = [float(row["uncertainty"]) for row in rows]
            failures = [int(not row["selected_output_correct"]) for row in rows]
            intercept, slope = _calibration_fit(risk, failures)
            uncertainty.append({
                "model": model,
                "method": method_label,
                "tasks": len(rows),
                "failure_rate": _mean(failures),
                "mean_uncertainty": _mean(risk),
                "auroc_failure": float(roc_auc_score(failures, risk)) if len(set(failures)) > 1 else None,
                "auprc_failure": float(average_precision_score(failures, risk)) if len(set(failures)) > 1 else None,
                "brier_failure": _mean((prediction - target) ** 2 for prediction, target in zip(risk, failures)),
                "ece_failure_10_bins": _ece(risk, failures),
                "aurc": _aurc(risk, failures),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            })
            repairs.append({
                "model": model,
                "method": method_label,
                "tasks": len(rows),
                "repair_attempted_tasks": len(attempts),
                "repair_attempt_rate": len(attempts) / len(rows),
                "repair_success_tasks": sum(bool(row["repair_success"]) for row in attempts),
                "repair_success_rate_given_attempt": (
                    _mean(float(row["repair_success"]) for row in attempts) if attempts else None
                ),
                "mean_rounds_given_attempt": (
                    _mean(float(row["repair_rounds"]) for row in attempts) if attempts else None
                ),
            })

    paired_passk: list[dict[str, Any]] = []
    paired_selected: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        open_rows = {str(row["task_id"]): row for row in grouped[(model, "OpenCoderX")]}
        for comparator_index, comparator in enumerate(COMPARATORS):
            comp_rows = {str(row["task_id"]): row for row in grouped[(model, comparator)]}
            if set(open_rows) != expected_set or set(comp_rows) != expected_set:
                issues.append(f"{model}/{comparator}: paired task mismatch")
                continue
            for metric_index, metric in enumerate(("pass_at_1", "pass_at_3", "pass_at_5")):
                comp = [float(comp_rows[task_id][metric]) for task_id in expected_ids]
                op = [float(open_rows[task_id][metric]) for task_id in expected_ids]
                low, high = _paired_bootstrap_ci(
                    comp,
                    op,
                    seed_offset=100 * model_index + 10 * comparator_index + metric_index,
                )
                paired_passk.append({
                    "model": model,
                    "comparator": comparator,
                    "metric": metric.replace("pass_at_", "pass@"),
                    "comparator_value": _mean(comp),
                    "opencoder_value": _mean(op),
                    "absolute_difference": _mean(op) - _mean(comp),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "opencoder_wins": sum(o > c for c, o in zip(comp, op)),
                    "opencoder_losses": sum(o < c for c, o in zip(comp, op)),
                    "ties": sum(o == c for c, o in zip(comp, op)),
                    "matched_tasks": len(expected_ids),
                })
            comp = [float(comp_rows[task_id]["selected_output_correct"]) for task_id in expected_ids]
            op = [float(open_rows[task_id]["selected_output_correct"]) for task_id in expected_ids]
            wins = sum(o > c for c, o in zip(comp, op))
            losses = sum(o < c for c, o in zip(comp, op))
            low, high = _paired_bootstrap_ci(comp, op, seed_offset=1_000 + 100 * model_index + comparator_index)
            paired_selected.append({
                "model": model,
                "comparator": comparator,
                "comparator_correctness": _mean(comp),
                "opencoder_correctness": _mean(op),
                "absolute_difference": _mean(op) - _mean(comp),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "opencoder_wins": wins,
                "opencoder_losses": losses,
                "ties": len(expected_ids) - wins - losses,
                "mcnemar_exact_p": _mcnemar_exact(wins, losses),
                "mcnemar_holm_p": None,
                "matched_tasks": len(expected_ids),
            })

    for comparator in COMPARATORS:
        family = [row for row in paired_selected if row["comparator"] == comparator]
        adjusted = _holm_adjust([float(row["mcnemar_exact_p"]) for row in family])
        for row, value in zip(family, adjusted):
            row["mcnemar_holm_p"] = value

    _write_csv(OUTPUT_ROOT / "task_level.csv", task_level)
    _write_csv(OUTPUT_ROOT / "summary.csv", summary)
    _write_csv(OUTPUT_ROOT / "paired_passk_statistics.csv", paired_passk)
    _write_csv(OUTPUT_ROOT / "selected_output_statistics.csv", paired_selected)
    _write_csv(OUTPUT_ROOT / "resource_summary.csv", resources)
    _write_csv(OUTPUT_ROOT / "uncertainty_calibration.csv", uncertainty)
    _write_csv(OUTPUT_ROOT / "repair_summary.csv", repairs)
    _build_main_table(OUTPUT_ROOT / "latex/table_exec120_effectiveness.tex", summary)
    _build_matched_table(OUTPUT_ROOT / "latex/table_exec120_matched.tex", paired_selected)

    campaign_cost = defaultdict(float)
    for row in resources:
        campaign_cost[str(row["currency"])] += float(row["estimated_cost"])
    provenance = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": _sha256(PROTOCOL),
        "scientific_protocol_sha256": protocol.get("scientific_protocol_sha256"),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "retrieval_leakage_report": str(LEAKAGE_REPORT.relative_to(ROOT)),
        "retrieval_leakage_report_sha256": _sha256(LEAKAGE_REPORT),
        "frozen_hash_verification": frozen_hashes,
        "raw_result_files": raw_files,
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "candidate_metric_definition": "Unbiased Pass@k over exactly five original generated candidates per task.",
        "selected_metric_definition": "Final executable-test outcome after the method's declared candidate selection and repair policy.",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "base_seed": BOOTSTRAP_SEED, "unit": "task"},
        "multiple_testing": "Holm adjustment across the four model backends within each selected-output comparator family.",
    }
    (OUTPUT_ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    integrity = {
        "passed": not issues,
        "paper_eligible": not issues,
        "task_count": len(expected_ids),
        "model_count": len(MODELS),
        "method_count": len(METHODS),
        "task_method_model_cells": len(task_level),
        "raw_batch_files": len(raw_files),
        "audited_provider_responses": len(response_ids),
        "duplicate_provider_response_ids": len(duplicate_response_ids),
        "repair_prompt_calls": repair_prompt_calls,
        "length_limited_candidates_scored_as_failures": length_limited_candidates,
        "retrieval_index_documents": leakage.get("index_documents"),
        "retrieval_leakage_failures": leakage.get("leakage_failures"),
        "confirmatory_result_cost_only": dict(campaign_cost),
        "campaign_spend_including_pilots_and_superseded_runs": campaign_status.get("campaign_spend"),
        "issues": issues,
        "warnings": warnings,
    }
    (OUTPUT_ROOT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    memo = [
        "# ExecRepoBench-120 Confirmatory Results",
        "",
        f"Integrity status: **{'PASSED' if not issues else 'FAILED'}**.",
        "",
        "Candidate Pass@k uses only the five original samples. Selected-output correctness is a separate endpoint that includes each method's declared verified-selection and repair behavior.",
        "",
        "## Effectiveness",
        "",
        "| Model | Method | Pass@1 | Pass@3 | Pass@5 | Selected |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        memo.append(
            f"| {row['model']} | {row['method']} | {_fmt_percent(row['pass_at_1'])} | "
            f"{_fmt_percent(row['pass_at_3'])} | {_fmt_percent(row['pass_at_5'])} | "
            f"{_fmt_percent(row['selected_output_correctness'])} |"
        )
    memo.extend(["", "## Matched Control", ""])
    for row in paired_selected:
        if row["comparator"] != "RAG + Verify/Repair":
            continue
        memo.append(
            f"- {row['model']}: OpenCoderX {100*row['absolute_difference']:+.1f} percentage points; "
            f"95% CI [{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}], "
            f"W/L/T={row['opencoder_wins']}/{row['opencoder_losses']}/{row['ties']}, "
            f"exact McNemar p={row['mcnemar_exact_p']:.3f}, Holm p={row['mcnemar_holm_p']:.3f}."
        )
    memo.extend([
        "",
        "## Interpretation",
        "",
        "The matched comparison isolates uncertainty-aware decomposition, filtering, fusion, and generation from ordinary verification/repair. Results are backend dependent: OpenCoderX does not universally outperform the matched control, and significance is claimed only where the adjusted paired test supports it.",
        "",
        f"The audit covered {len(response_ids):,} successful provider responses and {repair_prompt_calls:,} repair prompts. All length-limited candidates ({length_limited_candidates:,}) remain failures under the frozen integrity policy rather than being silently dropped.",
        "",
    ])
    (OUTPUT_ROOT / "results_memo.md").write_text("\n".join(memo), encoding="utf-8")

    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
