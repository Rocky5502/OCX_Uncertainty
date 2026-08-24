#!/usr/bin/env python3
"""Analyze selective autonomy and collaboration from frozen TOSEM records.

This script performs no model calls. Observed executable outcomes and simulated
reviewer outcomes are written to separate artifacts so simulated collaboration
results cannot be mistaken for human-subject evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from opencoderx.collaboration import (
    CollaborationPolicy,
    RiskTrace,
    allocate_review_budget,
    review_allocation_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
EXEC_RAW = ROOT / "results/tosem/confirmatory"
EXEC_ANALYSIS = ROOT / "results/tosem/confirmatory_analysis"
CROSS_ANALYSIS = ROOT / "results/tosem/crosscodeeval_confirmatory"
OUTPUT = ROOT / "results/tosem/collaboration_analysis"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
PROTOCOL = ROOT / "results/tosem/protocol_freeze.json"
CAMPAIGN = ROOT / "configs/tosem/campaign.yaml"

MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = (
    "Direct Generation",
    "Standard RAG",
    "RAG + Verify/Repair",
    "OpenCoderX",
)
POLICIES = (
    "always_autonomous",
    "random_deferral",
    "test_failure_deferral",
    "confidence_deferral",
    "entropy_deferral",
    "self_consistency_deferral",
    "aggregate_uncertainty_deferral",
    "source_specific_opencoderx_deferral",
    "oracle_deferral",
)
POLICY_LABELS = {
    "always_autonomous": "Always autonomous",
    "random_deferral": "Random",
    "test_failure_deferral": "Pre-repair test failure",
    "confidence_deferral": "Semantic confidence",
    "entropy_deferral": "Token entropy",
    "self_consistency_deferral": "Self-consistency",
    "aggregate_uncertainty_deferral": "Aggregate uncertainty",
    "source_specific_opencoderx_deferral": "Source-specific OpenCoderX",
    "oracle_deferral": "Oracle (upper bound)",
}
BOOTSTRAPS = 10_000
SEED = 20260809


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"field names required for empty artifact: {path}")
        fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def f(value: Any) -> float:
    return float(value)


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def avg(values: Iterable[float]) -> float:
    items = list(values)
    return float(mean(items)) if items else 0.0


def source_risk(stats: Mapping[str, Any] | None) -> float:
    """Scale-independent within-source instability in [0, 1]."""
    if not stats or int(stats.get("n_retrieved") or 0) <= 0:
        return 1.0
    top = float(stats.get("mean_top_retrieval_score") or 0.0)
    center = float(stats.get("mean_retrieval_score") or 0.0)
    concentration_loss = 1.0 if top <= 0.0 else 1.0 - max(0.0, min(1.0, center / top))
    retrieved = int(stats.get("n_retrieved") or 0)
    kept = int(stats.get("n_kept") or 0)
    filtering_loss = 1.0 - max(0.0, min(1.0, kept / retrieved))
    return float((concentration_loss + filtering_loss) / 2.0)


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def bootstrap_delta(before: Sequence[float], after: Sequence[float], offset: int) -> tuple[float, float]:
    left = np.asarray(before, dtype=float)
    right = np.asarray(after, dtype=float)
    if left.shape != right.shape or not len(left):
        raise ValueError("bootstrap data must be non-empty and matched")
    rng = np.random.default_rng(SEED + offset)
    estimates = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 1_000):
        stop = min(BOOTSTRAPS, start + 1_000)
        indices = rng.integers(0, len(left), size=(stop - start, len(left)))
        estimates[start:stop] = np.mean(right[indices] - left[indices], axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def bootstrap_stat(
    rows: Sequence[Mapping[str, Any]], statistic: Callable[[Sequence[Mapping[str, Any]]], float | None], offset: int
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    rng = np.random.default_rng(SEED + offset)
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        value = statistic(sample)
        if value is not None:
            values.append(float(value))
    if not values:
        return None, None
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}"


def load_open_records() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    raw_files: list[str] = []
    for model, directory in MODELS.items():
        for path in sorted((EXEC_RAW / directory / "opencoder").glob("batch_*.json")):
            raw_files.append(str(path.relative_to(ROOT)))
            for raw in read_json(path).get("with", []):
                diagnostics = raw.get("source_diagnostics") or {}
                risks = {source: source_risk(diagnostics.get(source)) for source in ("api", "context", "similar_code")}
                uncertainty = raw.get("u") or {}
                post_passed = bool((raw.get("post_selection_test_report") or {}).get("passed"))
                final_passed = bool(raw.get("passed"))
                rounds = int(raw.get("repair_rounds") or 0)
                repair_risk = 0.0 if not rounds else (0.5 * rounds / 2.0 if final_passed else 1.0)
                evidence = tuple(
                    str(item.get("id")) for item in (raw.get("fused_evidence_ids") or []) if item.get("id")
                )
                trace = RiskTrace(
                    api=risks["api"],
                    context=risks["context"],
                    similar_code=risks["similar_code"],
                    generation=float(uncertainty.get("aggregate") or 0.0),
                    verification=0.0 if post_passed else 1.0,
                    repair=repair_risk,
                    evidence_ids=evidence,
                )
                records.append({
                    "task_id": str(raw["id"]),
                    "model": model,
                    "selected_output_correct": final_passed,
                    "pre_repair_test_failed": not post_passed,
                    "initial_output_correct": bool((raw.get("initial_test_report") or {}).get("passed")),
                    "post_selection_correct": post_passed,
                    "repair_rounds": rounds,
                    "repair_attempted": rounds > 0,
                    "repair_success": rounds > 0 and not post_passed and final_passed,
                    "generation_uncertainty": float(uncertainty.get("semantic_variance") or 0.0),
                    "token_entropy": float(uncertainty.get("token_entropy") or 0.0),
                    "candidate_disagreement": float(uncertainty.get("self_consistency") or 0.0),
                    "aggregate_risk": float(uncertainty.get("aggregate") or 0.0),
                    "uncertainty_sources": risks,
                    "risk_trace": trace,
                    "evidence_ids": evidence,
                    "tokens": int((raw.get("llm_usage") or {}).get("total_tokens") or 0),
                    "latency_seconds": float(raw.get("run_latency_s") or 0.0),
                    "irrecoverable": not final_passed and rounds >= 2,
                })
    return records, raw_files


def make_decisions(records: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = {
        "uncertainty_only": CollaborationPolicy(
            name="source_and_generation_uncertainty_only",
            review_threshold=float(config["review_threshold"]),
            abstain_threshold=float(config["abstain_threshold"]),
            weights={
                "api": 1.0,
                "context": 1.0,
                "similar_code": 1.0,
                "generation": 1.0,
                "verification": 0.0,
                "repair": 0.0,
            },
        ),
        "lifecycle_gated": CollaborationPolicy(
            name="full_lifecycle_with_executable_gate",
            review_threshold=float(config["review_threshold"]),
            abstain_threshold=float(config["abstain_threshold"]),
        ),
    }
    decisions: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        uncertainty_trace = RiskTrace(
            api=record["uncertainty_sources"]["api"],
            context=record["uncertainty_sources"]["context"],
            similar_code=record["uncertainty_sources"]["similar_code"],
            generation=record["aggregate_risk"],
            evidence_ids=record["evidence_ids"],
        )
        for policy_name, policy in policies.items():
            trace = uncertainty_trace if policy_name == "uncertainty_only" else record["risk_trace"]
            decision = policy.decide(
                task_id=record["task_id"], model=record["model"], trace=trace,
                irrecoverable=record["irrecoverable"] if policy_name == "lifecycle_gated" else False,
            ).to_dict()
            decision.update({
                "decision_policy": policy_name,
                "selected_output_correct": record["selected_output_correct"],
                "pre_repair_test_failed": record["pre_repair_test_failed"],
                "repair_rounds": record["repair_rounds"],
                "uses_executable_validation": policy_name == "lifecycle_gated",
                "simulation": False,
            })
            decisions.append(decision)
            by_group[(policy_name, record["model"])].append(decision)

    summary: list[dict[str, Any]] = []
    for policy_index, policy_name in enumerate(policies):
      for model_index, model in enumerate(MODELS):
        rows = by_group[(policy_name, model)]
        for decision_name in ("autonomous", "request_review", "abstain"):
            selected = [row for row in rows if row["decision"] == decision_name]
            summary.append({
                "decision_policy": policy_name,
                "model": model,
                "decision": decision_name,
                "tasks": len(selected),
                "task_share": len(selected) / len(rows),
                "correct_tasks": sum(bool(row["selected_output_correct"]) for row in selected),
                "correctness": avg(float(row["selected_output_correct"]) for row in selected) if selected else None,
                "failure_rate": avg(float(not row["selected_output_correct"]) for row in selected) if selected else None,
                "mean_risk": avg(float(row["risk_score"]) for row in selected) if selected else None,
            })
        autonomous = [row for row in rows if row["decision"] == "autonomous"]
        intervened = [row for row in rows if row["decision"] != "autonomous"]
        failures = [row for row in rows if not row["selected_output_correct"]]

        def autonomous_accuracy(sample: Sequence[Mapping[str, Any]]) -> float | None:
            kept = [row for row in sample if row["decision"] == "autonomous"]
            return avg(float(row["selected_output_correct"]) for row in kept) if kept else None

        def failure_capture(sample: Sequence[Mapping[str, Any]]) -> float | None:
            failed = [row for row in sample if not row["selected_output_correct"]]
            return avg(float(row["decision"] != "autonomous") for row in failed) if failed else None

        auto_ci = bootstrap_stat(rows, autonomous_accuracy, 100 + 1000 * policy_index + model_index)
        capture_ci = bootstrap_stat(rows, failure_capture, 200 + 1000 * policy_index + model_index)
        risk_values = [float(row["risk_score"]) for row in rows]
        failure_values = [int(not row["selected_output_correct"]) for row in rows]
        summary.append({
            "decision_policy": policy_name,
            "model": model,
            "decision": "ALL_POLICY_METRICS",
            "tasks": len(rows),
            "task_share": 1.0,
            "correct_tasks": sum(bool(row["selected_output_correct"]) for row in rows),
            "correctness": avg(float(row["selected_output_correct"]) for row in rows),
            "failure_rate": avg(float(not row["selected_output_correct"]) for row in rows),
            "mean_risk": avg(float(row["risk_score"]) for row in rows),
            "autonomous_coverage": len(autonomous) / len(rows),
            "autonomous_accuracy": autonomous_accuracy(rows),
            "autonomous_accuracy_ci_low": auto_ci[0],
            "autonomous_accuracy_ci_high": auto_ci[1],
            "intervention_rate": len(intervened) / len(rows),
            "failure_capture_rate": len([row for row in failures if row["decision"] != "autonomous"]) / len(failures),
            "failure_capture_ci_low": capture_ci[0],
            "failure_capture_ci_high": capture_ci[1],
            "failure_auroc": float(roc_auc_score(failure_values, risk_values)),
            "failure_auprc": float(average_precision_score(failure_values, risk_values)),
        })
    return decisions, summary


def review_simulations(records: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    by_model = {model: [row for row in records if row["model"] == model] for model in MODELS}
    for model in MODELS:
        rows = by_model[model]
        for policy in POLICIES:
            for budget in config["review_budgets"]:
                for success in config["reviewer_success"]:
                    for seed in config["seeds"]:
                        selected = allocate_review_budget(rows, policy=policy, review_budget=float(budget), seed=int(seed))
                        metrics = review_allocation_metrics(
                            rows, selected, reviewer_success=float(success), seed=int(seed)
                        )
                        raw_rows.append({
                            "model": model,
                            "policy": policy,
                            "policy_label": POLICY_LABELS[policy],
                            "review_budget": float(budget),
                            "reviewer_success": float(success),
                            "seed": int(seed),
                            "tasks": len(rows),
                            "reviewed_tasks": len(selected),
                            "actual_review_rate": len(selected) / len(rows),
                            "simulation": True,
                            **metrics,
                        })

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (row["model"], row["policy"], row["review_budget"], row["reviewer_success"])
        grouped[key].append(row)
    metrics = (
        "team_success_rate", "selective_accuracy", "autonomous_coverage", "autonomous_failure_rate",
        "failure_capture_rate", "deferral_precision", "unnecessary_review_rate",
        "errors_prevented_per_review", "reviews_per_prevented_failure",
    )
    summary: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        out: dict[str, Any] = {
            "model": key[0], "policy": key[1], "policy_label": POLICY_LABELS[str(key[1])],
            "review_budget": key[2], "reviewer_success": key[3], "seeds": len(rows),
            "tasks": rows[0]["tasks"], "mean_reviewed_tasks": avg(float(row["reviewed_tasks"]) for row in rows),
            "mean_actual_review_rate": avg(float(row["actual_review_rate"]) for row in rows),
            "simulation": True,
        }
        for metric in metrics:
            finite = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            out[f"mean_{metric}"] = avg(finite) if finite else None
            out[f"sd_{metric}"] = pstdev(finite) if finite else None
            out[f"min_{metric}"] = min(finite) if finite else None
            out[f"max_{metric}"] = max(finite) if finite else None
        summary.append(out)
    return raw_rows, summary


def paired_effect(
    model: str, label: str, scope: str, before_rows: Mapping[str, Mapping[str, Any]],
    after_rows: Mapping[str, Mapping[str, Any]], before_field: str, after_field: str, offset: int,
    uncertainty_fields: tuple[str, str] | None = None,
) -> dict[str, Any]:
    task_ids = sorted(set(before_rows) & set(after_rows))
    before = [float(before_rows[task][before_field]) for task in task_ids]
    after = [float(after_rows[task][after_field]) for task in task_ids]
    wins = sum(right > left for left, right in zip(before, after))
    losses = sum(right < left for left, right in zip(before, after))
    low, high = bootstrap_delta(before, after, offset)
    out: dict[str, Any] = {
        "model": model, "intervention": label, "scope": scope, "status": "OBSERVED",
        "matched_tasks": len(task_ids), "before_correctness": avg(before), "after_correctness": avg(after),
        "absolute_difference": avg(after) - avg(before), "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high, "wins": wins, "losses": losses,
        "ties": len(task_ids) - wins - losses, "mcnemar_exact_p": exact_mcnemar(wins, losses),
        "simulation": False,
    }
    if uncertainty_fields:
        out["before_uncertainty"] = avg(float(before_rows[task][uncertainty_fields[0]]) for task in task_ids)
        out["after_uncertainty"] = avg(float(after_rows[task][uncertainty_fields[1]]) for task in task_ids)
        out["uncertainty_difference"] = out["after_uncertainty"] - out["before_uncertainty"]
    else:
        out["uncertainty_note"] = "Generation uncertainty was not recomputed after this within-run intervention."
    return out


def intervention_analysis(
    exec_rows: Sequence[dict[str, str]], open_records: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in exec_rows:
        grouped[(row["model"], row["method"])][row["task_id"]] = {
            **row,
            "selected_output_correct": b(row["selected_output_correct"]),
            "initial_output_correct": b(row["initial_output_correct"]),
            "post_selection_correct": b(row["post_selection_correct"]),
            "uncertainty": f(row["uncertainty"]),
            "total_tokens": f(row["total_tokens"]),
            "latency_seconds": f(row["latency_seconds"]),
        }
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        direct = grouped[(model, "Direct Generation")]
        rag = grouped[(model, "Standard RAG")]
        control = grouped[(model, "RAG + Verify/Repair")]
        opencoder = grouped[(model, "OpenCoderX")]
        comparisons = [
            ("Repository evidence provision", "Direct -> Standard RAG", direct, rag, "selected_output_correct", "selected_output_correct", ("uncertainty", "uncertainty")),
            ("Ordinary verification and repair", "Standard RAG -> RAG + Verify/Repair", rag, control, "selected_output_correct", "selected_output_correct", ("uncertainty", "uncertainty")),
            ("Verified candidate selection", "Within RAG + Verify/Repair", control, control, "initial_output_correct", "post_selection_correct", None),
            ("Executable repair", "Within RAG + Verify/Repair", control, control, "post_selection_correct", "selected_output_correct", None),
            ("Uncertainty-aware evidence and generation", "RAG + Verify/Repair -> OpenCoderX", control, opencoder, "selected_output_correct", "selected_output_correct", ("uncertainty", "uncertainty")),
            ("Verified candidate selection", "Within OpenCoderX", opencoder, opencoder, "initial_output_correct", "post_selection_correct", None),
            ("Executable repair", "Within OpenCoderX", opencoder, opencoder, "post_selection_correct", "selected_output_correct", None),
        ]
        for comparison_index, values in enumerate(comparisons):
            effect = paired_effect(
                model,
                *values[:6],
                offset=100 * model_index + comparison_index,
                uncertainty_fields=values[6],
            )
            if values[2] is not values[3]:
                ids = sorted(set(values[2]) & set(values[3]))
                effect["mean_token_difference"] = avg(
                    float(values[3][task]["total_tokens"]) - float(values[2][task]["total_tokens"]) for task in ids
                )
                effect["mean_latency_difference_seconds"] = avg(
                    float(values[3][task]["latency_seconds"]) - float(values[2][task]["latency_seconds"]) for task in ids
                )
            rows.append(effect)
    for source in ("API evidence correction", "Context correction", "Similar-code correction"):
        rows.append({
            "model": "all", "intervention": source, "scope": "Counterfactual correction",
            "status": "NOT_RUN", "simulation": False,
            "uncertainty_note": "No matched correction run exists; no effectiveness value is reported.",
        })

    observed = [row for row in rows if row["status"] == "OBSERVED"]
    for intervention in sorted({str(row["intervention"]) for row in observed}):
        family = [row for row in observed if row["intervention"] == intervention]
        adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in family])
        for row, value in zip(family, adjusted):
            row["mcnemar_holm_p"] = value

    repair_sources: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in open_records if row["model"] == model and row["repair_attempted"]]
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in model_rows:
            highest = max(("api", "context", "similar_code"), key=lambda source: row["uncertainty_sources"][source])
            by_source[highest].append(row)
        for source in ("api", "context", "similar_code"):
            selected = by_source[source]
            repair_sources.append({
                "model": model, "highest_source_risk": source, "repair_attempts": len(selected),
                "repair_successes": sum(bool(row["repair_success"]) for row in selected),
                "repair_success_rate": avg(float(row["repair_success"]) for row in selected) if selected else None,
                "mean_source_risk": avg(float(row["uncertainty_sources"][source]) for row in selected) if selected else None,
                "simulation": False,
            })
    return rows, repair_sources


def generalizability(
    exec_rows: Sequence[dict[str, str]], cross_rows: Sequence[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    language: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for language_index, language_name in enumerate(("python", "java", "typescript", "csharp")):
            direct = {row["task_id"]: row for row in cross_rows if row["model"] == model and row["method"] == "Direct Generation" and row["language"] == language_name}
            context = {row["task_id"]: row for row in cross_rows if row["model"] == model and row["method"] == "Cross-file Context RAG" and row["language"] == language_name}
            ids = sorted(set(direct) & set(context))
            before = [f(direct[task]["selected_exact_match"]) for task in ids]
            after = [f(context[task]["selected_exact_match"]) for task in ids]
            wins = sum(right > left for left, right in zip(before, after))
            losses = sum(right < left for left, right in zip(before, after))
            low, high = bootstrap_delta(before, after, 1_000 + 100 * model_index + language_index)
            language.append({
                "model": model, "language": language_name, "tasks": len(ids),
                "direct_exact_match": avg(before), "context_exact_match": avg(after),
                "exact_match_difference": avg(after) - avg(before), "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high, "wins": wins, "losses": losses,
                "ties": len(ids) - wins - losses, "mcnemar_exact_p": exact_mcnemar(wins, losses),
                "direct_edit_similarity": avg(f(direct[task]["selected_edit_similarity"]) for task in ids),
                "context_edit_similarity": avg(f(context[task]["selected_edit_similarity"]) for task in ids),
                "direct_identifier_f1": avg(f(direct[task]["selected_identifier_f1"]) for task in ids),
                "context_identifier_f1": avg(f(context[task]["selected_identifier_f1"]) for task in ids),
                "functional_execution": False,
            })

    for model in MODELS:
        family = [row for row in language if row["model"] == model]
        adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in family])
        for row, value in zip(family, adjusted):
            row["mcnemar_holm_p"] = value

    repository: list[dict[str, Any]] = []
    for model in MODELS:
        control = {row["task_id"]: row for row in exec_rows if row["model"] == model and row["method"] == "RAG + Verify/Repair"}
        opencoder = {row["task_id"]: row for row in exec_rows if row["model"] == model and row["method"] == "OpenCoderX"}
        by_repo: dict[str, list[str]] = defaultdict(list)
        for task in set(control) & set(opencoder):
            by_repo[control[task]["repository"]].append(task)
        for repo, ids in sorted(by_repo.items()):
            before = [float(b(control[task]["selected_output_correct"])) for task in ids]
            after = [float(b(opencoder[task]["selected_output_correct"])) for task in ids]
            repository.append({
                "model": model, "repository": repo, "tasks": len(ids),
                "control_correctness": avg(before), "opencoder_correctness": avg(after),
                "absolute_difference": avg(after) - avg(before),
                "descriptive_only": True,
            })

    model_summary: list[dict[str, Any]] = []
    exec_summary = read_csv(EXEC_ANALYSIS / "summary.csv")
    cross_summary = read_csv(CROSS_ANALYSIS / "summary.csv")
    for row in exec_summary:
        model_summary.append({
            "benchmark": "ExecRepoBench-120", "model": row["model"], "method": row["method"],
            "primary_metric": "selected executable correctness", "primary_value": f(row["selected_output_correctness"]),
            "secondary_metric": "candidate Pass@1", "secondary_value": f(row["pass_at_1"]),
            "functional_execution": True,
        })
    for row in cross_summary:
        model_summary.append({
            "benchmark": "CrossCodeEval-100", "model": row["model"], "method": row["method"],
            "primary_metric": "normalized exact match", "primary_value": f(row["selected_exact_match"]),
            "secondary_metric": "identifier F1", "secondary_value": f(row["selected_identifier_f1"]),
            "functional_execution": False,
        })
    return language, repository, model_summary


def build_latex(
    decisions: Sequence[dict[str, Any]], review: Sequence[dict[str, Any]],
    interventions: Sequence[dict[str, Any]], language: Sequence[dict[str, Any]],
) -> None:
    latex = OUTPUT / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    metrics = [
        row for row in decisions
        if row["decision"] == "ALL_POLICY_METRICS" and row["decision_policy"] == "uncertainty_only"
    ]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Fixed-threshold uncertainty-only OpenCoderX decisions on ExecRepoBench-120. Acc. is correctness among autonomous outputs; Capture is the fraction of final failures routed to review or abstention. No executable outcome enters the decision score. Values are percentages.}",
        r"\label{tab:selective_autonomy}", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & Auto. & Review & Abstain & Acc./Capture \\", r"\midrule",
    ]
    for row in metrics:
        model_rows = [
            item for item in decisions
            if item["model"] == row["model"] and item["decision"] != "ALL_POLICY_METRICS"
            and item["decision_policy"] == "uncertainty_only"
        ]
        counts = {item["decision"]: item["task_share"] for item in model_rows}
        lines.append(
            f"{latex_escape(row['model'])} & {pct(counts['autonomous'])} & {pct(counts['request_review'])} & "
            f"{pct(counts['abstain'])} & {pct(row.get('autonomous_accuracy'))}/{pct(row.get('failure_capture_rate'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (latex / "table_selective_autonomy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    chosen = [row for row in review if row["review_budget"] == 0.2 and row["reviewer_success"] == 0.75]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Offline review-allocation simulation at a 20\% nominal review budget and 75\% reviewer-success assumption. Team success includes simulated corrections and is not observed human performance.}",
        r"\label{tab:review_allocation}", r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Policy & Team success & Failure capture & Deferral precision & Unnecessary review \\", r"\midrule",
    ]
    for model in MODELS:
        for row in [item for item in chosen if item["model"] == model]:
            lines.append(
                f"{latex_escape(model)} & {latex_escape(row['policy_label'])} & {pct(row['mean_team_success_rate'])} & "
                f"{pct(row['mean_failure_capture_rate'])} & {pct(row['mean_deferral_precision'])} & "
                f"{pct(row['mean_unnecessary_review_rate'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (latex / "table_review_allocation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    chosen_effects = [row for row in interventions if row["status"] == "OBSERVED" and row["intervention"] in {
        "Repository evidence provision", "Ordinary verification and repair", "Uncertainty-aware evidence and generation"
    }]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Matched observed intervention effects on ExecRepoBench-120 selected-output correctness. Values are percentages; CIs are paired task bootstrap intervals.}",
        r"\label{tab:intervention_effects}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Intervention & Before & After & $\Delta$ & 95\% CI & W/L/T \\", r"\midrule",
    ]
    for row in chosen_effects:
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['intervention'])} & {pct(row['before_correctness'])} & "
            f"{pct(row['after_correctness'])} & {100*row['absolute_difference']:+.1f} & "
            f"[{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}] & "
            f"{row['wins']}/{row['losses']}/{row['ties']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (latex / "table_intervention_effects.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{CrossCodeEval-100 context effect by language. Exact match is a native non-executable metric and must not be interpreted as functional correctness.}",
        r"\label{tab:language_generalization}", r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Language & Direct EM & Context EM & $\Delta$ & 95\% CI \\", r"\midrule",
    ]
    for row in language:
        lines.append(
            f"{latex_escape(row['model'])} & {row['language'].title()} & {pct(row['direct_exact_match'])} & "
            f"{pct(row['context_exact_match'])} & {100*row['exact_match_difference']:+.1f} & "
            f"[{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (latex / "table_language_generalization.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_memo(
    decision_summary: Sequence[dict[str, Any]], review_summary: Sequence[dict[str, Any]],
    interventions: Sequence[dict[str, Any]], language: Sequence[dict[str, Any]], integrity: Mapping[str, Any],
) -> str:
    lines = [
        "# TOSEM Collaboration and Generalizability Analysis", "",
        f"Integrity status: **{'PASSED' if integrity['passed'] else 'FAILED'}**.", "",
        "## Selective autonomy", "",
        "The fixed policy uses frozen review/abstain thresholds (0.35/0.70). The primary uncertainty-only policy excludes executable outcomes; a separate lifecycle-gated policy is reported as an operational deployment analysis.", "",
    ]
    for row in decision_summary:
        if row["decision"] == "ALL_POLICY_METRICS" and row["decision_policy"] == "uncertainty_only":
            lines.append(
                f"- {row['model']} (uncertainty only): autonomous coverage {pct(row['autonomous_coverage'])}%, autonomous accuracy "
                f"{pct(row['autonomous_accuracy'])}%, failure capture {pct(row['failure_capture_rate'])}%, and AUROC {row['failure_auroc']:.3f}."
            )
    lines += ["", "When executable validation and exhausted repair enforce the delivery gate:", ""]
    for row in decision_summary:
        if row["decision"] == "ALL_POLICY_METRICS" and row["decision_policy"] == "lifecycle_gated":
            lines.append(
                f"- {row['model']}: autonomous coverage {pct(row['autonomous_coverage'])}%, autonomous accuracy "
                f"{pct(row['autonomous_accuracy'])}%, and failure capture {pct(row['failure_capture_rate'])}%."
            )
    lines += ["", "## Review allocation simulation", "",
              "Reviewer corrections are offline simulations, not observed developer behavior. At a 20% nominal budget and 75% reviewer success:", ""]
    for model in MODELS:
        candidates = [row for row in review_summary if row["model"] == model and row["review_budget"] == 0.2 and row["reviewer_success"] == 0.75 and row["policy"] == "source_specific_opencoderx_deferral"]
        row = candidates[0]
        lines.append(
            f"- {model}: simulated team success {pct(row['mean_team_success_rate'])}%, failure capture "
            f"{pct(row['mean_failure_capture_rate'])}%, unnecessary review {pct(row['mean_unnecessary_review_rate'])}%."
        )
    lines += ["", "## Observed interventions", ""]
    for row in interventions:
        if row["status"] == "OBSERVED" and row["intervention"] == "Uncertainty-aware evidence and generation":
            lines.append(
                f"- {row['model']}: OpenCoderX versus matched RAG + Verify/Repair {100*row['absolute_difference']:+.1f} points "
                f"(95% CI [{100*row['bootstrap_ci95_low']:.1f}, {100*row['bootstrap_ci95_high']:.1f}])."
            )
    lines += ["", "API-, context-, and similar-code correction effects are explicitly NOT_RUN because no matched correction records exist.", "",
              "## Generalizability", "",
              "ExecRepoBench reports executable correctness. CrossCodeEval reports native exact match, edit similarity, and identifier F1 only; these endpoints are not pooled.", ""]
    positive = sum(row["exact_match_difference"] > 0 for row in language)
    lines.append(f"Cross-file context increased exact match in {positive}/{len(language)} model-language cells; each cell contains 25 matched tasks.")
    lines += ["", "## Guardrails", "",
              "- No model calls were made by this analysis.",
              "- The test-failure policy uses post-selection, pre-repair validation, never final correctness.",
              "- The oracle policy is an explicit upper bound and uses final outcomes only for ranking.",
              "- Repository-level effects are descriptive because repositories contain only one to four tasks.",
              "- No claim of universal OpenCoderX superiority is supported by the matched results.", ""]
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    collaboration = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))["collaboration"]
    exec_rows = read_csv(EXEC_ANALYSIS / "task_level.csv")
    cross_rows = read_csv(CROSS_ANALYSIS / "task_level.csv")
    open_records, raw_files = load_open_records()
    decisions, decision_summary = make_decisions(open_records, collaboration)
    review_raw, review_summary = review_simulations(open_records, collaboration)
    interventions, repair_sources = intervention_analysis(exec_rows, open_records)
    language, repository, model_summary = generalizability(exec_rows, cross_rows)

    issues: list[str] = []
    if len(open_records) != 480:
        issues.append(f"expected 480 OpenCoderX records, found {len(open_records)}")
    if len({(row['model'], row['task_id']) for row in open_records}) != 480:
        issues.append("OpenCoderX model-task records are not unique")
    expected_review = len(MODELS) * len(POLICIES) * len(collaboration["review_budgets"]) * len(collaboration["reviewer_success"]) * len(collaboration["seeds"])
    if len(review_raw) != expected_review:
        issues.append(f"expected {expected_review} review simulations, found {len(review_raw)}")
    if len(language) != 16 or any(row["tasks"] != 25 for row in language):
        issues.append("language analysis is not 16 matched 25-task cells")
    if any(row["simulation"] for row in interventions if row["status"] == "OBSERVED"):
        issues.append("observed intervention row mislabeled as simulation")
    if any(row["decision"] not in {"autonomous", "request_review", "abstain"} for row in decisions):
        issues.append("unknown collaboration decision")
    if len(decisions) != 960:
        issues.append(f"expected 960 two-policy decisions, found {len(decisions)}")

    write_jsonl(OUTPUT / "decision_records.jsonl", decisions)
    write_csv(OUTPUT / "decision_summary.csv", decision_summary)
    write_csv(OUTPUT / "review_budget_simulations.csv", review_raw)
    write_csv(OUTPUT / "review_budget_summary.csv", review_summary)
    write_csv(OUTPUT / "intervention_effectiveness.csv", interventions)
    write_csv(OUTPUT / "repair_effectiveness_by_source.csv", repair_sources)
    write_csv(OUTPUT / "generalizability_language.csv", language)
    write_csv(OUTPUT / "generalizability_repository.csv", repository)
    write_csv(OUTPUT / "generalizability_model_benchmark.csv", model_summary)
    build_latex(decision_summary, review_summary, interventions, language)

    provenance = {
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "policy_module": "opencoderx/collaboration.py",
        "policy_module_sha256": sha256(ROOT / "opencoderx/collaboration.py"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST),
        "exec_task_level_sha256": sha256(EXEC_ANALYSIS / "task_level.csv"),
        "cross_task_level_sha256": sha256(CROSS_ANALYSIS / "task_level.csv"),
        "raw_opencoder_files": raw_files,
        "review_simulation": {
            "observed_human_data": False,
            "budgets": collaboration["review_budgets"],
            "reviewer_success": collaboration["reviewer_success"],
            "seeds": collaboration["seeds"],
        },
        "source_risk_definition": "Mean of within-source retrieval concentration loss (1 - mean/top) and filtering loss (1 - kept/retrieved).",
        "bootstrap": {"iterations": BOOTSTRAPS, "seed": SEED, "unit": "task"},
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    integrity = {
        "passed": not issues,
        "paper_eligible": not issues,
        "issues": issues,
        "opencoder_records": len(open_records),
        "decision_records": len(decisions),
        "review_simulation_rows": len(review_raw),
        "observed_intervention_rows": sum(row["status"] == "OBSERVED" for row in interventions),
        "not_run_intervention_rows": sum(row["status"] == "NOT_RUN" for row in interventions),
        "language_cells": len(language),
        "repository_cells": len(repository),
    }
    (OUTPUT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "results_memo.md").write_text(
        make_memo(decision_summary, review_summary, interventions, language, integrity), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
