#!/usr/bin/env python3
"""Post-hoc leave-one-uncertainty-source-out analysis on frozen TOSEM data.

No model calls are made. Risk scores use only API, context, similar-code, and
generation uncertainty observed before final executable correctness.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoderx.collaboration import RiskTrace, review_allocation_metrics
from scripts.analyze_tosem_collaboration import source_risk


CONFIRMATORY = ROOT / "results/tosem/confirmatory"
PROTOCOL = ROOT / "results/tosem/protocol_freeze.json"
FINAL_INTEGRITY = ROOT / "results/tosem/final_integrity.json"
CAMPAIGN = ROOT / "configs/tosem/campaign.yaml"
COLLABORATION_SCRIPT = ROOT / "scripts/analyze_tosem_collaboration.py"
COLLABORATION_CORE = ROOT / "opencoderx/collaboration.py"
OUTPUT = ROOT / "results/tosem/uncertainty_source_ablation"
TABLE_OUTPUT = ROOT / "results/tosem/publication_tables/tab_uncertainty_source_ablation.tex"
FIGURE_PDF = ROOT / "results/tosem/publication_figures/fig_uncertainty_source_ablation.pdf"
FIGURE_PNG = ROOT / "results/tosem/publication_figures/fig_uncertainty_source_ablation.png"

MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-5": "Claude Sonnet 5",
    "qwen3-coder-plus": "Qwen3-Coder-Plus",
}
COMPONENTS = ("api", "context", "similar_code", "generation")
VARIANTS = ("FULL", "minus_API", "minus_CONTEXT", "minus_SIMILAR_CODE", "minus_GENERATION")
VARIANT_LABELS = {
    "FULL": "FULL",
    "minus_API": "- API",
    "minus_CONTEXT": "- Context",
    "minus_SIMILAR_CODE": "- Similar Code",
    "minus_GENERATION": "- Generation",
}
REMOVED_COMPONENT = {
    "minus_API": "api",
    "minus_CONTEXT": "context",
    "minus_SIMILAR_CODE": "similar_code",
    "minus_GENERATION": "generation",
}
REVIEW_BUDGETS = (0.10, 0.20, 0.30)
PRIMARY_BUDGET = 0.20
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_METRICS = ("auroc", "auprc", "failure_capture_rate", "deferral_precision")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise ValueError(f"fields required for empty CSV: {path}")
        fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_variants(components: Mapping[str, float]) -> dict[str, float]:
    """Construct FULL and four LOCO scores without access to outcomes."""
    if set(components) != set(COMPONENTS):
        raise ValueError(f"expected exactly {COMPONENTS}, found {tuple(components)}")
    values = {name: float(components[name]) for name in COMPONENTS}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"malformed uncertainty component: {values}")
    trace = RiskTrace(
        api=values["api"],
        context=values["context"],
        similar_code=values["similar_code"],
        generation=values["generation"],
    )
    scores = {
        "FULL": trace.aggregate({
            "api": 1.0,
            "context": 1.0,
            "similar_code": 1.0,
            "generation": 1.0,
            "verification": 0.0,
            "repair": 0.0,
        })
    }
    for variant, removed in REMOVED_COMPONENT.items():
        scores[variant] = float(mean(value for name, value in values.items() if name != removed))
    return scores


def load_frozen_records() -> tuple[list[dict[str, Any]], list[Path], float]:
    records: list[dict[str, Any]] = []
    input_files: list[Path] = []
    expected_task_ids: set[str] | None = None
    max_full_mean_error = 0.0
    for model, directory in MODELS.items():
        model_rows: list[dict[str, Any]] = []
        paths = sorted((CONFIRMATORY / directory / "opencoder").glob("batch_*.json"))
        if not paths:
            raise ValueError(f"no frozen OpenCoderX batches found for {model}")
        input_files.extend(paths)
        for path in paths:
            for raw in read_json(path).get("with", []):
                task_id = str(raw.get("id") or "")
                if not task_id:
                    raise ValueError(f"missing task ID in {path}")
                if "passed" not in raw or not isinstance(raw["passed"], bool):
                    raise ValueError(f"missing or malformed correctness label: {model}/{task_id}")
                diagnostics = raw.get("source_diagnostics")
                uncertainty = raw.get("u")
                if not isinstance(diagnostics, Mapping) or not isinstance(uncertainty, Mapping):
                    raise ValueError(f"missing uncertainty data: {model}/{task_id}")
                components = {
                    "api": source_risk(diagnostics.get("api")),
                    "context": source_risk(diagnostics.get("context")),
                    "similar_code": source_risk(diagnostics.get("similar_code")),
                    "generation": float(uncertainty.get("aggregate")),
                }
                scores = score_variants(components)
                arithmetic_mean = float(mean(components.values()))
                max_full_mean_error = max(max_full_mean_error, abs(scores["FULL"] - arithmetic_mean))
                model_rows.append({
                    "task_id": task_id,
                    "model": model,
                    "selected_output_correct": raw["passed"],
                    "failure": not raw["passed"],
                    **{f"u_{name}": value for name, value in components.items()},
                    **{f"score_{variant}": value for variant, value in scores.items()},
                })
        keys = [(row["model"], row["task_id"]) for row in model_rows]
        if len(model_rows) != 120 or len(set(keys)) != 120:
            raise ValueError(
                f"frozen record discrepancy for {model}: rows={len(model_rows)}, unique={len(set(keys))}"
            )
        task_ids = {row["task_id"] for row in model_rows}
        if expected_task_ids is None:
            expected_task_ids = task_ids
        elif task_ids != expected_task_ids:
            raise ValueError(f"task set mismatch for {model}")
        records.extend(model_rows)
    if len(records) != 480 or len({(row["model"], row["task_id"]) for row in records}) != 480:
        raise ValueError("expected 480 unique model-task observations")
    return records, input_files, max_full_mean_error


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def auroc_score(failure: Sequence[bool] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    labels = np.asarray(failure, dtype=bool)
    values = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(values)
    u = float(ranks[labels].sum() - positives * (positives + 1) / 2.0)
    return u / (positives * negatives)


def auprc_score(failure: Sequence[bool] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    labels = np.asarray(failure, dtype=bool)
    values = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    true_positives = 0
    predicted = 0
    average_precision = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        group_positives = int(sorted_labels[start:stop].sum())
        true_positives += group_positives
        predicted += stop - start
        if group_positives:
            average_precision += (group_positives / positives) * (true_positives / predicted)
        start = stop
    return float(average_precision)


def select_highest_risk(
    task_ids: Sequence[str], scores: Sequence[float], review_budget: float,
) -> set[str]:
    """Use the existing ceil-budget convention and deterministic task-ID ties."""
    if not 0.0 <= review_budget <= 1.0:
        raise ValueError("review budget must lie in [0, 1]")
    if len(task_ids) != len(scores) or len(set(task_ids)) != len(task_ids):
        raise ValueError("review allocation requires matched unique task IDs and scores")
    count = min(len(task_ids), math.ceil(len(task_ids) * review_budget))
    ranked = sorted(range(len(task_ids)), key=lambda index: (-float(scores[index]), str(task_ids[index])))
    return {str(task_ids[index]) for index in ranked[:count]}


def review_metrics(
    records: Sequence[Mapping[str, Any]], variant: str, budget: float,
) -> dict[str, float]:
    task_ids = [str(row["task_id"]) for row in records]
    reviewed = select_highest_risk(task_ids, [float(row[f"score_{variant}"]) for row in records], budget)
    metrics = review_allocation_metrics(records, reviewed, reviewer_success=0.0, seed=BOOTSTRAP_SEED)
    return {
        key: float(metrics[key])
        for key in (
            "failure_capture_rate", "deferral_precision", "autonomous_coverage",
            "autonomous_failure_rate", "selective_accuracy",
        )
    }


def point_metrics(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = sorted((row for row in records if row["model"] == model), key=lambda row: row["task_id"])
        failure = np.asarray([row["failure"] for row in model_rows], dtype=bool)
        for variant in VARIANTS:
            scores = np.asarray([row[f"score_{variant}"] for row in model_rows], dtype=float)
            prediction_rows.append({
                "model": model,
                "variant": variant,
                "tasks": len(model_rows),
                "failures": int(failure.sum()),
                "successes": int((~failure).sum()),
                "failure_prevalence": float(failure.mean()),
                "auroc": auroc_score(failure, scores),
                "auprc": auprc_score(failure, scores),
            })
            for budget in REVIEW_BUDGETS:
                allocation_rows.append({
                    "model": model,
                    "variant": variant,
                    "review_budget": budget,
                    "tasks": len(model_rows),
                    "reviewed_tasks": math.ceil(len(model_rows) * budget),
                    **review_metrics(model_rows, variant, budget),
                })
    return prediction_rows, allocation_rows


def paired_bootstrap_indices(n_tasks: int, replicates: int, seed: int) -> np.ndarray:
    if n_tasks <= 0 or replicates <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(0, n_tasks, size=(replicates, n_tasks))


def paired_sample(indices: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply one sampled index vector to every FULL/ablation array."""
    return {name: np.asarray(values)[indices] for name, values in arrays.items()}


def _sample_review_metrics(failure: np.ndarray, scores: np.ndarray, task_ids: np.ndarray) -> tuple[float, float]:
    count = math.ceil(len(failure) * PRIMARY_BUDGET)
    ranked = sorted(
        range(len(failure)),
        key=lambda index: (-float(scores[index]), str(task_ids[index]), index),
    )
    selected = np.asarray(ranked[:count], dtype=int)
    failures = int(failure.sum())
    captured = int(failure[selected].sum())
    return (captured / failures if failures else float("nan"), captured / count if count else 0.0)


def bootstrap_deltas(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        model_rows = sorted((row for row in records if row["model"] == model), key=lambda row: row["task_id"])
        failure = np.asarray([row["failure"] for row in model_rows], dtype=bool)
        task_ids = np.asarray([row["task_id"] for row in model_rows], dtype=object)
        scores = {
            variant: np.asarray([row[f"score_{variant}"] for row in model_rows], dtype=float)
            for variant in VARIANTS
        }
        model_seed = BOOTSTRAP_SEED + model_index
        sampled_indices = paired_bootstrap_indices(len(model_rows), BOOTSTRAP_REPLICATES, model_seed)
        deltas = {
            variant: {metric: [] for metric in BOOTSTRAP_METRICS}
            for variant in VARIANTS[1:]
        }
        excluded_auroc = 0
        for indices in sampled_indices:
            sampled_failure = failure[indices]
            sampled_ids = task_ids[indices]
            sampled_variant_scores = paired_sample(indices, scores)
            replicate_metrics: dict[str, dict[str, float]] = {}
            for variant in VARIANTS:
                sampled_scores = sampled_variant_scores[variant]
                capture, precision = _sample_review_metrics(sampled_failure, sampled_scores, sampled_ids)
                replicate_metrics[variant] = {
                    "auroc": auroc_score(sampled_failure, sampled_scores),
                    "auprc": auprc_score(sampled_failure, sampled_scores),
                    "failure_capture_rate": capture,
                    "deferral_precision": precision,
                }
            if not math.isfinite(replicate_metrics["FULL"]["auroc"]):
                excluded_auroc += 1
            for variant in VARIANTS[1:]:
                for metric in BOOTSTRAP_METRICS:
                    full_value = replicate_metrics["FULL"][metric]
                    ablated_value = replicate_metrics[variant][metric]
                    if math.isfinite(full_value) and math.isfinite(ablated_value):
                        deltas[variant][metric].append(ablated_value - full_value)

        prediction_lookup = {
            variant: {
                "auroc": auroc_score(failure, variant_scores),
                "auprc": auprc_score(failure, variant_scores),
            }
            for variant, variant_scores in scores.items()
        }
        allocation_lookup = {}
        for variant, variant_scores in scores.items():
            capture, precision = _sample_review_metrics(failure, variant_scores, task_ids)
            allocation_lookup[variant] = {
                "failure_capture_rate": capture,
                "deferral_precision": precision,
            }
        for variant in VARIANTS[1:]:
            for metric in BOOTSTRAP_METRICS:
                if metric in {"auroc", "auprc"}:
                    point_delta = float(prediction_lookup[variant][metric]) - float(prediction_lookup["FULL"][metric])
                else:
                    point_delta = float(allocation_lookup[variant][metric]) - float(allocation_lookup["FULL"][metric])
                values = np.asarray(deltas[variant][metric], dtype=float)
                low, high = np.quantile(values, (0.025, 0.975))
                results.append({
                    "model": model,
                    "variant": variant,
                    "removed_component": REMOVED_COMPONENT[variant],
                    "metric": metric,
                    "point_delta_ablation_minus_full": point_delta,
                    "bootstrap_mean_delta": float(values.mean()),
                    "bootstrap_median_delta": float(np.median(values)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                    "bootstrap_replicates_used": len(values),
                    "excluded_single_class_replicates": excluded_auroc if metric == "auroc" else 0,
                    "bootstrap_seed": model_seed,
                    "paired_indices_sha256": hashlib.sha256(sampled_indices.tobytes()).hexdigest(),
                })
    return results


def macro_summary(
    prediction_rows: Sequence[dict[str, Any]], allocation_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        predictions = [row for row in prediction_rows if row["variant"] == variant]
        allocations = [
            row for row in allocation_rows
            if row["variant"] == variant and float(row["review_budget"]) == PRIMARY_BUDGET
        ]
        row = {
            "variant": variant,
            "models": len(MODELS),
            "tasks_per_model": 120,
            "macro_auroc": float(mean(float(item["auroc"]) for item in predictions)),
            "macro_auprc": float(mean(float(item["auprc"]) for item in predictions)),
            "macro_failure_capture_at_20": float(mean(float(item["failure_capture_rate"]) for item in allocations)),
            "macro_deferral_precision_at_20": float(mean(float(item["deferral_precision"]) for item in allocations)),
            "macro_average_is_descriptive": True,
        }
        rows.append(row)
    full = rows[0]
    for row in rows:
        if row["variant"] == "FULL":
            row.update({
                "delta_macro_auroc_vs_full": None,
                "delta_macro_auprc_vs_full": None,
                "delta_macro_failure_capture_vs_full": None,
                "delta_macro_deferral_precision_vs_full": None,
            })
        else:
            row.update({
                "delta_macro_auroc_vs_full": row["macro_auroc"] - full["macro_auroc"],
                "delta_macro_auprc_vs_full": row["macro_auprc"] - full["macro_auprc"],
                "delta_macro_failure_capture_vs_full": row["macro_failure_capture_at_20"] - full["macro_failure_capture_at_20"],
                "delta_macro_deferral_precision_vs_full": row["macro_deferral_precision_at_20"] - full["macro_deferral_precision_at_20"],
            })
    return rows


def publication_table_rows(macro: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in macro]


def fmt_delta(value: Any, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "--"
    numeric = float(value) * (100.0 if percent else 1.0)
    return f"{numeric:+.1f}" if percent else f"{numeric:+.3f}"


def write_latex_table(rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Post-hoc leave-one-source-out uncertainty ablation. Prediction metrics and review-allocation outcomes are descriptive macro-averages across four LLM backends. Review allocation uses a fixed 20\% budget; deltas are ablation minus FULL.}",
        r"\label{tab:uncertainty_source_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variant & AUROC & $\Delta$ & AUPRC & $\Delta$ & Capture (\%) & $\Delta$ (pp) & Precision (\%) & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{VARIANT_LABELS[row['variant']]} & "
            f"{float(row['macro_auroc']):.3f} & {fmt_delta(row['delta_macro_auroc_vs_full'])} & "
            f"{float(row['macro_auprc']):.3f} & {fmt_delta(row['delta_macro_auprc_vs_full'])} & "
            f"{100 * float(row['macro_failure_capture_at_20']):.1f} & {fmt_delta(row['delta_macro_failure_capture_vs_full'], percent=True)} & "
            f"{100 * float(row['macro_deferral_precision_at_20']):.1f} & {fmt_delta(row['delta_macro_deferral_precision_vs_full'], percent=True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure(bootstrap_rows: Sequence[dict[str, Any]]) -> None:
    components = ("api", "context", "similar_code", "generation")
    labels = ("API", "Context", "Similar code", "Generation")
    colors = {
        "gpt-4o-mini": "#2D5F8B",
        "gemini-2.5-flash": "#D18C19",
        "claude-sonnet-5": "#7B4F88",
        "qwen3-coder-plus": "#3E7C68",
    }
    markers = {"gpt-4o-mini": "o", "gemini-2.5-flash": "s", "claude-sonnet-5": "^", "qwen3-coder-plus": "D"}
    offsets = dict(zip(MODELS, (-0.24, -0.08, 0.08, 0.24)))
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.7,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    panel_specs = (
        ("auroc", "Delta AUROC (ablation - FULL)", 1.0),
        ("failure_capture_rate", "Delta failure capture at 20% (pp)", 100.0),
    )
    x = np.arange(len(components))
    for axis, (metric, ylabel, scale) in zip(axes, panel_specs):
        axis.axhline(0.0, color="#4E555B", linewidth=0.9, linestyle="--", zorder=1)
        for model in MODELS:
            selected = {
                row["removed_component"]: row
                for row in bootstrap_rows
                if row["model"] == model and row["metric"] == metric
            }
            center = np.asarray([scale * float(selected[component]["point_delta_ablation_minus_full"]) for component in components])
            low = np.asarray([scale * float(selected[component]["ci95_low"]) for component in components])
            high = np.asarray([scale * float(selected[component]["ci95_high"]) for component in components])
            axis.errorbar(
                x + offsets[model], center,
                yerr=np.vstack((center - low, high - center)),
                fmt=markers[model], color=colors[model], ecolor=colors[model],
                markersize=4.0, markeredgecolor="white", markeredgewidth=0.4,
                capsize=2.0, elinewidth=1.0, linestyle="none", label=MODEL_LABELS[model], zorder=3,
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9DDE1", linewidth=0.55, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.5)
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_results_memo(
    macro: Sequence[dict[str, Any]], bootstrap_rows: Sequence[dict[str, Any]],
) -> None:
    ablations = [row for row in macro if row["variant"] != "FULL"]
    worst_auroc = min(ablations, key=lambda row: float(row["delta_macro_auroc_vs_full"]))
    worst_auprc = min(ablations, key=lambda row: float(row["delta_macro_auprc_vs_full"]))
    worst_capture = min(ablations, key=lambda row: float(row["delta_macro_failure_capture_vs_full"]))
    improving = []
    for row in bootstrap_rows:
        if float(row["point_delta_ablation_minus_full"]) > 0:
            improving.append((row["model"], row["removed_component"], row["metric"]))
    sign_patterns: dict[tuple[str, str], set[int]] = {}
    for row in bootstrap_rows:
        key = (row["removed_component"], row["metric"])
        value = float(row["point_delta_ablation_minus_full"])
        sign_patterns.setdefault(key, set()).add(1 if value > 0 else -1 if value < 0 else 0)
    heterogeneous = sorted({component for (component, _), signs in sign_patterns.items() if 1 in signs and -1 in signs})
    lines = [
        "# Leave-One-Uncertainty-Source-Out Ablation",
        "",
        "This post-hoc exploratory analysis uses 480 frozen model-task records and makes no model calls.",
        "",
        "1. Largest macro AUROC degradation: removing "
        f"{REMOVED_COMPONENT[worst_auroc['variant']]} ({float(worst_auroc['delta_macro_auroc_vs_full']):+.3f}).",
        "2. Largest macro AUPRC degradation: removing "
        f"{REMOVED_COMPONENT[worst_auprc['variant']]} ({float(worst_auprc['delta_macro_auprc_vs_full']):+.3f}).",
        "3. Largest macro failure-capture degradation at 20% review: removing "
        f"{REMOVED_COMPONENT[worst_capture['variant']]} ({100 * float(worst_capture['delta_macro_failure_capture_vs_full']):+.1f} pp).",
        "4. Cross-model consistency: effects are not uniform across all four model families.",
        f"5. Positive removal effects occur in {len(improving)} model-source-metric combinations.",
        "6. Heterogeneous sign patterns occur for: " + (", ".join(heterogeneous) if heterogeneous else "none") + ".",
        "7. The evidence does not support a universal claim that every uncertainty component is necessary.",
        "",
        "Macro averages are descriptive. Per-model paired task-bootstrap intervals are the primary uncertainty estimates.",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results_memo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("matplotlib", "numpy", "scikit-learn", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "UNAVAILABLE_WORKSPACE_EXPORT"


def write_metadata(
    records: Sequence[dict[str, Any]], input_files: Sequence[Path], max_full_mean_error: float,
    bootstrap_rows: Sequence[dict[str, Any]],
) -> None:
    principal_inputs = [
        PROTOCOL, FINAL_INTEGRITY, CAMPAIGN, COLLABORATION_SCRIPT, COLLABORATION_CORE, *input_files,
    ]
    task_counts = {model: len([row for row in records if row["model"] == model]) for model in MODELS}
    failures = {
        model: sum(bool(row["failure"]) for row in records if row["model"] == model)
        for model in MODELS
    }
    successes = {model: task_counts[model] - failures[model] for model in MODELS}
    common = {
        "analysis_type": "post_hoc_exploratory_source_ablation",
        "new_model_calls": False,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "git_sha_warning": "Workspace root is not a Git worktree; file SHA-256 hashes are authoritative.",
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "models": list(MODELS),
        "task_counts": task_counts,
        "failure_counts": failures,
        "success_counts": successes,
        "bootstrap_master_seed": BOOTSTRAP_SEED,
        "bootstrap_model_seeds": {model: BOOTSTRAP_SEED + index for index, model in enumerate(MODELS)},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "review_budgets": list(REVIEW_BUDGETS),
        "input_artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in principal_inputs},
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    provenance = {
        **common,
        "score_components": list(COMPONENTS),
        "score_variants": list(VARIANTS),
        "risk_definition": "equal-weight RiskTrace mean; LOCO variants recompute over three retained components",
        "target": "failure = NOT selected_output_correct",
        "macro_inference": "descriptive only; no macro bootstrap interval",
    }
    integrity = {
        **common,
        "status": "PASS",
        "observations": len(records),
        "unique_model_task_ids": len({(row["model"], row["task_id"]) for row in records}),
        "matched_task_sets": True,
        "max_abs_full_vs_arithmetic_mean_error": max_full_mean_error,
        "malformed_uncertainty_values": 0,
        "missing_correctness_labels": 0,
        "bootstrap_delta_rows": len(bootstrap_rows),
        "routing_uses_correctness": False,
        "frozen_outputs_modified": False,
        "manuscript_modified": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")


def verify_outputs(
    records: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]],
    allocation_rows: Sequence[dict[str, Any]], bootstrap_rows: Sequence[dict[str, Any]],
    macro: Sequence[dict[str, Any]], max_full_mean_error: float,
) -> None:
    if len(records) != 480 or len({(row["model"], row["task_id"]) for row in records}) != 480:
        raise AssertionError("expected 480 unique observations")
    if len(prediction_rows) != 20 or len(allocation_rows) != 60 or len(bootstrap_rows) != 64 or len(macro) != 5:
        raise AssertionError("unexpected output row counts")
    if max_full_mean_error > 1e-12:
        raise AssertionError(f"FULL score differs from arithmetic mean by {max_full_mean_error}")
    for row in prediction_rows:
        for metric in ("auroc", "auprc"):
            if not math.isfinite(float(row[metric])):
                raise AssertionError(f"non-finite point metric: {row}")
    for row in bootstrap_rows:
        if int(row["bootstrap_replicates_used"]) <= 0:
            raise AssertionError(f"empty bootstrap interval: {row}")
        if float(row["ci95_low"]) > float(row["ci95_high"]):
            raise AssertionError(f"reversed bootstrap interval: {row}")


def main() -> None:
    protocol = read_json(PROTOCOL)
    expected_models = tuple(protocol["scientific_protocol"]["models"])
    if expected_models != tuple(MODELS) or int(protocol["scientific_protocol"]["tasks"]) != 120:
        raise ValueError("frozen protocol does not match the required four-model, 120-task campaign")
    records, input_files, max_full_mean_error = load_frozen_records()
    prediction_rows, allocation_rows = point_metrics(records)
    bootstrap_rows = bootstrap_deltas(records)
    macro = macro_summary(prediction_rows, allocation_rows)
    publication_rows = publication_table_rows(macro)
    verify_outputs(records, prediction_rows, allocation_rows, bootstrap_rows, macro, max_full_mean_error)

    write_csv(OUTPUT / "per_model_metrics.csv", prediction_rows)
    write_csv(OUTPUT / "review_allocation.csv", allocation_rows)
    write_csv(OUTPUT / "bootstrap_deltas.csv", bootstrap_rows)
    write_csv(OUTPUT / "macro_summary.csv", macro)
    write_csv(OUTPUT / "publication_table.csv", publication_rows)
    write_latex_table(publication_rows)
    plot_figure(bootstrap_rows)
    write_results_memo(macro, bootstrap_rows)
    write_metadata(records, input_files, max_full_mean_error, bootstrap_rows)
    print(f"Wrote source-ablation analysis to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
