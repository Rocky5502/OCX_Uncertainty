#!/usr/bin/env python3
"""Create publication figures from audited TOSEM CSV artifacts."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RQ12 = ROOT / "results/tosem/rq1_rq2_analysis"
COLLAB = ROOT / "results/tosem/collaboration_analysis"
EXEC = ROOT / "results/tosem/confirmatory_analysis"
CROSS = ROOT / "results/tosem/crosscodeeval_confirmatory"
OUTPUT = ROOT / "results/tosem/publication_figures"

MODELS = ("gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-5", "qwen3-coder-plus")
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-5": "Claude Sonnet 5",
    "qwen3-coder-plus": "Qwen3-Coder-Plus",
}
COLORS = {
    "gpt-4o-mini": "#2D5F8B",
    "gemini-2.5-flash": "#D18C19",
    "claude-sonnet-5": "#7B4F88",
    "qwen3-coder-plus": "#3E7C68",
}
SIGNAL_COLORS = {
    "api": "#2D5F8B",
    "context": "#D18C19",
    "similar_code": "#7B4F88",
    "generation": "#3E7C68",
    "aggregate": "#272B30",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    return float(value)


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def finish(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def quiet_axis(axis: plt.Axes, grid: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, color="#D9DDE1", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def plot_uncertainty_signals() -> None:
    data = rows(RQ12 / "uncertainty_signal_discrimination.csv")
    signals = ("api", "context", "similar_code", "generation", "aggregate")
    labels = ("API", "Context", "Similar code", "Generation", "Aggregate")
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.35), sharex=True, sharey=True)
    y = np.arange(len(signals))
    for axis, model in zip(axes, MODELS):
        selected = {row["signal"]: row for row in data if row["model"] == model}
        values = np.asarray([f(selected[signal]["auroc_failure"]) for signal in signals])
        low = np.asarray([f(selected[signal]["auroc_ci95_low"]) for signal in signals])
        high = np.asarray([f(selected[signal]["auroc_ci95_high"]) for signal in signals])
        for index, signal in enumerate(signals):
            axis.errorbar(
                values[index], y[index],
                xerr=[[values[index] - low[index]], [high[index] - values[index]]],
                fmt="o", color=SIGNAL_COLORS[signal], markeredgecolor="white", markeredgewidth=0.5,
                markersize=4.8, capsize=2.0, linewidth=1.0,
            )
        axis.axvline(0.5, color="#747A80", linestyle="--", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model], pad=7)
        axis.set_xlim(0.35, 0.9)
        axis.set_xticks((0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        quiet_axis(axis, "x")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.supxlabel("Failure AUROC with task-bootstrap 95% CI", y=-0.02)
    fig.tight_layout(w_pad=0.45)
    finish(fig, "fig_uncertainty_signal_discrimination")


def plot_selective_autonomy() -> None:
    data = rows(COLLAB / "decision_summary.csv")
    selected = [
        row for row in data
        if row["decision_policy"] == "uncertainty_only" and row["decision"] == "ALL_POLICY_METRICS"
    ]
    x = np.arange(len(MODELS))
    coverage = [100 * f(next(row for row in selected if row["model"] == model)["autonomous_coverage"]) for model in MODELS]
    accuracy = [100 * f(next(row for row in selected if row["model"] == model)["autonomous_accuracy"]) for model in MODELS]
    capture = [100 * f(next(row for row in selected if row["model"] == model)["failure_capture_rate"]) for model in MODELS]
    fig, axis = plt.subplots(figsize=(7.15, 2.75))
    width = 0.23
    axis.bar(x - width, coverage, width, label="Autonomous coverage", color="#AEB7C0", edgecolor="#59636D", linewidth=0.5, hatch="//")
    axis.bar(x, accuracy, width, label="Autonomous accuracy", color="#2D5F8B", edgecolor="#234967", linewidth=0.5, hatch="..")
    axis.bar(x + width, capture, width, label="Failure capture", color="#D18C19", edgecolor="#9D6813", linewidth=0.5, hatch="xx")
    for offset, values in ((-width, coverage), (0, accuracy), (width, capture)):
        for index, value in enumerate(values):
            axis.text(index + offset, value + 1.7, f"{value:.1f}", ha="center", va="bottom", fontsize=6.8)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_ylabel("Tasks / correctness (%)")
    axis.set_ylim(0, 102)
    axis.legend(frameon=False, ncol=3, loc="upper left", handlelength=2.2)
    quiet_axis(axis)
    fig.tight_layout()
    finish(fig, "fig_selective_autonomy")


def plot_review_budget() -> None:
    data = rows(COLLAB / "review_budget_summary.csv")
    policies = (
        "random_deferral", "test_failure_deferral", "aggregate_uncertainty_deferral",
        "source_specific_opencoderx_deferral", "oracle_deferral",
    )
    labels = {
        "random_deferral": "Random",
        "test_failure_deferral": "Pre-repair test",
        "aggregate_uncertainty_deferral": "Aggregate uncertainty",
        "source_specific_opencoderx_deferral": "Source-specific",
        "oracle_deferral": "Oracle",
    }
    styles = {
        "random_deferral": ("#9AA2AA", "--", "o"),
        "test_failure_deferral": ("#D18C19", "-", "s"),
        "aggregate_uncertainty_deferral": ("#2D5F8B", "-", "o"),
        "source_specific_opencoderx_deferral": ("#7B4F88", "-", "^"),
        "oracle_deferral": ("#272B30", ":", "D"),
    }
    budgets = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.55), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        for policy in policies:
            values = []
            for budget in budgets:
                row = next(
                    item for item in data
                    if item["model"] == model and item["policy"] == policy
                    and f(item["review_budget"]) == budget and f(item["reviewer_success"]) == 0.75
                )
                values.append(100 * f(row["mean_team_success_rate"]))
            color, line, marker = styles[policy]
            axis.plot(
                np.asarray(budgets) * 100, values, color=color, linestyle=line, marker=marker,
                markersize=3.2, linewidth=1.25, label=labels[policy],
            )
        axis.set_title(MODEL_LABELS[model], pad=7)
        axis.set_xticks((0, 20, 50, 100))
        axis.set_xlim(-2, 102)
        quiet_axis(axis)
    axes[0].set_ylabel("Simulated team success (%)")
    fig.supxlabel("Nominal review budget (%)", y=-0.01)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.45)
    finish(fig, "fig_review_budget_team_success")


def plot_failure_capture_budget() -> None:
    data = rows(COLLAB / "review_budget_summary.csv")
    policies = (
        "random_deferral", "test_failure_deferral", "aggregate_uncertainty_deferral",
        "source_specific_opencoderx_deferral", "oracle_deferral",
    )
    labels = {
        "random_deferral": "Random", "test_failure_deferral": "Pre-repair test",
        "aggregate_uncertainty_deferral": "Aggregate uncertainty",
        "source_specific_opencoderx_deferral": "Source-specific", "oracle_deferral": "Oracle",
    }
    styles = {
        "random_deferral": ("#9AA2AA", "--", "o"), "test_failure_deferral": ("#D18C19", "-", "s"),
        "aggregate_uncertainty_deferral": ("#2D5F8B", "-", "o"),
        "source_specific_opencoderx_deferral": ("#7B4F88", "-", "^"),
        "oracle_deferral": ("#272B30", ":", "D"),
    }
    budgets = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.55), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        for policy in policies:
            values = []
            for budget in budgets:
                row = next(
                    item for item in data if item["model"] == model and item["policy"] == policy
                    and f(item["review_budget"]) == budget and f(item["reviewer_success"]) == 0.75
                )
                values.append(100 * f(row["mean_failure_capture_rate"]))
            color, line, marker = styles[policy]
            axis.plot(np.asarray(budgets) * 100, values, color=color, linestyle=line, marker=marker,
                      markersize=3.2, linewidth=1.25, label=labels[policy])
        axis.set_title(MODEL_LABELS[model], pad=7)
        axis.set_xticks((0, 20, 50, 100))
        axis.set_xlim(-2, 102)
        axis.set_ylim(-2, 104)
        quiet_axis(axis)
    axes[0].set_ylabel("Failures routed to review (%)")
    fig.supxlabel("Nominal review budget (%)", y=-0.01)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.45)
    finish(fig, "fig_review_budget_failure_capture")


def plot_interventions() -> None:
    data = rows(COLLAB / "intervention_effectiveness.csv")
    interventions = (
        "Repository evidence provision",
        "Ordinary verification and repair",
        "Uncertainty-aware evidence and generation",
    )
    labels = ("Repository evidence", "Verification + repair", "Uncertainty layer")
    y = np.arange(len(MODELS))
    offsets = (-0.20, 0.0, 0.20)
    colors = ("#2D5F8B", "#D18C19", "#7B4F88")
    markers = ("o", "s", "^")
    fig, axis = plt.subplots(figsize=(7.15, 2.65))
    for intervention, label, offset, color, marker in zip(interventions, labels, offsets, colors, markers):
        selected = {row["model"]: row for row in data if row["intervention"] == intervention and row["status"] == "OBSERVED"}
        values = np.asarray([100 * f(selected[model]["absolute_difference"]) for model in MODELS])
        low = np.asarray([100 * f(selected[model]["bootstrap_ci95_low"]) for model in MODELS])
        high = np.asarray([100 * f(selected[model]["bootstrap_ci95_high"]) for model in MODELS])
        axis.errorbar(
            values, y + offset, xerr=np.vstack((values - low, high - values)), fmt=marker,
            color=color, markeredgecolor="white", markeredgewidth=0.5, markersize=5.0,
            linewidth=1.1, capsize=2.2, label=label,
        )
    axis.axvline(0, color="#747A80", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
    axis.invert_yaxis()
    axis.set_xlabel("Change in selected executable correctness (percentage points)")
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    quiet_axis(axis, "x")
    fig.tight_layout()
    finish(fig, "fig_intervention_effectiveness")


def plot_language_transfer() -> None:
    data = rows(COLLAB / "generalizability_language.csv")
    languages = ("python", "java", "typescript", "csharp")
    labels = ("Python", "Java", "TypeScript", "C#")
    x = np.arange(len(languages))
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    fig, axis = plt.subplots(figsize=(7.15, 2.75))
    markers = ("o", "s", "^", "D")
    for offset, model, marker in zip(offsets, MODELS, markers):
        selected = {row["language"]: row for row in data if row["model"] == model}
        values = np.asarray([100 * f(selected[language]["exact_match_difference"]) for language in languages])
        low = np.asarray([100 * f(selected[language]["bootstrap_ci95_low"]) for language in languages])
        high = np.asarray([100 * f(selected[language]["bootstrap_ci95_high"]) for language in languages])
        axis.errorbar(
            x + offset, values, yerr=np.vstack((values - low, high - values)), fmt=marker,
            color=COLORS[model], markeredgecolor="white", markeredgewidth=0.5,
            markersize=4.7, linewidth=1.0, capsize=2.0, label=MODEL_LABELS[model],
        )
    axis.axhline(0, color="#747A80", linewidth=0.8, linestyle="--")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Context effect on exact match (points)")
    axis.legend(frameon=False, ncol=4, loc="upper left")
    quiet_axis(axis)
    fig.tight_layout()
    finish(fig, "fig_cross_language_context_effect")


def plot_main_effectiveness() -> None:
    data = rows(EXEC / "summary.csv")
    methods = ("Direct Generation", "Standard RAG", "RAG + Verify/Repair", "OpenCoderX")
    labels = ("Direct", "RAG", "RAG + V/R", "OpenCoderX")
    x = np.arange(len(MODELS))
    width = 0.18
    method_colors = ("#AEB7C0", "#2D5F8B", "#D18C19", "#7B4F88")
    hatches = ("//", "..", "xx", "\\\\")
    fig, axis = plt.subplots(figsize=(7.15, 2.85))
    for index, (method, label, color, hatch) in enumerate(zip(methods, labels, method_colors, hatches)):
        values = [
            100 * f(next(row for row in data if row["model"] == model and row["method"] == method)["selected_output_correctness"])
            for model in MODELS
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color, edgecolor="#4B5258", linewidth=0.45, hatch=hatch)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS])
    axis.set_ylabel("Selected executable correctness (%)")
    axis.set_ylim(0, 82)
    axis.legend(frameon=False, ncol=4, loc="upper left")
    quiet_axis(axis)
    fig.tight_layout()
    finish(fig, "fig_four_family_effectiveness")


def plot_calibration() -> None:
    data = rows(EXEC / "task_level.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.35), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        selected = [row for row in data if row["model"] == model and row["method"] == "OpenCoderX"]
        risk = np.asarray([f(row["uncertainty"]) for row in selected])
        failure = np.asarray([0.0 if row["selected_output_correct"] == "True" else 1.0 for row in selected])
        order = np.argsort(risk, kind="stable")
        groups = np.array_split(order, 5)
        predicted = np.asarray([np.mean(risk[group]) for group in groups])
        observed = np.asarray([np.mean(failure[group]) for group in groups])
        axis.plot((0, 1), (0, 1), color="#747A80", linestyle="--", linewidth=0.8, label="Ideal")
        axis.plot(predicted, observed, color=COLORS[model], marker="o", markeredgecolor="white",
                  markeredgewidth=0.5, linewidth=1.4, label="Observed")
        axis.set_title(MODEL_LABELS[model], pad=7)
        axis.set_xlim(0.45, 0.85)
        axis.set_ylim(0, 1.02)
        quiet_axis(axis)
    axes[0].set_ylabel("Observed failure rate")
    fig.supxlabel("Mean predicted failure risk (equal-frequency bins)", y=-0.02)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.45)
    finish(fig, "fig_uncertainty_calibration")


def plot_risk_coverage() -> None:
    data = rows(RQ12 / "risk_coverage.csv")
    fig, axis = plt.subplots(figsize=(7.15, 2.75))
    markers = ("o", "s", "^", "D")
    for model, marker in zip(MODELS, markers):
        selected = sorted(
            (row for row in data if row["model"] == model and row["signal"] == "aggregate"),
            key=lambda row: f(row["coverage"]),
        )
        axis.plot(
            [100 * f(row["coverage"]) for row in selected],
            [100 * f(row["selective_accuracy"]) for row in selected],
            color=COLORS[model], marker=marker, markeredgecolor="white", markeredgewidth=0.5,
            linewidth=1.4, markersize=4.2, label=MODEL_LABELS[model],
        )
    axis.set_xlabel("Autonomous coverage (%)")
    axis.set_ylabel("Selective accuracy (%)")
    axis.set_xlim(7, 103)
    axis.set_ylim(20, 100)
    axis.legend(frameon=False, ncol=4, loc="upper right")
    quiet_axis(axis)
    fig.tight_layout()
    finish(fig, "fig_risk_coverage")


def main() -> None:
    setup()
    plot_uncertainty_signals()
    plot_selective_autonomy()
    plot_review_budget()
    plot_failure_capture_budget()
    plot_interventions()
    plot_language_transfer()
    plot_main_effectiveness()
    plot_calibration()
    plot_risk_coverage()
    print(f"Wrote publication figures to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
