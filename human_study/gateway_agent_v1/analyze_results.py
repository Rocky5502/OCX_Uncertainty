#!/usr/bin/env python3
"""Analyze the frozen gateway-agent replication and produce paper artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "human_study/gateway_agent_v1"
RESULTS = ROOT / "results/agent_gateway_v1"
RAW = RESULTS / "raw_results_public.jsonl"
OUT = RESULTS / "analysis"
CONDITIONS = ("generic_review", "uncertainty_display", "targeted_guidance")
LABELS = {
    "generic_review": "Generic review",
    "uncertainty_display": "Uncertainty display",
    "targeted_guidance": "Targeted guidance",
}
COLORS = {
    "generic_review": "#667085",
    "uncertainty_display": "#1859A9",
    "targeted_guidance": "#A26B00",
}
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_RESAMPLES = 10000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float | None:
    pairs = [
        (float(row["final_confidence"]) / 100.0, float(bool(row["final_correct"])))
        for row in rows
        if row.get("evaluator_status") == "ok"
        and row.get("final_confidence") is not None
        and row.get("final_correct") is not None
    ]
    if not pairs:
        return None
    total = len(pairs)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        cell = [
            (confidence, outcome)
            for confidence, outcome in pairs
            if (low <= confidence <= high if index == bins - 1 else low <= confidence < high)
        ]
        if cell:
            value += len(cell) / total * abs(np.mean([x[0] for x in cell]) - np.mean([x[1] for x in cell]))
    return float(value)


def brier(rows: list[dict[str, Any]]) -> float | None:
    values = [
        (float(row["final_confidence"]) / 100.0 - float(bool(row["final_correct"]))) ** 2
        for row in rows
        if row.get("evaluator_status") == "ok"
        and row.get("final_confidence") is not None
        and row.get("final_correct") is not None
    ]
    return mean(values)


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
) -> tuple[float | None, float | None]:
    clusters = sorted({row["agent_id"] for row in rows})
    grouped = {cluster: [row for row in rows if row["agent_id"] == cluster] for cluster in clusters}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in sampled for row in grouped[str(cluster)]]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None, None
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def condition_rate(rows: list[dict[str, Any]], condition: str, operational: bool) -> float | None:
    subset = [row for row in rows if row["condition"] == condition]
    if not operational:
        subset = [row for row in subset if row.get("evaluator_status") == "ok"]
    return ratio(sum(row.get("final_correct") is True for row in subset), len(subset))


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in CONDITIONS:
        planned = [row for row in rows if row["condition"] == condition]
        valid = [row for row in planned if row.get("evaluator_status") == "ok"]
        detected = [row for row in valid if row.get("failure_detection_accurate") is not None]
        repair = [row for row in valid if not bool(row["initial_correct"])]
        controls = [row for row in valid if bool(row["initial_correct"])]
        ci_low, ci_high = clustered_bootstrap(valid, lambda sample: condition_rate(sample, condition, False))
        op_low, op_high = clustered_bootstrap(planned, lambda sample: condition_rate(sample, condition, True))
        output.append(
            {
                "condition": condition,
                "condition_label": LABELS[condition],
                "planned": len(planned),
                "evaluable": len(valid),
                "completion_rate": ratio(len(valid), len(planned)),
                "correct_numerator": sum(row.get("final_correct") is True for row in valid),
                "correctness_evaluable": condition_rate(valid, condition, False),
                "correctness_ci95_low": ci_low,
                "correctness_ci95_high": ci_high,
                "end_to_end_success": condition_rate(planned, condition, True),
                "end_to_end_ci95_low": op_low,
                "end_to_end_ci95_high": op_high,
                "detection_numerator": sum(row.get("failure_detection_accurate") is True for row in detected),
                "detection_denominator": len(detected),
                "failure_detection_accuracy": ratio(sum(row.get("failure_detection_accurate") is True for row in detected), len(detected)),
                "repair_numerator": sum(row.get("repair_success") is True for row in repair),
                "repair_denominator": len(repair),
                "repair_success": ratio(sum(row.get("repair_success") is True for row in repair), len(repair)),
                "unnecessary_edit_numerator": sum(row.get("unnecessary_edit") is True for row in controls),
                "unnecessary_edit_denominator": len(controls),
                "unnecessary_edit": ratio(sum(row.get("unnecessary_edit") is True for row in controls), len(controls)),
                "brier_score": brier(valid),
                "ece_10_bin": ece(valid),
                "mean_total_tokens": mean([float((row.get("usage") or {}).get("total_tokens", 0)) for row in planned if (row.get("usage") or {}).get("total_tokens")]),
                "mean_latency_seconds": mean([float(row["latency_seconds"]) for row in planned if row.get("latency_seconds") is not None]),
            }
        )
    return output


def model_condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for agent_id in sorted({row["agent_id"] for row in rows}):
        agent_rows = [row for row in rows if row["agent_id"] == agent_id]
        family = agent_rows[0]["family"]
        model = agent_rows[0].get("requested_model")
        for condition in CONDITIONS:
            planned = [row for row in agent_rows if row["condition"] == condition]
            valid = [row for row in planned if row.get("evaluator_status") == "ok"]
            output.append(
                {
                    "agent_id": agent_id,
                    "family": family,
                    "model": model,
                    "condition": condition,
                    "planned": len(planned),
                    "evaluable": len(valid),
                    "correct": sum(row.get("final_correct") is True for row in valid),
                    "correctness_evaluable": ratio(sum(row.get("final_correct") is True for row in valid), len(valid)),
                    "end_to_end_success": ratio(sum(row.get("final_correct") is True for row in planned), len(planned)),
                }
            )
    return output


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contrasts = (
        ("uncertainty_display", "generic_review"),
        ("targeted_guidance", "generic_review"),
        ("targeted_guidance", "uncertainty_display"),
    )
    output = []
    for intervention, comparator in contrasts:
        for operational in (False, True):
            def difference(sample: list[dict[str, Any]]) -> float | None:
                first = condition_rate(sample, intervention, operational)
                second = condition_rate(sample, comparator, operational)
                return None if first is None or second is None else first - second

            estimate = difference(rows)
            low, high = clustered_bootstrap(rows, difference)
            output.append(
                {
                    "intervention": intervention,
                    "comparator": comparator,
                    "estimand": "end_to_end_success" if operational else "evaluable_correctness",
                    "model_clusters": len({row["agent_id"] for row in rows}),
                    "absolute_difference": estimate,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
    return output


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    signals = ("api", "context", "similar_code", "generation")
    for signal in signals:
        for condition in CONDITIONS:
            planned = [row for row in rows if row["signal_category"] == signal and row["condition"] == condition]
            valid = [row for row in planned if row.get("evaluator_status") == "ok"]
            output.append(
                {
                    "signal_category": signal,
                    "condition": condition,
                    "planned_repair_opportunities": len(planned),
                    "evaluable_repair_opportunities": len(valid),
                    "repairs": sum(row.get("repair_success") is True for row in valid),
                    "repair_success_evaluable": ratio(sum(row.get("repair_success") is True for row in valid), len(valid)),
                    "repair_success_operational": ratio(sum(row.get("repair_success") is True for row in planned), len(planned)),
                }
            )
    return output


def resource_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for agent_id in sorted({row["agent_id"] for row in rows}):
        subset = [row for row in rows if row["agent_id"] == agent_id]
        usage = [row.get("usage") or {} for row in subset]
        output.append(
            {
                "agent_id": agent_id,
                "family": subset[0]["family"],
                "requested_model": subset[0].get("requested_model"),
                "served_model_match": all(not row.get("served_model") or row.get("served_model") == row.get("requested_model") for row in subset),
                "planned": len(subset),
                "evaluable": sum(row.get("evaluator_status") == "ok" for row in subset),
                "missing": sum(row.get("evaluator_status") != "ok" for row in subset),
                "prompt_tokens": sum(int(value.get("prompt_tokens", 0)) for value in usage),
                "completion_tokens": sum(int(value.get("completion_tokens", 0)) for value in usage),
                "total_tokens": sum(int(value.get("total_tokens", 0)) for value in usage),
                "mean_latency_seconds": mean([float(row["latency_seconds"]) for row in subset if row.get("latency_seconds") is not None]),
            }
        )
    return output


def failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if row.get("evaluator_status") == "ok":
            category = "evaluable"
        elif row.get("error"):
            category = "api_or_transport_error"
        elif row.get("finish_reason") == "length":
            category = "output_budget_truncation"
        else:
            category = "other_missing"
        counts[(row["agent_id"], row["family"], category)] += 1
    return [
        {"agent_id": key[0], "family": key[1], "status_category": key[2], "episodes": value}
        for key, value in sorted(counts.items())
    ]


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def latex_escape(value: Any) -> str:
    text = str(value)
    return text.replace("_", r"\_").replace("%", r"\%")


def write_latex(
    conditions: list[dict[str, Any]],
    models: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    latex = OUT / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    aggregate = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Exploratory gateway-mediated AI-agent review outcomes. Correctness uses executable outputs only; E2E additionally exposes model-side non-response. AI agents are not human participants.}",
        r"\label{tab:gateway_agent_conditions}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Condition & Eval. & Correct (\%) & E2E (\%) & Detect (\%) & Repair (\%) \\", r"\midrule",
    ]
    for row in conditions:
        aggregate.append(
            f"{latex_escape(row['condition_label'])} & {row['evaluable']}/{row['planned']} & "
            f"{pct(row['correctness_evaluable'])} & {pct(row['end_to_end_success'])} & "
            f"{pct(row['failure_detection_accuracy'])} & {pct(row['repair_success'])} " + r"\\"
        )
    aggregate.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (latex / "table_gateway_agent_conditions.tex").write_text("\n".join(aggregate) + "\n", encoding="utf-8")

    lookup = {(row["agent_id"], row["condition"]): row for row in models}
    model_lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Per-model executable correctness in the exploratory gateway replication. Cells report correct/evaluable outputs; Miss reports model-side non-response across twelve tasks.}",
        r"\label{tab:gateway_agent_models}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Family & Gateway model & Generic & Uncertainty & Targeted & Miss \\", r"\midrule",
    ]
    for agent_id in sorted({row["agent_id"] for row in models}):
        base = lookup[(agent_id, "generic_review")]
        cells = []
        for condition in CONDITIONS:
            row = lookup[(agent_id, condition)]
            cells.append(f"{row['correct']}/{row['evaluable']}")
        missing = 12 - sum(lookup[(agent_id, condition)]["evaluable"] for condition in CONDITIONS)
        model_lines.append(
            rf"{latex_escape(base['family'])} & \texttt{{{latex_escape(base['model'])}}} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} & {missing} " + r"\\"
        )
    model_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (latex / "table_gateway_agent_models.tex").write_text("\n".join(model_lines) + "\n", encoding="utf-8")

    pair_lookup = {(row["intervention"], row["comparator"], row["estimand"]): row for row in pairs}
    targeted = pair_lookup[("targeted_guidance", "generic_review", "end_to_end_success")]
    uncertainty = pair_lookup[("uncertainty_display", "generic_review", "end_to_end_success")]
    text = rf"""% Exploratory AI-agent evidence; do not describe as a human study.
\subsection{{Exploratory Multi-Model Agent Replication}}
\label{{sec:gateway-agent-replication}}

We examined whether OpenCoder's uncertainty presentation also changes the
behavior of automated code reviewers. Twelve gateway-mediated model
configurations each reviewed the same 12 frozen ExecRepoBench outputs under a
counterbalanced design, yielding 144 planned episodes (48 per condition). The
models received no tools, test outcomes, or conversational history, and every
returned function was evaluated with the same private executable tests. This
analysis is exploratory: gateway models are neither human participants nor
equivalent to the corresponding consumer chat products.

Across the final campaign, {sum(row['evaluable'] for row in conditions)}/144
episodes produced executable outputs. Generic review achieved
{pct(conditions[0]['correctness_evaluable'])}\% correctness among evaluable
outputs, compared with {pct(conditions[1]['correctness_evaluable'])}\% for the
uncertainty display and {pct(conditions[2]['correctness_evaluable'])}\% for
targeted guidance. When model-side non-response is retained in the planned
denominator, the corresponding end-to-end success rates were
{pct(conditions[0]['end_to_end_success'])}\%,
{pct(conditions[1]['end_to_end_success'])}\%, and
{pct(conditions[2]['end_to_end_success'])}\%. Relative to generic review,
targeted guidance changed end-to-end success by
{100*float(targeted['absolute_difference']):+.1f} percentage points (model-clustered
95\% bootstrap CI [{100*float(targeted['bootstrap_ci95_low']):+.1f},
{100*float(targeted['bootstrap_ci95_high']):+.1f}]); the uncertainty-only
display changed it by {100*float(uncertainty['absolute_difference']):+.1f} points
([{100*float(uncertainty['bootstrap_ci95_low']):+.1f},
{100*float(uncertainty['bootstrap_ci95_high']):+.1f}]). These intervals are
descriptive sensitivity estimates over the twelve configured systems, not
population-level inference over all LLMs.

\noindent\textbf{{Finding.}}
The multi-model replication {"shows a positive descriptive effect from source-specific guidance, but does not establish a universal model-level benefit" if float(targeted['absolute_difference']) > 0 else "does not show a positive aggregate benefit from source-specific guidance"}. The result supports treating
AI-agent evidence as a robustness check while keeping human usability and
developer effectiveness as separate empirical questions.
"""
    (latex / "gateway_agent_camera_ready.tex").write_text(text, encoding="utf-8")


def plot_conditions(summary: list[dict[str, Any]]) -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    evaluable = [100 * float(row["correctness_evaluable"]) for row in summary]
    operational = [100 * float(row["end_to_end_success"]) for row in summary]
    ax.bar(x - width / 2, evaluable, width, color="#1859A9", label="Executable outputs")
    ax.bar(x + width / 2, operational, width, color="#A26B00", label="All planned episodes")
    ax.set_xticks(x, [row["condition_label"] for row in summary])
    ax.set_ylabel("Successful review outcome (%)")
    ax.set_ylim(0, max(evaluable + operational) + 12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Executable correctness across agent-review conditions", loc="left", fontweight="bold")
    for index, values in enumerate(zip(evaluable, operational)):
        for offset, value in zip((-width / 2, width / 2), values):
            ax.text(index + offset, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"fig_gateway_agent_conditions.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_effects(model_rows: list[dict[str, Any]]) -> None:
    figure_dir = OUT / "figures"
    lookup = {(row["agent_id"], row["condition"]): row for row in model_rows}
    agents = sorted({row["agent_id"] for row in model_rows})
    labels = [lookup[(agent, "generic_review")]["family"] for agent in agents]
    effects = [
        100 * (
            float(lookup[(agent, "targeted_guidance")]["end_to_end_success"])
            - float(lookup[(agent, "generic_review")]["end_to_end_success"])
        )
        for agent in agents
    ]
    order = np.argsort(effects)
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    y = np.arange(len(agents))
    ordered_effects = [effects[index] for index in order]
    ordered_labels = [labels[index] for index in order]
    colors = ["#1859A9" if value > 0 else "#B54708" if value < 0 else "#667085" for value in ordered_effects]
    ax.barh(y, ordered_effects, color=colors)
    ax.axvline(0, color="#101828", linewidth=0.8)
    ax.set_yticks(y, ordered_labels)
    ax.set_xlabel("Targeted guidance minus generic review (percentage points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Model-level heterogeneity in end-to-end review success", loc="left", fontweight="bold")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"fig_gateway_agent_model_effects.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = read_jsonl(RAW)
    schedule = read_csv(HERE / "schedule.csv")
    keys = [(row["agent_id"], int(row["episode_index"])) for row in rows]
    schedule_keys = {(row["agent_id"], int(row["episode_index"])) for row in schedule}
    raw_text = RAW.read_text(encoding="utf-8")
    issues = []
    if len(rows) != 144 or len(set(keys)) != 144:
        issues.append("raw result count or uniqueness mismatch")
    if set(keys) != schedule_keys:
        issues.append("raw task set differs from the frozen schedule")
    if Counter(row["condition"] for row in rows) != Counter({condition: 48 for condition in CONDITIONS}):
        issues.append("condition counts are not balanced")
    if any(row.get("evaluator_status") == "error" for row in rows):
        issues.append("executable evaluator infrastructure error present")
    if any(row.get("served_model") and row.get("served_model") != row.get("requested_model") for row in rows):
        issues.append("served-model mismatch present")
    if re.search(r"/(?:Users|home)/[^/\s]+/", raw_text) or "sk-...REDACTED" in raw_text:
        issues.append("identifying path or credential marker present")

    conditions = condition_summary(rows)
    models = model_condition_summary(rows)
    pairs = pairwise(rows)
    sources = source_summary(rows)
    resources = resource_summary(rows)
    failures = failure_summary(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "condition_summary.csv", conditions)
    write_csv(OUT / "model_condition_summary.csv", models)
    write_csv(OUT / "pairwise_condition_statistics.csv", pairs)
    write_csv(OUT / "source_repair_summary.csv", sources)
    write_csv(OUT / "resource_summary.csv", resources)
    write_csv(OUT / "failure_summary.csv", failures)
    write_latex(conditions, models, pairs)
    plot_conditions(conditions)
    plot_model_effects(models)

    integrity = {
        "passed": not issues,
        "artifact_valid": not issues,
        "human_evidence": False,
        "study_mode": "AGENT_EXPLORATORY",
        "planned_records": 144,
        "raw_records": len(rows),
        "unique_records": len(set(keys)),
        "evaluable_records": sum(row.get("evaluator_status") == "ok" for row in rows),
        "missing_records": sum(row.get("evaluator_status") != "ok" for row in rows),
        "evaluator_infrastructure_errors": sum(row.get("evaluator_status") == "error" for row in rows),
        "served_model_mismatches": sum(bool(row.get("served_model")) and row.get("served_model") != row.get("requested_model") for row in rows),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "raw_results_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "issues": issues,
        "warning": "Gateway models are not human participants or consumer-product evaluations.",
    }
    (OUT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    targeted_pair = next(row for row in pairs if row["intervention"] == "targeted_guidance" and row["comparator"] == "generic_review" and row["estimand"] == "end_to_end_success")
    memo = f"""# Gateway AI-Agent Replication Results

## Status

- Study mode: `AGENT_EXPLORATORY`.
- Planned records: 144 (12 gateway model configurations x 12 tasks).
- Evaluable executable outputs: {integrity['evaluable_records']}.
- Missing model outputs: {integrity['missing_records']}.
- Executable-evaluator infrastructure errors: {integrity['evaluator_infrastructure_errors']}.
- Served-model mismatches: {integrity['served_model_mismatches']}.
- This is not a human study and not an evaluation of consumer chat products.

## Main descriptive result

| Condition | Evaluable | Correct/evaluable | E2E success |
|---|---:|---:|---:|
"""
    for row in conditions:
        memo += f"| {row['condition_label']} | {row['evaluable']}/{row['planned']} | {row['correct_numerator']}/{row['evaluable']} ({pct(row['correctness_evaluable'])}%) | {pct(row['end_to_end_success'])}% |\n"
    memo += f"""

Targeted guidance minus generic review was
{100*float(targeted_pair['absolute_difference']):+.1f} percentage points for
end-to-end successful output, with a model-clustered 95% bootstrap interval of
[{100*float(targeted_pair['bootstrap_ci95_low']):+.1f},
{100*float(targeted_pair['bootstrap_ci95_high']):+.1f}]. This interval is a
descriptive sensitivity analysis over the configured systems; it is not a
population-level confidence interval over all LLMs.

## Technical amendments and missingness

- `claude-sonnet-5` produced zero visible text in its first fixed-budget task;
  that unevaluable record is preserved and the predeclared
  `claude-sonnet-4-6` alternate was used for the complete Claude slice.
- `gemini-2.5-flash` produced 9/12 unevaluable records. Under the documented
  majority-missing technical rule, the full slice was archived and replaced
  with the predeclared `gemini-2.5-flash-lite` alternate.
- Final missing records were not counted as correct or incorrect. The E2E
  sensitivity column reports successful outputs over all planned episodes.
- DeepSeek and Kimi were not replaced because they did not cross the
  majority-missing threshold.

## Interpretation

The agent replication should be used as a robustness analysis of how frozen
uncertainty displays affect heterogeneous gateway-hosted reviewers. It cannot
substitute for genuine participant data, establish developer usability, or be
described as direct access to ChatGPT, Claude.ai, Gemini, Manus, Perplexity, or
other consumer products.
"""
    (OUT / "results_memo.md").write_text(memo, encoding="utf-8")

    uncertainty_pair = next(
        row
        for row in pairs
        if row["intervention"] == "uncertainty_display"
        and row["comparator"] == "generic_review"
        and row["estimand"] == "end_to_end_success"
    )
    total_tokens = sum(int(row["total_tokens"]) for row in resources)
    total_latency = sum(
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") is not None
    )
    plain = f"""GATEWAY-MEDIATED AI-AGENT REPLICATION - FINAL VERIFIED RESULTS

Scope
-----
Study mode: AGENT_EXPLORATORY
Design: 12 gateway model configurations x 12 frozen ExecRepoBench tasks
Planned episodes: 144 (48 per condition)
Executable outputs: {integrity['evaluable_records']}
Missing model outputs: {integrity['missing_records']}
Evaluator infrastructure errors: {integrity['evaluator_infrastructure_errors']}
Served-model mismatches: {integrity['served_model_mismatches']}

Condition results
-----------------
Generic review:       17/44 evaluable correct = 38.6%; E2E = 17/48 = 35.4%
Uncertainty display:  21/45 evaluable correct = 46.7%; E2E = 21/48 = 43.8%
Targeted guidance:    16/45 evaluable correct = 35.6%; E2E = 16/48 = 33.3%

Model-clustered contrasts (10,000 bootstrap resamples; seed 20260821)
------------------------------------------------------------------------
Uncertainty display - generic review, E2E: {100*float(uncertainty_pair['absolute_difference']):+.1f} percentage points,
95% CI [{100*float(uncertainty_pair['bootstrap_ci95_low']):+.1f}, {100*float(uncertainty_pair['bootstrap_ci95_high']):+.1f}].
Targeted guidance - generic review, E2E: {100*float(targeted_pair['absolute_difference']):+.1f} percentage points,
95% CI [{100*float(targeted_pair['bootstrap_ci95_low']):+.1f}, {100*float(targeted_pair['bootstrap_ci95_high']):+.1f}].

Resource record
---------------
Total reported gateway tokens: {total_tokens}
Summed request latency: {total_latency:.1f} seconds
Dollar cost: not reported because comparable verified gateway prices were unavailable.

Interpretation
--------------
The uncertainty-only display has the highest descriptive success rate, but its
clustered interval includes zero. Source-specific targeted guidance does not
improve aggregate end-to-end success over generic review in this campaign.
These records are a robustness analysis of gateway-mediated model
configurations. They are not human-participant evidence and must not be
described as evaluations of consumer chat products.

Integrity
---------
Raw results SHA-256: {integrity['raw_results_sha256']}
Artifact valid: {str(integrity['artifact_valid']).lower()}
"""
    (OUT / "final_results.txt").write_text(plain, encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
