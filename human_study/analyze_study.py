#!/usr/bin/env python3
"""Analyze genuine human data or explicitly labeled synthetic dry-run records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opencoder-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "opencoder-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.formula.api import gee


ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = json.loads((ROOT / "human_study/study_config.json").read_text(encoding="utf-8"))
CONDITIONS = ("generic_review", "uncertainty_display", "targeted_guidance")
LABELS = {
    "generic_review": "Generic review",
    "uncertainty_display": "Uncertainty display",
    "targeted_guidance": "Targeted guidance",
}
COLORS = {
    "generic_review": "#667085",
    "uncertainty_display": "#1859a9",
    "targeted_guidance": "#a26b00",
}
SEED = 20260821
BOOTSTRAPS = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ethics_approval(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("empirical analysis requires --ethics-approval-file")
    approval = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "human_study/study_config.json").read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED_FOR_RECRUITMENT":
        raise RuntimeError("ethics approval status does not unlock recruitment")
    if approval.get("protocol_version") != config["protocol_version"]:
        raise RuntimeError("ethics approval protocol version does not match the frozen study")
    required = (
        "institution", "ethics_body_or_authority", "protocol_identifier",
        "determination", "determination_date", "principal_investigator", "contact",
        "compensation", "retention_period",
    )
    for field in required:
        if not str(approval.get(field) or "").strip() or str(approval[field]).startswith("REPLACE"):
            raise RuntimeError(f"ethics approval field {field!r} is incomplete")
    return approval


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def as_optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return as_bool(value)


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}"


def decimal(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def mean_bool(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [as_optional_bool(row.get(field)) for row in rows]
    known = [float(value) for value in values if value is not None]
    return float(np.mean(known)) if known else None


def calibration_error(confidences: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    if not len(confidences):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(confidences, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(confidences[mask])) - float(np.mean(outcomes[mask]))
            )
    return error


def bootstrap_participants(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None], seed_offset: int
) -> tuple[float | None, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_code"])].append(row)
    codes = sorted(grouped)
    if not codes:
        return None, None
    rng = np.random.default_rng(SEED + seed_offset)
    values = []
    for _ in range(BOOTSTRAPS):
        sample = []
        for code in rng.choice(codes, size=len(codes), replace=True):
            sample.extend(grouped[str(code)])
        value = statistic(sample)
        if value is not None:
            values.append(float(value))
    if not values:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def holm(rows: list[dict[str, Any]], p_field: str, output_field: str) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1][p_field]))
    running = 0.0
    adjusted: dict[int, float] = {}
    total = len(rows)
    for rank, (index, row) in enumerate(indexed):
        value = min(1.0, (total - rank) * float(row[p_field]))
        running = max(running, value)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row[output_field] = adjusted[index]


def exclusion_audit(
    participants: list[dict[str, Any]], episodes: list[dict[str, Any]], withdrawals: set[str],
    tutorial_passed: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in participants:
        latest[str(row["participant_code"])] = row
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_participant[str(row["participant_code"])].append(row)
    flow = []
    retained = []
    retained_codes = set()
    for code, profile in sorted(latest.items()):
        rows = by_participant.get(code, [])
        reasons = []
        if code in withdrawals or as_bool(profile.get("withdrawn", False)):
            reasons.append("withdrawn")
        if not as_bool(profile.get("consent")) or not as_bool(profile.get("adult")):
            reasons.append("missing_consent_or_adult_eligibility")
        if float(profile.get("programming_years", 0)) < 1 or float(profile.get("python_years", 0)) < 1:
            reasons.append("experience_ineligible")
        familiar = (
            float(profile.get("code_review_frequency", 0)) >= 1
            or float(profile.get("repository_development_frequency", 0)) >= 1
            or float(profile.get("testing_familiarity", 0)) >= 2
        )
        if not familiar:
            reasons.append("familiarity_ineligible")
        if tutorial_passed is not None and code not in tutorial_passed:
            reasons.append("tutorial_not_completed")
        if len(rows) < 9:
            reasons.append("fewer_than_nine_tasks")
        if sum(float(row.get("elapsed_seconds", 0)) < 30 for row in rows) >= 4:
            reasons.append("four_implausibly_short_tasks")
        if sum(row.get("evaluator_status") == "error" for row in rows) > 3:
            reasons.append("more_than_three_technical_failures")
        excluded = bool(reasons)
        flow.append({
            "participant_code": code, "completed_episodes": len(rows),
            "excluded": excluded, "exclusion_reasons": ";".join(reasons),
        })
        if not excluded:
            retained_codes.add(code)
            retained.append(profile)
    retained_episodes = [row for row in episodes if row["participant_code"] in retained_codes]
    return retained, retained_episodes, flow


def expertise(profile: dict[str, Any]) -> float:
    components = [
        min(float(profile["programming_years"]) / 10.0, 1.0),
        min(float(profile["python_years"]) / 8.0, 1.0),
        float(profile["code_review_frequency"]) / 5.0,
        float(profile["repository_development_frequency"]) / 5.0,
        float(profile["testing_familiarity"]) / 5.0,
        float(profile["dependency_tracing_familiarity"]) / 5.0,
    ]
    return float(np.mean(components))


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for offset, condition in enumerate(CONDITIONS):
        subset = [row for row in rows if row["condition"] == condition]
        evaluable = [row for row in subset if row.get("evaluator_status") == "ok" and as_optional_bool(row.get("final_correct")) is not None]
        correct = mean_bool(evaluable, "final_correct")
        ci = bootstrap_participants(evaluable, lambda sample: mean_bool(sample, "final_correct"), offset)
        repair = [row for row in evaluable if as_bool(row["repair_opportunity"])]
        controls = [row for row in evaluable if as_bool(row["initial_correct"])]
        confidences = np.asarray([float(row["final_confidence"]) / 100.0 for row in evaluable])
        outcomes = np.asarray([float(as_bool(row["final_correct"])) for row in evaluable])
        output.append({
            "condition": condition, "condition_label": LABELS[condition],
            "participants": len({row["participant_code"] for row in subset}),
            "episodes": len(subset), "evaluable_episodes": len(evaluable),
            "correct_numerator": sum(as_bool(row["final_correct"]) for row in evaluable),
            "correctness": correct, "correctness_ci95_low": ci[0], "correctness_ci95_high": ci[1],
            "failure_detection_accuracy": mean_bool(subset, "failure_detection_accurate"),
            "repair_opportunities": len(repair), "repair_success": mean_bool(repair, "repair_success"),
            "accepted_incorrect": mean_bool(repair, "accepted_incorrect"),
            "correct_controls": len(controls), "unnecessary_edit": mean_bool(controls, "unnecessary_edit"),
            "median_elapsed_seconds": float(np.median([float(row["elapsed_seconds"]) for row in subset])) if subset else None,
            "timeout_rate": mean_bool(subset, "timed_out"),
            "mean_difficulty": float(np.mean([float(row["difficulty"]) for row in subset])) if subset else None,
            "mean_guidance_usefulness": float(np.mean([float(row["guidance_usefulness"]) for row in subset if row.get("guidance_usefulness") not in (None, "")])) if any(row.get("guidance_usefulness") not in (None, "") for row in subset) else None,
            "brier_score": float(np.mean(np.square(confidences - outcomes))) if len(evaluable) else None,
            "ece_10_bin": calibration_error(confidences, outcomes) if len(evaluable) else None,
        })
    return output


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    for code in {row["participant_code"] for row in rows}:
        participant_rows = [row for row in rows if row["participant_code"] == code and row.get("evaluator_status") == "ok"]
        for condition in CONDITIONS:
            value = mean_bool([row for row in participant_rows if row["condition"] == condition], "final_correct")
            if value is not None:
                rates[code][condition] = value
    comparisons = [
        ("uncertainty_display", "generic_review"),
        ("targeted_guidance", "generic_review"),
        ("targeted_guidance", "uncertainty_display"),
    ]
    output = []
    rng = np.random.default_rng(SEED + 100)
    for intervention, comparator in comparisons:
        codes = [code for code in sorted(rates) if intervention in rates[code] and comparator in rates[code]]
        differences = np.asarray([rates[code][intervention] - rates[code][comparator] for code in codes], dtype=float)
        if not len(differences):
            raise RuntimeError(f"no paired participants for {intervention} vs {comparator}")
        boot = np.mean(rng.choice(differences, size=(BOOTSTRAPS, len(differences)), replace=True), axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        p_value = 1.0 if np.allclose(differences, 0.0) else float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
        output.append({
            "intervention": intervention, "comparator": comparator,
            "matched_participants": len(codes), "absolute_difference": float(np.mean(differences)),
            "bootstrap_ci95_low": float(low), "bootstrap_ci95_high": float(high),
            "wilcoxon_p": p_value,
        })
    holm(output, "wilcoxon_p", "holm_p")
    return output


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for signal in ("api", "context", "similar_code", "generation"):
        for condition in CONDITIONS:
            subset = [
                row for row in rows
                if row["signal_category"] == signal and row["condition"] == condition
                and row.get("evaluator_status") == "ok"
            ]
            output.append({
                "signal_category": signal, "condition": condition,
                "episodes": len(subset), "repair_success": mean_bool(subset, "repair_success"),
                "failure_detection_accuracy": mean_bool(subset, "failure_detection_accurate"),
                "median_elapsed_seconds": float(np.median([float(row["elapsed_seconds"]) for row in subset])) if subset else None,
            })
    return output


def demographic_summary(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize roles without forcing mutually exclusive expertise categories."""
    output: list[dict[str, Any]] = []
    for role in ("software_engineer", "ai_ml_researcher", "phd_student", "other"):
        count = sum(
            profile.get("primary_role") == role or role in profile.get("additional_roles", [])
            for profile in profiles
        )
        output.append({
            "measure": f"role_{role}", "n": count,
            "value": count / len(profiles) if profiles else None,
            "summary": "overlapping role prevalence",
        })
    continuous = (
        ("programming_years", "years"),
        ("python_years", "years"),
        ("ai_tool_usage", "0--5 scale"),
        ("code_review_frequency", "0--5 scale"),
        ("repository_development_frequency", "0--5 scale"),
        ("testing_familiarity", "1--5 scale"),
        ("dependency_tracing_familiarity", "1--5 scale"),
    )
    for field, unit in continuous:
        values = np.asarray([float(profile[field]) for profile in profiles], dtype=float)
        output.append({
            "measure": field, "n": len(values), "value": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "summary": unit,
        })
    expertise_values = np.asarray([expertise(profile) for profile in profiles], dtype=float)
    output.append({
        "measure": "continuous_expertise_score", "n": len(expertise_values),
        "value": float(np.mean(expertise_values)), "median": float(np.median(expertise_values)),
        "sd": float(np.std(expertise_values, ddof=1)) if len(expertise_values) > 1 else 0.0,
        "summary": "0--1 preregistered composite",
    })
    return output


def gee_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluable = [
        row for row in rows
        if row.get("evaluator_status") == "ok"
        and as_optional_bool(row.get("final_correct")) is not None
    ]
    frame = pd.DataFrame(evaluable)
    frame["final_correct_binary"] = frame["final_correct"].map(as_bool).astype(int)
    frame["episode_index"] = frame["episode_index"].astype(float)
    frame["expertise_z"] = frame["expertise_z"].astype(float)
    model = gee(
        "final_correct_binary ~ C(condition, Treatment(reference='generic_review')) + "
        "C(task_id) + episode_index + expertise_z",
        groups=frame["participant_code"], data=frame,
        cov_struct=Exchangeable(), family=Binomial(),
    ).fit(maxiter=200)
    intervals = model.conf_int()
    output = []
    for term in model.params.index:
        coefficient = float(model.params[term])
        low, high = (float(value) for value in intervals.loc[term])
        output.append({
            "term": term, "coefficient_log_odds": coefficient,
            "odds_ratio": math.exp(coefficient),
            "ci95_low_or": math.exp(low), "ci95_high_or": math.exp(high),
            "robust_se": float(model.bse[term]), "p_value": float(model.pvalues[term]),
            "episodes": len(frame),
            "participant_clusters": int(frame["participant_code"].nunique()),
        })
    return output


def completion_time_gee(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([
        row for row in rows if float(row.get("elapsed_seconds", 0)) > 0
    ])
    frame["log_elapsed"] = np.log(frame["elapsed_seconds"].astype(float))
    frame["episode_index"] = frame["episode_index"].astype(float)
    frame["expertise_z"] = frame["expertise_z"].astype(float)
    model = gee(
        "log_elapsed ~ C(condition, Treatment(reference='generic_review')) + "
        "C(task_id) + episode_index + expertise_z",
        groups=frame["participant_code"], data=frame,
        cov_struct=Exchangeable(), family=Gaussian(),
    ).fit(maxiter=200)
    intervals = model.conf_int()
    output = []
    for term in model.params.index:
        coefficient = float(model.params[term])
        low, high = (float(value) for value in intervals.loc[term])
        output.append({
            "term": term, "coefficient_log_seconds": coefficient,
            "geometric_mean_ratio": math.exp(coefficient),
            "ci95_low_ratio": math.exp(low), "ci95_high_ratio": math.exp(high),
            "robust_se": float(model.bse[term]), "p_value": float(model.pvalues[term]),
            "episodes": len(frame),
            "participant_clusters": int(frame["participant_code"].nunique()),
        })
    return output


def poststudy_summary(path: Path | None, retained_codes: set[str]) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        code = str(row["participant_code"])
        if code in retained_codes:
            latest[code] = row
    output = []
    for field, scale in (
        ("mental_demand", "0--20"), ("effort", "0--20"),
        ("frustration", "0--20"), ("temporal_demand", "0--20"),
        ("adoption_intent", "1--5"),
    ):
        values = np.asarray([
            float(row[field]) for row in latest.values() if row.get(field) not in (None, "")
        ])
        output.append({
            "measure": field, "respondents": len(values),
            "missing": len(retained_codes) - len(values),
            "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "scale": scale,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_condition(summary: list[dict[str, Any]], output: Path, stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    x = np.arange(len(summary))
    values = np.asarray([float(row["correctness"]) for row in summary])
    low = np.asarray([float(row["correctness_ci95_low"]) for row in summary])
    high = np.asarray([float(row["correctness_ci95_high"]) for row in summary])
    for index, row in enumerate(summary):
        ax.errorbar(index, values[index], yerr=[[values[index]-low[index]], [high[index]-values[index]]],
                    fmt="o", color=COLORS[row["condition"]], markersize=8, capsize=4, linewidth=2)
        ax.text(index, values[index] + 0.035, f"{100*values[index]:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, [row["condition_label"] for row in summary])
    ax.set_ylabel("Final executable correctness")
    ax.set_ylim(0, min(1.0, max(0.65, float(np.max(high)) + 0.12)))
    ax.yaxis.set_major_formatter(lambda value, _: f"{100*value:.0f}%")
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Executable correctness by review condition", loc="left", fontweight="bold", pad=12)
    fig.text(0.01, 0.01, stamp, fontsize=7.5, color="#667085")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sources(summary: list[dict[str, Any]], output: Path, stamp: str) -> None:
    signals = ("api", "context", "similar_code", "generation")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.22
    x = np.arange(len(signals))
    for offset, condition in enumerate(CONDITIONS):
        lookup = {(row["signal_category"], row["condition"]): row for row in summary}
        values = [
            np.nan if lookup[(signal, condition)]["repair_success"] is None
            else float(lookup[(signal, condition)]["repair_success"])
            for signal in signals
        ]
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width=width, color=COLORS[condition], label=LABELS[condition], edgecolor="#31363f", linewidth=0.5)
    ax.set_xticks(x, [signal.replace("_", " ").title() for signal in signals])
    ax.set_ylabel("Repair success")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{100*value:.0f}%")
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=8)
    ax.set_title("Repair success by uncertainty signal", loc="left", fontweight="bold", pad=12)
    fig.text(0.01, 0.01, stamp, fontsize=7.5, color="#667085")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(rows: list[dict[str, Any]], output: Path, stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    edges = np.linspace(0.0, 1.0, 6)
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="#667085", linewidth=1,
        label="Perfect calibration",
    )
    for condition in CONDITIONS:
        subset = [
            row for row in rows
            if row["condition"] == condition and row.get("evaluator_status") == "ok"
            and as_optional_bool(row.get("final_correct")) is not None
        ]
        confidence = np.asarray([float(row["final_confidence"]) / 100.0 for row in subset])
        outcome = np.asarray([float(as_bool(row["final_correct"])) for row in subset])
        assignment = np.minimum(
            np.digitize(confidence, edges[1:-1], right=False), len(edges) - 2
        )
        x_values, y_values = [], []
        for index in range(len(edges) - 1):
            mask = assignment == index
            if np.any(mask):
                x_values.append(float(np.mean(confidence[mask])))
                y_values.append(float(np.mean(outcome[mask])))
        ax.plot(
            x_values, y_values, marker="o", linewidth=1.8,
            color=COLORS[condition], label=LABELS[condition],
        )
    ax.set_xlabel("Mean reported confidence")
    ax.set_ylabel("Observed executable correctness")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#d7dce2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Confidence calibration", loc="left", fontweight="bold", pad=12)
    fig.text(0.01, 0.01, stamp, fontsize=7.5, color="#667085")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def agent_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = read_jsonl(path)
    if any(row.get("study_mode") != "AGENT_EXPLORATORY" for row in rows):
        raise RuntimeError("agent input contains a non-agent study mode")
    output = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition and row.get("evaluator_status") == "ok" and row.get("final_correct") is not None]
        output.append({
            "condition": condition, "condition_label": LABELS[condition],
            "agent_sessions": len({row["agent_session_id"] for row in subset}),
            "episodes": len(subset), "correctness": mean_bool(subset, "final_correct"),
            "mean_tokens": float(np.mean([int((row.get("usage") or {}).get("total_tokens", 0)) for row in subset])) if subset else None,
            "mean_latency_seconds": float(np.mean([float(row["latency_seconds"]) for row in subset])) if subset else None,
        })
    return output


def latex_outputs(
    output: Path, mode: str, summary: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    sources: list[dict[str, Any]], agents: list[dict[str, Any]], demographics: list[dict[str, Any]],
    poststudy: list[dict[str, Any]], retained: int, episodes: int
) -> None:
    latex = output / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    stamp = "SIMULATED DRY RUN--NOT EMPIRICAL EVIDENCE" if mode == "SIMULATED_DRY_RUN" else "Human participant study"
    table = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{stamp}. Human review outcomes on the frozen constructed stimulus set. Confidence intervals use participant-clustered bootstrap resampling.}}",
        r"\label{tab:human_review_outcomes}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        "Condition & Episodes & Correct & Detect & Repair & Unnec. edit & Median time (s) " + r"\\",
        r"\midrule",
    ]
    for row in summary:
        table.append(
            f"{latex_escape(row['condition_label'])} & {row['evaluable_episodes']} & "
            f"{pct(row['correctness'])} & {pct(row['failure_detection_accuracy'])} & "
            f"{pct(row['repair_success'])} & {pct(row['unnecessary_edit'])} & "
            f"{float(row['median_elapsed_seconds']):.1f} " + r"\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (latex / "table_human_review.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    pairs = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{stamp}. Paired participant-level correctness contrasts.}}",
        r"\label{tab:human_review_contrasts}", r"\begin{tabular}{lrrrr}", r"\toprule",
        "Contrast & $N$ & $\\Delta$ (pp) & 95\\% CI & Holm $p$ " + r"\\",
        r"\midrule",
    ]
    for row in comparisons:
        label = f"{LABELS[row['intervention']]} vs. {LABELS[row['comparator']]}"
        pairs.append(
            f"{latex_escape(label)} & {row['matched_participants']} & {100*float(row['absolute_difference']):+.1f} & "
            f"[{100*float(row['bootstrap_ci95_low']):+.1f}, {100*float(row['bootstrap_ci95_high']):+.1f}] & "
            f"{float(row['holm_p']):.3f} " + r"\\"
        )
    pairs.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (latex / "table_human_contrasts.tex").write_text("\n".join(pairs) + "\n", encoding="utf-8")

    if agents:
        agent_lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{Exploratory AI-agent replication. Agent sessions are not human participants and are not pooled with the human analysis.}",
            r"\label{tab:agent_review_replication}", r"\begin{tabular}{lrrr}", r"\toprule",
            "Condition & Sessions & Episodes & Correct (\\%) " + r"\\", r"\midrule",
        ]
        for row in agents:
            agent_lines.append(
                f"{latex_escape(row['condition_label'])} & {row['agent_sessions']} & "
                f"{row['episodes']} & {pct(row['correctness'])} " + r"\\"
            )
        agent_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    else:
        agent_lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{Exploratory AI-agent replication status.}", r"\label{tab:agent_review_replication}",
            r"\begin{tabular}{ll}", r"\toprule", "Status & Interpretation " + r"\\", r"\midrule",
            "Not run & No agent outcome is reported. " + r"\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    (latex / "table_agent_replication.tex").write_text("\n".join(agent_lines) + "\n", encoding="utf-8")

    by_name = {row["measure"]: row for row in demographics}
    demographic_lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{stamp}. Participant characteristics. Role percentages may overlap.}}",
        r"\label{tab:human_participants}", r"\begin{tabular}{lrr}", r"\toprule",
        "Characteristic & Value & Scale " + r"\\", r"\midrule",
    ]
    role_labels = {
        "role_software_engineer": "Software engineer",
        "role_ai_ml_researcher": "AI/ML researcher",
        "role_phd_student": "PhD student",
        "role_other": "Other role",
    }
    for field, label in role_labels.items():
        row = by_name[field]
        demographic_lines.append(
            f"{label} & {row['n']} ({pct(row['value'])}\\%) & overlapping " + r"\\"
        )
    for field, label in (
        ("programming_years", "Programming experience"),
        ("python_years", "Python experience"),
    ):
        row = by_name[field]
        demographic_lines.append(
            f"{label} & {float(row['value']):.1f} mean & years " + r"\\"
        )
    demographic_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (latex / "table_human_participants.tex").write_text(
        "\n".join(demographic_lines) + "\n", encoding="utf-8"
    )

    if poststudy:
        questionnaire_lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            rf"\caption{{{stamp}. Post-study questionnaire summary.}}",
            r"\label{tab:human_questionnaire}", r"\begin{tabular}{lrrrr}", r"\toprule",
            "Measure & $N$ & Missing & Mean & Median " + r"\\", r"\midrule",
        ]
        labels = {
            "mental_demand": "Mental demand", "effort": "Effort",
            "frustration": "Frustration", "temporal_demand": "Temporal demand",
            "adoption_intent": "Adoption intent",
        }
        for row in poststudy:
            questionnaire_lines.append(
                f"{labels[row['measure']]} & {row['respondents']} & {row['missing']} & "
                f"{decimal(row['mean'], 2)} & {decimal(row['median'], 2)} " + r"\\"
            )
        questionnaire_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    else:
        questionnaire_lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{Post-study questionnaire status.}",
            r"\label{tab:human_questionnaire}", r"\begin{tabular}{ll}", r"\toprule",
            "Status & Interpretation " + r"\\", r"\midrule",
            "Unavailable & No questionnaire outcomes are reported. " + r"\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
    (latex / "table_human_questionnaire.tex").write_text(
        "\n".join(questionnaire_lines) + "\n", encoding="utf-8"
    )

    primary = next(row for row in comparisons if row["intervention"] == "targeted_guidance" and row["comparator"] == "generic_review")
    targeted = next(row for row in summary if row["condition"] == "targeted_guidance")
    generic = next(row for row in summary if row["condition"] == "generic_review")
    if float(primary["holm_p"]) < 0.05 and float(primary["bootstrap_ci95_low"]) > 0:
        interpretation = "increased"
    elif float(primary["holm_p"]) < 0.05 and float(primary["bootstrap_ci95_high"]) < 0:
        interpretation = "reduced"
    else:
        interpretation = "did not measurably change"
    banner = "\\noindent\\textbf{SIMULATED DRY RUN--NOT EMPIRICAL EVIDENCE.}\\par\n" if mode == "SIMULATED_DRY_RUN" else ""
    section = rf"""{banner}\subsection{{Human Evaluation of Uncertainty-Guided Review}}

We retained {retained} participants and {episodes} task episodes after applying the preregistered exclusion rules. Participants reviewed the same frozen code and evidence under generic review, uncertainty display, and targeted guidance; only the displayed uncertainty and recommended action varied.

Targeted guidance achieved {pct(targeted['correctness'])}\% final executable correctness, compared with {pct(generic['correctness'])}\% under generic review. The paired difference was {100*float(primary['absolute_difference']):+.1f} percentage points (participant-clustered 95\% CI [{100*float(primary['bootstrap_ci95_low']):+.1f}, {100*float(primary['bootstrap_ci95_high']):+.1f}]; Holm-adjusted $p={float(primary['holm_p']):.3f}$). Thus, targeted guidance {interpretation} correctness on the constructed stimulus set. This result does not estimate benchmark-wide failure prevalence.

\input{{generated/table_human_review}}
\input{{generated/table_human_contrasts}}
\input{{generated/table_human_participants}}
\input{{generated/table_human_questionnaire}}

The separate AI-agent replication is descriptive and is never pooled with human observations.
\input{{generated/table_agent_replication}}
"""
    (latex / "human_study_results.tex").write_text(section, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--withdrawals", type=Path)
    parser.add_argument("--agents", type=Path)
    parser.add_argument("--tutorials", type=Path)
    parser.add_argument("--poststudy", type=Path)
    parser.add_argument("--ethics-approval-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-simulated", action="store_true")
    args = parser.parse_args()

    participants = read_jsonl(args.participants)
    episodes = read_csv(args.episodes)
    if not participants or not episodes:
        raise RuntimeError("participant and scored episode inputs must be non-empty")
    modes = {str(row.get("study_mode")) for row in participants} | {str(row.get("study_mode")) for row in episodes}
    ethics_verified = False
    if modes == {"EMPIRICAL"}:
        validate_ethics_approval(args.ethics_approval_file)
        ethics_verified = True
        output = args.out or ROOT / "results/human_study/analysis"
    elif modes == {"SIMULATED_DRY_RUN"} and args.allow_simulated:
        output = args.out or ROOT / "human_study/dry_run/analysis"
    else:
        raise RuntimeError("mixed modes are forbidden; simulated records require --allow-simulated")
    mode = next(iter(modes))
    empirical_root = (ROOT / "results/human_study").resolve()
    if mode == "SIMULATED_DRY_RUN" and output.resolve().is_relative_to(empirical_root):
        raise RuntimeError("simulated analyses cannot be written to empirical results")
    withdrawals = {
        str(row["participant_code"])
        for row in (read_jsonl(args.withdrawals) if args.withdrawals and args.withdrawals.is_file() else [])
    }
    tutorial_passed: set[str] | None = None
    if args.tutorials is not None and args.tutorials.is_file():
        tutorial_passed = {
            str(row["participant_code"])
            for row in read_jsonl(args.tutorials) if as_bool(row.get("passed"))
        }
    if mode == "EMPIRICAL":
        if args.tutorials is None or not args.tutorials.is_file():
            raise RuntimeError(
                "empirical analysis requires --tutorials to verify tutorial completion"
            )
    retained_profiles, retained_episodes, flow = exclusion_audit(
        participants, episodes, withdrawals, tutorial_passed
    )
    if not retained_profiles:
        raise RuntimeError("no participants remain after preregistered exclusions")
    profiles = {row["participant_code"]: row for row in retained_profiles}
    raw_expertise = {code: expertise(row) for code, row in profiles.items()}
    values = np.asarray(list(raw_expertise.values()), dtype=float)
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    for row in retained_episodes:
        row["expertise_score"] = raw_expertise[row["participant_code"]]
        row["expertise_z"] = 0.0 if sd == 0 else (raw_expertise[row["participant_code"]] - mean) / sd

    summary = condition_summary(retained_episodes)
    comparisons = pairwise(retained_episodes)
    sources = source_summary(retained_episodes)
    demographics = demographic_summary(retained_profiles)
    gee_rows = gee_sensitivity(retained_episodes)
    time_gee_rows = completion_time_gee(retained_episodes)
    questionnaire = poststudy_summary(args.poststudy, set(profiles))
    agents = agent_summary(args.agents)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "participant_flow.csv", flow)
    write_csv(output / "condition_summary.csv", summary)
    write_csv(output / "pairwise_statistics.csv", comparisons)
    write_csv(output / "source_summary.csv", sources)
    write_csv(output / "participant_characteristics.csv", demographics)
    write_csv(output / "gee_condition_effects.csv", gee_rows)
    write_csv(output / "gee_completion_time.csv", time_gee_rows)
    if questionnaire:
        write_csv(output / "poststudy_summary.csv", questionnaire)
    if agents:
        write_csv(output / "agent_summary.csv", agents)
    stamp = "SIMULATED DRY RUN--NOT EMPIRICAL EVIDENCE" if mode == "SIMULATED_DRY_RUN" else f"N={len(retained_profiles)} participants; constructed 12-task set"
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    plot_condition(summary, figures / "human_condition_correctness", stamp)
    plot_sources(sources, figures / "human_source_repair", stamp)
    plot_calibration(retained_episodes, figures / "human_confidence_calibration", stamp)
    latex_outputs(
        output, mode, summary, comparisons, sources, agents, demographics, questionnaire,
        len(retained_profiles), len(retained_episodes),
    )

    primary = next(row for row in comparisons if row["intervention"] == "targeted_guidance" and row["comparator"] == "generic_review")
    integrity = {
        "passed": True,
        "paper_eligible": (
            mode == "EMPIRICAL" and ethics_verified
            and len(retained_profiles) >= int(STUDY_CONFIG["participants_target_analyzable"])
        ),
        "target_analyzable_participants": int(STUDY_CONFIG["participants_target_analyzable"]),
        "study_mode": mode,
        "ethics_verified": ethics_verified,
        "participants_received": len({row["participant_code"] for row in participants}),
        "participants_retained": len(retained_profiles),
        "episodes_retained": len(retained_episodes),
        "evaluator_errors": sum(row.get("evaluator_status") == "error" for row in retained_episodes),
        "primary_contrast": primary,
        "agent_records_included": 0 if not args.agents or not args.agents.is_file() else len(read_jsonl(args.agents)),
        "human_agent_pooling": False,
        "analysis_seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "expertise_formula": "mean(min(programming_years/10,1), min(python_years/8,1), code_review_frequency/5, repository_development_frequency/5, testing_familiarity/5, dependency_tracing_familiarity/5)",
        "analysis_input_sha256": {
            str(path.name): sha256_file(path)
            for path in (
                args.participants, args.episodes, args.tutorials, args.withdrawals,
                args.poststudy, args.agents,
            )
            if path is not None and path.is_file()
        },
    }
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    warning = "SIMULATED DRY RUN; DO NOT REPORT AS HUMAN EVIDENCE." if mode == "SIMULATED_DRY_RUN" else "Empirical analysis; verify ethics and participant-flow documentation before publication."
    (output / "results_memo.md").write_text(
        f"# Human Study Analysis\n\n**{warning}**\n\n"
        f"Retained participants: {len(retained_profiles)}. Retained episodes: {len(retained_episodes)}.\n\n"
        f"Targeted guidance versus generic review: {100*float(primary['absolute_difference']):+.1f} percentage points; "
        f"95% CI [{100*float(primary['bootstrap_ci95_low']):+.1f}, {100*float(primary['bootstrap_ci95_high']):+.1f}]; "
        f"Holm-adjusted p={float(primary['holm_p']):.3f}.\n",
        encoding="utf-8",
    )
    print(json.dumps(integrity, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
