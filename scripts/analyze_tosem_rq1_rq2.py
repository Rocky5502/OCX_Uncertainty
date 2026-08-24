#!/usr/bin/env python3
"""Consolidate RQ1 evidence interactions and RQ2 uncertainty transfer.

No model calls are made. The causal source-factorial evidence is retained from
the audited 10-task GPT/Gemini campaign; the four-family results are analyzed
as confirmatory uncertainty discrimination and multilingual context transfer.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_ROOT = ROOT / "results/rq12_corrected_10"
COLLAB_ROOT = ROOT / "results/tosem/collaboration_analysis"
EXEC_ROOT = ROOT / "results/tosem/confirmatory_analysis"
CROSS_ROOT = ROOT / "results/tosem/crosscodeeval_confirmatory"
OUTPUT = ROOT / "results/tosem/rq1_rq2_analysis"
BACKENDS = {"gpt": "GPT", "gemini": "Gemini"}
SOURCES = ("api", "context", "similar_code")
SIGNALS = ("api", "context", "similar_code", "generation", "aggregate")
MODELS = ("gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-5", "qwen3-coder-plus")
BOOTSTRAPS = 10_000
SEED = 20260809


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def avg(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def ece(risk: Sequence[float], failures: Sequence[int], bins: int = 10) -> float:
    prediction = np.clip(np.asarray(risk, dtype=float), 0.0, 1.0)
    target = np.asarray(failures, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (prediction >= low) & (prediction < high if index < bins - 1 else prediction <= high)
        if np.any(selected):
            result += float(np.mean(selected)) * abs(float(np.mean(prediction[selected])) - float(np.mean(target[selected])))
    return result


def aurc(risk: Sequence[float], failures: Sequence[int]) -> float:
    order = np.argsort(np.asarray(risk, dtype=float), kind="stable")
    ordered = np.asarray(failures, dtype=float)[order]
    return float(np.mean(np.cumsum(ordered) / np.arange(1, len(ordered) + 1)))


def bootstrap_auc(risk: Sequence[float], failures: Sequence[int], offset: int) -> tuple[float, float]:
    values = np.asarray(risk, dtype=float)
    target = np.asarray(failures, dtype=int)
    rng = np.random.default_rng(SEED + offset)
    estimates: list[float] = []
    while len(estimates) < BOOTSTRAPS:
        indices = rng.integers(0, len(values), size=len(values))
        sampled = target[indices]
        if len(np.unique(sampled)) == 2:
            estimates.append(float(roc_auc_score(sampled, values[indices])))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def factorial_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    effects: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for backend, label in BACKENDS.items():
        data = read_json(FACTORIAL_ROOT / backend / "rq1.json")
        rows = data.get("rows") or []
        keys = {(row.get("example_id"), row.get("condition")) for row in rows}
        if len(rows) != 80 or len(keys) != 80 or any("error" in row for row in rows):
            raise RuntimeError(f"{backend} factorial is not the audited 10x8 matrix")
        factorial = data["summary"]["factorial"]
        for source in SOURCES:
            uncertainty = factorial["uncertainty"]["main_effects"][source]
            correctness = factorial["pass@1"]["main_effects"][source]
            effects.append({
                "backend": label,
                "source": source,
                "tasks": 10,
                "factorial_conditions": 8,
                "candidate_count": 3,
                "uncertainty_effect": uncertainty["effect_present_minus_absent"],
                "uncertainty_ci_low": uncertainty["ci95"][0],
                "uncertainty_ci_high": uncertainty["ci95"][1],
                "uncertainty_holm_p": uncertainty["p_holm"],
                "pass_at_1_effect": correctness["effect_present_minus_absent"],
                "pass_at_1_ci_low": correctness["ci95"][0],
                "pass_at_1_ci_high": correctness["ci95"][1],
                "pass_at_1_holm_p": correctness["p_holm"],
                "evidence_status": "EXISTING_VERIFIED_RESULT",
            })
        for name, uncertainty in factorial["uncertainty"]["two_way_interactions"].items():
            correctness = factorial["pass@1"]["two_way_interactions"][name]
            interactions.append({
                "backend": label,
                "interaction": name,
                "tasks": 10,
                "uncertainty_effect": uncertainty["effect"],
                "uncertainty_ci_low": uncertainty["ci95"][0],
                "uncertainty_ci_high": uncertainty["ci95"][1],
                "uncertainty_holm_p": uncertainty["p_holm"],
                "pass_at_1_effect": correctness["effect"],
                "pass_at_1_ci_low": correctness["ci95"][0],
                "pass_at_1_ci_high": correctness["ci95"][1],
                "pass_at_1_holm_p": correctness["p_holm"],
                "evidence_status": "EXISTING_VERIFIED_RESULT",
            })
    return effects, interactions


def uncertainty_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = [json.loads(line) for line in (COLLAB_ROOT / "decision_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in raw if row["decision_policy"] == "uncertainty_only"]
    discrimination: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        failures = [int(not row["selected_output_correct"]) for row in model_rows]
        for signal_index, signal in enumerate(SIGNALS):
            if signal == "aggregate":
                risk = [float(row["risk_score"]) for row in model_rows]
            else:
                risk = [float(row["uncertainty_sources"][signal]) for row in model_rows]
            low, high = bootstrap_auc(risk, failures, 100 * model_index + signal_index)
            rho, rho_p = spearmanr(risk, failures)
            discrimination.append({
                "benchmark": "ExecRepoBench-120",
                "model": model,
                "signal": signal,
                "tasks": len(model_rows),
                "failure_rate": avg(failures),
                "mean_risk": avg(risk),
                "auroc_failure": float(roc_auc_score(failures, risk)),
                "auroc_ci95_low": low,
                "auroc_ci95_high": high,
                "auprc_failure": float(average_precision_score(failures, risk)),
                "brier_failure": avg((prediction - target) ** 2 for prediction, target in zip(risk, failures)),
                "ece_failure_10_bins": ece(risk, failures),
                "aurc": aurc(risk, failures),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
            })
            order = np.argsort(np.asarray(risk), kind="stable")
            for accepted in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
                count = max(1, int(round(len(model_rows) * accepted)))
                selected = order[:count]
                coverage.append({
                    "benchmark": "ExecRepoBench-120",
                    "model": model,
                    "signal": signal,
                    "coverage": count / len(model_rows),
                    "accepted_tasks": count,
                    "selective_failure_rate": float(np.mean(np.asarray(failures)[selected])),
                    "selective_accuracy": 1.0 - float(np.mean(np.asarray(failures)[selected])),
                })
    return discrimination, coverage


def latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = {"api": "API", "context": "Context", "similar_code": "Similar code", "generation": "Generation", "aggregate": "Aggregate"}
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Failure discrimination of source and generation uncertainty on 120 executable tasks per model. CIs use 10,000 task bootstrap resamples. AUPRC should be interpreted relative to each model's failure prevalence.}",
        r"\label{tab:uncertainty_signal_transfer}", r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Signal & AUROC & 95\% CI & AUPRC & ECE \\", r"\midrule",
    ]
    for model in MODELS:
        for row in [item for item in rows if item["model"] == model]:
            model_label = model.replace("_", r"\_")
            lines.append(
                f"{model_label} & {labels[row['signal']]} & {row['auroc_failure']:.3f} & "
                f"[{row['auroc_ci95_low']:.3f}, {row['auroc_ci95_high']:.3f}] & "
                f"{row['auprc_failure']:.3f} & {row['ece_failure_10_bins']:.3f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    effects, interactions = factorial_artifacts()
    discrimination, coverage = uncertainty_artifacts()
    context_transfer = read_csv(CROSS_ROOT / "paired_statistics.csv")
    cross_uncertainty = read_csv(CROSS_ROOT / "uncertainty_calibration.csv")
    exec_calibration = read_csv(EXEC_ROOT / "uncertainty_calibration.csv")

    write_csv(OUTPUT / "factorial_main_effects.csv", effects)
    write_csv(OUTPUT / "factorial_interactions.csv", interactions)
    write_csv(OUTPUT / "uncertainty_signal_discrimination.csv", discrimination)
    write_csv(OUTPUT / "risk_coverage.csv", coverage)
    write_csv(OUTPUT / "context_transfer_crosscodeeval.csv", context_transfer)
    write_csv(OUTPUT / "uncertainty_transfer_crosscodeeval.csv", cross_uncertainty)
    write_csv(OUTPUT / "uncertainty_calibration_exec120.csv", exec_calibration)
    latex_table(OUTPUT / "latex/table_uncertainty_signal_transfer.tex", discrimination)

    issues: list[str] = []
    if len(effects) != 6 or len(interactions) != 6:
        issues.append("factorial result dimensions are incomplete")
    if len(discrimination) != 20 or len(coverage) != 140:
        issues.append("four-family uncertainty result dimensions are incomplete")
    if any(int(row["tasks"]) != 120 for row in discrimination):
        issues.append("uncertainty signals do not contain 120 tasks per model")
    if len(context_transfer) != 12:
        issues.append("CrossCodeEval paired transfer table is incomplete")
    integrity = {
        "passed": not issues,
        "paper_eligible": not issues,
        "issues": issues,
        "factorial_backends": 2,
        "factorial_tasks_per_backend": 10,
        "factorial_conditions": 8,
        "exec_uncertainty_model_task_records": 480,
        "uncertainty_signals": list(SIGNALS),
        "crosscodeeval_tasks": 100,
        "crosscodeeval_languages": 4,
        "crosscodeeval_functional_execution": False,
    }
    (OUTPUT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    provenance_files = [
        FACTORIAL_ROOT / "gpt/rq1.json", FACTORIAL_ROOT / "gemini/rq1.json",
        COLLAB_ROOT / "decision_records.jsonl", EXEC_ROOT / "uncertainty_calibration.csv",
        CROSS_ROOT / "paired_statistics.csv", CROSS_ROOT / "uncertainty_calibration.csv",
        Path(__file__),
    ]
    provenance = {
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in provenance_files],
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "bootstrap": {"iterations": BOOTSTRAPS, "seed": SEED, "unit": "task"},
        "scope_note": "The 2^3 source factorial is GPT/Gemini on 10 execution-backed tasks. Four-family source risks are observational failure signals, not causal source ablations.",
        "crosscodeeval_note": "CrossCodeEval endpoints are native non-executable metrics.",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    aggregate = [row for row in discrimination if row["signal"] == "aggregate"]
    significant_interactions = [row for row in interactions if min(row["uncertainty_holm_p"], row["pass_at_1_holm_p"]) < 0.05]
    memo = [
        "# RQ1-RQ2 Consolidated Results", "",
        f"Integrity: **{'PASSED' if not issues else 'FAILED'}**.", "",
        "## RQ1 Evidence interactions", "",
        "The complete 2^3 factorial remains the audited 10-task GPT/Gemini campaign. It is not relabeled as a four-family experiment.", "",
    ]
    for row in significant_interactions:
        memo.append(
            f"- {row['backend']} {row['interaction']}: uncertainty interaction {row['uncertainty_effect']:+.3f} "
            f"(Holm p={row['uncertainty_holm_p']:.3f}); Pass@1 interaction {100*row['pass_at_1_effect']:+.1f} points "
            f"(Holm p={row['pass_at_1_holm_p']:.3f})."
        )
    memo += ["", "CrossCodeEval-100 independently shows positive context exact-match effects for GPT, Claude, and Qwen after Holm correction; Gemini is inconclusive. These are native completion metrics, not executable correctness.", "",
             "## RQ2 Uncertainty generalization", ""]
    for row in aggregate:
        memo.append(
            f"- {row['model']}: aggregate source+generation risk AUROC {row['auroc_failure']:.3f} "
            f"(95% CI [{row['auroc_ci95_low']:.3f}, {row['auroc_ci95_high']:.3f}]), AUPRC {row['auprc_failure']:.3f}."
        )
    memo += ["", "Uncertainty is directionally informative but backend-dependent. Calibration and discrimination must be reported separately; no universal threshold claim is supported.", ""]
    (OUTPUT / "results_memo.md").write_text("\n".join(memo), encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
