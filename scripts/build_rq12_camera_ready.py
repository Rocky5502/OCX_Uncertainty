"""Build audited camera-ready artifacts from the corrected RQ1/RQ2 runs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "rq12_corrected_10"
BACKENDS = ("gpt", "gemini")
SOURCES = ("api", "context", "similar_code")
SOURCE_LABELS = {"api": "API", "context": "Context", "similar_code": "Similar code"}
CONDITIONS = ("without", "decomposition", "filtering", "guidance", "selection", "with")
CONDITION_LABELS = {
    "without": "Baseline RAG",
    "decomposition": "+ Decomposition",
    "filtering": "+ Unc. filtering",
    "guidance": "+ Guided generation",
    "selection": "+ Verified selection",
    "with": "OpenCoder (+ repair)",
}
COLORS = {"gpt": "#2463A5", "gemini": "#D97706"}
BACKEND_LABELS = {"gpt": "GPT", "gemini": "Gemini"}


def _load(backend: str, rq: str) -> dict:
    return json.loads((RESULT_ROOT / backend / f"{rq}.json").read_text(encoding="utf-8"))


def _audit(rq1: dict, rq2: dict, backend: str) -> dict:
    rows = rq1.get("rows") or []
    keys = {(row.get("example_id"), row.get("condition")) for row in rows}
    if len(rows) != 80 or len(keys) != 80 or any("error" in row for row in rows):
        raise RuntimeError(f"{backend} RQ1 is not a complete 10x8 matrix")

    for condition in CONDITIONS:
        condition_rows = rq2.get(condition) or []
        ids = {row.get("id") for row in condition_rows}
        if len(condition_rows) != 10 or len(ids) != 10 or any("error" in row for row in condition_rows):
            raise RuntimeError(f"{backend} RQ2/{condition} is not a complete 10-task block")
        if any(int(row.get("candidate_count", 0)) != 3 for row in condition_rows):
            raise RuntimeError(f"{backend} RQ2/{condition} does not use exactly three raw samples")

    paired = True
    for task_id in {row["id"] for row in rq2["with"]}:
        signatures = []
        for condition in CONDITIONS[1:]:
            row = next(item for item in rq2[condition] if item["id"] == task_id)
            signatures.append(row.get("phase2_plan_sha256"))
        paired = paired and all(signature == signatures[0] for signature in signatures[1:])
    if not paired:
        raise RuntimeError(f"{backend} RQ2 contains mismatched Phase-II plans")
    return {"rq1_rows": len(rows), "rq2_rows": 60, "paired_plans": paired}


def _pct(value):
    return "--" if value is None else _fixed(100.0 * float(value), 1)


def _num(value, digits=3):
    return "--" if value is None else _fixed(float(value), digits)


def _fixed(value: float, digits: int) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _ci(values, *, scale=1.0, digits=3):
    if not values:
        return "--"
    return f"[{_fixed(scale * values[0], digits)}, {_fixed(scale * values[1], digits)}]"


def _mcnemar_exact(baseline_rows, method_rows):
    baseline = {row["id"]: bool(row["passed"]) for row in baseline_rows}
    method = {row["id"]: bool(row["passed"]) for row in method_rows}
    improved = sum(not baseline[key] and method[key] for key in baseline)
    regressed = sum(baseline[key] and not method[key] for key in baseline)
    discordant = improved + regressed
    if not discordant:
        return 1.0, improved, regressed
    tail = sum(math.comb(discordant, index) for index in range(min(improved, regressed) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant)), improved, regressed


def _write_rq1_tables(data: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Task-paired factorial effects of retrieval evidence. $\Delta U$ and $\Delta$Pass@1 are marginal present-minus-absent effects over the remaining source factors. Confidence intervals use 10,000 task-level bootstrap resamples; $p_{\mathrm{Holm}}$ adjusts the three source tests within each backend and metric.}",
        r"\label{tab:source_factorial_effects}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Backend & Source & $\Delta U$ & 95\% CI & $p_{\mathrm{Holm}}$ & $\Delta$Pass@1 (pp) & 95\% CI (pp) & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for backend in BACKENDS:
        summary = data[backend]["rq1"]["summary"]
        for source in SOURCES:
            row = summary[source]
            lines.append(
                f"{BACKEND_LABELS[backend]} & {SOURCE_LABELS[source]} & "
                f"{_fixed(row['delta_u_present_minus_absent'], 3)} & {_ci(row['delta_u_ci95'])} & "
                f"{row['delta_u_p_holm']:.3f} & {_fixed(100 * row['delta_pass_at_1'], 1)} & "
                f"{_ci(row['delta_pass_at_1_ci95'], scale=100, digits=1)} & "
                f"{row['delta_pass_at_1_p_holm']:.3f} \\\\"
            )
        if backend == "gpt":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    lines.extend([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Two-way evidence interactions. Effects use difference-in-differences over the third source.}",
        r"\label{tab:source_interactions}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Backend & Interaction & $\Delta U$ & $p_{\mathrm{Holm}}$ & $\Delta$Pass@1 (pp) & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ])
    for backend in BACKENDS:
        factorial = data[backend]["rq1"]["summary"]["factorial"]
        interactions = factorial["uncertainty"]["two_way_interactions"]
        pass_interactions = factorial["pass@1"]["two_way_interactions"]
        for name, row in interactions.items():
            label = name.replace("api", "API").replace("context", "Context").replace("similar_code", "Similar")
            pass_row = pass_interactions[name]
            lines.append(
                f"{BACKEND_LABELS[backend]} & {label} & {_fixed(row['effect'], 3)} & {row['p_holm']:.3f} & "
                f"{_fixed(100 * pass_row['effect'], 1)} & {pass_row['p_holm']:.3f} \\\\ "
            )
        if backend == "gpt":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (RESULT_ROOT / "table_source_factorial_effects.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_rq2_tables(data: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Component analysis of uncertainty-aware generation and mitigation. Effective Pass@1 evaluates the final selected or repaired output; Sample Pass@$k$ evaluates the three raw candidates. Pass@5 is omitted because this experiment generated three candidates per task.}",
        r"\label{tab:uncertainty_component_effectiveness}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Backend & Method & Effective P@1 & Sample P@1 & Sample P@3 & $\Delta$Effective P@1 & ECE$\downarrow$ & AURC$\downarrow$ \\",
        r"\midrule",
    ]
    for backend in BACKENDS:
        summary = data[backend]["rq2"]["summary"]
        for condition in CONDITIONS:
            row = summary[condition]
            delta = row.get("delta_effective_pass@1_vs_baseline")
            delta_text = "--" if delta is None else _fixed(100 * delta, 1)
            lines.append(
                f"{BACKEND_LABELS[backend]} & {CONDITION_LABELS[condition]} & {_pct(row['effective_pass@1'])} & "
                f"{_pct(row.get('pass@1'))} & {_pct(row.get('pass@3'))} & {delta_text} & "
                f"{_num(row.get('ece'))} & {_num(row.get('aurc'))} \\\\"
            )
        if backend == "gpt":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    lines.extend([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mitigation diagnostics for complete OpenCoder. AUROC predicts initial generation failure before validation.}",
        r"\label{tab:mitigation_diagnostics}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Backend & Failure AUROC & AUPRC & Selection success & Repair success & Time (s) \\",
        r"\midrule",
    ])
    for backend in BACKENDS:
        row = data[backend]["rq2"]["summary"]["with"]
        lines.append(
            f"{BACKEND_LABELS[backend]} & {_num(row.get('failure_auroc'))} & {_num(row.get('failure_auprc'))} & "
            f"{_pct(row.get('selection_success_rate'))} & {_pct(row.get('repair_success_rate'))} & "
            f"{row['mean_run_latency_s']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (RESULT_ROOT / "table_uncertainty_components.tex").write_text("\n".join(lines), encoding="utf-8")


def _plot_rq1(data: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    y = np.arange(len(SOURCES))
    offsets = {"gpt": -0.10, "gemini": 0.10}
    for backend in BACKENDS:
        rows = data[backend]["rq1"]["summary"]
        effects_u = np.array([rows[source]["delta_u_present_minus_absent"] for source in SOURCES])
        ci_u = np.array([rows[source]["delta_u_ci95"] for source in SOURCES])
        effects_p = 100 * np.array([rows[source]["delta_pass_at_1"] for source in SOURCES])
        ci_p = 100 * np.array([rows[source]["delta_pass_at_1_ci95"] for source in SOURCES])
        axes[0].errorbar(effects_u, y + offsets[backend], xerr=np.vstack([effects_u-ci_u[:, 0], ci_u[:, 1]-effects_u]),
                         fmt="o", color=COLORS[backend], capsize=2.5, label=BACKEND_LABELS[backend])
        axes[1].errorbar(effects_p, y + offsets[backend], xerr=np.vstack([effects_p-ci_p[:, 0], ci_p[:, 1]-effects_p]),
                         fmt="o", color=COLORS[backend], capsize=2.5, label=BACKEND_LABELS[backend])
    for axis in axes:
        axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_yticks(y, [SOURCE_LABELS[source] for source in SOURCES])
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Marginal change in uncertainty")
    axes[1].set_xlabel("Change in Pass@1 (percentage points)")
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
    for suffix in ("pdf", "png"):
        fig.savefig(RESULT_ROOT / f"fig_source_factorial_effects.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_rq1_interactions(data: dict) -> None:
    names = ("api:context", "api:similar_code", "context:similar_code")
    labels = ("API + Context", "API + Similar", "Context + Similar")
    y = np.arange(len(names))
    offsets = {"gpt": -0.10, "gemini": 0.10}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    for backend in BACKENDS:
        factorial = data[backend]["rq1"]["summary"]["factorial"]
        for axis, metric, scale in ((axes[0], "uncertainty", 1.0), (axes[1], "pass@1", 100.0)):
            rows = factorial[metric]["two_way_interactions"]
            effects = scale * np.array([rows[name]["effect"] for name in names])
            cis = scale * np.array([rows[name]["ci95"] for name in names])
            axis.errorbar(
                effects,
                y + offsets[backend],
                xerr=np.vstack([effects - cis[:, 0], cis[:, 1] - effects]),
                fmt="o",
                color=COLORS[backend],
                capsize=2.5,
                label=BACKEND_LABELS[backend],
            )
    for axis in axes:
        axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_yticks(y, labels)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Uncertainty interaction")
    axes[1].set_xlabel("Pass@1 interaction (percentage points)")
    axes[1].tick_params(labelleft=False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.94))
    for suffix in ("pdf", "png"):
        fig.savefig(RESULT_ROOT / f"fig_source_interactions.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_rq2(data: dict) -> None:
    labels = ("Baseline", "Decomp.", "Filtering", "Guidance", "Selection", "+ Repair")
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for axis, backend in zip(axes, BACKENDS):
        summary = data[backend]["rq2"]["summary"]
        effective = [100 * summary[name]["effective_pass@1"] for name in CONDITIONS]
        sample = [100 * summary[name]["pass@1"] for name in CONDITIONS]
        axis.plot(x, effective, marker="o", linewidth=1.8, color=COLORS[backend], label="Effective Pass@1")
        axis.plot(x, sample, marker="s", linewidth=1.4, linestyle="--", color="#3F7D5A", label="Sample Pass@1")
        axis.set_title("GPT" if backend == "gpt" else "Gemini")
        axis.set_xticks(x, labels, rotation=34, ha="right")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pass rate (\%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        fig.savefig(RESULT_ROOT / f"fig_component_effectiveness.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _write_findings(data: dict) -> None:
    gpt_rq1 = data["gpt"]["rq1"]["summary"]["factorial"]
    gem_rq1 = data["gemini"]["rq1"]["summary"]["factorial"]
    gpt_rq2 = data["gpt"]["rq2"]
    gem_rq2 = data["gemini"]["rq2"]
    gpt_p, gpt_up, gpt_down = _mcnemar_exact(gpt_rq2["without"], gpt_rq2["with"])
    gem_p, gem_up, gem_down = _mcnemar_exact(gem_rq2["without"], gem_rq2["with"])
    gpt_full = gpt_rq2["summary"]["with"]
    gem_full = gem_rq2["summary"]["with"]

    text = rf"""\subsection{{Influence of Retrieved Evidence}}

We evaluate all $2^3$ combinations of API knowledge, repository context, and
similar code on ten execution-backed tasks for each backend. This yields 160
task--condition runs and 480 test-executed candidate generations. All source
conditions for a task share the same query decomposition and retrieval intent.
Table~\ref{{tab:source_factorial_effects}} reports task-paired marginal effects,
while Table~\ref{{tab:source_interactions}} reports difference-in-differences
interactions.

No individual source exhibits a statistically reliable marginal effect on
aggregate uncertainty after Holm correction. For GPT, the marginal uncertainty
effects of API, context, and similar-code evidence are
{gpt_rq1['uncertainty']['main_effects']['api']['effect_present_minus_absent']:.3f},
{gpt_rq1['uncertainty']['main_effects']['context']['effect_present_minus_absent']:.3f}, and
{gpt_rq1['uncertainty']['main_effects']['similar_code']['effect_present_minus_absent']:.3f},
respectively. The corresponding Gemini effects are
{gem_rq1['uncertainty']['main_effects']['api']['effect_present_minus_absent']:.3f},
{gem_rq1['uncertainty']['main_effects']['context']['effect_present_minus_absent']:.3f}, and
{gem_rq1['uncertainty']['main_effects']['similar_code']['effect_present_minus_absent']:.3f}.
The interaction analysis is more informative: context and similar code have a
positive GPT Pass@1 interaction of
{100*gpt_rq1['pass@1']['two_way_interactions']['context:similar_code']['effect']:.1f}
percentage points ($p_{{\mathrm{{Holm}}}}=
{gpt_rq1['pass@1']['two_way_interactions']['context:similar_code']['p_holm']:.3f}$),
whereas API and similar-code evidence reduce Gemini uncertainty jointly by
{abs(gem_rq1['uncertainty']['two_way_interactions']['api:similar_code']['effect']):.3f}
($p_{{\mathrm{{Holm}}}}=
{gem_rq1['uncertainty']['two_way_interactions']['api:similar_code']['p_holm']:.3f}$).

\noindent\textbf{{Finding 1.}}
Retrieved evidence does not have a backend-independent additive effect on
uncertainty. Instead, uncertainty and correctness are shaped by interactions
among heterogeneous sources, supporting consensus-aware fusion rather than a
fixed universal source ordering.

\subsection{{Uncertainty-Aware Mitigation}}

We isolate query decomposition, uncertainty-aware filtering, guided generation,
verified candidate selection, and repair using six paired configurations per
task and backend. Table~\ref{{tab:uncertainty_component_effectiveness}}
distinguishes raw-sample performance from the final effective output.
OpenCoder increases effective Pass@1 from
{_pct(gpt_rq2['summary']['without']['effective_pass@1'])}\% to
{_pct(gpt_full['effective_pass@1'])}\% for GPT and from
{_pct(gem_rq2['summary']['without']['effective_pass@1'])}\% to
{_pct(gem_full['effective_pass@1'])}\% for Gemini. The paired gains are
{100*gpt_full['delta_effective_pass@1_vs_baseline']:.1f} points
(95\% CI {_ci(gpt_full['delta_effective_pass@1_ci95'], scale=100, digits=1)};
exact McNemar $p={gpt_p:.3f}$) and
{100*gem_full['delta_effective_pass@1_vs_baseline']:.1f} points
(95\% CI {_ci(gem_full['delta_effective_pass@1_ci95'], scale=100, digits=1)};
exact McNemar $p={gem_p:.3f}$), with {gpt_up}/{gpt_down} and
{gem_up}/{gem_down} improved/regressed task pairs.

The mitigation pathway differs by backend. For GPT, verified selection reaches
the same {_pct(gpt_full['effective_pass@1'])}\% effective Pass@1 as the full
system, and repair does not recover an additional post-selection failure. For
Gemini, selection succeeds on {_pct(gem_full['selection_success_rate'])}\% of
initial failures and repair succeeds on {_pct(gem_full['repair_success_rate'])}\%
of attempted post-selection failures, raising final Pass@1 to
{_pct(gem_full['effective_pass@1'])}\%. Calibration is not uniformly improved:
ECE changes from {data['gpt']['rq2']['summary']['without']['ece']:.3f} to
{gpt_full['ece']:.3f} for GPT, but from
{data['gemini']['rq2']['summary']['without']['ece']:.3f} to
{gem_full['ece']:.3f} for Gemini. Failure AUROC for full OpenCoder is
{gpt_full['failure_auroc']:.3f} and {gem_full['failure_auroc']:.3f}, respectively.

\noindent\textbf{{Finding 2.}}
Uncertainty-aware decomposition, evidence control, verification, and repair
substantially improve final functional correctness, but raw uncertainty
calibration remains backend-dependent. OpenCoder is therefore supported as an
effective mitigation architecture; stronger cross-backend calibration remains
an explicit limitation rather than a claimed universal property.
"""
    (RESULT_ROOT / "overleaf_rq1_rq2.tex").write_text(text, encoding="utf-8")


def _write_status(data: dict, audits: dict) -> None:
    gpt = data["gpt"]["rq2"]["summary"]
    gemini = data["gemini"]["rq2"]["summary"]
    status = f"""# Corrected RQ1/RQ2 Experiment Status

## Integrity

- GPT RQ1: {audits['gpt']['rq1_rows']}/80 valid task-condition rows.
- Gemini RQ1: {audits['gemini']['rq1_rows']}/80 valid task-condition rows.
- GPT RQ2: {audits['gpt']['rq2_rows']}/60 valid task-condition rows.
- Gemini RQ2: {audits['gemini']['rq2_rows']}/60 valid task-condition rows.
- All RQ2 uncertainty-aware conditions reuse identical Phase-II plans within each task/backend: yes.
- Correctness mode: reconstructed repository tests on 10 audited ExecRepoBench tasks.
- Raw candidates: 3 per task-condition. Pass@5 is intentionally not reported.
- All numbers in the paper artifacts are generated from the JSON result files; no assumed cells are used.

## Main Results

- GPT effective Pass@1: {100*gpt['without']['effective_pass@1']:.1f}% -> {100*gpt['with']['effective_pass@1']:.1f}%.
- Gemini effective Pass@1: {100*gemini['without']['effective_pass@1']:.1f}% -> {100*gemini['with']['effective_pass@1']:.1f}%.
- GPT sample Pass@1/3: {100*gpt['without']['pass@1']:.1f}/{100*gpt['without']['pass@3']:.1f} -> {100*gpt['with']['pass@1']:.1f}/{100*gpt['with']['pass@3']:.1f}.
- Gemini sample Pass@1/3: {100*gemini['without']['pass@1']:.1f}/{100*gemini['without']['pass@3']:.1f} -> {100*gemini['with']['pass@1']:.1f}/{100*gemini['with']['pass@3']:.1f}.
- Calibration is backend-dependent: GPT ECE {gpt['without']['ece']:.3f} -> {gpt['with']['ece']:.3f}; Gemini ECE {gemini['without']['ece']:.3f} -> {gemini['with']['ece']:.3f}.

## Camera-Ready Files

- `overleaf_rq1_rq2.tex`
- `table_source_factorial_effects.tex`
- `table_uncertainty_components.tex`
- `fig_source_factorial_effects.pdf` / `.png`
- `fig_source_interactions.pdf` / `.png`
- `fig_component_effectiveness.pdf` / `.png`

## Scope

These are complete real-API results for the local 10-task execution-backed subset and one three-candidate run per condition. Confidence intervals use tasks as the inference unit. The results support strong mitigation claims but not universal calibration or broad benchmark-level generalization.
"""
    (RESULT_ROOT / "RQ1_RQ2_STATUS.md").write_text(status, encoding="utf-8")


def main() -> None:
    data = {}
    audits = {}
    for backend in BACKENDS:
        rq1, rq2 = _load(backend, "rq1"), _load(backend, "rq2")
        data[backend] = {"rq1": rq1, "rq2": rq2}
        audits[backend] = _audit(rq1, rq2, backend)
    _write_rq1_tables(data)
    _write_rq2_tables(data)
    _plot_rq1(data)
    _plot_rq1_interactions(data)
    _plot_rq2(data)
    _write_findings(data)
    _write_status(data, audits)
    print(json.dumps({"audits": audits, "out": str(RESULT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
