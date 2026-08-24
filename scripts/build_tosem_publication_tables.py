#!/usr/bin/env python3
"""Build the TOSEM T1--T13 table package from audited result artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/tosem/publication_tables"
LATEX = OUTPUT / "latex"
EXEC = ROOT / "results/tosem/confirmatory_analysis"
CROSS = ROOT / "results/tosem/crosscodeeval_confirmatory"
COLLAB = ROOT / "results/tosem/collaboration_analysis"
RQ12 = ROOT / "results/tosem/rq1_rq2_analysis"
OLD_RQ12 = ROOT / "results/rq12_corrected_10"
MODELS = ("gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-5", "qwen3-coder-plus")
MODEL_SHORT = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-5": "Claude Sonnet 5",
    "qwen3-coder-plus": "Qwen3-Coder-Plus",
}
STATUS_SHORT = {
    "EXISTING_VERIFIED_RESULT": "Retained",
    "COMPLETED": "Completed",
    "COMPLETED_NON_EXECUTABLE": "Completed (native)",
    "BLOCKED_NO_DOCKER": "Blocked (no Docker)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tex(name: str, lines: Sequence[str]) -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    (LATEX / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any) -> float:
    return float(value)


def pct(value: Any) -> str:
    return f"{100.0 * f(value):.1f}"


def esc(value: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "--"
    return f"{f(value):.{digits}f}"


def table1() -> None:
    rows = [
        {"benchmark": "RepoExec-inline", "tasks": 32, "repositories": "multiple", "languages": "Python", "endpoint": "executable tests", "role": "retained two-family evaluation", "status": "EXISTING_VERIFIED_RESULT"},
        {"benchmark": "CoderEval API subset", "tasks": 13, "repositories": "multiple", "languages": "Python/Java", "endpoint": "API-set quality", "role": "retained API analysis", "status": "EXISTING_VERIFIED_RESULT"},
        {"benchmark": "ExecRepoBench-120", "tasks": 120, "repositories": 37, "languages": "Python", "endpoint": "executable tests", "role": "main confirmatory evaluation", "status": "COMPLETED"},
        {"benchmark": "CrossCodeEval-100", "tasks": 100, "repositories": 88, "languages": "Python/Java/TypeScript/C#", "endpoint": "native EM/edit/identifier F1", "role": "multilingual retrieval transfer", "status": "COMPLETED_NON_EXECUTABLE"},
        {"benchmark": "Multi-SWE-bench-Flash-35", "tasks": 35, "repositories": "--", "languages": "multilingual", "endpoint": "official container tests", "role": "issue-resolution stress test", "status": "BLOCKED_NO_DOCKER"},
    ]
    write_csv(OUTPUT / "T1_benchmarks.csv", rows)
    for row in rows:
        row["status"] = STATUS_SHORT[str(row["status"])]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Benchmark scope and evaluation status. CrossCodeEval uses native non-executable metrics; Multi-SWE-bench is excluded from quantitative claims because the official Docker evaluator was unavailable.}",
             r"\label{tab:benchmark_characteristics}", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lrrlll}", r"\toprule",
             r"Benchmark & Tasks & Repositories & Languages & Endpoint & Status \\", r"\midrule"]
    for row in rows:
        lines.append(f"{esc(str(row['benchmark']))} & {row['tasks']} & {row['repositories']} & {esc(str(row['languages']))} & {esc(str(row['endpoint']))} & {esc(str(row['status']))} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    write_tex("T1_benchmark_characteristics.tex", lines)


def table2() -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = ROOT / "configs/tosem/models" / {
            "gpt-4o-mini": "gpt4o_mini.yaml", "gemini-2.5-flash": "gemini_2_5_flash.yaml",
            "claude-sonnet-5": "claude_sonnet_5.yaml", "qwen3-coder-plus": "qwen3_coder_plus.yaml",
        }[model]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows.append({
            "family": MODEL_SHORT[model].split()[0], "model": model, "gateway": config["llm"]["backend"],
            "temperature": config["llm"]["temperature"], "max_tokens": config["llm"]["max_tokens"],
            "candidates": config["llm"]["n_samples_for_uncertainty"],
            "repair_rounds": config["verification"]["max_repair_rounds"], "status": "CONFIRMATORY_COMPLETED",
        })
    write_csv(OUTPUT / "T2_models.csv", rows)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Frozen four-family model configuration. Claude used the provider default temperature because the gateway rejected an explicit value.}",
             r"\label{tab:model_config_final}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Temp. & Max tokens & Candidates & Repairs \\", r"\midrule"]
    for row in rows:
        temperature = "default" if row["temperature"] is None else str(row["temperature"])
        lines.append(f"{esc(MODEL_SHORT[row['model']])} & {temperature} & {row['max_tokens']} & {row['candidates']} & {row['repair_rounds']} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("T2_model_configurations.tex", lines)


def table3() -> None:
    rows = read_csv(EXEC / "summary.csv")
    write_csv(OUTPUT / "T3_four_family_effectiveness.csv", rows)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Four-family effectiveness on ExecRepoBench-120. Pass@$k$ uses exactly five original candidates; Selected is executable correctness after each method's declared selection/repair policy. Values are percentages.}",
             r"\label{tab:four_family_effectiveness}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Method & Pass@1 & Pass@3 & Pass@5 & Selected \\", r"\midrule"]
    for model_index, model in enumerate(MODELS):
        for row in [item for item in rows if item["model"] == model]:
            lines.append(f"{esc(MODEL_SHORT[model])} & {esc(row['method'])} & {pct(row['pass_at_1'])} & {pct(row['pass_at_3'])} & {pct(row['pass_at_5'])} & {pct(row['selected_output_correctness'])} \\\\ ")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T3_four_family_effectiveness.tex", lines)


def table4() -> None:
    rows = read_csv(RQ12 / "factorial_interactions.csv")
    write_csv(OUTPUT / "T4_evidence_interactions.csv", rows)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Task-paired two-way evidence interactions on the audited 10-task, three-candidate factorial campaign. Effects do not represent four-family estimates.}",
             r"\label{tab:evidence_interactions_final}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Backend & Interaction & $\Delta U$ & $p_H$ & $\Delta$P@1 & $p_H$ \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['backend']} & {esc(row['interaction'])} & {f(row['uncertainty_effect']):+.3f} & {f(row['uncertainty_holm_p']):.3f} & {100*f(row['pass_at_1_effect']):+.1f} & {f(row['pass_at_1_holm_p']):.3f} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("T4_evidence_interactions.tex", lines)


def table5() -> None:
    rows = read_csv(RQ12 / "uncertainty_signal_discrimination.csv")
    write_csv(OUTPUT / "T5_uncertainty_discrimination.csv", rows)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Failure discrimination on ExecRepoBench-120. CIs are task-bootstrap intervals; AUPRC is interpreted relative to model-specific failure prevalence.}",
             r"\label{tab:uncertainty_discrimination_final}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Signal & AUROC & 95\% CI & AUPRC & Brier & ECE \\", r"\midrule"]
    labels = {"api": "API", "context": "Context", "similar_code": "Similar code", "generation": "Generation", "aggregate": "Aggregate"}
    for model_index, model in enumerate(MODELS):
        for row in [item for item in rows if item["model"] == model]:
            lines.append(f"{esc(MODEL_SHORT[model])} & {labels[row['signal']]} & {f(row['auroc_failure']):.3f} & [{f(row['auroc_ci95_low']):.3f}, {f(row['auroc_ci95_high']):.3f}] & {f(row['auprc_failure']):.3f} & {f(row['brier_failure']):.3f} & {f(row['ece_failure_10_bins']):.3f} \\\\ ")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T5_uncertainty_discrimination.tex", lines)


def table6() -> None:
    all_rows = read_csv(EXEC / "selected_output_statistics.csv")
    rows = [row for row in all_rows if row["comparator"] == "RAG + Verify/Repair"]
    write_csv(OUTPUT / "T6_matched_baseline.csv", rows)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Matched selected-output comparison: OpenCoderX versus RAG + Verify/Repair on 120 tasks. Values are percentages; $p_H$ is Holm-adjusted across model families.}",
             r"\label{tab:matched_baseline_final}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Control/OpenCoderX & $\Delta$ & 95\% CI & W/L/T ($p_H$) \\", r"\midrule"]
    for row in rows:
        lines.append(f"{esc(MODEL_SHORT[row['model']])} & {pct(row['comparator_correctness'])}/{pct(row['opencoder_correctness'])} & {100*f(row['absolute_difference']):+.1f} & [{100*f(row['bootstrap_ci95_low']):.1f}, {100*f(row['bootstrap_ci95_high']):.1f}] & {row['opencoder_wins']}/{row['opencoder_losses']}/{row['ties']} ({f(row['mcnemar_holm_p']):.3f}) \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("T6_matched_baseline.tex", lines)


def table7() -> None:
    condition_labels = {
        "without": "Baseline RAG", "decomposition": "+ Decomposition", "filtering": "+ Filtering",
        "guidance": "+ Guided generation", "selection": "+ Verified selection", "with": "OpenCoder + repair",
    }
    rows: list[dict[str, Any]] = []
    for backend in ("gpt", "gemini"):
        data = json.loads((OLD_RQ12 / backend / "rq2.json").read_text(encoding="utf-8"))["summary"]
        for condition, label in condition_labels.items():
            row = data[condition]
            rows.append({"backend": backend.upper(), "condition": label, "tasks": 10, "candidates": 3,
                         "effective_pass_at_1": row["effective_pass@1"], "sample_pass_at_1": row["pass@1"],
                         "sample_pass_at_3": row["pass@3"], "ece": row["ece"], "aurc": row["aurc"]})
    write_csv(OUTPUT / "T7_component_ablation.csv", rows)
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Retained component analysis on the audited 10-task, three-candidate subset. Effective Pass@1 includes selection and repair where enabled.}",
             r"\label{tab:component_ablation_final}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Backend & Condition & Eff. P@1 & Raw P@1 & Raw P@3 & ECE \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['backend']} & {esc(row['condition'])} & {pct(row['effective_pass_at_1'])} & {pct(row['sample_pass_at_1'])} & {pct(row['sample_pass_at_3'])} & {f(row['ece']):.3f} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("T7_component_ablation.tex", lines)


def table8() -> None:
    rows = read_csv(COLLAB / "generalizability_language.csv")
    write_csv(OUTPUT / "T8_cross_language.csv", rows)
    language_labels = {
        "python": "Python",
        "java": "Java",
        "typescript": "TypeScript",
        "csharp": r"C\#",
    }
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{CrossCodeEval-100 context effect by language (25 tasks per cell). Exact match is a native, non-executable completion metric.}",
             r"\label{tab:cross_language_final}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Language & Direct EM & Context EM & $\Delta$ & 95\% CI & $p_H$ \\", r"\midrule"]
    for model_index, model in enumerate(MODELS):
        for row in [item for item in rows if item["model"] == model]:
            lines.append(f"{esc(MODEL_SHORT[model])} & {language_labels[row['language']]} & {pct(row['direct_exact_match'])} & {pct(row['context_exact_match'])} & {100*f(row['exact_match_difference']):+.1f} & [{100*f(row['bootstrap_ci95_low']):.1f}, {100*f(row['bootstrap_ci95_high']):.1f}] & {f(row['mcnemar_holm_p']):.3f} \\\\ ")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T8_cross_language.tex", lines)


def table9() -> None:
    rows = read_csv(COLLAB / "generalizability_model_benchmark.csv")
    write_csv(OUTPUT / "T9_cross_benchmark.csv", rows)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Cross-benchmark model summary. ExecRepoBench values are executable selected correctness/Pass@1; CrossCodeEval values are native exact match/identifier F1 and are not pooled.}",
             r"\label{tab:cross_benchmark_final}", r"\begin{tabular}{lllrrl}", r"\toprule",
             r"Benchmark & Model & Method & Primary & Secondary & Execution \\", r"\midrule"]
    for row in rows:
        lines.append(f"{esc(row['benchmark'])} & {esc(MODEL_SHORT[row['model']])} & {esc(row['method'])} & {pct(row['primary_value'])} & {pct(row['secondary_value'])} & {'Yes' if row['functional_execution']=='True' else 'No'} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T9_cross_benchmark.tex", lines)


def table10() -> None:
    all_rows = read_csv(COLLAB / "review_budget_summary.csv")
    policies = ("random_deferral", "test_failure_deferral", "aggregate_uncertainty_deferral", "source_specific_opencoderx_deferral", "oracle_deferral")
    budgets = (0.1, 0.2, 0.5)
    rows = [row for row in all_rows if row["policy"] in policies and f(row["reviewer_success"]) == 0.75 and f(row["review_budget"]) in budgets]
    write_csv(OUTPUT / "T10_deferral_budget.csv", rows)
    labels = {"random_deferral": "Random", "test_failure_deferral": "Pre-repair test", "aggregate_uncertainty_deferral": "Aggregate uncertainty", "source_specific_opencoderx_deferral": "Source-specific", "oracle_deferral": "Oracle"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Simulated team success under a 75\% reviewer-success assumption. These values are offline simulations, not observed developer performance.}",
             r"\label{tab:deferral_budget_final}", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Model & Policy & 10\% budget & 20\% budget & 50\% budget \\", r"\midrule"]
    for model_index, model in enumerate(MODELS):
        for policy in policies:
            values = {f(row["review_budget"]): row for row in rows if row["model"] == model and row["policy"] == policy}
            lines.append(f"{esc(MODEL_SHORT[model])} & {labels[policy]} & {pct(values[0.1]['mean_team_success_rate'])} & {pct(values[0.2]['mean_team_success_rate'])} & {pct(values[0.5]['mean_team_success_rate'])} \\\\ ")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T10_deferral_budget.tex", lines)


def table11() -> None:
    all_rows = read_csv(COLLAB / "review_budget_summary.csv")
    rows = [row for row in all_rows if row["policy"] == "source_specific_opencoderx_deferral" and f(row["review_budget"]) == 0.2]
    write_csv(OUTPUT / "T11_reviewer_sensitivity.csv", rows)
    levels = (0.6, 0.75, 0.9, 1.0)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Reviewer-capability sensitivity for source-specific routing at a 20\% review budget. Values are simulated team success percentages.}",
             r"\label{tab:reviewer_sensitivity_final}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & 60\% & 75\% & 90\% & 100\% \\", r"\midrule"]
    for model in MODELS:
        values = {f(row["reviewer_success"]): row for row in rows if row["model"] == model}
        lines.append(f"{esc(MODEL_SHORT[model])} & " + " & ".join(pct(values[level]["mean_team_success_rate"]) for level in levels) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("T11_reviewer_sensitivity.tex", lines)


def table12() -> None:
    all_rows = read_csv(COLLAB / "intervention_effectiveness.csv")
    interventions = ("Repository evidence provision", "Ordinary verification and repair", "Uncertainty-aware evidence and generation")
    rows = [row for row in all_rows if row["status"] == "OBSERVED" and row["intervention"] in interventions]
    write_csv(OUTPUT / "T12_interventions.csv", rows)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Observed matched intervention effects on ExecRepoBench-120 selected correctness. Source-specific evidence-correction interventions are omitted because no matched runs exist.}",
             r"\label{tab:interventions_final}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Intervention & Before/After & $\Delta$ & 95\% CI & $p_H$ \\", r"\midrule"]
    for row in rows:
        p_value = f(row["mcnemar_holm_p"])
        p_text = r"$<.001$" if p_value < 0.001 else f"{p_value:.3f}"
        lines.append(f"{esc(MODEL_SHORT[row['model']])} & {esc(row['intervention'])} & {pct(row['before_correctness'])}/{pct(row['after_correctness'])} & {100*f(row['absolute_difference']):+.1f} & [{100*f(row['bootstrap_ci95_low']):.1f}, {100*f(row['bootstrap_ci95_high']):.1f}] & {p_text} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T12_interventions.tex", lines)


def table13() -> None:
    rows = read_csv(EXEC / "resource_summary.csv")
    write_csv(OUTPUT / "T13_resources.csv", rows)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{ExecRepoBench-120 resource use per task. Costs use the frozen gateway pricing audit and are reported in each model's billing currency.}",
             r"\label{tab:resources_final}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Method & Mean tokens & Mean latency (s) & Mean repairs & Total cost & Currency \\", r"\midrule"]
    for model_index, model in enumerate(MODELS):
        for row in [item for item in rows if item["model"] == model]:
            lines.append(f"{esc(MODEL_SHORT[model])} & {esc(row['method'])} & {f(row['mean_tokens']):.0f} & {f(row['mean_latency_seconds']):.1f} & {f(row['mean_repair_rounds']):.2f} & {f(row['estimated_cost']):.3f} & {row['currency']} \\\\ ")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex("T13_resources.tex", lines)


def main() -> None:
    for function in (table1, table2, table3, table4, table5, table6, table7, table8, table9, table10, table11, table12, table13):
        function()
    manifest = {"tables": 13, "latex_files": sorted(path.name for path in LATEX.glob("*.tex")), "status": "GENERATED_FROM_AUDITED_ARTIFACTS"}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
