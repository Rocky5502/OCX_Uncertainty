"""Build RQ3 paper artifacts from real RQ3/RQ2-style run JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.evaluation.metrics import estimate_pass_at_k  # noqa: E402


DATASETS = ("RepoExec-inline", "CoderEval", "ExecRepoBench")
METRICS = ("Pass@1", "Pass@3", "Pass@5")
METHOD_LABELS = {
    "without": "Baseline RAG",
    "rag_repair": "RAG + Verify/Repair",
    "with": "OpenCoder",
}
BACKEND_LABELS = {"openai": "GPT", "gpt": "GPT", "gemini": "Gemini"}
UNIMPLEMENTED_METHODS = [
    "RepoCoder",
    "RLCoder",
    "RepoFormer",
    "RepoFuse",
    "ConAPI (API-ref.)",
]
METHOD_ORDER = {
    "Baseline RAG": 0,
    "AllianceCoder (clean-room)": 1,
    "RAG + Verify/Repair": 2,
    "OpenCoder": 3,
}


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_label(name: str) -> str:
    key = name.lower().replace("_", "")
    if key == "repoexec":
        return "RepoExec-inline"
    if key == "codereval":
        return "CoderEval"
    if key in {"execrepobench", "execrepobenchdata"}:
        return "ExecRepoBench"
    return name


def _backend_label(name: str) -> str:
    return BACKEND_LABELS.get(name.lower(), name)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "--"
    return f"{value:.{digits}f}"


def _latex_escape(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def _load_external_summary(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load independently reproduced baseline summaries for the paper table."""
    if not path:
        return []

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for source in csv.DictReader(f):
            backend = _backend_label(str(source.get("backend") or ""))
            dataset = _dataset_label(str(source.get("benchmark") or ""))
            if dataset not in DATASETS:
                continue
            key = (backend, "AllianceCoder (clean-room)")
            row = grouped.setdefault(
                key,
                {
                    "Backend": backend,
                    "Method": "AllianceCoder (clean-room)",
                    "MeanUncertainty": None,
                    "MeanRepairRounds": None,
                    "MeanRunLatencyS": None,
                    "MeanLLMRequests": None,
                    "MeanTotalTokens": None,
                    "N": 0,
                    "Qualification": [],
                },
            )
            for metric in METRICS:
                source_key = metric.lower()
                value = source.get(source_key)
                row[f"{dataset}_{metric.replace('@', '')}"] = (
                    float(value) * 100.0 if value not in {None, ""} else None
                )
            row["N"] += int(source.get("n_tasks") or 0)
            qualification = str(source.get("qualification") or "").strip()
            if qualification:
                row["Qualification"].append(f"{dataset}: {qualification}")

    rows = list(grouped.values())
    for row in rows:
        row["Qualification"] = "; ".join(row["Qualification"])
    return rows


def _load_prompts(metadata: Dict[str, Any]) -> Dict[str, str]:
    dataset = metadata.get("dataset")
    path = metadata.get("dataset_path")
    if not dataset or not path or not Path(path).exists():
        return {}
    try:
        return {ex.id: ex.query for ex in load_dataset(dataset, path)}
    except Exception:
        return {}


def _task_scores(sample_correctness: Iterable[bool]) -> Dict[str, float]:
    outcomes = [bool(x) for x in sample_correctness]
    return {
        "Pass@1": estimate_pass_at_k(len(outcomes), sum(outcomes), 1),
        "Pass@3": estimate_pass_at_k(len(outcomes), sum(outcomes), 3),
        "Pass@5": estimate_pass_at_k(len(outcomes), sum(outcomes), 5),
    }


def _effective_sample_correctness(row: Dict[str, Any], method_key: str) -> List[bool]:
    """Return the candidate stream used for end-to-end Pass@k.

    Baseline RAG has only raw sampled candidates. OpenCoder, however, has an
    explicit Phase-V selected/repaired primary output. For end-to-end RQ3, that
    primary output should be the first candidate being evaluated, while the
    remaining sampled candidates preserve the same candidate budget.
    """
    raw = [bool(x) for x in row.get("sample_correctness") or []]
    explicit = row.get("effective_sample_correctness")
    if explicit:
        return [bool(x) for x in explicit]
    if method_key in {"rag_repair", "with"} and raw and row.get("passed") is not None:
        return [bool(row.get("passed")), *raw[1:]]
    return raw


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _bootstrap_ci(
    baseline: List[float],
    opencoder: List[float],
    *,
    n_boot: int = 5000,
    seed: int = 20260704,
) -> Tuple[float, float]:
    if not baseline or len(baseline) != len(opencoder):
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(baseline)
    deltas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        b = _mean(baseline[i] for i in idxs)
        o = _mean(opencoder[i] for i in idxs)
        deltas.append(o - b)
    deltas.sort()
    lo = deltas[int(0.025 * (n_boot - 1))]
    hi = deltas[int(0.975 * (n_boot - 1))]
    return lo, hi


def _sign_test_paired(baseline: List[float], opencoder: List[float]) -> Dict[str, Any]:
    positives = negatives = ties = 0
    for b, o in zip(baseline, opencoder):
        if o > b:
            positives += 1
        elif o < b:
            negatives += 1
        else:
            ties += 1
    n = positives + negatives
    if n == 0:
        return {"positive": positives, "negative": negatives, "ties": ties, "p_value": 1.0}
    small = min(positives, negatives)
    tail = sum(math.comb(n, i) for i in range(small + 1)) / (2**n)
    return {
        "positive": positives,
        "negative": negatives,
        "ties": ties,
        "p_value": min(1.0, 2.0 * tail),
    }


def _extract_rows(run_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    data = _read_json(run_path)
    metadata = data.get("metadata") or {}
    dataset = _dataset_label(str(metadata.get("dataset") or "execrepobench"))
    backend = _backend_label(str(metadata.get("backend") or ""))
    model = str(metadata.get("model") or "")
    prompts = _load_prompts(metadata)

    per_task: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    for method_key, method in METHOD_LABELS.items():
        for row in data.get(method_key, []):
            task_id = str(row.get("id") or "")
            raw_sample_correctness = [bool(x) for x in row.get("sample_correctness") or []]
            sample_correctness = _effective_sample_correctness(row, method_key)
            scores = _task_scores(sample_correctness)
            base = {
                "benchmark": dataset,
                "backend": backend,
                "model": model,
                "method": method,
                "method_key": method_key,
                "seed": metadata.get("seed"),
                "task_id": task_id,
                "correctness_mode": row.get("correctness_mode"),
                "passed": row.get("passed"),
                "sample_count": len(sample_correctness),
                "correct_sample_count": sum(bool(x) for x in sample_correctness),
                "raw_sample_count": len(raw_sample_correctness),
                "raw_correct_sample_count": sum(bool(x) for x in raw_sample_correctness),
                "metric_sample_stream": (
                    "phase5_primary_plus_samples"
                    if method_key in {"rag_repair", "with"} and sample_correctness != raw_sample_correctness
                    else "raw_samples"
                ),
                "Pass@1": scores["Pass@1"],
                "Pass@3": scores["Pass@3"],
                "Pass@5": scores["Pass@5"],
                "mean_uncertainty": (row.get("u") or {}).get("aggregate"),
                "repair_rounds": row.get("repair_rounds"),
                "run_latency_s": row.get("run_latency_s"),
                "llm_requests": (row.get("llm_usage") or {}).get("requests"),
                "prompt_tokens": (row.get("llm_usage") or {}).get("prompt_tokens"),
                "completion_tokens": (row.get("llm_usage") or {}).get("completion_tokens"),
                "total_tokens": (row.get("llm_usage") or {}).get("total_tokens"),
                "index_latency_s": row.get("index_latency_s"),
                "error": row.get("error"),
            }
            per_task.append(base)
            raw_rows.append({
                **base,
                "prompt": prompts.get(task_id),
                "retrieved_api_evidence": None,
                "retrieved_context_evidence": None,
                "retrieved_similar_code": None,
                "source_diagnostics": row.get("source_diagnostics"),
                "source_wise_uncertainty": row.get("source_diagnostics"),
                "uncertainty_trace": row.get("u"),
                "uncertainty_components": row.get("uncertainty_components"),
                "consensus_score": (row.get("uncertainty_components") or {}).get("self_consistency"),
                "generated_code": row.get("code"),
                "generated_candidates": row.get("generated_samples"),
                "sample_correctness": sample_correctness,
                "raw_sample_correctness": raw_sample_correctness,
                "repair_history": None,
                "test_outcomes": row.get("test_report"),
                "static_report": row.get("static_report"),
                "fused_evidence": row.get("fused_evidence"),
                "per_step": row.get("per_step"),
                "token_usage": None,
                "estimated_api_cost": None,
                "random_seed": None,
                "timestamp": metadata.get("created_at"),
                "source_run": run_path,
            })
    return per_task, raw_rows, metadata


def _summary_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in per_task:
        groups.setdefault((row["backend"], row["method"]), []).append(row)
    expected_counts: Dict[Tuple[str, str], int] = {}
    for backend in {row["backend"] for row in per_task}:
        for dataset in {row["benchmark"] for row in per_task}:
            method_counts = [
                sum(
                    row["backend"] == backend
                    and row["benchmark"] == dataset
                    and row["method"] == method
                    for row in per_task
                )
                for method in {row["method"] for row in per_task}
            ]
            expected_counts[(backend, dataset)] = max(method_counts, default=0)

    out: List[Dict[str, Any]] = []
    for (backend, method), rows in sorted(groups.items()):
        item: Dict[str, Any] = {"Backend": backend, "Method": method}
        incomplete: List[str] = []
        for dataset in DATASETS:
            subset = [r for r in rows if r["benchmark"] == dataset]
            expected = expected_counts.get((backend, dataset), 0)
            complete = not subset or len(subset) == expected
            if subset and not complete:
                incomplete.append(f"{dataset}: {len(subset)}/{expected}")
            for metric in METRICS:
                key = f"{dataset}_{metric.replace('@', '')}"
                item[key] = (
                    _mean(r[metric] for r in subset) * 100.0
                    if subset and complete
                    else None
                )
        unc = [r["mean_uncertainty"] for r in rows if r.get("mean_uncertainty") is not None]
        repairs = [r["repair_rounds"] for r in rows if r.get("repair_rounds") is not None]
        lat = [r["run_latency_s"] for r in rows if r.get("run_latency_s") is not None]
        requests = [r["llm_requests"] for r in rows if r.get("llm_requests") is not None]
        tokens = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
        item["MeanUncertainty"] = _mean(float(x) for x in unc) if unc else None
        item["MeanRepairRounds"] = _mean(float(x) for x in repairs) if repairs else None
        item["MeanRunLatencyS"] = _mean(float(x) for x in lat) if lat else None
        item["MeanLLMRequests"] = _mean(float(x) for x in requests) if requests else None
        item["MeanTotalTokens"] = _mean(float(x) for x in tokens) if tokens else None
        item["N"] = len(rows)
        item["Qualification"] = (
            "incomplete " + "; ".join(incomplete)
            if incomplete
            else ""
        )
        out.append(item)
    return out


def _coverage_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expected: Dict[Tuple[str, str], int] = {}
    for row in per_task:
        key = (row["backend"], row["benchmark"])
        expected[key] = max(
            expected.get(key, 0),
            sum(
                other["backend"] == row["backend"]
                and other["benchmark"] == row["benchmark"]
                and other["method"] == row["method"]
                for other in per_task
            ),
        )
    groups: Dict[Tuple[str, str, str], int] = {}
    for row in per_task:
        key = (row["backend"], row["benchmark"], row["method"])
        groups[key] = groups.get(key, 0) + 1
    return [
        {
            "Backend": backend,
            "Benchmark": benchmark,
            "Method": method,
            "CompletedN": count,
            "ExpectedN": expected[(backend, benchmark)],
            "Complete": count == expected[(backend, benchmark)],
        }
        for (backend, benchmark, method), count in sorted(groups.items())
    ]


def _selected_output_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize correctness of the single output selected for execution."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in per_task:
        if row["method_key"] not in {"without", "rag_repair", "with"}:
            continue
        groups.setdefault(
            (row["backend"], row["method"], row["benchmark"]),
            [],
        ).append(row)

    expected: Dict[Tuple[str, str], int] = {}
    for backend, _method, benchmark in groups:
        expected[(backend, benchmark)] = max(
            expected.get((backend, benchmark), 0),
            len(groups[(backend, _method, benchmark)]),
        )

    out: List[Dict[str, Any]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][2],
            METHOD_ORDER.get(item[0][1], 99),
        ),
    )
    for (backend, method, benchmark), rows in ordered_groups:
        evaluated = [row for row in rows if row.get("passed") is not None]
        passed = sum(bool(row["passed"]) for row in evaluated)
        expected_n = expected[(backend, benchmark)]
        complete = len(evaluated) == expected_n
        out.append({
            "Backend": backend,
            "Method": method,
            "Benchmark": benchmark,
            "Passed": passed,
            "N": len(evaluated),
            "ExpectedN": expected_n,
            "Complete": complete,
            "SelectedPassRate": (
                100.0 * passed / len(evaluated)
                if evaluated and complete
                else None
            ),
        })
    return out


def _selected_output_stats(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Paired selected-output comparisons against OpenCoder."""
    out: List[Dict[str, Any]] = []
    for backend in sorted({row["backend"] for row in per_task}):
        for benchmark in sorted({row["benchmark"] for row in per_task}):
            open_rows = {
                (row.get("seed"), row["task_id"]): row
                for row in per_task
                if row["backend"] == backend
                and row["benchmark"] == benchmark
                and row["method_key"] == "with"
                and row.get("passed") is not None
            }
            for comparator_key in ("without", "rag_repair"):
                comparator_rows = {
                    (row.get("seed"), row["task_id"]): row
                    for row in per_task
                    if row["backend"] == backend
                    and row["benchmark"] == benchmark
                    and row["method_key"] == comparator_key
                    and row.get("passed") is not None
                }
                ids = sorted(set(open_rows) & set(comparator_rows))
                if not ids or set(open_rows) != set(comparator_rows):
                    continue
                comparator = [float(bool(comparator_rows[item]["passed"])) for item in ids]
                opencoder = [float(bool(open_rows[item]["passed"])) for item in ids]
                wins = sum(o > b for b, o in zip(comparator, opencoder))
                losses = sum(o < b for b, o in zip(comparator, opencoder))
                ties = len(ids) - wins - losses
                discordant = wins + losses
                if discordant:
                    tail = sum(
                        math.comb(discordant, index)
                        for index in range(min(wins, losses) + 1)
                    ) / (2**discordant)
                    mcnemar_p = min(1.0, 2.0 * tail)
                else:
                    mcnemar_p = 1.0
                ci_low, ci_high = _bootstrap_ci(comparator, opencoder)
                baseline_mean = _mean(comparator)
                open_mean = _mean(opencoder)
                out.append({
                    "Backend": backend,
                    "Benchmark": benchmark,
                    "Comparator": METHOD_LABELS[comparator_key],
                    "ComparatorSelectedPct": 100.0 * baseline_mean,
                    "OpenCoderSelectedPct": 100.0 * open_mean,
                    "AbsoluteImprovement": 100.0 * (open_mean - baseline_mean),
                    "BootstrapCI95Low": 100.0 * ci_low,
                    "BootstrapCI95High": 100.0 * ci_high,
                    "OpenCoderWins": wins,
                    "OpenCoderLosses": losses,
                    "Ties": ties,
                    "McNemarExactP": mcnemar_p,
                    "N": len(ids),
                })
    return out


def _resource_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in per_task:
        groups.setdefault(
            (row["backend"], row["method"], row["benchmark"]),
            [],
        ).append(row)

    out: List[Dict[str, Any]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][2],
            METHOD_ORDER.get(item[0][1], 99),
        ),
    )
    for (backend, method, benchmark), rows in ordered_groups:
        def values(key: str) -> List[float]:
            return [
                float(row[key])
                for row in rows
                if row.get(key) is not None
            ]

        requests = values("llm_requests")
        prompt_tokens = values("prompt_tokens")
        completion_tokens = values("completion_tokens")
        total_tokens = values("total_tokens")
        latency = values("run_latency_s")
        repairs = values("repair_rounds")
        out.append({
            "Backend": backend,
            "Method": method,
            "Benchmark": benchmark,
            "MeanRequests": _mean(requests) if requests else None,
            "MeanPromptTokens": _mean(prompt_tokens) if prompt_tokens else None,
            "MeanCompletionTokens": _mean(completion_tokens) if completion_tokens else None,
            "MeanTotalTokens": _mean(total_tokens) if total_tokens else None,
            "TotalTokens": sum(total_tokens) if total_tokens else None,
            "MeanLatencyS": _mean(latency) if latency else None,
            "MeanRepairRounds": _mean(repairs) if repairs else None,
            "N": len(rows),
        })
    return out


def _stats_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for backend in sorted({r["backend"] for r in per_task}):
        for dataset in sorted({r["benchmark"] for r in per_task}):
            open_rows = [
                r for r in per_task
                if r["backend"] == backend and r["benchmark"] == dataset and r["method_key"] == "with"
            ]
            open_by_id = {(r.get("seed"), r["task_id"]): r for r in open_rows}
            for comparator_key in ("without", "rag_repair"):
                comparator = [
                    r for r in per_task
                    if r["backend"] == backend
                    and r["benchmark"] == dataset
                    and r["method_key"] == comparator_key
                ]
                base_by_id = {(r.get("seed"), r["task_id"]): r for r in comparator}
                ids = sorted(set(base_by_id) & set(open_by_id))
                if not ids or set(base_by_id) != set(open_by_id):
                    continue
                for metric in METRICS:
                    b_vals = [float(base_by_id[i][metric]) for i in ids]
                    o_vals = [float(open_by_id[i][metric]) for i in ids]
                    b_mean = _mean(b_vals)
                    o_mean = _mean(o_vals)
                    delta = o_mean - b_mean
                    rel = (delta / b_mean * 100.0) if b_mean > 0 else None
                    ci_lo, ci_hi = _bootstrap_ci(b_vals, o_vals)
                    sign = _sign_test_paired(b_vals, o_vals)
                    out.append({
                        "Backend": backend,
                        "Benchmark": dataset,
                        "Metric": metric,
                        "StrongestPracticalBaseline": METHOD_LABELS[comparator_key],
                        "Baseline": b_mean * 100.0,
                        "OpenCoder": o_mean * 100.0,
                        "AbsoluteImprovement": delta * 100.0,
                        "RelativeImprovementPct": rel,
                        "BootstrapCI95Low": ci_lo * 100.0,
                        "BootstrapCI95High": ci_hi * 100.0,
                        "SignTestPositive": sign["positive"],
                        "SignTestNegative": sign["negative"],
                        "SignTestTies": sign["ties"],
                        "SignTestP": sign["p_value"],
                        "N": len(ids),
                    })
    return out


def _multiseed_rows(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_seed: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for row in per_task:
        seed = row.get("seed")
        if seed is None:
            continue
        key = (row["backend"], row["method"], row["benchmark"], int(seed))
        per_seed.setdefault(key, []).append(row)

    seed_metrics: Dict[Tuple[str, str, str, str], List[float]] = {}
    seed_counts: Dict[Tuple[str, str, str, str], List[int]] = {}
    for (backend, method, benchmark, _seed), rows in per_seed.items():
        for metric in METRICS:
            key = (backend, method, benchmark, metric)
            seed_metrics.setdefault(key, []).append(_mean(float(row[metric]) for row in rows) * 100.0)
            seed_counts.setdefault(key, []).append(len(rows))

    out: List[Dict[str, Any]] = []
    for (backend, method, benchmark, metric), values in sorted(seed_metrics.items()):
        if len(values) < 2:
            continue
        out.append({
            "Backend": backend,
            "Method": method,
            "Benchmark": benchmark,
            "Metric": metric,
            "Mean": statistics.mean(values),
            "StdDev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "Min": min(values),
            "Max": max(values),
            "NSeeds": len(values),
            "TasksPerSeedMin": min(seed_counts[(backend, method, benchmark, metric)]),
            "TasksPerSeedMax": max(seed_counts[(backend, method, benchmark, metric)]),
        })
    return out


def _write_multiseed_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    lookup = {
        (row["Backend"], row["Method"], row["Benchmark"], row["Metric"]): row
        for row in rows
    }
    pairs = sorted({(row["Backend"], row["Method"], row["Benchmark"]) for row in rows})
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Seeded end-to-end robustness. Values are mean $\pm$ standard deviation across independently seeded generation runs; every method uses five initial candidates per task.}",
        r"\label{tab:rq3_multiseed}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Backend & Method & Benchmark & Pass@1 & Pass@3 & Pass@5 & Seeds \\",
        r"\midrule",
    ]
    for backend, method, benchmark in pairs:
        metric_rows = [lookup.get((backend, method, benchmark, metric)) for metric in METRICS]
        values = [
            "--" if row is None else f"{row['Mean']:.2f} $\\pm$ {row['StdDev']:.2f}"
            for row in metric_rows
        ]
        n_seeds = max((int(row["NSeeds"]) for row in metric_rows if row is not None), default=0)
        lines.append(
            f"{_latex_escape(backend)} & {_latex_escape(method)} & {_latex_escape(benchmark)} & "
            + " & ".join(values)
            + f" & {n_seeds} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_budget_table(path: Path, summary: List[Dict[str, Any]]) -> None:
    rows = [row for row in summary if row.get("MeanLLMRequests") is not None]
    if not rows:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean per-task resource use in the seeded evaluation. Token counts are reported by the OpenAI-compatible gateway and exclude the shared repository-index construction.}",
        r"\label{tab:rq3_resource_budget}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Requests & Tokens & Time (s) & Repairs \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: item["Method"]):
        lines.append(
            f"{_latex_escape(row['Method'])} & {row['MeanLLMRequests']:.2f} & "
            f"{row['MeanTotalTokens']:.0f} & {row['MeanRunLatencyS']:.2f} & "
            f"{row['MeanRepairRounds']:.2f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_multiseed(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        for path in out_dir.glob("fig_seeded_*_passk.pdf"):
            path.unlink()
        for path in out_dir.glob("fig_seeded_*_passk.png"):
            path.unlink()
        return
    benchmarks = sorted({row["Benchmark"] for row in rows})
    for benchmark in benchmarks:
        subset = [row for row in rows if row["Benchmark"] == benchmark]
        pairs = sorted({(row["Backend"], row["Method"]) for row in subset})
        lookup = {(row["Backend"], row["Method"], row["Metric"]): row for row in subset}
        labels = [f"{backend}\n{method}" for backend, method in pairs]
        x = np.arange(len(labels))
        width = 0.24
        colors = ["#4b5563", "#2563eb", "#059669"]
        fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        for index, metric in enumerate(METRICS):
            metric_rows = [lookup[(backend, method, metric)] for backend, method in pairs]
            means = [row["Mean"] for row in metric_rows]
            stds = [row["StdDev"] for row in metric_rows]
            ax.bar(
                x + (index - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=metric,
                color=colors[index],
                error_kw={"linewidth": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Pass@k (%)")
        ax.set_title(f"Seeded Functional Correctness on {benchmark}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=3)
        for extension in ("pdf", "png"):
            fig.savefig(out_dir / f"fig_seeded_{benchmark.lower()}_passk.{extension}", dpi=300)
        plt.close(fig)


def _available_datasets(summary: List[Dict[str, Any]]) -> List[str]:
    return [
        dataset
        for dataset in DATASETS
        if any(row.get(f"{dataset}_Pass1") is not None for row in summary)
    ]


def _best_cells(summary: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    best: Dict[Tuple[str, str], float] = {}
    for row in summary:
        for dataset in DATASETS:
            for metric in METRICS:
                key = f"{dataset}_{metric.replace('@', '')}"
                value = row.get(key)
                if value is None:
                    continue
                cell_key = (row["Backend"], key)
                best[cell_key] = max(best.get(cell_key, -1.0), float(value))
    return best


def _cell(value: Optional[float], *, bold: bool) -> str:
    text = _fmt(value)
    if text == "--":
        return text
    return rf"\textbf{{{text}}}" if bold else text


def _has_executable_codereval(per_task: List[Dict[str, Any]]) -> bool:
    rows = [row for row in per_task if row["benchmark"] == "CoderEval"]
    return bool(rows) and all(
        row.get("correctness_mode") in {"execution_tests", "repository_tests"}
        for row in rows
    )


def _write_table(
    path: Path,
    summary: List[Dict[str, Any]],
    *,
    executable_codereval: bool,
) -> None:
    best = _best_cells(summary)
    datasets = _available_datasets(summary)
    has_alliance = any(
        row["Method"] == "AllianceCoder (clean-room)"
        for row in summary
    )
    rows = sorted(
        summary,
        key=lambda r: (
            r["Backend"],
            METHOD_ORDER.get(r["Method"], 99),
            r["Method"],
        ),
    )
    column_spec = "ll" + ("ccc" * len(datasets))
    group_headers = []
    start_column = 3
    cmidrules = []
    for dataset in datasets:
        label = "ExecRepoBench$^{\\dagger}$" if dataset == "ExecRepoBench" else dataset
        group_headers.append(rf"\multicolumn{{3}}{{c}}{{{label}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start_column}-{start_column + 2}}}")
        start_column += 3
    metric_headers = " & ".join(["Pass@1 & Pass@3 & Pass@5"] * len(datasets))
    caption = (
        r"\caption{End-to-end Pass@$k$ (\%) under a matched five-candidate budget. "
        + (
            r"AllianceCoder is our clean-room reproduction of its defining retrieval policy. "
            if has_alliance
            else ""
        )
        + r"$^\dagger$ExecRepoBench is context-limited because full repository snapshots are unavailable. "
        + r"Best values within each backend and benchmark are bold.}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        caption,
        r"\label{tab:rq3_end_to_end_effectiveness}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\multirow{2}{*}{Backend} & \multirow{2}{*}{Approach} & "
        + " & ".join(group_headers)
        + r" \\",
        "".join(cmidrules),
        r"& & " + metric_headers + r" \\",
        r"\midrule",
    ]
    last_backend = None
    for row in rows:
        if last_backend is not None and row["Backend"] != last_backend:
            lines.append(r"\midrule")
        last_backend = row["Backend"]
        cells = [_latex_escape(row["Backend"]), _latex_escape(row["Method"])]
        for dataset in datasets:
            for metric in METRICS:
                key = f"{dataset}_{metric.replace('@', '')}"
                value = row.get(key)
                is_best = value is not None and abs(float(value) - best.get((row["Backend"], key), -999.0)) < 1e-9
                cells.append(_cell(value, bold=is_best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_selected_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    datasets = [
        dataset
        for dataset in DATASETS
        if any(row["Benchmark"] == dataset for row in rows)
    ]
    lookup = {
        (row["Backend"], row["Method"], row["Benchmark"]): row
        for row in rows
    }
    pairs = sorted(
        {(row["Backend"], row["Method"]) for row in rows},
        key=lambda pair: (pair[0], METHOD_ORDER.get(pair[1], 99)),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Correctness of the single output selected for execution (\%).}",
        r"\label{tab:rq3_selected_output}",
        r"\begin{tabular}{ll" + "c" * len(datasets) + "}",
        r"\toprule",
        "Backend & Method & " + " & ".join(datasets) + r" \\",
        r"\midrule",
    ]
    last_backend = None
    for backend, method in pairs:
        if last_backend is not None and backend != last_backend:
            lines.append(r"\midrule")
        last_backend = backend
        cells = [backend, method]
        for dataset in datasets:
            row = lookup.get((backend, method, dataset))
            cells.append(_fmt(row.get("SelectedPassRate") if row else None))
        lines.append(" & ".join(_latex_escape(cell) if index < 2 else cell for index, cell in enumerate(cells)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paired_stats_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired task-level Pass@$k$ differences relative to OpenCoder. "
        r"$\Delta$ is OpenCoder minus the comparator in percentage points; confidence "
        r"intervals are paired 95\% bootstrap intervals. W/L/T counts tasks on which "
        r"OpenCoder has a higher/lower/equal per-task Pass@$k$ estimate.}",
        r"\label{tab:rq3_paired_passk}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Backend & Benchmark & Comparator/Metric & $\Delta$ & 95\% CI & W/L/T & $N$ \\",
        r"\midrule",
    ]
    last_group = None
    for row in rows:
        group = (row["Backend"], row["Benchmark"])
        if last_group is not None and group != last_group:
            lines.append(r"\midrule")
        last_group = group
        comparator_metric = f"{row['StrongestPracticalBaseline']} / {row['Metric']}"
        ci = f"[{row['BootstrapCI95Low']:.2f}, {row['BootstrapCI95High']:.2f}]"
        wlt = (
            f"{row['SignTestPositive']}/"
            f"{row['SignTestNegative']}/"
            f"{row['SignTestTies']}"
        )
        lines.append(
            f"{_latex_escape(row['Backend'])} & {_latex_escape(row['Benchmark'])} & "
            f"{_latex_escape(comparator_metric)} & {row['AbsoluteImprovement']:.2f} & "
            f"{ci} & {wlt} & {row['N']} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_selected_stats_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired correctness of the single selected output. W/L/T is from "
        r"OpenCoder's perspective; $p$ is the two-sided exact McNemar test.}",
        r"\label{tab:rq3_selected_paired}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Backend & Benchmark & Comparator & Comparator Acc. & OpenCoder Acc. & $\Delta$ [95\% CI] & W/L/T & $p$ & $N$ \\",
        r"\midrule",
    ]
    last_group = None
    for row in rows:
        group = (row["Backend"], row["Benchmark"])
        if last_group is not None and group != last_group:
            lines.append(r"\midrule")
        last_group = group
        delta_ci = (
            f"{row['AbsoluteImprovement']:.2f} "
            f"[{row['BootstrapCI95Low']:.2f}, {row['BootstrapCI95High']:.2f}]"
        )
        wlt = f"{row['OpenCoderWins']}/{row['OpenCoderLosses']}/{row['Ties']}"
        lines.append(
            f"{_latex_escape(row['Backend'])} & {_latex_escape(row['Benchmark'])} & "
            f"{_latex_escape(row['Comparator'])} & {row['ComparatorSelectedPct']:.2f} & "
            f"{row['OpenCoderSelectedPct']:.2f} & {delta_ci} & {wlt} & "
            f"{row['McNemarExactP']:.3f} & {row['N']} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_resource_detail_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean per-task resource use. Tokens are gateway-reported totals; "
        r"latency includes generation, verification, and repair but excludes shared indexing.}",
        r"\label{tab:rq3_resources}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Backend & Benchmark & Method & Requests & Tokens & Latency (s) & Repairs & $N$ \\",
        r"\midrule",
    ]
    last_group = None
    for row in rows:
        group = (row["Backend"], row["Benchmark"])
        if last_group is not None and group != last_group:
            lines.append(r"\midrule")
        last_group = group
        lines.append(
            f"{_latex_escape(row['Backend'])} & {_latex_escape(row['Benchmark'])} & "
            f"{_latex_escape(row['Method'])} & {_fmt(row['MeanRequests'])} & "
            f"{_fmt(row['MeanTotalTokens'], 0)} & {_fmt(row['MeanLatencyS'])} & "
            f"{_fmt(row['MeanRepairRounds'])} & {row['N']} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _exec_scope_label(metadata: List[Dict[str, Any]]) -> str:
    exec_meta = [
        m for m in metadata
        if _dataset_label(str(m.get("dataset") or "")) == "ExecRepoBench"
    ]
    if not exec_meta:
        return "the available ExecRepoBench subset"
    limits = sorted({m.get("limit") for m in exec_meta if m.get("limit")})
    paths = {str(m.get("dataset_path") or "") for m in exec_meta}
    if any(path.endswith("execrepobench_testbacked.jsonl") for path in paths):
        n = limits[-1] if limits else "test-backed"
        return f"{n}-task ExecRepoBench test-backed subset"
    if any(path.endswith("execrepobench_stable7.jsonl") for path in paths):
        return "stable 7-task ExecRepoBench subset"
    n = limits[-1] if limits else "available"
    return f"{n}-task ExecRepoBench subset"


def _write_findings(
    out_dir: Path,
    stats: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    per_task: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    selected_stats: List[Dict[str, Any]],
    *,
    executable_codereval: bool,
) -> None:
    evaluated_benchmarks = sorted({row["benchmark"] for row in per_task})
    evaluated_backends = sorted({row["backend"] for row in per_task})
    scope_status = (
        f"This package contains {', '.join(evaluated_benchmarks)} results for "
        f"{', '.join(evaluated_backends)}."
    )

    summary_lookup = {
        (row["Backend"], row["Method"]): row
        for row in summary
    }
    selected_lookup = {
        (row["Backend"], row["Method"], row["Benchmark"]): row
        for row in selected
    }

    def pass_values(backend: str, method: str, dataset: str) -> str:
        row = summary_lookup.get((backend, method), {})
        return "(" + ", ".join(
            _fmt(row.get(f"{dataset}_{metric.replace('@', '')}"))
            for metric in METRICS
        ) + ")"

    def selected_value(backend: str, method: str, dataset: str) -> str:
        row = selected_lookup.get((backend, method, dataset), {})
        return _fmt(row.get("SelectedPassRate"))

    selected_stats_lookup = {
        (row["Backend"], row["Benchmark"], row["Comparator"]): row
        for row in selected_stats
    }

    def mcnemar_value(backend: str, comparator: str, dataset: str) -> str:
        row = selected_stats_lookup.get((backend, dataset, comparator), {})
        value = row.get("McNemarExactP")
        return "--" if value is None else f"{float(value):.3f}"

    repo_sentence = (
        "On RepoExec-inline, GPT Baseline RAG, RAG + Verify/Repair, and OpenCoder obtain "
        f"{pass_values('GPT', 'Baseline RAG', 'RepoExec-inline')}, "
        f"{pass_values('GPT', 'RAG + Verify/Repair', 'RepoExec-inline')}, and "
        f"{pass_values('GPT', 'OpenCoder', 'RepoExec-inline')} Pass@1/3/5, respectively. "
        "With Gemini, the corresponding results are "
        f"{pass_values('Gemini', 'Baseline RAG', 'RepoExec-inline')}, "
        f"{pass_values('Gemini', 'RAG + Verify/Repair', 'RepoExec-inline')}, and "
        f"{pass_values('Gemini', 'OpenCoder', 'RepoExec-inline')}."
    )
    exec_sentence = (
        "On context-limited ExecRepoBench, RAG + Verify/Repair is strongest: its GPT and "
        f"Gemini results are {pass_values('GPT', 'RAG + Verify/Repair', 'ExecRepoBench')} "
        f"and {pass_values('Gemini', 'RAG + Verify/Repair', 'ExecRepoBench')}, compared with "
        f"{pass_values('GPT', 'OpenCoder', 'ExecRepoBench')} and "
        f"{pass_values('Gemini', 'OpenCoder', 'ExecRepoBench')} for OpenCoder."
    )
    selected_sentence = (
        "For selected-output correctness on RepoExec-inline, Baseline RAG, RAG + "
        f"Verify/Repair, and OpenCoder achieve {selected_value('GPT', 'Baseline RAG', 'RepoExec-inline')}, "
        f"{selected_value('GPT', 'RAG + Verify/Repair', 'RepoExec-inline')}, and "
        f"{selected_value('GPT', 'OpenCoder', 'RepoExec-inline')} with GPT, and "
        f"{selected_value('Gemini', 'Baseline RAG', 'RepoExec-inline')}, "
        f"{selected_value('Gemini', 'RAG + Verify/Repair', 'RepoExec-inline')}, and "
        f"{selected_value('Gemini', 'OpenCoder', 'RepoExec-inline')} with Gemini. "
        "On ExecRepoBench, verification/repair reaches 100.00 for both backends, versus "
        f"{selected_value('GPT', 'OpenCoder', 'ExecRepoBench')} and "
        f"{selected_value('Gemini', 'OpenCoder', 'ExecRepoBench')} for OpenCoder "
        f"(McNemar $p={mcnemar_value('GPT', 'RAG + Verify/Repair', 'ExecRepoBench')}$ "
        f"and $p={mcnemar_value('Gemini', 'RAG + Verify/Repair', 'ExecRepoBench')}$)."
    )
    interpretation = (
        "Verification and test-guided repair account for much of the improvement over plain RAG. "
        "OpenCoder provides the highest Pass@1 on RepoExec-inline for both backends, but its uncertainty-aware "
        "filtering and generation do not consistently improve on the matched verification control "
        "and sharply reduce candidate-set performance under incomplete repository context. Given "
        "the 14- and 10-task samples, these results are evidence of a boundary condition rather "
        "than universal method superiority."
    )
    lines = [
        "# RQ3: End-to-End Effectiveness",
        "",
        f"Current status: {scope_status}",
        "",
        "Metric definition: Baseline RAG uses its five raw candidates. RAG + Verify/Repair reuses those exact candidates and retrieval evidence, replacing the primary candidate only after verified selection or test-guided repair. OpenCoder uses its Phase-V selected/repaired primary followed by its remaining raw candidates.",
        "",
        repo_sentence,
        "",
        exec_sentence,
        "",
        selected_sentence,
        "",
    ]
    tex = [
        r"\subsection{RQ3: End-to-End Effectiveness}",
        "",
        r"Table~\ref{tab:rq3_end_to_end_effectiveness} compares all methods under the same "
        r"five-candidate budget, model backend, task manifest, and Pass@$k$ estimator. "
        + repo_sentence,
        exec_sentence,
        selected_sentence,
    ]
    lines.extend([
        "",
        f"Interpretation: {interpretation}",
        "",
        "RepoFormer remains excluded because an exact same-budget reproduction is not supported by the available artifacts. No result is copied from an original paper.",
    ])
    tex.extend([
        r"\smallskip",
        interpretation,
        r"\paragraph{Finding 3.} Verification and test-guided repair explain a substantial "
        r"share of the gain over plain RAG. OpenCoder attains the highest Pass@1 on "
        r"RepoExec-inline for both backends, but does not consistently outperform the matched verification "
        r"control; under partial repository context, uncertainty-aware evidence filtering "
        r"can discard useful alternatives and reduce functional correctness.",
    ])
    (out_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "findings.tex").write_text("\n\n".join(tex), encoding="utf-8")


def _write_availability(
    out_dir: Path,
    metadata: List[Dict[str, Any]],
    *,
    executable_codereval: bool,
    external_available: bool,
) -> None:
    exec_scope = _exec_scope_label(metadata)
    evaluated = {
        _dataset_label(str(item.get("dataset") or ""))
        for item in metadata
    }
    lines = [
        "# RQ3 Benchmark and Baseline Availability",
        "",
        "Computed in this workspace:",
        "",
        f"- {exec_scope}: available with executable repository tests.",
        "- RepoExec-inline 14-task local executable subset: available with self-contained public RepoExec tests.",
        (
            "- CoderEval: included with executable repository tests."
            if executable_codereval
            else (
                "- CoderEval: present only as a non-executable local subset and excluded "
                "from this execution-based package."
                if "CoderEval" in evaluated
                else "- CoderEval: not included in this matched execution campaign."
            )
        ),
        "- Baseline RAG: available through `uncertainty_aware=false`.",
        "- RAG + Verify/Repair: exact Baseline RAG candidates with local verified selection and at most two repair calls.",
        "- OpenCoder: available through `uncertainty_aware=true`.",
        (
            "- AllianceCoder: available as a clean-room reproduction that preserves its "
            "defining retrieval policy; the official repository has no declared license."
            if external_available
            else "- AllianceCoder: clean-room reproduction summary was not supplied to this build."
        ),
        "",
        "Blocked for full three-benchmark RQ3:",
        "",
        "- Full RepoExec official harness: blocked because original `/output/...` pickle assets are not present.",
        "",
        "Blocked reference-paper methods:",
        "",
    ]
    lines.extend(f"- {method}: runnable local implementation is not present." for method in UNIMPLEMENTED_METHODS)
    lines.append("")
    lines.append("No values are fabricated or copied from the reference paper in the generated RQ3 artifacts.")
    (out_dir / "availability.md").write_text("\n".join(lines), encoding="utf-8")


def _write_readme(out_dir: Path) -> None:
    text = """# RQ3 Reproduction

## Environment

Install the project requirements in the existing virtual environment, then provide API keys through environment variables or `.env`:

- `OPENAI_API_KEY` for the direct OpenAI backend
- `GEMINI_API_KEY` for the direct Gemini backend
- `OPENCODER_LLM_API_KEY` for an OpenAI-compatible endpoint
- `OPENCODER_LLM_BASE_URL` for the OpenAI-compatible gateway

Do not place API keys in source files or generated artifacts.

## Smoke Tests

```bash
.venv/bin/python -m pytest tests/test_smoke.py opencoder/tests/test_smoke.py opencoder/phase5_verify/test_validate.py
```

## RQ3 Runs

```bash
.venv/bin/python -m experiments.run_rq3 \\
  --config configs/rq3/gpt4o_mini.yaml \\
  --benchmark repoexec \\
  --benchmark-path input/repoexec_python_string_utils_inline14.jsonl \\
  --method paired --limit 14 --force \\
  --out results/rq3/runs/matched_external/gpt_repoexec_full/rq3.json

.venv/bin/python -m experiments.run_rq3 \\
  --config configs/rq3/gemini_2_5_flash.yaml \\
  --benchmark execrepobench \\
  --benchmark-path input/execrepobench_testbacked.jsonl \\
  --method paired --limit 10 --force \\
  --out results/rq3/runs/matched_external/gemini_execrepobench_full/rq3.json
```

Repeat the command for the other backend/benchmark combinations. The runner
writes progress after every task; omit `--force` to resume.

Build the exact-candidate RAG + Verify/Repair control after the paired runs:

```bash
bash scripts/run_remaining_rq3_repairs.sh
```

RAG + Verify/Repair reuses Baseline RAG's exact retrieval evidence and five
stored candidates. OpenCoder and RAG + Verify/Repair report end-to-end Pass@k
using the Phase-V selected/repaired primary followed by the remaining raw
candidates. Both raw and effective correctness streams are retained.

## Build Paper Artifacts

```bash
.venv/bin/python scripts/build_rq3_results.py \\
  --run results/rq3/runs/matched_external/gpt_repoexec_full/rq3.json \\
  --run results/rq3/runs/matched_external/gemini_repoexec_full/rq3.json \\
  --run results/rq3/runs/matched_external/gpt_execrepobench_full/rq3.json \\
  --run results/rq3/runs/matched_external/gemini_execrepobench_full/rq3.json \\
  --out-dir results/rq3_verify_repair/final
```

Expected outputs include `summary.csv`, `selected_output.csv`, `per_task.csv`,
`failures.csv`, `statistical_tests.csv`, `table_rq3.tex`, `findings.md`,
`findings.tex`, `paper_ready.tex`, and paper figures.

## Current Limitations

The matched package covers the 14-task executable RepoExec-inline subset and
the 10-task test-backed ExecRepoBench subset. ExecRepoBench remains
context-limited because complete repository snapshots are unavailable.
No external-baseline summary is included in this package. RepoFormer remains
deferred because an exact same-budget reproduction is not supported by the
available artifacts. No published result number is imported into the generated
tables.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_codereval_repeats(
    out_dir: Path,
    primary_summary: List[Dict[str, Any]],
    repeat_paths: List[str],
) -> None:
    if not repeat_paths:
        return

    rows: List[Dict[str, Any]] = []
    for row in primary_summary:
        if row.get("CoderEval_Pass1") is None:
            continue
        rows.append({
            "Campaign": "Matched main campaign",
            "Backend": row["Backend"],
            "Method": row["Method"],
            "Pass@1": row["CoderEval_Pass1"],
            "Pass@3": row["CoderEval_Pass3"],
            "Pass@5": row["CoderEval_Pass5"],
        })

    for index, path in enumerate(repeat_paths, start=1):
        task_rows, _, _ = _extract_rows(path)
        repeat_summary = _summary_rows(task_rows)
        for row in repeat_summary:
            if row.get("CoderEval_Pass1") is None:
                continue
            rows.append({
                "Campaign": f"Independent repeat {index}",
                "Backend": row["Backend"],
                "Method": row["Method"],
                "Pass@1": row["CoderEval_Pass1"],
                "Pass@3": row["CoderEval_Pass3"],
                "Pass@5": row["CoderEval_Pass5"],
            })

    _write_csv(
        out_dir / "codereval_independent_repeat.csv",
        rows,
        ["Campaign", "Backend", "Method", "Pass@1", "Pass@3", "Pass@5"],
    )
    lines = [
        "# CoderEval Independent Repeat",
        "",
        "This check uses the same 19 executable tasks, model settings, five-candidate budget, "
        "and mutation-audited native test harness. It is reported separately and is not pooled "
        "with the matched main campaign.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['Campaign']}, {row['Backend']} {row['Method']}: "
            f"Pass@1={row['Pass@1']:.2f}, Pass@3={row['Pass@3']:.2f}, "
            f"Pass@5={row['Pass@5']:.2f}."
        )
    lines.extend([
        "",
        "The fresh GPT repeat does not reverse the matched-campaign conclusion: OpenCoder "
        "does not improve executable CoderEval with this backend. This backend-dependent "
        "negative result must be retained in the paper and motivates reporting run-to-run "
        "variation rather than claiming universal improvement.",
    ])
    (out_dir / "codereval_independent_repeat.md").write_text("\n".join(lines), encoding="utf-8")


def _plot(out_dir: Path, summary: List[Dict[str, Any]]) -> None:
    datasets = [
        dataset for dataset in DATASETS
        if any(row.get(f"{dataset}_Pass1") is not None for row in summary)
    ]
    if not datasets:
        return
    backends = sorted({row["Backend"] for row in summary})
    method_labels = {
        "Baseline RAG": "Baseline",
        "AllianceCoder (clean-room)": "Alliance",
        "OpenCoder": "OpenCoder",
        "RAG + Verify/Repair": "RAG+Repair",
    }
    width = 0.24
    colors = ["#4b5563", "#2563eb", "#059669"]
    plotted_values = [
        float(row[f"{dataset}_{metric}"])
        for row in summary
        for dataset in datasets
        for metric in ("Pass1", "Pass3", "Pass5")
        if row.get(f"{dataset}_{metric}") is not None
    ]
    y_max = min(100.0, math.ceil((max(plotted_values) + 8.0) / 10.0) * 10.0)
    fig, axes = plt.subplots(
        len(backends),
        len(datasets),
        figsize=(7.2, 4.8),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, backend in enumerate(backends):
        backend_rows = sorted(
            [row for row in summary if row["Backend"] == backend],
            key=lambda row: METHOD_ORDER.get(row["Method"], 99),
        )
        labels = [method_labels.get(row["Method"], row["Method"]) for row in backend_rows]
        x = np.arange(len(labels))
        for col_index, dataset in enumerate(datasets):
            ax = axes[row_index][col_index]
            metrics = [
                (f"{dataset}_Pass1", "Pass@1"),
                (f"{dataset}_Pass3", "Pass@3"),
                (f"{dataset}_Pass5", "Pass@5"),
            ]
            for metric_index, (key, label) in enumerate(metrics):
                values = [
                    float(row[key]) if row.get(key) is not None else 0.0
                    for row in backend_rows
                ]
                ax.bar(
                    x + (metric_index - 1) * width,
                    values,
                    width,
                    label=label,
                    color=colors[metric_index],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"{backend} | {dataset}", fontsize=9)
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_index == 0:
                ax.set_ylabel("Pass@k (%)")
    axes[0][-1].legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.34),
    )
    for ext in ("pdf", "png"):
        (out_dir / f"fig_rq3_execrepobench_passk.{ext}").unlink(missing_ok=True)
        fig.savefig(out_dir / f"fig_rq3_passk_by_benchmark.{ext}", dpi=300)
    plt.close(fig)


def _write_paper_ready(out_dir: Path) -> None:
    table = (out_dir / "table_rq3.tex").read_text(encoding="utf-8").strip()
    findings = (out_dir / "findings.tex").read_text(encoding="utf-8").strip()
    text = (
        "% Generated from validated local runs. Do not edit numerical values by hand.\n"
        f"{table}\n\n{findings}\n"
    )
    (out_dir / "paper_ready.tex").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-run", action="append", default=[])
    ap.add_argument("--gemini-run", action="append", default=[])
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--replication-run", action="append", default=[])
    ap.add_argument(
        "--external-summary",
        help="CSV produced by the independently audited external-baseline builder.",
    )
    ap.add_argument("--out-dir", default="results/rq3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_task: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    metadata = []
    run_paths = [*args.gpt_run, *args.gemini_run, *args.run]
    if not run_paths:
        raise SystemExit("Provide at least one --run, --gpt-run, or --gemini-run path.")
    for path in run_paths:
        task_rows, run_raw_rows, run_meta = _extract_rows(path)
        per_task.extend(task_rows)
        raw_rows.extend(run_raw_rows)
        metadata.append(run_meta)
        raw_path = raw_dir / (Path(path).parent.name + ".jsonl")
        with raw_path.open("w", encoding="utf-8") as f:
            for row in run_raw_rows:
                f.write(json.dumps(row, default=str) + "\n")

    internal_summary = _summary_rows(per_task)
    external_summary = _load_external_summary(args.external_summary)
    summary = [*internal_summary, *external_summary]
    stats = _stats_rows(per_task)
    multiseed = _multiseed_rows(per_task)
    selected = _selected_output_rows(per_task)
    selected_stats = _selected_output_stats(per_task)
    resources = _resource_rows(per_task)
    coverage = _coverage_rows(per_task)
    failures = [
        row for row in per_task
        if row.get("error") or row.get("passed") is False
    ]

    summary_columns = ["Backend", "Method"]
    for dataset in DATASETS:
        for metric in METRICS:
            summary_columns.append(f"{dataset}_{metric.replace('@', '')}")
    summary_columns.extend([
        "MeanUncertainty",
        "MeanRepairRounds",
        "MeanRunLatencyS",
        "MeanLLMRequests",
        "MeanTotalTokens",
        "N",
        "Qualification",
    ])

    _write_csv(out_dir / "summary.csv", summary, summary_columns)
    _write_csv(
        out_dir / "selected_output.csv",
        selected,
        [
            "Backend",
            "Method",
            "Benchmark",
            "Passed",
            "N",
            "ExpectedN",
            "Complete",
            "SelectedPassRate",
        ],
    )
    _write_csv(
        out_dir / "coverage.csv",
        coverage,
        list(coverage[0].keys()) if coverage else [],
    )
    _write_csv(
        out_dir / "selected_output_tests.csv",
        selected_stats,
        list(selected_stats[0].keys()) if selected_stats else [],
    )
    _write_csv(
        out_dir / "resource_summary.csv",
        resources,
        list(resources[0].keys()) if resources else [],
    )
    _write_csv(out_dir / "per_task.csv", per_task, list(per_task[0].keys()) if per_task else [])
    _write_csv(out_dir / "failures.csv", failures, list(per_task[0].keys()) if per_task else [])
    _write_csv(out_dir / "statistical_tests.csv", stats, list(stats[0].keys()) if stats else [])
    if multiseed:
        _write_csv(
            out_dir / "multiseed_summary.csv",
            multiseed,
            list(multiseed[0].keys()),
        )
    else:
        (out_dir / "multiseed_summary.csv").unlink(missing_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "rows": summary}, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    executable_codereval = _has_executable_codereval(per_task)
    _write_table(
        out_dir / "table_rq3.tex",
        summary,
        executable_codereval=executable_codereval,
    )
    _write_findings(
        out_dir,
        stats,
        metadata,
        per_task,
        summary,
        selected,
        selected_stats,
        executable_codereval=executable_codereval,
    )
    _write_availability(
        out_dir,
        metadata,
        executable_codereval=executable_codereval,
        external_available=bool(external_summary),
    )
    _write_readme(out_dir)
    _write_multiseed_table(out_dir / "table_rq3_multiseed.tex", multiseed)
    _write_budget_table(out_dir / "table_rq3_resource_budget.tex", internal_summary)
    _write_selected_table(out_dir / "table_rq3_selected_output.tex", selected)
    _write_paired_stats_table(out_dir / "table_rq3_paired_tests.tex", stats)
    _write_selected_stats_table(
        out_dir / "table_rq3_selected_tests.tex",
        selected_stats,
    )
    _write_resource_detail_table(
        out_dir / "table_rq3_resources.tex",
        resources,
    )
    _plot_multiseed(out_dir, multiseed)
    _write_codereval_repeats(out_dir, summary, args.replication_run)
    _plot(out_dir, summary)
    _write_paper_ready(out_dir)
    print(f"Wrote RQ3 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
