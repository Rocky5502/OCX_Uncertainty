"""RQ2 ablation: with vs. without uncertainty-aware scoring.

Runs the full pipeline twice per example: uncertainty_aware=True and
False. Compares pass@1, mean uncertainty, and calibration (ECE) of the
uncertainty signal against correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.evaluation.metrics import (  # noqa: E402
    pass_at_k,
    pass_at_ks_from_samples,
    pass_rate_variance,
    uncertainty_calibration_ece,
)
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402
from opencoder.phase2_query import ImplementationStep, RetrievalIntent  # noqa: E402


COMPONENTS = (
    "uncertainty_decomposition",
    "uncertainty_filtering",
    "uncertainty_guided_generation",
    "uncertainty_verified_selection",
    "uncertainty_triggered_repair",
)

CONDITIONS = {
    "without": (False, False, False, False, False),
    "decomposition": (True, False, False, False, False),
    "filtering": (True, True, False, False, False),
    "guidance": (True, True, True, False, False),
    "selection": (True, True, True, True, False),
    "with": (True, True, True, True, True),
}


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _bootstrap_ci(values: Sequence[float], *, seed: int = 2027, n_boot: int = 10000):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _risk_metrics(uncertainties: Sequence[float], passed: Sequence[bool]) -> Dict[str, float | None]:
    if not uncertainties or len(uncertainties) != len(passed):
        return {"failure_auroc": None, "failure_auprc": None, "brier": None, "aurc": None}
    scores = np.asarray(uncertainties, dtype=float)
    failures = 1.0 - np.asarray(passed, dtype=float)
    positives = scores[failures == 1.0]
    negatives = scores[failures == 0.0]
    auroc = None
    if len(positives) and len(negatives):
        comparisons = [float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives]
        auroc = float(np.mean(comparisons))

    order_desc = np.argsort(-scores, kind="stable")
    sorted_failures = failures[order_desc]
    total_positive = float(np.sum(sorted_failures))
    auprc = None
    if total_positive:
        precision = np.cumsum(sorted_failures) / np.arange(1, len(sorted_failures) + 1)
        auprc = float(np.sum(precision * sorted_failures) / total_positive)

    order_asc = np.argsort(scores, kind="stable")
    accepted_failures = failures[order_asc]
    coverage_risk = np.cumsum(accepted_failures) / np.arange(1, len(accepted_failures) + 1)
    return {
        "failure_auroc": auroc,
        "failure_auprc": auprc,
        "brier": float(np.mean((scores - failures) ** 2)),
        "aurc": float(np.mean(coverage_risk)),
    }


def _config_for(base: PipelineConfig, flags: Sequence[bool]) -> PipelineConfig:
    values = dict(base.__dict__)
    values["uncertainty_aware"] = any(flags)
    values.update(dict(zip(COMPONENTS, flags)))
    return PipelineConfig(**values)


def _plan_from_row(row: dict | None):
    debug = (row or {}).get("per_step") or []
    if not debug:
        return None
    steps, intents = [], []
    for index, item in enumerate(debug, start=1):
        steps.append(ImplementationStep(
            index=index,
            description=str(item.get("step") or ""),
            uncertainty=float(item.get("step_uncertainty") or 0.0),
        ))
        queries = {
            source: str(stats.get("query") or item.get("step") or "")
            for source, stats in (item.get("sources") or {}).items()
        }
        intents.append(RetrievalIntent(
            step_index=index,
            source_weights=dict(item.get("intent") or {}),
            queries=queries,
        ))
    return steps, intents


def _summarize(out: dict) -> dict:
    summary = {}
    for k, rows in out.items():
        if k not in CONDITIONS:
            continue
        known = [r for r in rows if r.get("passed") is not None and "u" in r]
        passed = [bool(r["passed"]) for r in known]
        uncs = [r["u"]["aggregate"] for r in known]
        initial_known = [
            r for r in rows
            if (r.get("initial_test_report") or {}).get("passed") is not None and "u" in r
        ]
        initial_passed = [bool(r["initial_test_report"]["passed"]) for r in initial_known]
        initial_uncs = [float(r["u"]["aggregate"]) for r in initial_known]
        sample_sets = [r["sample_correctness"] for r in rows if r.get("sample_correctness")]
        uncertainty_values = [r["u"]["aggregate"] for r in rows if "u" in r]
        repair_values = [r["repair_rounds"] for r in rows if "repair_rounds" in r]
        item = {
            "effective_pass@1": pass_at_k(passed, k=1) if known else None,
            "pass_rate_variance": pass_rate_variance(sample_sets),
            "ece": uncertainty_calibration_ece(initial_uncs, initial_passed) if initial_known else None,
            "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else None,
            "mean_repair_rounds": float(np.mean(repair_values)) if repair_values else None,
            "initial_pass@1": _mean([float(value) for value in initial_passed]),
            "n": len(rows),
            "n_known_correctness": len(known),
            "n_errors": sum(1 for r in rows if "error" in r),
        }
        if sample_sets:
            max_supported_k = min(len(samples) for samples in sample_sets)
            valid_ks = tuple(k for k in (1, 3, 5) if k <= max_supported_k)
            item.update(dict(pass_at_ks_from_samples(sample_sets, ks=valid_ks)))
        item.update(_risk_metrics(initial_uncs, initial_passed))

        initial_failures = [
            r for r in rows if (r.get("initial_test_report") or {}).get("passed") is False
        ]
        post_selection_failures = [
            r for r in rows if (r.get("post_selection_test_report") or {}).get("passed") is False
        ]
        item["selection_success_rate"] = _mean([
            float((r.get("post_selection_test_report") or {}).get("passed") is True)
            for r in initial_failures
        ])
        item["repair_success_rate"] = _mean([
            float(r.get("passed") is True) for r in post_selection_failures if r.get("repair_rounds", 0) > 0
        ])
        initially_correct = [
            r for r in rows if (r.get("initial_test_report") or {}).get("passed") is True
        ]
        item["regression_rate"] = _mean([
            float(r.get("passed") is False) for r in initially_correct
        ])
        item["mean_run_latency_s"] = _mean([
            float(r["run_latency_s"]) for r in rows if r.get("run_latency_s") is not None
        ])
        summary[k] = item
    baseline_rows = {
        row.get("id"): row for row in out.get("without", [])
        if row.get("id") is not None and "error" not in row
    }
    for key in CONDITIONS:
        if key == "without" or key not in summary:
            continue
        condition_rows = {
            row.get("id"): row for row in out.get(key, [])
            if row.get("id") is not None and "error" not in row
        }
        common = sorted(set(baseline_rows) & set(condition_rows))
        final_diffs = [
            float(bool(condition_rows[task].get("passed")))
            - float(bool(baseline_rows[task].get("passed")))
            for task in common
            if condition_rows[task].get("passed") is not None
            and baseline_rows[task].get("passed") is not None
        ]
        sample_diffs = []
        for task in common:
            condition_value = (condition_rows[task].get("pass_at_k") or {}).get("pass@1")
            baseline_value = (baseline_rows[task].get("pass_at_k") or {}).get("pass@1")
            if condition_value is not None and baseline_value is not None:
                sample_diffs.append(float(condition_value) - float(baseline_value))
        summary[key]["delta_effective_pass@1_vs_baseline"] = _mean(final_diffs)
        summary[key]["delta_effective_pass@1_ci95"] = _bootstrap_ci(final_diffs)
        summary[key]["delta_sample_pass@1_vs_baseline"] = _mean(sample_diffs)
        summary[key]["delta_sample_pass@1_ci95"] = _bootstrap_ci(sample_diffs)
        summary[key]["n_paired_with_baseline"] = len(final_diffs)
    return summary


def _write_payload(path: str, out: dict, metadata: dict) -> None:
    payload = dict(out)
    payload["summary"] = _summarize(out)
    payload["metadata"] = metadata
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--describe-limit", type=int, default=100)
    ap.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated component conditions: " + ",".join(CONDITIONS),
    )
    ap.add_argument("--out", default="results/rq2.json")
    ap.add_argument("--resume", action="store_true", help="Resume completed task-condition rows from --out.")
    ap.add_argument("--rerun-ids", default=None, help="Comma-separated task ids to replace as complete paired blocks.")
    args = ap.parse_args()

    base = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    base.llm_backend = os.environ.get("OPENCODER_LLM_BACKEND", base.llm_backend)
    base.llm_model = os.environ.get("OPENCODER_LLM_MODEL", base.llm_model)

    requested = list(CONDITIONS)
    if args.conditions:
        requested = [name.strip() for name in args.conditions.split(",") if name.strip()]
        unknown = [name for name in requested if name not in CONDITIONS]
        if unknown:
            raise ValueError(f"Unknown RQ2 conditions: {', '.join(unknown)}")
    pipes = {name: Pipeline(_config_for(base, CONDITIONS[name])) for name in requested}
    indexing_pipe = pipes.get("with") or next(iter(pipes.values()))
    examples = list(load_dataset(args.dataset, args.dataset_path, limit=args.limit))

    out = {name: [] for name in requested}
    if args.resume and os.path.exists(args.out):
        try:
            existing = json.load(open(args.out))
            for name in requested:
                out[name] = list(existing.get(name) or [])
            print(
                f"Resuming {sum(len(rows) for rows in out.values())} existing RQ2 rows from {args.out}",
                flush=True,
            )
        except Exception as exc:
            print(f"Could not resume {args.out}: {exc}; starting a new result.", flush=True)
            out = {name: [] for name in requested}
    rerun_ids = {value.strip() for value in (args.rerun_ids or "").split(",") if value.strip()}
    if rerun_ids:
        for name in requested:
            out[name] = [row for row in out[name] if row.get("id") not in rerun_ids]
        print(f"Replacing paired RQ2 blocks for: {', '.join(sorted(rerun_ids))}", flush=True)
    completed = {
        name: {row.get("id") for row in rows if "error" not in row}
        for name, rows in out.items()
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rq": "RQ2",
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "repo_root": args.repo_root,
        "limit": args.limit,
        "describe_limit": args.describe_limit,
        "backend": base.llm_backend,
        "model": base.llm_model,
        "base_url": os.environ.get("OPENCODER_LLM_BASE_URL"),
        "conditions": {
            name: dict(zip(COMPONENTS, CONDITIONS[name])) for name in requested
        },
        "pretest_uncertainty_note": "Failure-prediction metrics use generation uncertainty computed before validation against the initial generated candidate.",
    }
    for ex in examples:
        missing = [name for name in requested if ex.id not in completed[name]]
        if not missing:
            print(f"  {ex.id:<24} all requested conditions already complete", flush=True)
            continue
        # Build Phase I once so both paired conditions use identical evidence,
        # descriptions, knowledge uncertainty, and vector indexes.
        index_start = time.perf_counter()
        try:
            _, retrievers = indexing_pipe.index_example(
                ex,
                fallback_repo_root=args.repo_root,
                describe_limit=args.describe_limit,
            )
            index_latency_s = time.perf_counter() - index_start
        except Exception as e:
            for key in missing:
                out[key].append({"id": ex.id, "error": f"index_example: {e}"})
            print(f"  {ex.id:<24} index ERROR: {e}", flush=True)
            _write_payload(args.out, out, metadata)
            continue
        aware_plan = None
        aware_plan_error = None
        if any(CONDITIONS[name][0] for name in missing):
            prior = next(
                (
                    row for name in requested if CONDITIONS[name][0]
                    for row in out[name]
                    if row.get("id") == ex.id and "error" not in row
                ),
                None,
            )
            aware_plan = _plan_from_row(prior)
            if aware_plan is None:
                try:
                    aware_plan = indexing_pipe.prepare_query(ex, uncertainty_decomposition=True)
                except Exception as exc:
                    aware_plan_error = f"prepare_query: {exc}"
        baseline_plan = indexing_pipe.prepare_query(ex, uncertainty_decomposition=False)
        for key in missing:
            pipe = pipes[key]
            run_start = time.perf_counter()
            try:
                if CONDITIONS[key][0] and aware_plan is None:
                    raise RuntimeError(aware_plan_error or "uncertainty-aware query plan unavailable")
                plan = aware_plan if CONDITIONS[key][0] else baseline_plan
                r = pipe.run(ex, retrievers, prepared_query=plan)
                run_latency_s = time.perf_counter() - run_start
                out[key] = [row for row in out[key] if row.get("id") != ex.id]
                out[key].append({
                    "id": ex.id,
                    "code": r.code,
                    "generated_samples": r.generated_samples,
                    "passed": r.test_report.get("passed"),
                    "correctness_mode": r.correctness_mode,
                    "static_report": r.static_report,
                    "test_report": r.test_report,
                    "u": r.uncertainty_trace,
                    "uncertainty_components": r.uncertainty_components,
                    "per_step": r.per_step,
                    "fused_evidence": r.fused_evidence,
                    "source_diagnostics": r.source_diagnostics,
                    "sample_correctness": r.sample_correctness,
                    "pass_at_k": r.pass_at_k,
                    "pass_rate_variance": r.pass_rate_variance,
                    "repair_rounds": r.repair_rounds,
                    "initial_test_report": r.initial_test_report,
                    "post_selection_test_report": r.post_selection_test_report,
                    "verified_selection_applied": r.verified_selection_applied,
                    "index_latency_s": index_latency_s,
                    "run_latency_s": run_latency_s,
                })
                print(
                    f"  {ex.id:<24} {key:<8} "
                    f"u={r.uncertainty_trace['aggregate']:.3f} "
                    f"pass={r.test_report.get('passed')}",
                    flush=True,
                )
            except Exception as e:
                out[key] = [row for row in out[key] if row.get("id") != ex.id]
                out[key].append({"id": ex.id, "error": str(e)})
                print(f"  {ex.id:<24} {key:<8} ERROR: {e}", flush=True)
            _write_payload(args.out, out, metadata)

    summary = _summarize(out)
    _write_payload(args.out, out, metadata)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
