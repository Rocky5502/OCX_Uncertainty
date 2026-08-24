"""RQ1 ablation: per-source uncertainty correlation.

For each example, run the pipeline 4 times:
  - all sources
  - api only
  - context only
  - similar_code only

Record uncertainty trace + correctness. Then compute Spearman
correlation between (presence of each source) and (aggregate
uncertainty / pass@1). The deliverable is a per-source contribution
table for the paper.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402
from opencoder.phase2_query import ImplementationStep, RetrievalIntent  # noqa: E402


SOURCES = ("api", "context", "similar_code")
SOURCE_LABELS = {source: source for source in SOURCES}


def _safe_spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"rho": None, "p": None}
    rho, p = spearmanr(xs, ys)
    if np.isnan(rho) or np.isnan(p):
        return {"rho": None, "p": None}
    return {"rho": float(rho), "p": float(p)}


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _bootstrap_ci(values: Sequence[float], *, seed: int = 2027, n_boot: int = 10000) -> List[float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_flip_p(values: Sequence[float]) -> float | None:
    """Two-sided paired randomization test over task-level effects."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(arr):
        return None
    observed = abs(float(np.mean(arr)))
    if observed == 0.0:
        return 1.0
    if len(arr) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(arr))))
    else:
        rng = np.random.default_rng(2027)
        signs = rng.choice((-1.0, 1.0), size=(100000, len(arr)))
    null = np.abs(np.mean(signs * arr, axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def _holm_adjust(pairs: Sequence[tuple[str, float | None]]) -> Dict[str, float | None]:
    valid = sorted(((name, p) for name, p in pairs if p is not None), key=lambda item: item[1])
    adjusted: Dict[str, float | None] = {name: None for name, _ in pairs}
    running = 0.0
    total = len(valid)
    for rank, (name, p_value) in enumerate(valid):
        running = max(running, min(1.0, (total - rank) * float(p_value)))
        adjusted[name] = running
    return adjusted


def _row_metric(row: dict, metric: str) -> float | None:
    if "error" in row:
        return None
    if metric == "uncertainty":
        value = (row.get("u") or {}).get("aggregate")
        return float(value) if value is not None else None
    if metric.startswith("pass@"):
        value = (row.get("pass_at_k") or {}).get(metric)
        return float(value) if value is not None else None
    if metric == "final_pass":
        value = row.get("passed")
        return float(bool(value)) if value is not None else None
    raise KeyError(metric)


def _factorial_effects(rows: List[dict], metric: str) -> Dict[str, Any]:
    by_task: Dict[str, Dict[tuple[int, int, int], float]] = {}
    for row in rows:
        value = _row_metric(row, metric)
        task_id = row.get("example_id")
        if task_id is None or value is None:
            continue
        enabled = set(row.get("enabled") or [])
        mask = tuple(int(source in enabled) for source in SOURCES)
        by_task.setdefault(str(task_id), {})[mask] = value

    main: Dict[str, Any] = {}
    for source_index, source in enumerate(SOURCES):
        task_effects = []
        present_values, absent_values = [], []
        for task_rows in by_task.values():
            contrasts = []
            other = [i for i in range(3) if i != source_index]
            for bits in itertools.product((0, 1), repeat=2):
                absent = [0, 0, 0]
                for idx, bit in zip(other, bits):
                    absent[idx] = bit
                present = list(absent)
                present[source_index] = 1
                absent_key, present_key = tuple(absent), tuple(present)
                if absent_key in task_rows and present_key in task_rows:
                    contrasts.append(task_rows[present_key] - task_rows[absent_key])
                    present_values.append(task_rows[present_key])
                    absent_values.append(task_rows[absent_key])
            if contrasts:
                task_effects.append(float(np.mean(contrasts)))
        main[source] = {
            "effect_present_minus_absent": _mean(task_effects),
            "ci95": _bootstrap_ci(task_effects),
            "p_randomization": _sign_flip_p(task_effects),
            "mean_when_present": _mean(present_values),
            "mean_when_absent": _mean(absent_values),
            "n_tasks": len(task_effects),
        }

    adjusted = _holm_adjust([(source, item["p_randomization"]) for source, item in main.items()])
    for source, value in adjusted.items():
        main[source]["p_holm"] = value

    interactions: Dict[str, Any] = {}
    for first, second in itertools.combinations(range(3), 2):
        third = next(index for index in range(3) if index not in {first, second})
        effects = []
        for task_rows in by_task.values():
            task_values = []
            for third_value in (0, 1):
                keys = {}
                for a, b in itertools.product((0, 1), repeat=2):
                    mask = [0, 0, 0]
                    mask[first], mask[second], mask[third] = a, b, third_value
                    keys[(a, b)] = tuple(mask)
                if all(key in task_rows for key in keys.values()):
                    task_values.append(
                        task_rows[keys[(1, 1)]] - task_rows[keys[(1, 0)]]
                        - task_rows[keys[(0, 1)]] + task_rows[keys[(0, 0)]]
                    )
            if task_values:
                effects.append(float(np.mean(task_values)))
        name = f"{SOURCES[first]}:{SOURCES[second]}"
        interactions[name] = {
            "effect": _mean(effects),
            "ci95": _bootstrap_ci(effects),
            "p_randomization": _sign_flip_p(effects),
            "n_tasks": len(effects),
        }
    adjusted_interactions = _holm_adjust([
        (name, item["p_randomization"]) for name, item in interactions.items()
    ])
    for name, value in adjusted_interactions.items():
        interactions[name]["p_holm"] = value
    return {"metric": metric, "main_effects": main, "two_way_interactions": interactions}


def _run_with(pipe, ex, retrievers, enabled, prepared_query=None):
    pipe.cfg.enable_sources = tuple(enabled)
    return pipe.run(ex, retrievers, prepared_query=prepared_query)


def _plan_from_row(row: dict):
    debug = row.get("per_step") or []
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


def _summarize(rows: List[dict]) -> dict:
    summary: Dict[str, Any] = {}
    factorial_u = _factorial_effects(rows, "uncertainty")
    factorial_pass1 = _factorial_effects(rows, "pass@1")
    for s in SOURCES:
        present = []
        agg_u = []
        pass_var = []
        known_present = []
        known_passed = []
        for r in rows:
            if "u" not in r:
                continue
            is_present = 1 if s in r["enabled"] else 0
            present.append(is_present)
            agg_u.append(r["u"]["aggregate"])
            pass_var.append(float(r.get("pass_rate_variance") or 0.0))
            if r.get("passed") is not None:
                known_present.append(is_present)
                known_passed.append(1 if r["passed"] else 0)
        if len(set(present)) > 1:
            uncertainty_corr = _safe_spearman(present, agg_u)
            pass_corr = _safe_spearman(known_present, known_passed)
            present_u = [u for u, x in zip(agg_u, present) if x]
            absent_u = [u for u, x in zip(agg_u, present) if not x]
            present_var = [v for v, x in zip(pass_var, present) if x]
            absent_var = [v for v, x in zip(pass_var, present) if not x]
            effect_u = factorial_u["main_effects"].get(s, {})
            effect_pass = factorial_pass1["main_effects"].get(s, {})
            summary[s] = {
                "spearman_uncertainty": uncertainty_corr,
                "spearman_passed": pass_corr,
                "mean_u_when_present": effect_u.get("mean_when_present", float(np.mean(present_u))),
                "mean_u_when_absent": effect_u.get("mean_when_absent", float(np.mean(absent_u))),
                "delta_u_present_minus_absent": effect_u.get(
                    "effect_present_minus_absent", float(np.mean(present_u) - np.mean(absent_u))
                ),
                "delta_u_ci95": effect_u.get("ci95"),
                "delta_u_p_randomization": effect_u.get("p_randomization"),
                "delta_u_p_holm": effect_u.get("p_holm"),
                "delta_pass_at_1": effect_pass.get("effect_present_minus_absent"),
                "delta_pass_at_1_ci95": effect_pass.get("ci95"),
                "delta_pass_at_1_p_holm": effect_pass.get("p_holm"),
                "n_paired_tasks": effect_u.get("n_tasks"),
                "mean_pass_rate_variance_when_present": float(np.mean(present_var)) if present_var else None,
                "mean_pass_rate_variance_when_absent": float(np.mean(absent_var)) if absent_var else None,
            }
    condition_summary = {}
    for condition in sorted({row.get("condition") for row in rows if row.get("condition")}):
        condition_rows = [row for row in rows if row.get("condition") == condition and "error" not in row]
        condition_summary[condition] = {
            "n": len(condition_rows),
            "mean_uncertainty": _mean([v for row in condition_rows if (v := _row_metric(row, "uncertainty")) is not None]),
            "pass@1": _mean([v for row in condition_rows if (v := _row_metric(row, "pass@1")) is not None]),
            "pass@3": _mean([v for row in condition_rows if (v := _row_metric(row, "pass@3")) is not None]),
            "pass@5": _mean([v for row in condition_rows if (v := _row_metric(row, "pass@5")) is not None]),
        }
    summary["condition_summary"] = condition_summary
    summary["factorial"] = {
        "uncertainty": factorial_u,
        "pass@1": factorial_pass1,
        "pass@3": _factorial_effects(rows, "pass@3"),
        "pass@5": _factorial_effects(rows, "pass@5"),
        "inference_unit": "task",
        "ci": "task bootstrap, 10000 resamples",
        "test": "paired task-level sign-flip randomization with Holm correction",
    }
    return summary


def _write_payload(path: str, rows: List[dict], metadata: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "rows": rows, "summary": _summarize(rows)}, f, indent=2, default=str)


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
        help=(
            "Comma-separated RQ1 conditions to run. Defaults to all eight: "
            "no_retrieval,all,only_api,only_context,only_similar_code,no_api,"
            "no_context,no_similar_code."
        ),
    )
    ap.add_argument("--out", default="results/rq1.json")
    ap.add_argument("--resume", action="store_true", help="Resume completed task-condition rows from --out.")
    ap.add_argument("--rerun-ids", default=None, help="Comma-separated task ids to replace as complete paired blocks.")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    cfg.llm_backend = os.environ.get("OPENCODER_LLM_BACKEND", cfg.llm_backend)
    cfg.llm_model = os.environ.get("OPENCODER_LLM_MODEL", cfg.llm_model)
    pipe = Pipeline(cfg)

    examples = list(load_dataset(args.dataset, args.dataset_path, limit=args.limit))
    # Conditions: all, single-source x3, leave-one-out x3, none.
    all_conditions = [
        ("no_retrieval", ()),
        ("all", SOURCES),
        *[(f"only_{s}", (s,)) for s in SOURCES],
        *[(f"no_{s}", tuple(x for x in SOURCES if x != s)) for s in SOURCES],
    ]
    requested = None
    if args.conditions:
        requested = {x.strip() for x in args.conditions.split(",") if x.strip()}
    conditions = [
        (name, enabled)
        for name, enabled in all_conditions
        if requested is None or name in requested
    ]
    if not conditions:
        known = ", ".join(name for name, _ in all_conditions)
        raise ValueError(f"No known RQ1 conditions selected. Known: {known}")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rq": "RQ1",
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "repo_root": args.repo_root,
        "limit": args.limit,
        "describe_limit": args.describe_limit,
        "backend": cfg.llm_backend,
        "model": cfg.llm_model,
        "base_url": os.environ.get("OPENCODER_LLM_BASE_URL"),
        "conditions": [
            name for name, _ in conditions
        ],
    }
    rows: List[dict] = []
    if args.resume and os.path.exists(args.out):
        try:
            rows = list(json.load(open(args.out)).get("rows") or [])
            print(f"Resuming {len(rows)} existing RQ1 rows from {args.out}", flush=True)
        except Exception as exc:
            print(f"Could not resume {args.out}: {exc}; starting a new result.", flush=True)
            rows = []
    rerun_ids = {value.strip() for value in (args.rerun_ids or "").split(",") if value.strip()}
    if rerun_ids:
        rows = [row for row in rows if row.get("example_id") not in rerun_ids]
        print(f"Replacing paired RQ1 blocks for: {', '.join(sorted(rerun_ids))}", flush=True)
    completed = {
        (row.get("example_id"), row.get("condition"))
        for row in rows if "error" not in row
    }
    for ex in examples:
        missing_conditions = [
            (name, enabled) for name, enabled in conditions
            if (ex.id, name) not in completed
        ]
        if not missing_conditions:
            print(f"  {ex.id:<24} all requested conditions already complete", flush=True)
            continue
        try:
            _, retrievers = pipe.index_example(
                ex,
                fallback_repo_root=args.repo_root,
                describe_limit=args.describe_limit,
            )
            prior = next(
                (row for row in rows if row.get("example_id") == ex.id and "error" not in row),
                None,
            )
            prepared_query = _plan_from_row(prior) if prior else None
            if prepared_query is None:
                prepared_query = pipe.prepare_query(ex)
        except Exception as e:
            for name, enabled in missing_conditions:
                rows.append({
                    "example_id": ex.id,
                    "condition": name,
                    "enabled": list(enabled),
                    "error": f"prepare: {e}",
                })
            print(f"  {ex.id:<24} prepare ERROR: {e}", flush=True)
            _write_payload(args.out, rows, metadata)
            continue
        for name, enabled in missing_conditions:
            try:
                r = _run_with(pipe, ex, retrievers, enabled, prepared_query=prepared_query)
                rows = [
                    row for row in rows
                    if (row.get("example_id"), row.get("condition")) != (ex.id, name)
                ]
                rows.append({
                    "example_id": ex.id,
                    "condition": name,
                    "enabled": list(enabled),
                    "u": r.uncertainty_trace,
                    "passed": r.test_report.get("passed"),
                    "correctness_mode": r.correctness_mode,
                    "source_diagnostics": r.source_diagnostics,
                    "per_step": r.per_step,
                    "sample_correctness": r.sample_correctness,
                    "pass_at_k": r.pass_at_k,
                    "pass_rate_variance": r.pass_rate_variance,
                })
                print(
                    f"  {ex.id:<24} {name:<18} "
                    f"u={r.uncertainty_trace['aggregate']:.3f} "
                    f"pass={r.test_report.get('passed')}",
                    flush=True,
                )
            except Exception as e:
                rows = [
                    row for row in rows
                    if (row.get("example_id"), row.get("condition")) != (ex.id, name)
                ]
                rows.append({"example_id": ex.id, "condition": name, "error": str(e)})
                print(f"  {ex.id:<24} {name:<18} ERROR: {e}", flush=True)
            _write_payload(args.out, rows, metadata)

    summary = _summarize(rows)
    _write_payload(args.out, rows, metadata)
    print(f"\nRQ1 summary:\n{json.dumps(summary, indent=2)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
