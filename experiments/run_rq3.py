"""Unified RQ3 runner.

This runner executes the currently implemented practical comparison:
Baseline RAG versus uncertainty-aware OpenCoder. Reference-paper baselines
such as RepoCoder, RLCoder, RepoFormer, RepoFuse, AllianceCoder, and ConAPI
are intentionally not simulated here; the command exits with an explicit error
if asked to run a method that is not implemented in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import Example, load_dataset  # noqa: E402
from opencoder.evaluation.metrics import (  # noqa: E402
    pass_at_k,
    pass_at_ks_from_samples,
    pass_rate_variance,
    uncertainty_calibration_ece,
)
from opencoder.llm.client import _load_dotenv  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402
from opencoderx.frozen_retrieval import build_frozen_retrievers  # noqa: E402


METHODS = {
    "direct": ("direct", False),
    "baseline_rag": ("without", False),
    "rag_verify_repair": ("rag_repair", False),
    "opencoder": ("with", True),
}
RESULT_KEYS = ("direct", "without", "rag_repair", "with")
UNCERTAINTY_FEATURES = (
    "uncertainty_decomposition",
    "uncertainty_filtering",
    "uncertainty_guided_generation",
    "uncertainty_verified_selection",
    "uncertainty_triggered_repair",
    "whole_task_retrieval_anchor",
    "source_balanced_fusion",
)

UNIMPLEMENTED_METHODS = {
    "repocoder",
    "rlcoder",
    "repoformer",
    "repofuse",
    "alliancecoder",
    "conapi",
}


def _parse_description_limit(value: str) -> int | None:
    if value.lower() in {"all", "none", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "description limit must be a positive integer or 'all'"
        )
    return parsed


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_from_config(data: Dict[str, Any]) -> Tuple[str, str | None]:
    dataset = data.get("dataset") or {}
    return str(dataset.get("name") or "execrepobench"), dataset.get("path")


def _out_from_config(data: Dict[str, Any], backend: str) -> str:
    exp = data.get("experiment") or {}
    out_dir = exp.get("output_dir") or f"results/rq3/runs/{backend}"
    return os.fspath(Path(out_dir) / "rq3.json")


def _summarize(out: dict) -> dict:
    summary = {}
    for key, rows in out.items():
        if key not in RESULT_KEYS:
            continue
        known = [r for r in rows if r.get("passed") is not None and "u" in r]
        passed = [bool(r["passed"]) for r in known]
        uncs = [r["u"]["aggregate"] for r in known]
        sample_sets = [
            r.get("effective_sample_correctness") or r["sample_correctness"]
            for r in rows
            if r.get("effective_sample_correctness") or r.get("sample_correctness")
        ]
        uncertainty_values = [r["u"]["aggregate"] for r in rows if "u" in r]
        repair_values = [r["repair_rounds"] for r in rows if "repair_rounds" in r]
        run_latencies = [r["run_latency_s"] for r in rows if "run_latency_s" in r]
        item = {
            "pass@1": pass_at_k(passed, k=1) if known else None,
            "pass_rate_variance": pass_rate_variance(sample_sets),
            "ece": uncertainty_calibration_ece(uncs, passed) if known else None,
            "mean_uncertainty": (
                sum(uncertainty_values) / len(uncertainty_values)
                if uncertainty_values
                else None
            ),
            "mean_repair_rounds": (
                sum(repair_values) / len(repair_values) if repair_values else None
            ),
            "mean_run_latency_s": (
                sum(run_latencies) / len(run_latencies) if run_latencies else None
            ),
            "n": len(rows),
            "n_known_correctness": len(known),
            "n_errors": sum(1 for r in rows if "error" in r),
        }
        if sample_sets:
            item.update(dict(pass_at_ks_from_samples(sample_sets, ks=(1, 3, 5))))
        summary[key] = item
    return summary


def _write_payload(path: str, out: dict, metadata: dict) -> None:
    payload = dict(out)
    payload["summary"] = _summarize(out)
    payload["metadata"] = metadata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _runtime_cost_gate(
    config_path: str | None,
    model: str,
    pipes: Iterable[Pipeline],
) -> dict[str, Any] | None:
    if not config_path:
        return None
    controls = _load_yaml(config_path).get("cost_controls") or {}
    pricing = (controls.get("gateway_pricing") or {}).get(model)
    if not pricing:
        raise RuntimeError(f"COST_GATE_UNPRICED_MODEL:{model}")
    currency = str(pricing.get("currency") or "USD").upper()
    limit = (controls.get("per_run_limits") or {}).get(currency)
    if limit is None:
        raise RuntimeError(f"COST_GATE_MISSING_LIMIT:{currency}")
    prompt_tokens = completion_tokens = requests = 0
    for pipe in pipes:
        usage = pipe.llm.usage_snapshot()
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        requests += usage["requests"]
    amount = (
        prompt_tokens * float(pricing["input_per_million"])
        + completion_tokens * float(pricing["output_per_million"])
    ) / 1_000_000
    return {
        "amount": amount,
        "currency": currency,
        "limit": float(limit),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests": requests,
        "allowed": amount <= float(limit),
    }


def _load_existing(path: str) -> dict:
    if not Path(path).exists():
        return {key: [] for key in RESULT_KEYS}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: list(data.get(key) or []) for key in RESULT_KEYS}


def _completed_ids(rows: Iterable[dict]) -> set[str]:
    return {
        str(row.get("id"))
        for row in rows
        if (
            row.get("id") is not None
            and "error" not in row
            and (
                not row.get("generation_integrity")
                or bool((row.get("generation_integrity") or {}).get("valid"))
            )
        )
    }


def _upsert_row(rows: List[dict], row: dict) -> None:
    row_id = str(row.get("id"))
    rows[:] = [item for item in rows if str(item.get("id")) != row_id]
    rows.append(row)


def _pipeline_config_for(
    base: PipelineConfig,
    key: str,
    uncertainty_aware: bool,
) -> PipelineConfig:
    """Construct one matched RQ3 condition without changing shared budgets."""
    overrides = {**base.__dict__, "uncertainty_aware": uncertainty_aware}
    if key == "direct":
        overrides.update({
            "enable_sources": (),
            "uncertainty_decomposition": False,
            "uncertainty_filtering": False,
            "uncertainty_guided_generation": False,
            "uncertainty_verified_selection": False,
            "uncertainty_triggered_repair": False,
            "whole_task_retrieval_anchor": False,
            "source_balanced_fusion": False,
        })
    elif key == "rag_repair":
        overrides.update({
            "uncertainty_decomposition": False,
            "uncertainty_filtering": False,
            "uncertainty_guided_generation": False,
            "uncertainty_verified_selection": True,
            "uncertainty_triggered_repair": True,
            "whole_task_retrieval_anchor": False,
            "source_balanced_fusion": False,
        })
    return PipelineConfig(**overrides)


def _feature_flags(config: PipelineConfig) -> Dict[str, bool]:
    return {
        name: config.feature_enabled(name)
        for name in UNCERTAINTY_FEATURES
    }


def _record_run(
    *,
    pipe: Pipeline,
    example: Example,
    retrievers: Dict[str, Any],
    index_latency_s: float,
) -> dict:
    run_start = time.perf_counter()
    usage_before = pipe.llm.usage_snapshot()
    response_audit_before = len(pipe.llm.response_audit_snapshot())
    result = pipe.run(example, retrievers)
    usage_after = pipe.llm.usage_snapshot()
    provider_response_audit = pipe.llm.response_audit_snapshot()[response_audit_before:]
    sample_correctness = list(result.sample_correctness)
    effective_sample_correctness = list(sample_correctness)
    uses_phase5_primary = (
        pipe.cfg.feature_enabled("uncertainty_verified_selection")
        or pipe.cfg.feature_enabled("uncertainty_triggered_repair")
    )
    if sample_correctness and result.test_report.get("passed") is not None and uses_phase5_primary:
        effective_sample_correctness = [
            bool(result.test_report.get("passed")),
            *sample_correctness[1:],
        ]
    return {
        "id": example.id,
        "code": result.code,
        "generated_samples": result.generated_samples,
        "generation_raw_responses": result.generation_raw_responses,
        "generation_response_metadata": result.generation_response_metadata,
        "generation_integrity": result.generation_integrity,
        "passed": result.test_report.get("passed"),
        "correctness_mode": result.correctness_mode,
        "static_report": result.static_report,
        "test_report": result.test_report,
        "initial_test_report": result.initial_test_report,
        "post_selection_test_report": result.post_selection_test_report,
        "verified_selection_applied": result.verified_selection_applied,
        "u": result.uncertainty_trace,
        "uncertainty_components": result.uncertainty_components,
        "per_step": result.per_step,
        "fused_evidence": result.fused_evidence,
        "fused_evidence_ids": result.fused_evidence_ids,
        "source_diagnostics": result.source_diagnostics,
        "sample_correctness": sample_correctness,
        "effective_sample_correctness": effective_sample_correctness,
        "pass_at_k": result.pass_at_k,
        "pass_rate_variance": result.pass_rate_variance,
        "repair_rounds": result.repair_rounds,
        "repair_history": result.repair_history,
        "index_latency_s": index_latency_s,
        "run_latency_s": time.perf_counter() - run_start,
        "llm_usage": {
            key: usage_after[key] - usage_before[key]
            for key in usage_after
        },
        "provider_response_audit": provider_response_audit,
    }


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--benchmark-path", default=None)
    ap.add_argument(
        "--method",
        default="paired",
        help=(
            "direct, opencoder, baseline_rag, rag_verify_repair, paired, all "
            "(legacy three-method comparison), or tosem_all (four core methods). "
            "Reference-paper methods fail explicitly."
        ),
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--describe-limit",
        type=_parse_description_limit,
        default=5,
        help="Number of APIs to describe, or 'all' for the complete index.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--frozen-index",
        default=None,
        help="Leakage-audited JSONL retrieval index shared by every method.",
    )
    ap.add_argument(
        "--cost-config",
        default=None,
        help="Campaign YAML used to enforce a live per-run token-cost cap.",
    )
    args = ap.parse_args()

    method = args.method.lower()
    if method in UNIMPLEMENTED_METHODS:
        raise SystemExit(
            f"Method {args.method!r} is not implemented in this repository. "
            "Install or implement it before reporting RQ3 numbers."
        )
    if method not in {"paired", "all", "tosem_all"} and method not in METHODS:
        raise SystemExit(f"Unknown method: {args.method}")

    cfg_data = _load_yaml(args.config)
    dataset_name, dataset_path = _dataset_from_config(cfg_data)
    dataset_name = args.benchmark or dataset_name
    dataset_path = args.benchmark_path or dataset_path

    base = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        base.llm_seed = args.seed

    out_path = args.out or _out_from_config(cfg_data, base.llm_backend)
    out = {key: [] for key in RESULT_KEYS} if args.force else _load_existing(out_path)
    condition_configs = {
        key: _pipeline_config_for(base, key, aware)
        for key, aware in (
            ("direct", False),
            ("without", False),
            ("rag_repair", False),
            ("with", True),
        )
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rq": "RQ3",
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "repo_root": args.repo_root,
        "target_implementation_excluded_from_retrieval": bool(args.repo_root),
        "limit": args.limit,
        "describe_limit": args.describe_limit,
        "backend": base.llm_backend,
        "model": base.llm_model,
        "temperature": base.llm_temperature,
        "reasoning_effort": base.llm_reasoning_effort,
        "thinking_budget": base.llm_thinking_budget,
        "n_samples_for_uncertainty": base.n_samples_for_uncertainty,
        "seed": base.llm_seed,
        "initial_candidate_budget": base.n_samples_for_uncertainty,
        "max_repair_rounds": base.max_repair_rounds,
        "retrieval_budget": {
            "api_top_k": base.api_top_k,
            "context_top_k": base.context_top_k,
            "similar_code_top_k": base.similar_code_top_k,
            "fused_top_k": base.fused_top_k,
        },
        "max_generation_tokens_per_candidate": base.llm_max_tokens,
        "request_timeout_s": base.llm_timeout,
        "integrity_policy": "length-limited candidates are scored as failures; unexplained truncations invalidate the task",
        "base_url": os.environ.get("OPENCODER_LLM_BASE_URL"),
        "frozen_retrieval_index": args.frozen_index,
        "cost_config": args.cost_config,
        "conditions": {
            "direct": "Direct generation from the native task prompt without retrieved evidence, verification, or repair.",
            "without": "Baseline RAG without uncertainty-aware decomposition, filtering, generation, or repair.",
            "rag_repair": "Standard RAG with verified candidate selection and test-guided repair, but without uncertainty-aware decomposition, filtering, or generation.",
            "with": "OpenCoderX with uncertainty-aware decomposition, whole-task retrieval anchoring, source-balanced evidence fusion, generation, verification, and repair.",
        },
        "condition_feature_flags": {
            key: _feature_flags(config)
            for key, config in condition_configs.items()
        },
    }

    if method == "paired":
        run_specs = [("without", False), ("with", True)]
    elif method == "all":
        run_specs = [
            METHODS["baseline_rag"],
            METHODS["rag_verify_repair"],
            METHODS["opencoder"],
        ]
    elif method == "tosem_all":
        run_specs = [
            METHODS["direct"],
            METHODS["baseline_rag"],
            METHODS["rag_verify_repair"],
            METHODS["opencoder"],
        ]
    else:
        run_specs = [METHODS[method]]

    examples = list(load_dataset(dataset_name, dataset_path, limit=args.limit))
    index_pipe = Pipeline(PipelineConfig(**{**base.__dict__, "uncertainty_aware": True}))
    pipes = {
        key: Pipeline(condition_configs[key])
        for key, _ in run_specs
    }
    frozen_retriever_cache: Dict[str, tuple[list[Any], Dict[str, Any]]] = {}
    needs_retrieval = any(key != "direct" for key, _ in run_specs)

    for ex in examples:
        if all(ex.id in _completed_ids(out[key]) for key, _ in run_specs):
            print(f"  {ex.id:<24} skipped", flush=True)
            continue
        index_start = time.perf_counter()
        try:
            if not needs_retrieval:
                retrievers = {}
            elif args.frozen_index:
                repository = str((ex.raw or {}).get("repo_name") or "")
                if not repository:
                    raise ValueError("frozen retrieval requires repo_name in the task manifest")
                if repository not in frozen_retriever_cache:
                    frozen_retriever_cache[repository] = build_frozen_retrievers(
                        args.frozen_index,
                        repository,
                        index_pipe.encoder,
                    )
                _, retrievers = frozen_retriever_cache[repository]
            else:
                _, retrievers = index_pipe.index_example(
                    ex,
                    fallback_repo_root=args.repo_root,
                    describe_limit=args.describe_limit,
                )
            index_latency_s = (
                time.perf_counter() - index_start if needs_retrieval else 0.0
            )
        except Exception as exc:
            for key, _ in run_specs:
                if ex.id not in _completed_ids(out[key]):
                    _upsert_row(out[key], {"id": ex.id, "error": f"index_example: {exc}"})
            print(f"  {ex.id:<24} index ERROR: {exc}", flush=True)
            _write_payload(out_path, out, metadata)
            gate = _runtime_cost_gate(
                args.cost_config,
                base.llm_model,
                [index_pipe, *pipes.values()],
            )
            if gate is not None:
                metadata["runtime_cost_gate"] = gate
                _write_payload(out_path, out, metadata)
                if not gate["allowed"]:
                    print(json.dumps(gate, indent=2), flush=True)
                    print("COST_LIMIT_EXCEEDED: stopping before the next cell", flush=True)
                    return 3
            continue

        for key, _ in run_specs:
            if ex.id in _completed_ids(out[key]):
                continue
            try:
                row = _record_run(
                    pipe=pipes[key],
                    example=ex,
                    retrievers={} if key == "direct" else retrievers,
                    index_latency_s=0.0 if key == "direct" else index_latency_s,
                )
                _upsert_row(out[key], row)
                print(
                    f"  {ex.id:<24} {key:<8} "
                    f"u={row['u']['aggregate']:.3f} pass={row['passed']}",
                    flush=True,
                )
            except Exception as exc:
                _upsert_row(out[key], {"id": ex.id, "error": str(exc)})
                print(f"  {ex.id:<24} {key:<8} ERROR: {exc}", flush=True)
            _write_payload(out_path, out, metadata)
            gate = _runtime_cost_gate(
                args.cost_config,
                base.llm_model,
                [index_pipe, *pipes.values()],
            )
            if gate is not None:
                metadata["runtime_cost_gate"] = gate
                _write_payload(out_path, out, metadata)
                if not gate["allowed"]:
                    print(json.dumps(gate, indent=2), flush=True)
                    print("COST_LIMIT_EXCEEDED: stopping before the next cell", flush=True)
                    return 3

    _write_payload(out_path, out, metadata)
    print(json.dumps(_summarize(out), indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
