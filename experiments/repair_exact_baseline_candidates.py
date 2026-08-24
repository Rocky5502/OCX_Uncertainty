"""Repair failed Baseline RAG candidates without regenerating them.

This completes the RAG + Verify/Repair ablation for tasks on which none of the
five stored Baseline RAG candidates passes. It preserves the exact candidate
stream and retrieval evidence, then applies only OpenCoder's Phase-V repair
loop.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_rq3 import (  # noqa: E402
    _load_dotenv,
    _pipeline_config_for,
    _summarize,
    _upsert_row,
)
from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.evaluation.metrics import (  # noqa: E402
    pass_at_ks_from_samples,
    pass_rate_variance,
)
from opencoder.phase5_verify.repair import repair_code  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _add_usage(
    baseline: Dict[str, Any],
    repair_usage: Dict[str, int],
) -> Dict[str, int]:
    baseline_usage = baseline.get("llm_usage") or {}
    return {
        key: int(baseline_usage.get(key) or 0) + int(repair_usage.get(key) or 0)
        for key in (
            "requests",
            "retries",
            "failed_attempts",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }


def _is_completed_exact_repair(
    baseline: Dict[str, Any],
    existing: Dict[str, Any] | None,
) -> bool:
    if not existing or existing.get("error"):
        return False
    provenance = existing.get("rag_verify_repair_provenance") or {}
    return (
        provenance.get("candidate_source")
        == "exact_matched_baseline_rag_candidates"
        and provenance.get("repair_required") is True
        and existing.get("generated_samples") == baseline.get("generated_samples")
        and int(existing.get("repair_rounds") or 0) <= 2
    )


def _repair_row(
    baseline: Dict[str, Any],
    example: Any,
    pipeline: Pipeline,
) -> Dict[str, Any]:
    samples = list(baseline.get("generated_samples") or [])
    raw_correctness = [
        bool(value) for value in baseline.get("sample_correctness") or []
    ]
    expected = pipeline.cfg.n_samples_for_uncertainty
    if len(samples) != expected or len(raw_correctness) != expected:
        raise ValueError(
            f"{example.id}: expected {expected} candidates and correctness labels; "
            f"found {len(samples)} and {len(raw_correctness)}"
        )

    verification_start = time.perf_counter()
    selection_outcomes: List[bool] = []
    for sample in samples:
        candidate = pipeline._normalize_completion_code(sample, example)
        _, report = pipeline._validate_code(candidate, example)
        selection_outcomes.append(report.passed is True)
    if any(selection_outcomes):
        raise ValueError(
            f"{example.id}: a stored candidate already passes; derive verified "
            "selection locally instead of invoking repair"
        )

    code = pipeline._normalize_completion_code(
        str(baseline.get("code") or samples[0]),
        example,
    )
    static_report, test_report = pipeline._validate_code(code, example)
    initial_test_report = dict(test_report.__dict__)
    post_selection_test_report = dict(test_report.__dict__)
    verification_latency_s = time.perf_counter() - verification_start

    usage_before = pipeline.llm.usage_snapshot()
    repair_start = time.perf_counter()
    rounds = 0
    repair_history: List[Dict[str, Any]] = []
    repairable = pipeline._should_repair(example, static_report, test_report)
    while (
        test_report.passed is False
        and repairable
        and rounds < pipeline.cfg.max_repair_rounds
    ):
        before_report = dict(test_report.__dict__)
        diagnostics = (
            f"static: {static_report.__dict__}\n"
            f"tests:\n{test_report.stderr}\n{test_report.stdout}"
        )
        test_context = pipeline._repair_test_context(example)
        if test_context:
            diagnostics += f"\n\n# Test Context For Repair\n{test_context}"
        code = repair_code(
            code,
            diagnostics,
            pipeline.llm,
            task=example.query,
            completion_mode=pipeline._is_completion_record(example),
            expected_indent=pipeline._expected_completion_indent(example),
        )
        code = pipeline._normalize_completion_code(code, example)
        static_report, test_report = pipeline._validate_code(code, example)
        repairable = pipeline._should_repair(example, static_report, test_report)
        rounds += 1
        repair_history.append({
            "round": rounds,
            "input_test_report": before_report,
            "output_static_report": dict(static_report.__dict__),
            "output_test_report": dict(test_report.__dict__),
        })
    repair_latency_s = time.perf_counter() - repair_start
    usage_after = pipeline.llm.usage_snapshot()
    repair_usage = {
        key: usage_after[key] - usage_before[key]
        for key in usage_after
    }

    effective_correctness = [
        test_report.passed is True,
        *raw_correctness[1:],
    ]
    row = copy.deepcopy(baseline)
    row.update({
        "code": code,
        "passed": test_report.passed,
        "static_report": dict(static_report.__dict__),
        "test_report": dict(test_report.__dict__),
        "initial_test_report": initial_test_report,
        "post_selection_test_report": post_selection_test_report,
        "verified_selection_applied": False,
        "selected_sample_index": None,
        "selection_candidate_correctness": selection_outcomes,
        "sample_correctness": raw_correctness,
        "effective_sample_correctness": effective_correctness,
        "pass_at_k": dict(
            pass_at_ks_from_samples([effective_correctness], ks=(1, 3, 5))
        ),
        "pass_rate_variance": pass_rate_variance([effective_correctness]),
        "repair_rounds": rounds,
        "repair_history": repair_history,
        "verification_latency_s": verification_latency_s,
        "repair_latency_s": repair_latency_s,
        "run_latency_s": (
            float(baseline.get("run_latency_s") or 0.0)
            + verification_latency_s
            + repair_latency_s
        ),
        "llm_usage": _add_usage(baseline, repair_usage),
        "rag_verify_repair_provenance": {
            "candidate_source": "exact_matched_baseline_rag_candidates",
            "verification": "local_executable_tests",
            "repair": "phase_v_test_guided_repair_only",
            "repair_required": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    return row


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = Path(args.run)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_config = PipelineConfig.from_yaml(args.config)
    pipeline = Pipeline(_pipeline_config_for(base_config, "rag_repair", False))
    examples = {
        str(example.id): example
        for example in load_dataset(
            args.benchmark,
            args.benchmark_path,
            limit=args.limit,
        )
    }
    requested = set(args.task_id)
    baseline_rows = {
        str(row.get("id")): row
        for row in payload.get("without") or []
    }
    repair_rows = list(payload.get("rag_repair") or [])
    existing_by_id = {
        str(row.get("id")): row
        for row in repair_rows
    }

    completed: List[Dict[str, Any]] = []
    for task_id, baseline in baseline_rows.items():
        if requested and task_id not in requested:
            continue
        example = examples.get(task_id)
        if example is None:
            raise KeyError(f"{task_id}: task missing from manifest")
        if any(bool(value) for value in baseline.get("sample_correctness") or []):
            continue
        if (
            not args.force
            and _is_completed_exact_repair(baseline, existing_by_id.get(task_id))
        ):
            print(f"{task_id}: skipped (exact repair already complete)", flush=True)
            continue
        row = _repair_row(baseline, example, pipeline)
        _upsert_row(repair_rows, row)
        existing_by_id[task_id] = row
        completed.append({
            "id": task_id,
            "passed": row.get("passed"),
            "repair_rounds": row.get("repair_rounds"),
            "repair_requests": (
                int((row.get("llm_usage") or {}).get("requests") or 0)
                - int((baseline.get("llm_usage") or {}).get("requests") or 0)
            ),
        })
        payload["rag_repair"] = repair_rows
        payload["summary"] = _summarize(payload)
        _atomic_write(path, payload)
        print(
            f"{task_id}: pass={row.get('passed')} "
            f"rounds={row.get('repair_rounds')}",
            flush=True,
        )

    baseline_order = [str(row.get("id")) for row in payload.get("without") or []]
    by_id = {str(row.get("id")): row for row in repair_rows}
    payload["rag_repair"] = [
        by_id[task_id] for task_id in baseline_order if task_id in by_id
    ]
    payload["summary"] = _summarize(payload)
    exact_repair_ids = [
        task_id
        for task_id, baseline in baseline_rows.items()
        if _is_completed_exact_repair(baseline, by_id.get(task_id))
    ]
    metadata = payload.setdefault("metadata", {})
    metadata["rag_verify_repair_exact_candidate_policy"] = {
        "candidate_source": "exact stored Baseline RAG candidates",
        "repair_only": True,
        "completed_task_ids": exact_repair_ids,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(path, payload)
    print(json.dumps({"run": os.fspath(path), "completed": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
