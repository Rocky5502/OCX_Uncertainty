"""Derive the verification-only portion of RAG + Verify/Repair locally.

The matched ablation should reuse Baseline RAG's exact five candidates. This
script applies the same verified-sample selection used by OpenCoder and writes
completed rows only when one of those candidates passes. Tasks for which no
candidate passes are left pending for genuine model-based repair.
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
    _pipeline_config_for,
    _summarize,
    _upsert_row,
)
from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.evaluation.metrics import (  # noqa: E402
    pass_at_ks_from_samples,
    pass_rate_variance,
)
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _derive_row(
    baseline: Dict[str, Any],
    example: Any,
    pipeline: Pipeline,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    verification_start = time.perf_counter()
    samples = list(baseline.get("generated_samples") or [])
    raw_correctness = [bool(value) for value in baseline.get("sample_correctness") or []]
    if len(samples) != pipeline.cfg.n_samples_for_uncertainty:
        return None, {
            "id": example.id,
            "reason": "candidate_count_mismatch",
            "candidate_count": len(samples),
        }
    if len(raw_correctness) != len(samples):
        return None, {
            "id": example.id,
            "reason": "correctness_count_mismatch",
            "candidate_count": len(samples),
            "correctness_count": len(raw_correctness),
        }

    selection_outcomes: List[bool] = []
    selected = None
    for index, sample in enumerate(samples):
        code = pipeline._normalize_completion_code(sample, example)
        static_report, test_report = pipeline._validate_code(code, example)
        passed = test_report.passed is True
        selection_outcomes.append(passed)
        if selected is None and passed:
            selected = (index, code, static_report, test_report)

    if selected is None:
        return None, {
            "id": example.id,
            "reason": "repair_required",
            "selection_candidate_correctness": selection_outcomes,
        }

    selected_index, code, static_report, test_report = selected
    verification_latency_s = time.perf_counter() - verification_start
    effective_correctness = [True, *raw_correctness[1:]]
    row = copy.deepcopy(baseline)
    row.update({
        "code": code,
        "passed": True,
        "static_report": dict(static_report.__dict__),
        "test_report": dict(test_report.__dict__),
        "initial_test_report": copy.deepcopy(baseline.get("test_report") or {}),
        "post_selection_test_report": dict(test_report.__dict__),
        "verified_selection_applied": True,
        "selected_sample_index": selected_index,
        "selection_candidate_correctness": selection_outcomes,
        "sample_correctness": raw_correctness,
        "effective_sample_correctness": effective_correctness,
        "pass_at_k": dict(
            pass_at_ks_from_samples([effective_correctness], ks=(1, 3, 5))
        ),
        "pass_rate_variance": pass_rate_variance([effective_correctness]),
        "repair_rounds": 0,
        "repair_history": [],
        "verification_latency_s": verification_latency_s,
        "run_latency_s": (
            float(baseline.get("run_latency_s") or 0.0)
            + verification_latency_s
        ),
        "rag_verify_repair_provenance": {
            "candidate_source": "exact_matched_baseline_rag_candidates",
            "verification": "local_executable_tests",
            "repair_required": False,
            "derived_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    return row, {
        "id": example.id,
        "reason": "verified_candidate_selected",
        "selected_sample_index": selected_index,
        "selection_candidate_correctness": selection_outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    path = Path(args.run)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = PipelineConfig.from_yaml(args.config)
    config = _pipeline_config_for(base, "rag_repair", False)
    pipeline = Pipeline(config)
    examples = {
        example.id: example
        for example in load_dataset(
            args.benchmark,
            args.benchmark_path,
            limit=args.limit,
        )
    }

    baseline_rows = list(payload.get("without") or [])
    existing_rows = list(payload.get("rag_repair") or [])
    audit: List[Dict[str, Any]] = []
    derived_count = 0
    for baseline in baseline_rows:
        task_id = str(baseline.get("id") or "")
        example = examples.get(task_id)
        if example is None:
            audit.append({"id": task_id, "reason": "task_missing_from_manifest"})
            continue
        row, status = _derive_row(baseline, example, pipeline)
        audit.append(status)
        if row is not None:
            _upsert_row(existing_rows, row)
            derived_count += 1

    pending = [
        item["id"]
        for item in audit
        if item["reason"] == "repair_required"
    ]
    report = {
        "run": os.fspath(path),
        "baseline_rows": len(baseline_rows),
        "derived_rows": derived_count,
        "pending_repair_rows": len(pending),
        "pending_repair_task_ids": pending,
        "audit": audit,
    }
    print(json.dumps(report, indent=2))

    if args.apply:
        payload["rag_repair"] = existing_rows
        payload["summary"] = _summarize(payload)
        metadata = payload.setdefault("metadata", {})
        metadata["rag_verify_repair_derivation"] = {
            "policy": (
                "reuse exact Baseline RAG candidates; apply local verified "
                "selection; leave candidate-set failures pending for model repair"
            ),
            "derived_rows": len(existing_rows),
            "pending_repair_task_ids": pending,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write(path, payload)
        audit_path = path.parent / "rag_verify_repair_derivation_audit.json"
        _atomic_write(audit_path, report)
        print(f"Updated {path}")
        print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
