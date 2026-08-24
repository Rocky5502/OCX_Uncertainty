"""Recompute authoritative RQ4 task metrics without making API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_rq4 import (  # noqa: E402
    _record_to_metrics,
    aggregate_quality_for_table,
    summarize_metrics,
)


QUALITY_BENCHMARKS = {"RepoExec", "CoderEval"}
SOURCE_RUNS = {
    ("RepoExec", "GPT"): "results/rq3/runs/repoexec_inline14_gpt/rq3.json",
    ("RepoExec", "Gemini"): "results/rq3/runs/repoexec_inline14_gemini/rq3.json",
    ("CoderEval", "GPT"): "results/rq3/codereval_exec19/replication_gpt/rq3.json",
    ("CoderEval", "Gemini"): "results/rq3/codereval_exec19/replication_gemini/rq3.json",
    (
        "ExecRepoBench",
        "GPT",
    ): "results/rq3/runs/execrepobench_testbacked10_gpt/rq3.json",
    (
        "ExecRepoBench",
        "Gemini",
    ): "results/rq3/runs/execrepobench_testbacked10_gemini/rq3.json",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_saved_metrics(
    recomputed: list[dict[str, Any]],
    path: Path,
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if len(saved) != len(recomputed):
        raise AssertionError(
            f"saved/recomputed row count differs: {len(saved)} != "
            f"{len(recomputed)}"
        )
    fields = (
        "gt_count",
        "pred_count",
        "true_positive_count",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "exact_set_match",
    )
    for expected, actual in zip(recomputed, saved):
        key = (
            expected["benchmark"],
            expected["backend"],
            expected["method"],
            expected["task_id"],
        )
        actual_key = (
            actual["benchmark"],
            actual["backend"],
            actual["method"],
            actual["task_id"],
        )
        if key != actual_key:
            raise AssertionError(f"row ordering/key mismatch: {key} != {actual_key}")
        for field in fields:
            if abs(float(expected[field]) - float(actual[field])) > 1e-12:
                raise AssertionError(
                    f"{key}/{field}: {expected[field]} != {actual[field]}"
                )


def _assert_final_anchors(recomputed: list[dict[str, Any]]) -> None:
    _count, quality, _uncertainty, _success = summarize_metrics(recomputed)
    aggregated = aggregate_quality_for_table(quality)
    lookup = {
        (row["backend"], row["method"]): row
        for row in aggregated
    }
    expected = {
        ("GPT", "Baseline RAG"): (43.4, 0.0),
        ("GPT", "OpenCoder-NoAPIRefine"): (45.1, 3.8),
        ("GPT", "OpenCoder"): (64.8, 57.7),
        ("Gemini", "Baseline RAG"): (43.4, 0.0),
        ("Gemini", "OpenCoder-NoAPIRefine"): (39.9, 0.0),
        ("Gemini", "OpenCoder"): (59.1, 50.0),
    }
    for key, (f1, exact) in expected.items():
        row = lookup[key]
        observed = (
            round(100.0 * float(row["f1"]), 1),
            round(100.0 * float(row["exact_set_match"]), 1),
        )
        if observed != (f1, exact):
            raise AssertionError(
                f"authoritative anchor mismatch for {key}: "
                f"{observed} != {(f1, exact)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="results/rq4/raw_api_predictions.jsonl",
    )
    parser.add_argument(
        "--saved-metrics",
        default="results/rq4/per_task_api_metrics.csv",
    )
    parser.add_argument(
        "--out",
        default="results/rq4_reconciliation_by_task.csv",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    raw_records = _read_jsonl(raw_path)
    recomputed = [_record_to_metrics(record) for record in raw_records]
    _assert_saved_metrics(recomputed, Path(args.saved_metrics))
    _assert_final_anchors(recomputed)

    config_paths = {
        "GPT": "configs/rq3/gpt4o_mini.yaml",
        "Gemini": "configs/rq3/gemini_2_5_flash.yaml",
    }
    evaluator_hash = _sha256_file(ROOT / "experiments/run_rq4.py")
    campaign_hash = _sha256_file(raw_path)
    output_rows: list[dict[str, Any]] = []
    for record, metrics in zip(raw_records, recomputed):
        benchmark = str(record["benchmark"])
        backend = str(record["backend_label"])
        source_run = SOURCE_RUNS[(benchmark, backend)]
        source_run_path = ROOT / source_run
        source_metadata = json.loads(
            source_run_path.read_text(encoding="utf-8")
        ).get("metadata") or {}
        config_path = config_paths[backend]
        api_bearing = int(metrics["gt_count"]) > 0
        quality_included = benchmark in QUALITY_BENCHMARKS and api_bearing
        if quality_included:
            exclusion_reason = ""
        elif benchmark == "ExecRepoBench":
            exclusion_reason = (
                "no_resolvable_repository_api_in_selected_benchmark"
            )
        else:
            exclusion_reason = "empty_ground_truth_api_set"
        output_rows.append(
            {
                "campaign": "authoritative_corrected_rq4",
                "campaign_sha256": campaign_hash,
                "evaluator": "experiments/run_rq4.py",
                "evaluator_sha256": evaluator_hash,
                "created_at": record.get("created_at"),
                "backend": backend,
                "model": record.get("model"),
                "config_path": config_path,
                "config_sha256_at_audit": _sha256_file(ROOT / config_path),
                "config_sha256_at_run": "",
                "config_hash_status": "not_recorded_in_campaign",
                "source_run_file": source_run,
                "source_run_sha256": _sha256_file(source_run_path),
                "source_run_created_at": source_metadata.get("created_at"),
                "benchmark": benchmark,
                "task_id": record.get("task_id"),
                "method": record.get("method"),
                "prediction_stage": (
                    "post_target_aware_refinement"
                    if record.get("method") == "OpenCoder"
                    else (
                        "post_uncertainty_filter_pre_api_refinement"
                        if record.get("method") == "OpenCoder-NoAPIRefine"
                        else "method_final_api_set"
                    )
                ),
                "ground_truth_method": record.get("ground_truth_method"),
                "ground_truth_api_norms": ";".join(
                    record.get("ground_truth_api_norms") or []
                ),
                "predicted_final_apis": ";".join(
                    record.get("final_apis") or []
                ),
                "gt_count": metrics["gt_count"],
                "pred_count": metrics["pred_count"],
                "true_positive_count": metrics["true_positive_count"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_set_match": metrics["exact_set_match"],
                "api_bearing_task": api_bearing,
                "included_in_api_quality": quality_included,
                "exclusion_reason": exclusion_reason,
                "source_record_sha256": _record_hash(record),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"Wrote {len(output_rows)} independently recomputed task-method "
        f"records to {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
