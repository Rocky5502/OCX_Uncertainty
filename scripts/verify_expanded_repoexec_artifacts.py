"""Reconstruct expanded RepoExec tables from the published raw JSONL records."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_expanded_repoexec_results import (  # noqa: E402
    METHODS,
    _paired,
    _resources,
    _summaries,
    _task_scores,
)


SECRET_OR_PATH = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|"
    r"/" r"(?:Users|home)/[^/\s]+|/private/var/folders/)"
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _same_value(expected: Any, actual: str) -> bool:
    if expected is None:
        return actual == ""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        except ValueError:
            return False
    return str(expected) == actual


def _compare_csv(
    path: Path,
    expected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    actual = _read_csv(path)
    if len(actual) != len(expected):
        return {
            "passed": False,
            "reason": f"row count {len(actual)} != {len(expected)}",
        }
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected, actual),
        start=2,
    ):
        if list(expected_row) != list(actual_row):
            return {
                "passed": False,
                "reason": f"column mismatch at CSV row {row_index}",
            }
        for column, expected_value in expected_row.items():
            if not _same_value(expected_value, actual_row[column]):
                return {
                    "passed": False,
                    "reason": (
                        f"row {row_index}, {column}: "
                        f"{actual_row[column]!r} != {expected_value!r}"
                    ),
                }
    return {"passed": True, "rows": len(actual)}


def _analytical_record(record: Dict[str, Any]) -> Dict[str, Any]:
    method_keys = {label: key for key, label in METHODS}
    outcomes = [
        bool(value)
        for value in record["effective_candidate_test_outcomes"]
    ]
    scores = _task_scores(outcomes)
    tokens = record.get("tokens") or {}
    latency = record.get("latency_s") or {}
    telemetry = record.get("api_telemetry") or {}
    return {
        "cohort": record["cohort"],
        "backend": record["backend"],
        "model": record.get("model"),
        "method_key": method_keys[record["method"]],
        "method": record["method"],
        "task_id": record["task_id"],
        "selected_correct": bool(record["selected_output_correct"]),
        **scores,
        "tokens": tokens.get("total"),
        "latency_s": latency.get("run"),
        "repair_rounds": record.get("repair_rounds"),
        "requests": telemetry.get("requests"),
        "retries": (
            telemetry.get("retries")
            if telemetry.get("retry_telemetry_available")
            else None
        ),
        "prompt_tokens": tokens.get("prompt"),
        "completion_tokens": tokens.get("completion"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results/rq3_expanded_repoexec",
    )
    args = parser.parse_args()
    out_dir = Path(args.results_dir)
    raw_path = out_dir / "raw_results.jsonl"
    raw_text = raw_path.read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in raw_text.splitlines()
        if line.strip()
    ]

    keys = [
        (row["task_id"], row["cohort"], row["backend"], row["method"])
        for row in records
    ]
    methods = {label for _key, label in METHODS}
    task_sets_identical = True
    for cohort in ("original", "new"):
        for backend in ("GPT", "Gemini"):
            sets = [
                {
                    row["task_id"]
                    for row in records
                    if row["cohort"] == cohort
                    and row["backend"] == backend
                    and row["method"] == method
                }
                for method in methods
            ]
            task_sets_identical &= all(item == sets[0] for item in sets[1:])

    analytical = [_analytical_record(record) for record in records]
    passk, selected = _paired(analytical)
    checks: Dict[str, Any] = {
        "raw_record_count": {
            "passed": len(records) == 192,
            "observed": len(records),
        },
        "unique_task_backend_method_records": {
            "passed": len(keys) == len(set(keys)),
        },
        "five_candidates_per_record": {
            "passed": all(
                len(row.get("generated_candidates") or []) == 5
                for row in records
            ),
        },
        "five_raw_outcomes_per_record": {
            "passed": all(
                len(row.get("candidate_test_outcomes") or []) == 5
                for row in records
            ),
        },
        "five_effective_outcomes_per_record": {
            "passed": all(
                len(row.get("effective_candidate_test_outcomes") or []) == 5
                for row in records
            ),
        },
        "identical_method_task_sets": {"passed": task_sets_identical},
        "no_missing_selected_results": {
            "passed": all(
                row.get("selected_output_correct") is not None
                and row.get("selected_output_test_result") is not None
                for row in records
            ),
        },
        "no_recorded_failures": {
            "passed": all(not row.get("failure") for row in records),
        },
        "all_execution_integrity_checks_pass": {
            "passed": all(
                (row.get("generation_integrity") or {}).get("valid") is True
                for row in records
            ),
        },
        "no_credentials_or_identifying_paths": {
            "passed": SECRET_OR_PATH.search(raw_text) is None,
        },
        "summary_reproduced": _compare_csv(
            out_dir / "summary.csv",
            _summaries(analytical),
        ),
        "selected_statistics_reproduced": _compare_csv(
            out_dir / "selected_output_statistics.csv",
            selected,
        ),
        "paired_passk_reproduced": _compare_csv(
            out_dir / "paired_passk_statistics.csv",
            passk,
        ),
        "resources_reproduced": _compare_csv(
            out_dir / "resource_summary.csv",
            _resources(analytical),
        ),
    }
    passed = all(check["passed"] for check in checks.values())
    payload = {
        "passed": passed,
        "source": "raw_results.jsonl",
        "checks": checks,
    }
    verification_path = out_dir / "artifact_verification.json"
    verification_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
