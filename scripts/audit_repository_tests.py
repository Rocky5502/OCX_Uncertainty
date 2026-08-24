"""Audit repository-test reconstruction for ExecRepoBench-style records."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import load_dataset  # noqa: E402
from opencoder.phase5_verify.test_validate import run_repo_completion_tests  # noqa: E402


def _label_to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"true", "pass", "passed", "1"}:
            return True
        if text in {"false", "fail", "failed", "0"}:
            return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out-json", default="results/repository_test_harness_audit.json")
    ap.add_argument("--out-csv", default="results/repository_test_harness_audit.csv")
    args = ap.parse_args()

    rows = []
    for ex in load_dataset(args.dataset, args.dataset_path, limit=args.limit):
        raw = ex.raw or {}
        reference_report = run_repo_completion_tests(
            ex.reference_code or "",
            raw,
            timeout=args.timeout,
        )
        generated = raw.get("generated_middle_code")
        generated_report = None
        if isinstance(generated, str) and generated.strip():
            generated_report = run_repo_completion_tests(
                generated,
                raw,
                timeout=args.timeout,
            )
        expected_generated = _label_to_bool(raw.get("is_pass"))
        generated_passed = None if generated_report is None else generated_report.passed
        row = {
            "id": ex.id,
            "reference_passed": reference_report.passed,
            "reference_returncode": reference_report.returncode,
            "generated_available": bool(isinstance(generated, str) and generated.strip()),
            "generated_passed": generated_passed,
            "expected_generated_passed": expected_generated,
            "generated_label_matches": (
                None
                if generated_passed is None or expected_generated is None
                else generated_passed == expected_generated
            ),
            "reference_stdout_head": reference_report.stdout[:300],
            "reference_stderr_head": reference_report.stderr[:300],
            "generated_stdout_head": "" if generated_report is None else generated_report.stdout[:300],
            "generated_stderr_head": "" if generated_report is None else generated_report.stderr[:300],
        }
        rows.append(row)
        print(
            f"{ex.id:<24} ref={reference_report.passed} "
            f"generated={row['generated_passed']} expected={expected_generated} "
            f"match={row['generated_label_matches']}",
            flush=True,
        )

    known_reference = [r for r in rows if r["reference_passed"] is not None]
    known_generated = [r for r in rows if r["generated_label_matches"] is not None]
    summary = {
        "dataset": args.dataset,
        "limit": args.limit,
        "n": len(rows),
        "n_reference_checked": len(known_reference),
        "reference_pass_rate": (
            sum(r["reference_passed"] is True for r in known_reference) / len(known_reference)
            if known_reference else 0.0
        ),
        "generated_label_match_rate": (
            sum(r["generated_label_matches"] is True for r in known_generated)
            / len(known_generated)
            if known_generated else 0.0
        ),
        "n_generated_checked": len(known_generated),
    }
    payload = {"summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.out_json} and {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
