#!/usr/bin/env python3
"""Score submitted human or dry-run functions with frozen repository tests."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.phase5_verify.test_validate import run_execrepobench_function_tests  # noqa: E402


STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def code_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def error_category(returncode: int, stderr: str) -> str:
    lowered = stderr.lower()
    if returncode == 0:
        return "none"
    if returncode in {2, -1} or "missing official repository evaluator" in lowered or "timeout" in lowered:
        return "infrastructure_error"
    if "syntax" in lowered or "parse" in lowered or "indent" in lowered:
        return "syntax_or_normalization_failure"
    if "assert" in lowered or "failed" in lowered:
        return "test_failure"
    return "runtime_or_test_failure"


def edit_lines(before: str, after: str) -> int:
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--allow-simulated", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    responses = read_jsonl(args.responses)
    if not responses:
        raise RuntimeError("no episode records found")
    modes = {str(row.get("study_mode")) for row in responses}
    if modes == {"EMPIRICAL"}:
        default_out = ROOT / "results/human_study/scored_human_episodes.csv"
    elif modes == {"SIMULATED_DRY_RUN"} and args.allow_simulated:
        default_out = STUDY / "dry_run/scored_human_episodes.csv"
    else:
        raise RuntimeError(
            "responses must contain one study mode; simulated data require --allow-simulated"
        )
    out = args.out or default_out
    empirical_root = (ROOT / "results/human_study").resolve()
    if "SIMULATED_DRY_RUN" in modes and out.resolve().is_relative_to(empirical_root):
        raise RuntimeError("simulated records cannot be written to empirical results")

    public = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_public.jsonl")}
    private = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_private.jsonl")}
    manifest = {row["task_id"]: row for row in read_jsonl(MANIFEST)}
    keys = [(row["participant_code"], int(row["episode_index"])) for row in responses]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate participant-episode records")

    def score(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task_id = str(row["task_id"])
        if task_id not in public or task_id not in private or task_id not in manifest:
            raise RuntimeError(f"unknown frozen task: {task_id}")
        stimulus = public[task_id]
        if str(row.get("starting_code_sha256")) != str(stimulus["starting_code_sha256"]):
            raise RuntimeError(f"starting-code hash mismatch: {row['participant_code']}:{task_id}")
        final_code = str(row.get("final_code") or "")
        report = run_execrepobench_function_tests(final_code, manifest[task_id], timeout=args.timeout)
        category = error_category(report.returncode, report.stderr)
        evaluator_ok = category != "infrastructure_error"
        final_correct: bool | None = bool(report.passed) if evaluator_ok else None
        initial_correct = bool(private[task_id]["initial_correct"])
        changed = code_sha(final_code) != stimulus["starting_code_sha256"]
        judgment = bool_text(row["starting_judgment_correct"])
        output = {
            "study_mode": row["study_mode"],
            "participant_code": row["participant_code"],
            "episode_index": int(row["episode_index"]),
            "task_id": task_id,
            "condition": row["condition"],
            "assignment_group": int(row["assignment_group"]),
            "signal_category": private[task_id]["signal_category"],
            "initial_correct": initial_correct,
            "starting_judgment_correct": judgment,
            "failure_detection_accurate": judgment == initial_correct,
            "starting_confidence": int(row["starting_confidence"]),
            "final_confidence": int(row["final_confidence"]),
            "final_correct": "" if final_correct is None else final_correct,
            "evaluator_status": "ok" if evaluator_ok else "error",
            "failure_category": category,
            "code_changed": changed,
            "changed_lines": edit_lines(str(stimulus["starting_code"]), final_code),
            "repair_opportunity": not initial_correct,
            "repair_success": bool(not initial_correct and final_correct is True),
            "accepted_incorrect": bool(not initial_correct and not changed and final_correct is False),
            "unnecessary_edit": bool(initial_correct and changed),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "timed_out": float(row["elapsed_seconds"]) >= 360.0,
            "difficulty": int(row["difficulty"]),
            "guidance_usefulness": "" if row.get("guidance_usefulness") is None else int(row["guidance_usefulness"]),
            "action_category": row["action_category"],
            "starting_code_sha256": stimulus["starting_code_sha256"],
            "final_code_sha256": code_sha(final_code),
        }
        return index, output

    scored: list[dict[str, Any] | None] = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(score, index, row) for index, row in enumerate(responses)]
        for future in as_completed(futures):
            index, row = future.result()
            scored[index] = row
    rows = [row for row in scored if row is not None]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = sum(row["evaluator_status"] == "error" for row in rows)
    summary = {
        "study_mode": next(iter(modes)),
        "episodes": len(rows),
        "evaluator_errors": errors,
        "output": str(out.relative_to(ROOT)),
        "paper_eligible": next(iter(modes)) == "EMPIRICAL" and errors == 0,
    }
    summary_path = out.with_suffix(".integrity.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
