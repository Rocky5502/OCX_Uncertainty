#!/usr/bin/env python3
"""Generate neutral synthetic records solely to test the analysis software."""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
OUT = STUDY / "dry_run"
SEED = 20260822


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    rng = random.Random(SEED)
    with (FROZEN / "human_randomization_schedule.csv").open(newline="", encoding="utf-8") as handle:
        schedule = list(csv.DictReader(handle))
    private = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_private.jsonl")}
    public = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_public.jsonl")}
    participant_codes = sorted({row["participant_code"] for row in schedule})
    role_cycle = ["software_engineer", "ai_ml_researcher", "phd_student"]
    roles = [role_cycle[index % len(role_cycle)] for index in range(len(participant_codes))]
    rng.shuffle(roles)
    OUT.mkdir(parents=True, exist_ok=True)
    participants = []
    for code, role in zip(participant_codes, roles):
        number = int(code[1:])
        programming = round(rng.uniform(1.5, 14.0), 1)
        python = round(rng.uniform(1.0, min(programming, 10.0)), 1)
        participants.append({
            "record_type": "participant", "study_mode": "SIMULATED_DRY_RUN",
            "participant_code": code, "assignment_group": 1 + (number - 1) % 3,
            "consent": True, "adult": True, "primary_role": role,
            "additional_roles": [], "programming_years": programming,
            "python_years": python, "ai_tool_usage": rng.randint(1, 5),
            "code_review_frequency": rng.randint(1, 5),
            "repository_development_frequency": rng.randint(1, 5),
            "testing_familiarity": rng.randint(2, 5),
            "dependency_tracing_familiarity": rng.randint(2, 5),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "withdrawn": False,
        })
    (OUT / "participants.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in participants), encoding="utf-8"
    )
    tutorials = [
        {
            "participant_code": row["participant_code"], "passed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "study_mode": "SIMULATED_DRY_RUN",
        }
        for row in participants
    ]
    (OUT / "tutorials.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tutorials),
        encoding="utf-8",
    )
    poststudy = [
        {
            "record_type": "poststudy", "study_mode": "SIMULATED_DRY_RUN",
            "participant_code": row["participant_code"],
            "mental_demand": rng.randint(5, 17), "effort": rng.randint(5, 17),
            "frustration": rng.randint(2, 15), "temporal_demand": rng.randint(4, 17),
            "most_useful": "SIMULATED_DRY_RUN",
            "misleading": "SIMULATED_DRY_RUN",
            "adoption_intent": rng.randint(2, 5),
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for row in participants
    ]
    (OUT / "poststudy.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in poststudy),
        encoding="utf-8",
    )

    episodes = []
    for row in schedule:
        truth = private[row["task_id"]]
        initial_correct = bool(truth["initial_correct"])
        # Condition-invariant probabilities intentionally avoid simulating a favorable effect.
        detection_accurate = rng.random() < 0.72
        judgment = initial_correct if detection_accurate else not initial_correct
        code_changed = rng.random() < (0.72 if not initial_correct else 0.22)
        if initial_correct:
            final_correct = rng.random() < (0.86 if not code_changed else 0.76)
        else:
            final_correct = rng.random() < (0.38 if code_changed else 0.03)
        elapsed = min(360.0, max(35.0, rng.gauss(205, 70)))
        episodes.append({
            "study_mode": "SIMULATED_DRY_RUN", "participant_code": row["participant_code"],
            "episode_index": int(row["episode_index"]), "task_id": row["task_id"],
            "condition": row["condition"], "assignment_group": int(row["assignment_group"]),
            "signal_category": truth["signal_category"], "initial_correct": initial_correct,
            "starting_judgment_correct": judgment,
            "failure_detection_accurate": detection_accurate,
            "starting_confidence": rng.randint(45, 95), "final_confidence": rng.randint(50, 98),
            "final_correct": final_correct, "evaluator_status": "ok", "failure_category": "none" if final_correct else "test_failure",
            "code_changed": code_changed, "changed_lines": rng.randint(0 if not code_changed else 1, 12),
            "repair_opportunity": not initial_correct,
            "repair_success": bool(not initial_correct and final_correct),
            "accepted_incorrect": bool(not initial_correct and not code_changed and not final_correct),
            "unnecessary_edit": bool(initial_correct and code_changed),
            "elapsed_seconds": elapsed, "timed_out": elapsed >= 360,
            "difficulty": rng.randint(2, 5),
            "guidance_usefulness": "" if row["condition"] == "generic_review" else rng.randint(2, 5),
            "action_category": "changed_logic" if code_changed else "accepted",
            "starting_code_sha256": public[row["task_id"]]["starting_code_sha256"],
            "final_code_sha256": "SIMULATED_DRY_RUN",
        })
    path = OUT / "scored_human_episodes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader(); writer.writerows(episodes)
    marker = {
        "status": "SIMULATED_DRY_RUN_NOT_EMPIRICAL",
        "seed": SEED, "participants": len(participants), "episodes": len(episodes),
        "warning": "These records are synthetic software-test fixtures and cannot be reported as human results."
    }
    (OUT / "SIMULATION_MARKER.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
