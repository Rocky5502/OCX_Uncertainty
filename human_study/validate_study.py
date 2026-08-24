#!/usr/bin/env python3
"""Fail-closed validation for the OpenCoderX human-study package."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
CONFIG = json.loads((STUDY / "study_config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    issues: list[str] = []
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            issues.append(message)

    freeze_path = FROZEN / "protocol_freeze.json"
    require(freeze_path.is_file(), "missing protocol freeze")
    if not freeze_path.is_file():
        print(json.dumps({"passed": False, "issues": issues}, indent=2))
        return 1
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(freeze.get("status") == "TECHNICAL_FREEZE_PENDING_ETHICS", "unexpected study status")
    require(freeze.get("ethics_recruitment_allowed") is False, "technical package must not unlock recruitment")

    for relative, expected in {**freeze["source_file_sha256"], **freeze["output_file_sha256"]}.items():
        path = ROOT / relative
        require(path.is_file(), f"missing hashed file: {relative}")
        if path.is_file():
            require(sha256(path) == expected, f"hash mismatch: {relative}")

    public = read_jsonl(FROZEN / "stimuli_public.jsonl")
    private = read_jsonl(FROZEN / "stimuli_private.jsonl")
    require(len(public) == 12, "expected twelve public stimuli")
    require(len(private) == 12, "expected twelve private stimuli")
    require(len({row["task_id"] for row in public}) == 12, "public task IDs are not unique")
    require(len({row["repository"] for row in public}) == 12, "stimulus repositories are not unique")
    require(sum(not row["initial_correct"] for row in private) == 8, "expected eight incorrect cases")
    require(sum(row["initial_correct"] for row in private) == 4, "expected four correct controls")
    require(Counter(row["signal_category"] for row in private) == Counter({
        "api": 2, "context": 2, "similar_code": 2, "generation": 2, "correct_control": 4
    }), "signal-category balance mismatch")

    forbidden_fields = {
        "solution", "reference_code", "initial_correct", "passed", "test_report",
        "execution_prefix_code", "execution_suffix_code", "test_command"
    }
    for row in public:
        require(not (forbidden_fields & set(row)), f"private field exposed for {row['task_id']}")
        require(len(str(row["starting_code"])) > 0, f"empty starting code for {row['task_id']}")
        require(len(str(row["starting_code_sha256"])) == 64, f"invalid code hash for {row['task_id']}")
        require(set(row["source_risks"]) == {"api", "context", "similar_code", "generation"}, f"risk schema mismatch for {row['task_id']}")
        require(all(0.0 <= float(value) <= 1.0 for value in row["source_risks"].values()), f"risk out of range for {row['task_id']}")

    source = {row["task_id"]: row for row in read_jsonl(MANIFEST)}
    public_blob = (FROZEN / "stimuli_public.jsonl").read_text(encoding="utf-8")
    for row in public:
        solution = str(source[row["task_id"]].get("solution") or "").strip()
        require(not solution or solution not in public_blob, f"reference solution leaked for {row['task_id']}")

    schedule = read_csv(FROZEN / "human_randomization_schedule.csv")
    expected_invites = int(CONFIG["invitation_codes_planned"])
    expected_episodes = expected_invites * int(CONFIG["tasks_per_participant"])
    require(len(schedule) == expected_episodes, f"human schedule must contain {expected_episodes} episodes")
    by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    task_condition: Counter[tuple[str, str]] = Counter()
    for row in schedule:
        by_participant[row["participant_code"]].append(row)
        task_condition[(row["task_id"], row["condition"])] += 1
    require(len(by_participant) == expected_invites, f"human schedule must contain {expected_invites} invitation codes")
    for code, rows in by_participant.items():
        require(len(rows) == 12, f"{code} does not have twelve episodes")
        require(len({row["task_id"] for row in rows}) == 12, f"{code} sees a repeated task")
        require(Counter(row["condition"] for row in rows) == Counter({
            "generic_review": 4, "uncertainty_display": 4, "targeted_guidance": 4
        }), f"condition imbalance for {code}")
        require(sorted(int(row["episode_index"]) for row in rows) == list(range(1, 13)), f"episode order invalid for {code}")
    for task_id in {row["task_id"] for row in schedule}:
        counts = [task_condition[(task_id, condition)] for condition in (
            "generic_review", "uncertainty_display", "targeted_guidance"
        )]
        require(max(counts) - min(counts) <= 1, f"task-condition imbalance exceeds one for {task_id}")

    invitation_codes = read_csv(FROZEN / "invitation_codes.csv")
    expected_codes = {f"H{number:03d}" for number in range(1, expected_invites + 1)}
    require(len(invitation_codes) == expected_invites, "invitation-code file has the wrong row count")
    require(
        {row["participant_code"] for row in invitation_codes} == expected_codes,
        "invitation-code file does not match the configured pool",
    )
    require(set(by_participant) == expected_codes, "schedule and invitation-code pool differ")

    agents = read_csv(FROZEN / "agent_randomization_schedule.csv")
    require(len(agents) == 120, "agent schedule must contain 120 episodes")
    require(len({row["agent_session_id"] for row in agents}) == 10, "agent schedule must contain ten sessions")
    for code in {row["agent_session_id"] for row in agents}:
        rows = [row for row in agents if row["agent_session_id"] == code]
        require(len(rows) == 12, f"{code} does not have twelve episodes")
        require(Counter(row["condition"] for row in rows) == Counter({
            "generic_review": 4, "uncertainty_display": 4, "targeted_guidance": 4
        }), f"agent condition imbalance for {code}")

    for schema_name in ("participant", "episode", "agent_episode"):
        schema_path = STUDY / "schemas" / f"{schema_name}.schema.json"
        require(schema_path.is_file(), f"missing JSON schema: {schema_name}")
        if schema_path.is_file():
            try:
                Draft202012Validator.check_schema(
                    json.loads(schema_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                require(False, f"invalid JSON schema {schema_name}: {type(exc).__name__}")

    output = {
        "passed": not issues,
        "paper_eligible": False,
        "study_status": freeze["status"],
        "checks": checks,
        "issues": issues,
        "note": "Technical validation does not create or validate human evidence.",
    }
    (FROZEN / "integrity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
