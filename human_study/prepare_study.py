#!/usr/bin/env python3
"""Prepare and hash the OpenCoderX human-study stimuli and schedules."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
CONFIG_PATH = STUDY / "study_config.json"
MANIFEST_PATH = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
DECISIONS_PATH = ROOT / "results/tosem/collaboration_analysis/decision_records.jsonl"
RAW_ROOT = ROOT / "results/tosem/confirmatory/gpt4o_mini/opencoder"
FROZEN = STUDY / "frozen"


GUIDANCE = {
    "api": "Inspect repository API calls, signatures, return values, and documented edge-case semantics.",
    "context": "Inspect the surrounding repository file and cross-file dependencies for missing assumptions or required state.",
    "similar_code": "Check whether retrieved similar-code examples rely on incompatible types, versions, or repository conventions.",
    "generation": "Recheck control flow, boundary conditions, exceptions, and completeness; consider a structurally different implementation.",
    "correct_control": "The aggregate signal is comparatively low. Verify the implementation before editing and avoid unnecessary changes.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def clip(value: str, limit: int = 6000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    half = (limit - 46) // 2
    return value[:half] + "\n\n# ... study display clipped ...\n\n" + value[-half:]


def load_raw_records() -> tuple[dict[str, dict[str, Any]], list[Path]]:
    records: dict[str, dict[str, Any]] = {}
    files = sorted(RAW_ROOT.glob("batch_*.json"))
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("with", []):
            records[str(row["id"])] = row
    return records, files


def condition_schedule(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions = list(config["conditions"])
    task_ids = [
        task_id
        for source in ("api", "context", "similar_code", "generation", "correct_control")
        for task_id in config["source_task_ids"][source]
    ]
    human_rows: list[dict[str, Any]] = []
    for number in range(1, int(config["invitation_codes_planned"]) + 1):
        code = f"H{number:03d}"
        group = 1 + (number - 1) % int(config["assignment_groups"])
        assigned = [
            (task_id, conditions[(task_index + group - 1) % len(conditions)])
            for task_index, task_id in enumerate(task_ids)
        ]
        rng = random.Random(int(config["selection_seed"]) + number)
        rng.shuffle(assigned)
        for episode_index, (task_id, condition) in enumerate(assigned, start=1):
            human_rows.append({
                "participant_code": code,
                "assignment_group": group,
                "episode_index": episode_index,
                "task_id": task_id,
                "condition": condition,
            })

    agent_rows: list[dict[str, Any]] = []
    for number in range(1, 11):
        code = f"A{number:03d}"
        group = 1 + (number - 1) % int(config["assignment_groups"])
        assigned = [
            (task_id, conditions[(task_index + group - 1) % len(conditions)])
            for task_index, task_id in enumerate(task_ids)
        ]
        rng = random.Random(int(config["selection_seed"]) + 10_000 + number)
        rng.shuffle(assigned)
        for episode_index, (task_id, condition) in enumerate(assigned, start=1):
            agent_rows.append({
                "agent_session_id": code,
                "assignment_group": group,
                "episode_index": episode_index,
                "task_id": task_id,
                "condition": condition,
                "model": str(config["generator_model"]),
                "seed": int(config["selection_seed"]) + 20_000 + number,
            })
    return human_rows, agent_rows


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = {str(row["task_id"]): row for row in read_jsonl(MANIFEST_PATH)}
    raw, raw_files = load_raw_records()
    decisions = {
        str(row["task_id"]): row
        for row in read_jsonl(DECISIONS_PATH)
        if row.get("model") == config["generator_model"]
        and row.get("decision_policy") == "uncertainty_only"
    }

    ordered = [
        (source, task_id)
        for source in ("api", "context", "similar_code", "generation", "correct_control")
        for task_id in config["source_task_ids"][source]
    ]
    if len(ordered) != int(config["tasks_per_participant"]):
        raise RuntimeError("configured stimulus count does not match tasks_per_participant")
    if len({task_id for _, task_id in ordered}) != len(ordered):
        raise RuntimeError("duplicate task ID in configured stimulus set")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    repositories: set[str] = set()
    for index, (signal, task_id) in enumerate(ordered, start=1):
        if task_id not in manifest or task_id not in raw or task_id not in decisions:
            raise RuntimeError(f"missing source record for {task_id}")
        task = manifest[task_id]
        result = raw[task_id]
        decision = decisions[task_id]
        repository = str(task["repo_name"])
        if repository in repositories:
            raise RuntimeError(f"stimulus repositories must be unique: {repository}")
        repositories.add(repository)
        initial_correct = bool(result.get("passed"))
        expected_correct = signal == "correct_control"
        if initial_correct != expected_correct:
            raise RuntimeError(f"configured case/control outcome mismatch for {task_id}")
        source_risks = {
            key: float(decision["uncertainty_sources"][key])
            for key in ("api", "context", "similar_code", "generation")
        }
        if signal in {"api", "context", "similar_code"}:
            dominant_retrieval = max(("api", "context", "similar_code"), key=source_risks.get)
            if dominant_retrieval != signal:
                raise RuntimeError(f"retrieval-signal classification mismatch for {task_id}")
        if signal == "generation" and source_risks["generation"] <= max(
            source_risks[key] for key in ("api", "context", "similar_code")
        ):
            raise RuntimeError(f"generation signal is not dominant for {task_id}")

        context_blocks = []
        for item in task.get("context_code") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                context_blocks.append(f"# File: {item[0]}\n{item[1]}")
        evidence = clip(str(result.get("fused_evidence") or ""), 7000)
        public = {
            "stimulus_id": f"S{index:02d}",
            "task_id": task_id,
            "repository": repository,
            "file_name": str(task.get("file_name") or ""),
            "function_name": str(task.get("function_name") or ""),
            "task_text": str(task.get("target_function_prompt") or task.get("prompt") or ""),
            "repository_context": clip("\n\n".join(context_blocks), 5000),
            "retrieved_evidence": evidence,
            "starting_code": str(result.get("code") or ""),
            "starting_code_sha256": sha256_text(str(result.get("code") or "")),
            "aggregate_risk": float(decision["risk_score"]),
            "source_risks": source_risks,
            "signal_category": signal,
            "targeted_guidance": GUIDANCE[signal],
            "display_note": "Executable correctness is intentionally hidden until submission.",
        }
        private = {
            "task_id": task_id,
            "stimulus_id": public["stimulus_id"],
            "initial_correct": initial_correct,
            "signal_category": signal,
            "source_manifest_artifact_hash": str(task.get("artifact_hash") or ""),
            "source_raw_record_sha256": sha256_text(json.dumps(result, sort_keys=True)),
        }
        public_rows.append(public)
        private_rows.append(private)

    FROZEN.mkdir(parents=True, exist_ok=True)
    public_path = FROZEN / "stimuli_public.jsonl"
    private_path = FROZEN / "stimuli_private.jsonl"
    human_schedule_path = FROZEN / "human_randomization_schedule.csv"
    agent_schedule_path = FROZEN / "agent_randomization_schedule.csv"
    invitation_codes_path = FROZEN / "invitation_codes.csv"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    human_schedule, agent_schedule = condition_schedule(config)
    write_csv(human_schedule_path, human_schedule)
    write_csv(agent_schedule_path, agent_schedule)
    write_csv(invitation_codes_path, [
        {"participant_code": f"H{number:03d}"}
        for number in range(1, int(config["invitation_codes_planned"]) + 1)
    ])

    source_files = {
        str(CONFIG_PATH.relative_to(ROOT)): sha256_file(CONFIG_PATH),
        str(MANIFEST_PATH.relative_to(ROOT)): sha256_file(MANIFEST_PATH),
        str(DECISIONS_PATH.relative_to(ROOT)): sha256_file(DECISIONS_PATH),
    }
    for path in raw_files:
        if any(row["task_id"] in {str(item["id"]) for item in json.loads(path.read_text(encoding="utf-8")).get("with", [])} for row in private_rows):
            source_files[str(path.relative_to(ROOT))] = sha256_file(path)
    output_files = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            public_path, private_path, human_schedule_path, agent_schedule_path,
            invitation_codes_path,
        )
    }
    freeze = {
        "protocol_version": config["protocol_version"],
        "status": config["status"],
        "ethics_recruitment_allowed": False,
        "recorded_date": "2026-08-20",
        "stimuli": len(public_rows),
        "incorrect_cases": sum(not row["initial_correct"] for row in private_rows),
        "correct_controls": sum(row["initial_correct"] for row in private_rows),
        "repositories": len(repositories),
        "planned_human_participants": config["participants_planned"],
        "target_analyzable_participants": config["participants_target_analyzable"],
        "invitation_codes": config["invitation_codes_planned"],
        "planned_agent_sessions": 10,
        "source_file_sha256": source_files,
        "output_file_sha256": output_files,
        "warning": "No human outcomes exist. Recruitment remains blocked pending institutional ethics determination.",
    }
    freeze_path = FROZEN / "protocol_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": freeze["status"],
        "stimuli": freeze["stimuli"],
        "incorrect_cases": freeze["incorrect_cases"],
        "correct_controls": freeze["correct_controls"],
        "human_schedule_rows": len(human_schedule),
        "invitation_codes": config["invitation_codes_planned"],
        "agent_schedule_rows": len(agent_schedule),
        "freeze": str(freeze_path.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
