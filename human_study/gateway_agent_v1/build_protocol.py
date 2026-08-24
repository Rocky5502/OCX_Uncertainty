#!/usr/bin/env python3
"""Freeze prompts and the balanced schedule for the gateway agent study."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "human_study/gateway_agent_v1"
PUBLIC = ROOT / "human_study/frozen/stimuli_public.jsonl"
HUMAN_SCHEDULE = ROOT / "human_study/frozen/human_randomization_schedule.csv"

OUTPUT_CONTRACT = """# Response contract
Review the starting implementation using only the supplied material. Do not
browse the web, run code, use external tools, infer hidden tests, request more
context, or describe repository files that are not shown.

Return exactly this plain-text structure, with no text outside it:

STARTING_CORRECT: true|false
STARTING_CONFIDENCE: integer 0-100
FINAL_CONFIDENCE: integer 0-100
FINAL_CODE_BEGIN
<exactly one complete target function; no imports or surrounding declarations>
FINAL_CODE_END

STARTING_CONFIDENCE is the probability that your STARTING_CORRECT judgment is
correct. FINAL_CONFIDENCE is the probability that FINAL_CODE passes the private
executable tests. Always provide FINAL_CODE, even when no change is needed.
The evaluator accepts exactly one function definition. Put any necessary local
import inside that function body; do not return module-level imports, classes,
tests, Markdown fences, or explanatory prose.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prompt(stimulus: dict[str, Any], condition: str) -> str:
    parts = [
        OUTPUT_CONTRACT,
        "# Repository review task",
        str(stimulus["task_text"]),
        "\n# Repository context",
        str(stimulus["repository_context"]),
        "\n# Retrieved evidence",
        str(stimulus["retrieved_evidence"]),
        "\n# Starting implementation",
        str(stimulus["starting_code"]),
    ]
    if condition in {"uncertainty_display", "targeted_guidance"}:
        risks = stimulus["source_risks"]
        parts.extend(
            [
                "\n# Uncertainty trace",
                "\n".join(
                    f"{key}: {100 * float(risks[key]):.1f}% risk"
                    for key in ("api", "context", "similar_code", "generation")
                ),
                f"aggregate review risk: {100 * float(stimulus['aggregate_risk']):.1f}%",
            ]
        )
    if condition == "targeted_guidance":
        parts.extend(["\n# Recommended review action", str(stimulus["targeted_guidance"])])
    return "\n".join(parts).strip() + "\n"


def main() -> int:
    manifest = json.loads((HERE / "model_manifest.json").read_text(encoding="utf-8"))
    stimuli = {row["task_id"]: row for row in read_jsonl(PUBLIC)}
    human = read_csv(HUMAN_SCHEDULE)
    templates = {
        group: sorted(
            [row for row in human if row["participant_code"] == f"H{group:03d}"],
            key=lambda row: int(row["episode_index"]),
        )
        for group in (1, 2, 3)
    }
    if any(len(rows) != 12 for rows in templates.values()):
        raise RuntimeError("counterbalanced schedule templates are incomplete")

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for model in manifest["models"]:
        agent_id = model["agent_id"]
        directory = HERE / "prompts" / agent_id
        directory.mkdir(parents=True, exist_ok=True)
        for item in templates[int(model["assignment_group"])]:
            episode = int(item["episode_index"])
            task_id = item["task_id"]
            content = prompt(stimuli[task_id], item["condition"])
            path = directory / f"episode_{episode:02d}.txt"
            path.write_text(content, encoding="utf-8")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            relative = str(path.relative_to(ROOT))
            hashes[relative] = digest
            rows.append(
                {
                    "agent_id": agent_id,
                    "family": model["family"],
                    "assignment_group": model["assignment_group"],
                    "episode_index": episode,
                    "task_id": task_id,
                    "condition": item["condition"],
                    "prompt_path": relative,
                    "prompt_sha256": digest,
                }
            )

    counts = Counter(row["condition"] for row in rows)
    cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cell_counts[row["task_id"]][row["condition"]] += 1
    expected = {"generic_review": 4, "uncertainty_display": 4, "targeted_guidance": 4}
    if counts != Counter({key: 48 for key in expected}):
        raise RuntimeError(f"condition imbalance: {counts}")
    if any(dict(value) != expected for value in cell_counts.values()):
        raise RuntimeError("task-condition schedule is not counterbalanced")

    schedule = HERE / "schedule.csv"
    with schedule.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    freeze = {
        "protocol": manifest["protocol"],
        "study_mode": manifest["study_mode"],
        "models": manifest["models"],
        "planned_episodes": len(rows),
        "episodes_per_condition": dict(counts),
        "task_condition_count": 4,
        "prompts_sha256": hashes,
        "source_public_stimuli_sha256": hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),
        "source_schedule_sha256": hashlib.sha256(HUMAN_SCHEDULE.read_bytes()).hexdigest(),
        "warning": manifest["warning"],
    }
    (HERE / "protocol_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "FROZEN", "episodes": len(rows), "conditions": dict(counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
