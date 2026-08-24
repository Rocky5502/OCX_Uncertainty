#!/usr/bin/env python3
"""Create a labeled participant subset from synthetic dry-run fixtures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "human_study/dry_run"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=24)
    parser.add_argument("--out", type=Path, default=SOURCE / "mock24")
    args = parser.parse_args()
    if args.participants < 3 or args.participants > 100:
        raise ValueError("participant subset must be between 3 and 100")

    participants = read_jsonl(SOURCE / "participants.jsonl")[: args.participants]
    if any(row.get("study_mode") != "SIMULATED_DRY_RUN" for row in participants):
        raise RuntimeError("subset source contains non-synthetic participant records")
    codes = {str(row["participant_code"]) for row in participants}
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "participants.jsonl", participants)
    for name in ("tutorials.jsonl", "poststudy.jsonl"):
        rows = [row for row in read_jsonl(SOURCE / name) if str(row["participant_code"]) in codes]
        write_jsonl(args.out / name, rows)

    with (SOURCE / "scored_human_episodes.csv").open(newline="", encoding="utf-8") as handle:
        episodes = [
            row for row in csv.DictReader(handle) if str(row["participant_code"]) in codes
        ]
    with (args.out / "scored_human_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    marker = {
        "status": "SIMULATED_MOCK_NOT_EMPIRICAL",
        "participants": len(participants), "episodes": len(episodes),
        "source": "human_study/dry_run",
        "warning": "Layout-testing fixture only; never report as human evidence.",
    }
    (args.out / "SIMULATION_MARKER.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
