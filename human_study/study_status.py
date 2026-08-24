#!/usr/bin/env python3
"""Report privacy-preserving recruitment progress for the human study."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
CONFIG = json.loads((STUDY / "study_config.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        rows = [json.loads(line) for line in handle if line.strip()]
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return rows


def invitation_codes() -> list[str]:
    with (FROZEN / "invitation_codes.csv").open(newline="", encoding="utf-8") as handle:
        return [row["participant_code"] for row in csv.DictReader(handle)]


def snapshot(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    codes = invitation_codes()
    participants = read_jsonl(data_dir / "participants.jsonl")
    episodes = read_jsonl(data_dir / "episodes.jsonl")
    tutorials = read_jsonl(data_dir / "tutorials.jsonl")
    poststudy = read_jsonl(data_dir / "poststudy.jsonl")
    withdrawals = read_jsonl(data_dir / "withdrawals.jsonl")

    enrolled = {str(row["participant_code"]) for row in participants}
    tutorial_done = {
        str(row["participant_code"]) for row in tutorials if row.get("passed") is True
    }
    completed = {str(row["participant_code"]) for row in poststudy}
    withdrawn = {str(row["participant_code"]) for row in withdrawals}
    episode_counts = {code: 0 for code in codes}
    for row in episodes:
        code = str(row["participant_code"])
        if code in episode_counts:
            episode_counts[code] += 1

    rows = []
    for code in codes:
        if code in completed:
            state = "completed"
        elif code in withdrawn:
            state = "withdrawn"
        elif code in enrolled:
            state = "in_progress"
        else:
            state = "not_started"
        rows.append({
            "participant_code": code, "state": state,
            "enrolled": code in enrolled, "tutorial_completed": code in tutorial_done,
            "episodes_completed": episode_counts[code],
            "poststudy_completed": code in completed, "withdrawn": code in withdrawn,
        })

    summary = {
        "invitation_codes": len(codes),
        "completion_target": int(CONFIG["participants_planned"]),
        "analyzable_target": int(CONFIG["participants_target_analyzable"]),
        "enrolled": len(enrolled), "tutorial_completed": len(tutorial_done),
        "in_progress": sum(row["state"] == "in_progress" for row in rows),
        "completed": len(completed), "withdrawn": len(withdrawn),
        "not_started": sum(row["state"] == "not_started" for row in rows),
        "total_task_submissions": len(episodes),
        "recruitment_closed": len(completed) >= int(CONFIG["participants_planned"]),
        "next_available_codes": [
            row["participant_code"] for row in rows if row["state"] == "not_started"
        ][:10],
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(summary: dict[str, Any]) -> None:
    print("OpenCoderX recruitment status")
    print(f"  Completed:          {summary['completed']} / {summary['completion_target']}")
    print(f"  Enrolled:           {summary['enrolled']}")
    print(f"  In progress:        {summary['in_progress']}")
    print(f"  Tutorial completed: {summary['tutorial_completed']}")
    print(f"  Withdrawn:          {summary['withdrawn']}")
    print(f"  Not started:        {summary['not_started']}")
    print(f"  Task submissions:   {summary['total_task_submissions']}")
    print(f"  Recruitment closed: {summary['recruitment_closed']}")
    print(f"  Next codes:         {', '.join(summary['next_available_codes']) or '--'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=STUDY / "data")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    args = parser.parse_args()

    while True:
        summary, rows = snapshot(args.data_dir)
        if args.out_csv:
            write_csv(args.out_csv, rows)
        if args.watch_seconds > 0:
            print("\033[2J\033[H", end="")
        print(json.dumps(summary, indent=2) if args.json else "", end="" if args.json else "")
        if not args.json:
            display(summary)
        elif args.watch_seconds <= 0:
            print()
        if args.watch_seconds <= 0:
            return 0
        time.sleep(max(1.0, args.watch_seconds))


if __name__ == "__main__":
    sys.exit(main())
