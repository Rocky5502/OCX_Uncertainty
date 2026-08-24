"""Fail fast when an expanded RQ3 run is incomplete or malformed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ids_from_manifest(path: Path) -> list[str]:
    return [
        str(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["without", "with"],
    )
    args = parser.parse_args()

    expected_ids = _ids_from_manifest(Path(args.manifest))
    issues: list[str] = []
    for run_value in args.run:
        path = Path(run_value)
        if not path.is_file():
            issues.append(f"{path}: missing run file")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for condition in args.conditions:
            rows = list(payload.get(condition) or [])
            ids = [str(row.get("id") or "") for row in rows]
            if ids != expected_ids:
                issues.append(
                    f"{path}:{condition}: task IDs/order differ "
                    f"({len(ids)}/{len(expected_ids)})"
                )
            for row in rows:
                task_id = str(row.get("id") or "")
                if row.get("error"):
                    issues.append(f"{path}:{condition}:{task_id}: {row['error']}")
                if len(row.get("generated_samples") or []) != 5:
                    issues.append(
                        f"{path}:{condition}:{task_id}: candidate count is not five"
                    )
                if len(row.get("sample_correctness") or []) != 5:
                    issues.append(
                        f"{path}:{condition}:{task_id}: correctness count is not five"
                    )
                if not (row.get("generation_integrity") or {}).get("valid"):
                    issues.append(
                        f"{path}:{condition}:{task_id}: invalid generation integrity"
                    )
                if row.get("passed") is None:
                    issues.append(
                        f"{path}:{condition}:{task_id}: no executable selected result"
                    )

    if issues:
        print("Expanded RQ3 run integrity FAILED:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(
        f"Expanded RQ3 run integrity passed for {len(expected_ids)} tasks "
        f"and conditions {', '.join(args.conditions)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
