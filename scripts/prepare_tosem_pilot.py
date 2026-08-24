#!/usr/bin/env python3
"""Freeze the pre-output ExecRepoBench pilot subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/execrepobench_opencoderx_120_v1.jsonl")
    parser.add_argument("--output", default="data/manifests/execrepobench_opencoderx_pilot10_v1.jsonl")
    parser.add_argument("--report", default="results/tosem/pilot/subset_freeze.json")
    parser.add_argument("--tasks", type=int, default=10)
    args = parser.parse_args()
    source = Path(args.manifest)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection_rules = {
        str((row.get("adaptation_metadata") or {}).get("selection_order") or "")
        for row in rows
    }
    if selection_rules != {"ascending pre-output artifact_hash"}:
        raise SystemExit(f"unexpected manifest selection rules: {sorted(selection_rules)}")
    rows.sort(key=lambda row: str(row.get("artifact_hash") or ""))
    selected = rows[: args.tasks]
    if len(selected) != args.tasks or any(not row.get("reference_tests_pass") for row in selected):
        raise SystemExit("pilot selection requires the requested number of reference-valid tasks")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "selection_rule": "ascending pre-output artifact_hash, inherited from the frozen source manifest",
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "pilot_manifest": str(output),
        "pilot_manifest_sha256": sha256(output),
        "tasks": len(selected),
        "task_ids": [row["task_id"] for row in selected],
        "repositories": sorted({row["repo_name"] for row in selected}),
        "method_outputs_inspected_for_selection": False,
        "paper_eligible": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
