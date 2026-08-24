#!/usr/bin/env python3
"""Freeze a pre-output multilingual CrossCodeEval pilot subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LANGUAGES = ("python", "java", "typescript", "csharp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/crosscodeeval_opencoderx_100_v1.jsonl")
    parser.add_argument("--output", default="data/manifests/crosscodeeval_opencoderx_pilot8_v1.jsonl")
    parser.add_argument("--report", default="results/tosem/crosscodeeval_pilot/subset_freeze.json")
    parser.add_argument("--per-language", type=int, default=2)
    args = parser.parse_args()
    source = Path(args.manifest)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for language in LANGUAGES:
        candidates = [row for row in rows if row.get("language") == language]
        candidates.sort(key=lambda row: str((row.get("provenance") or {}).get("artifact_hash") or ""))
        selected.extend(candidates[: args.per_language])
    if len(selected) != len(LANGUAGES) * args.per_language:
        raise SystemExit("failed to freeze the requested multilingual pilot")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "selection_rule": "lowest pre-output artifact hashes within each frozen language stratum",
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "pilot_manifest": str(output),
        "pilot_manifest_sha256": _sha256(output),
        "tasks": len(selected),
        "tasks_per_language": args.per_language,
        "task_ids": [row["task_id"] for row in selected],
        "languages": list(LANGUAGES),
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
