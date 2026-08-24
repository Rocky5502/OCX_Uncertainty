"""Build a JSONL subset whose reference solutions pass reconstructed repo tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="input/execrepobench_data.jsonl")
    ap.add_argument("--audit", default="results/repository_test_harness_audit_full.json")
    ap.add_argument("--out", default="input/execrepobench_testbacked.jsonl")
    ap.add_argument(
        "--ids",
        default=None,
        help="Optional comma-separated task ids to keep after applying the audit filter.",
    )
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    keep_ids = {
        row["id"]
        for row in audit.get("rows", [])
        if row.get("reference_passed") is True
    }
    if args.ids:
        requested_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
        keep_ids &= requested_ids
    written = 0
    with open(args.source, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or row.get("id") or "")
            if task_id in keep_ids:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} reference-passing test-backed rows to {args.out}")
    if written != len(keep_ids):
        missing = sorted(keep_ids - {json.loads(line).get("task_id") for line in Path(args.out).read_text(encoding="utf-8").splitlines() if line.strip()})
        if missing:
            print(f"WARNING: missing IDs: {missing[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
