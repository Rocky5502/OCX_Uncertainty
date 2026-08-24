#!/usr/bin/env python3
"""Split the frozen ExecRepoBench-120 manifest into immutable run batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
OUT_DIR = ROOT / "data/manifests/execrepobench_opencoderx_120_batches_v1"
REPORT = ROOT / "results/tosem/confirmatory/batch_freeze.json"
BATCH_SIZE = 10
EXPECTED_SOURCE_SHA256 = "4b14e4a648e80b11c7a7011b6c9f878fcae5ec243f5c1b260e905e693c3f459d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if _sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("frozen ExecRepoBench-120 manifest hash mismatch")
    lines = [line for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    task_ids = [str(row["task_id"]) for row in rows]
    if len(rows) != 120 or len(set(task_ids)) != 120:
        raise SystemExit("expected 120 unique frozen tasks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batches = []
    for start in range(0, len(lines), BATCH_SIZE):
        batch_index = start // BATCH_SIZE
        path = OUT_DIR / f"batch_{batch_index:02d}.jsonl"
        batch_lines = lines[start : start + BATCH_SIZE]
        path.write_text("\n".join(batch_lines) + "\n", encoding="utf-8")
        batches.append({
            "batch": batch_index,
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "task_count": len(batch_lines),
            "task_ids": task_ids[start : start + BATCH_SIZE],
        })
    payload = {
        "status": "FROZEN",
        "source_manifest": str(SOURCE.relative_to(ROOT)),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "batch_size": BATCH_SIZE,
        "batch_count": len(batches),
        "task_count": len(rows),
        "selection_depends_on_model_outputs": False,
        "batches": batches,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "batch_count", "task_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
