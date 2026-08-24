#!/usr/bin/env python3
"""Archive pre-amendment repair results without deleting provenance."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/tosem/confirmatory"
DESTINATION = ROOT / "results/tosem/confirmatory_superseded_pre_repair_budget"
METHODS = ("rag_verify_repair", "opencoder")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_tree(source: Path, destination: Path, records: list[dict[str, object]]) -> None:
    if not source.exists():
        return
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {destination}")
    for path in files:
        records.append({
            "source": str(path.relative_to(ROOT)),
            "destination": str((destination / path.relative_to(source)).relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> int:
    manifest = DESTINATION / "migration_manifest.json"
    if manifest.exists():
        raise SystemExit(f"archive already completed: {manifest.relative_to(ROOT)}")
    records: list[dict[str, object]] = []
    model_dirs = sorted(
        path for path in SOURCE.iterdir()
        if path.is_dir() and path.name != "logs"
    )
    for model_dir in model_dirs:
        for method in METHODS:
            _move_tree(
                model_dir / method,
                DESTINATION / model_dir.name / method,
                records,
            )
            _move_tree(
                SOURCE / "logs" / model_dir.name / method,
                DESTINATION / "logs" / model_dir.name / method,
                records,
            )
    if not records:
        raise SystemExit("no pre-amendment repair outputs found; refusing empty archive")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    payload = {
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "reason": "frozen repair-prompt budget amendment after provider context rejection",
        "excluded_from_confirmatory_analysis": True,
        "api_usage_still_counted": True,
        "methods": list(METHODS),
        "file_count": len(records),
        "files": records,
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest.relative_to(ROOT)),
        "file_count": len(records),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
