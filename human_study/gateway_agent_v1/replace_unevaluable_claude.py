#!/usr/bin/env python3
"""Replace the zero-text Claude format pilot with its predeclared alternate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/agent_gateway_v1"
RAW = RESULTS / "raw_results.jsonl"
SUPERSEDED = RESULTS / "superseded_format_pilot.jsonl"
RESOLVED = RESULTS / "resolved_model_manifest.json"
ALT_PREFLIGHT = RESULTS / "preflight/G002_alternate.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    preflight = json.loads(ALT_PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "COMPLETED" or preflight.get("requested_model") != "claude-sonnet-4-6":
        raise RuntimeError("the predeclared Claude alternate has not passed preflight")
    rows = read_jsonl(RAW)
    target = [row for row in rows if row.get("agent_id") == "G002" and int(row.get("episode_index", -1)) == 1]
    if len(target) != 1:
        raise RuntimeError(f"expected exactly one G002 episode-1 pilot, found {len(target)}")
    record = target[0]
    if record.get("evaluator_status") != "not_scored" or record.get("raw_response"):
        raise RuntimeError("Claude replacement is allowed only for a zero-text unevaluable pilot")
    archived = dict(record)
    archived["superseded_reason"] = (
        "Primary claude-sonnet-5 exhausted the fixed output budget with finish_reason=length "
        "and returned zero visible text; no correctness outcome existed."
    )
    prior = read_jsonl(SUPERSEDED)
    key = (archived["agent_id"], int(archived["episode_index"]), archived.get("requested_model"))
    if key not in {(row["agent_id"], int(row["episode_index"]), row.get("requested_model")) for row in prior}:
        prior.append(archived)
        write_jsonl(SUPERSEDED, prior)
    write_jsonl(RAW, [row for row in rows if row is not record])

    manifest = json.loads(RESOLVED.read_text(encoding="utf-8"))
    model = next(row for row in manifest["models"] if row["agent_id"] == "G002")
    if model["resolved_model_id"] not in {"claude-sonnet-5", "claude-sonnet-4-6"}:
        raise RuntimeError("unexpected resolved Claude model")
    model["superseded_model_id"] = "claude-sonnet-5"
    model["resolved_model_id"] = "claude-sonnet-4-6"
    model["replacement_reason"] = archived["superseded_reason"]
    model["replacement_preflight"] = preflight
    RESOLVED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "decision_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_id": "G002",
        "family": "Claude",
        "superseded_model_id": "claude-sonnet-5",
        "replacement_model_id": "claude-sonnet-4-6",
        "outcome_observed_before_replacement": False,
        "superseded_record": str(SUPERSEDED.relative_to(ROOT)),
        "reason": archived["superseded_reason"],
    }
    (RESULTS / "model_replacement_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
