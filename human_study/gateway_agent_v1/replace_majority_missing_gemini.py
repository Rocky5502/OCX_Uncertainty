#!/usr/bin/env python3
"""Replace Gemini's majority-missing slice with its predeclared alternate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/agent_gateway_v1"
RAW = RESULTS / "raw_results.jsonl"
SUPERSEDED = RESULTS / "superseded_majority_missing_gemini.jsonl"
RESOLVED = RESULTS / "resolved_model_manifest.json"
ALT_PREFLIGHT = RESULTS / "preflight/G003_alternate.json"


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
    if preflight.get("status") != "COMPLETED" or preflight.get("requested_model") != "gemini-2.5-flash-lite":
        raise RuntimeError("the predeclared Gemini alternate has not passed preflight")
    rows = read_jsonl(RAW)
    target = [row for row in rows if row.get("agent_id") == "G003"]
    if len(target) != 12:
        raise RuntimeError(f"expected the complete 12-record Gemini slice, found {len(target)}")
    missing = sum(row.get("evaluator_status") != "ok" for row in target)
    if missing <= len(target) / 2:
        raise RuntimeError(f"replacement requires majority missingness; observed {missing}/12")
    reason = (
        f"gemini-2.5-flash produced {missing}/12 unevaluable responses under the fixed "
        "2,048-token output budget, exceeding the protocol-amendment majority-missing rule."
    )
    archived = []
    for row in target:
        copy = dict(row)
        copy["superseded_reason"] = reason
        archived.append(copy)
    write_jsonl(SUPERSEDED, archived)
    write_jsonl(RAW, [row for row in rows if row.get("agent_id") != "G003"])

    manifest = json.loads(RESOLVED.read_text(encoding="utf-8"))
    model = next(row for row in manifest["models"] if row["agent_id"] == "G003")
    model["superseded_model_id"] = "gemini-2.5-flash"
    model["resolved_model_id"] = "gemini-2.5-flash-lite"
    model["replacement_reason"] = reason
    model["replacement_preflight"] = preflight
    RESOLVED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "decision_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_id": "G003",
        "family": "Gemini",
        "superseded_model_id": "gemini-2.5-flash",
        "replacement_model_id": "gemini-2.5-flash-lite",
        "superseded_missing_records": missing,
        "superseded_total_records": len(target),
        "correctness_fields_were_present": True,
        "selection_basis_used_correctness": False,
        "selection_basis": "technical missingness above 50% only",
        "superseded_record": str(SUPERSEDED.relative_to(ROOT)),
        "reason": reason,
    }
    (RESULTS / "gemini_protocol_amendment.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
