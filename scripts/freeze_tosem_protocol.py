#!/usr/bin/env python3
"""Freeze the accepted OpenCoderX confirmatory scientific protocol."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tosem/protocol_freeze.json"
FILES = (
    "data/manifests/execrepobench_opencoderx_120_v1.jsonl",
    "data/indexes/execrepobench_120_repository_knowledge_v1.jsonl",
    "configs/tosem/models/gpt4o_mini.yaml",
    "configs/tosem/models/gemini_2_5_flash.yaml",
    "configs/tosem/models/claude_sonnet_5.yaml",
    "configs/tosem/models/qwen3_coder_plus.yaml",
    "opencoder/pipeline.py",
    "opencoder/phase5_verify/repair.py",
    "opencoder/phase3_retrieval/score_filter.py",
    "experiments/run_rq3.py",
    "docs/tosem/CONFIRMATORY_PROTOCOL.md",
    "docs/tosem/PREFREEZE_RETRIEVAL_CORRECTION.md",
    "docs/tosem/PROTOCOL_AMENDMENT_REPAIR_BUDGET.md",
    "results/tosem/pilot/anchor_repilot_decision.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    decision = json.loads(
        (ROOT / "results/tosem/pilot/anchor_repilot_decision.json").read_text(encoding="utf-8")
    )
    if decision.get("accepted") is not True or decision.get("integrity_pass") is not True:
        raise SystemExit("retrieval correction was not accepted; refusing to freeze")
    hashes = {path: _sha256(ROOT / path) for path in FILES}
    scientific_payload = {
        "protocol_version": "opencoderx_tosem_confirmatory_v1",
        "dataset": "ExecRepoBench-120",
        "tasks": 120,
        "models": [
            "gpt-4o-mini",
            "gemini-2.5-flash",
            "claude-sonnet-5",
            "qwen3-coder-plus",
        ],
        "methods": [
            "Direct Generation",
            "Standard RAG",
            "RAG + Verify/Repair",
            "OpenCoderX",
        ],
        "candidate_count": 5,
        "temperature": 0.7,
        "claude_temperature": None,
        "max_output_tokens": 2048,
        "max_repair_rounds": 2,
        "repair_prompt_budget_chars": {
            "task": 24000,
            "failed_code": 32000,
            "diagnostics": 32000,
            "policy": "frozen_head_tail_character_budget_v1",
        },
        "retrieval_budget": {
            "api_top_k": 8,
            "context_top_k": 8,
            "similar_code_top_k": 8,
            "fused_top_k": 10,
            "opencoderx_whole_task_anchor": True,
            "opencoderx_source_balanced_fusion": True,
            "opencoderx_max_source_fraction": 0.5,
        },
        "primary_outcomes": ["pass@1", "selected_output_correctness"],
        "secondary_passk": ["pass@3", "pass@5"],
        "paired_tests": ["exact McNemar", "paired bootstrap 95% CI"],
        "multiple_testing": "Holm correction within each declared comparison family",
    }
    payload = {
        "status": "FROZEN_APPROVED",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": True,
        "scientific_protocol": scientific_payload,
        "scientific_protocol_sha256": hashlib.sha256(
            json.dumps(scientific_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "file_sha256": hashes,
        "operational_note": "Campaign caps were explicitly approved at USD 80 and CNY 65. Superseded pre-amendment API usage remains charged to these caps.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
