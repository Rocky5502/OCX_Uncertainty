#!/usr/bin/env python3
"""Freeze balanced CrossCodeEval-100 batches and its measured cost projection."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl"
BATCH_DIR = ROOT / "data/manifests/crosscodeeval_opencoderx_100_batches_v1"
OUTPUT_DIR = ROOT / "results/tosem/crosscodeeval_confirmatory"
PILOT_RESOURCES = ROOT / "results/tosem/crosscodeeval_pilot/resource_summary.csv"
CAMPAIGN_STATUS = ROOT / "results/tosem/confirmatory/campaign_status.json"
CAMPAIGN_CONFIG = ROOT / "configs/tosem/campaign.yaml"
DATA_QUALITY = ROOT / "results/data_quality/crosscodeeval_freeze.json"
MODEL_CONFIGS = (
    "configs/tosem/models/gpt4o_mini.yaml",
    "configs/tosem/models/gemini_2_5_flash.yaml",
    "configs/tosem/models/claude_sonnet_5.yaml",
    "configs/tosem/models/qwen3_coder_plus.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    tasks = _read_jsonl(MANIFEST)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_language[str(task["language"])].append(task)
    expected_languages = ("python", "java", "typescript", "csharp")
    if len(tasks) != 100 or any(len(by_language[language]) != 25 for language in expected_languages):
        raise SystemExit("CrossCodeEval manifest must contain 25 frozen tasks per language")

    # Round-robin assignment preserves the pre-output manifest order while
    # balancing every operational batch across all four languages.
    batches: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for language_index, language in enumerate(expected_languages):
        for task_index, task in enumerate(by_language[language]):
            batches[(task_index + language_index) % len(batches)].append(task)
    if sorted(len(batch) for batch in batches) != [25, 25, 25, 25]:
        raise SystemExit("balanced batch construction failed")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_records = []
    for index, batch in enumerate(batches):
        path = BATCH_DIR / f"batch_{index:02d}.jsonl"
        ordered = sorted(batch, key=lambda row: str(row["task_id"]))
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
            encoding="utf-8",
        )
        counts = defaultdict(int)
        for row in ordered:
            counts[str(row["language"])] += 1
        batch_records.append({
            "batch": index,
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "tasks": len(ordered),
            "language_counts": dict(sorted(counts.items())),
            "task_ids": [str(row["task_id"]) for row in ordered],
        })

    config = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    limits = config["cost_controls"]["campaign_limits"]
    current_spend = json.loads(CAMPAIGN_STATUS.read_text(encoding="utf-8"))["campaign_spend"]
    projected_increment: dict[str, float] = defaultdict(float)
    projection_rows = []
    with PILOT_RESOURCES.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pilot_tasks = 8
            scale = len(tasks) / pilot_tasks
            projected = float(row["estimated_cost"]) * scale
            currency = str(row["currency"])
            projected_increment[currency] += projected
            projection_rows.append({
                "model": row["model"],
                "method": row["method"],
                "pilot_tasks": pilot_tasks,
                "confirmatory_tasks": len(tasks),
                "pilot_cost": float(row["estimated_cost"]),
                "projected_confirmatory_cost": projected,
                "currency": currency,
                "basis": "linear scaling of the audited eight-task native-metric pilot",
            })
    totals = {
        currency: float(current_spend.get(currency, 0.0)) + projected_increment.get(currency, 0.0)
        for currency in set(current_spend) | set(projected_increment)
    }
    allowed = all(totals[currency] <= float(limits[currency]) for currency in totals)
    projection = {
        "status": "APPROVED_WITHIN_CONFIGURED_CAP" if allowed else "BLOCKED_BY_CONFIGURED_CAP",
        "paper_eligible": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "current_campaign_spend": current_spend,
        "projected_crosscodeeval_increment": dict(projected_increment),
        "projected_campaign_total": totals,
        "campaign_limits": limits,
        "allowed": allowed,
        "rows": projection_rows,
    }
    cost_path = ROOT / "results/tosem/cost/crosscodeeval_confirmatory_projection.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
    if not allowed:
        raise SystemExit("CrossCodeEval projection exceeds a configured campaign cap")

    quality = json.loads(DATA_QUALITY.read_text(encoding="utf-8"))
    if quality.get("status") != "FROZEN_NATIVE_COMPLETION" or int(quality.get("tasks") or 0) != 100:
        raise SystemExit("CrossCodeEval data-quality freeze is invalid")
    if quality.get("manifest_sha256") != _sha256(MANIFEST):
        raise SystemExit("CrossCodeEval data-quality manifest hash mismatch")

    frozen_files = [
        MANIFEST,
        DATA_QUALITY,
        ROOT / "experiments/run_crosscodeeval.py",
        ROOT / "opencoder/evaluation/crosscodeeval.py",
        CAMPAIGN_CONFIG,
        *[ROOT / relative for relative in MODEL_CONFIGS],
        *[ROOT / record["path"] for record in batch_records],
    ]
    protocol = {
        "status": "FROZEN_APPROVED",
        "paper_eligible": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CrossCodeEval-100",
        "experiment_version": "crosscodeeval_native_confirmatory_v1",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "tasks": 100,
        "tasks_per_language": 25,
        "languages": list(expected_languages),
        "models": [
            "gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-5", "qwen3-coder-plus"
        ],
        "methods": ["Direct Generation", "Cross-file Context RAG"],
        "candidate_count": 5,
        "temperature": 0.7,
        "claude_temperature": None,
        "max_output_tokens": 128,
        "prefix_char_budget": 6000,
        "context_char_budget": 2048,
        "functional_execution": False,
        "native_metrics": ["exact_match", "edit_similarity", "identifier_f1"],
        "selection_rule": "first generated candidate; candidate means are reported separately",
        "batch_selection_depends_on_model_outputs": False,
        "batches": batch_records,
        "cost_projection": str(cost_path.relative_to(ROOT)),
        "cost_projection_sha256": _sha256(cost_path),
        "file_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in frozen_files},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT_DIR / "protocol_freeze.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": str(protocol_path.relative_to(ROOT)),
        "batches": len(batch_records),
        "projection": projection,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
