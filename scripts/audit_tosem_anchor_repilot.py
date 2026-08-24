#!/usr/bin/env python3
"""Audit the one-shot whole-task-anchor OpenCoderX development re-pilot."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "results/tosem/pilot"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_pilot10_v1.jsonl"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _score(row: dict[str, Any], k: int) -> float:
    outcomes = list(row.get("effective_sample_correctness") or row.get("sample_correctness") or [])
    n = len(outcomes)
    correct = sum(bool(value) for value in outcomes)
    if n == 0 or correct == 0:
        return 0.0
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    task_ids = [
        str(json.loads(line)["task_id"])
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = set(task_ids)
    issues: list[str] = []
    summary: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    model_noninferiority: dict[str, bool] = {}
    pooled_old_selected = pooled_new_selected = 0
    pooled_old_pass5 = pooled_new_pass5 = 0

    for model, directory in MODELS.items():
        original_path = PILOT_DIR / directory / "rq3.json"
        repilot_path = PILOT_DIR / directory / "opencoder_anchor_repilot.json"
        if not repilot_path.is_file():
            issues.append(f"missing re-pilot: {repilot_path.relative_to(ROOT)}")
            continue
        original = json.loads(original_path.read_text(encoding="utf-8"))
        repilot = json.loads(repilot_path.read_text(encoding="utf-8"))
        old = {str(row["id"]): row for row in original.get("with") or []}
        repair = {str(row["id"]): row for row in original.get("rag_repair") or []}
        new = {str(row["id"]): row for row in repilot.get("with") or []}
        if set(old) != expected or set(repair) != expected or set(new) != expected:
            issues.append(f"{model}: task set differs from frozen pilot")
            continue
        flags = ((repilot.get("metadata") or {}).get("condition_feature_flags") or {}).get("with") or {}
        if flags.get("whole_task_retrieval_anchor") is not True:
            issues.append(f"{model}: whole-task anchor flag is not enabled")
        if flags.get("source_balanced_fusion") is not True:
            issues.append(f"{model}: source-balanced fusion flag is not enabled")

        for task_id in task_ids:
            row = new[task_id]
            prefix = f"{model}/{task_id}"
            if row.get("error"):
                issues.append(f"{prefix}: {row['error']}")
            if len(row.get("generated_samples") or []) != 5:
                issues.append(f"{prefix}: candidate count is not five")
            if (row.get("generation_integrity") or {}).get("valid") is not True:
                issues.append(f"{prefix}: generation integrity failed")
            if row.get("correctness_mode") != "repository_tests":
                issues.append(f"{prefix}: non-executable correctness mode")
            evidence = list(row.get("fused_evidence_ids") or [])
            if len(evidence) != 10:
                issues.append(f"{prefix}: fused evidence count is not ten")
            counts = Counter(str(item.get("source")) for item in evidence)
            if len(counts) > 1 and max(counts.values()) > 5:
                issues.append(f"{prefix}: one source exceeds 50% of evidence")
            anchors = [
                step for step in row.get("per_step") or []
                if step.get("phase") == "whole_task_anchor"
            ]
            if len(anchors) != 1:
                issues.append(f"{prefix}: whole-task anchor trace missing")
            audit = list(row.get("provider_response_audit") or [])
            requests = int((row.get("llm_usage") or {}).get("requests") or 0)
            if len(audit) != requests:
                issues.append(f"{prefix}: provider audit count mismatch")
            for item in audit:
                if item.get("requested_model") != model or item.get("served_model") != model:
                    issues.append(f"{prefix}: requested/served model mismatch")
                    break
                if not item.get("response_id"):
                    issues.append(f"{prefix}: missing response ID")
                    break
            detail.append({
                "model": model,
                "task_id": task_id,
                "old_selected_correct": bool(old[task_id].get("passed")),
                "new_selected_correct": bool(row.get("passed")),
                "repair_selected_correct": bool(repair[task_id].get("passed")),
                "old_pass_at_5": _score(old[task_id], 5),
                "new_pass_at_5": _score(row, 5),
                "repair_pass_at_5": _score(repair[task_id], 5),
                "api_items": counts.get("api", 0),
                "context_items": counts.get("context", 0),
                "similar_code_items": counts.get("similar_code", 0),
            })

        old_rows = [old[task_id] for task_id in task_ids]
        new_rows = [new[task_id] for task_id in task_ids]
        repair_rows = [repair[task_id] for task_id in task_ids]
        old_selected = sum(bool(row.get("passed")) for row in old_rows)
        new_selected = sum(bool(row.get("passed")) for row in new_rows)
        old_pass5 = sum(_score(row, 5) for row in old_rows)
        new_pass5 = sum(_score(row, 5) for row in new_rows)
        model_noninferiority[model] = new_selected >= old_selected - 1
        pooled_old_selected += old_selected
        pooled_new_selected += new_selected
        pooled_old_pass5 += int(old_pass5)
        pooled_new_pass5 += int(new_pass5)
        for label, rows in (
            ("Original OpenCoderX", old_rows),
            ("Corrected OpenCoderX", new_rows),
            ("RAG + Verify/Repair", repair_rows),
        ):
            summary.append({
                "model": model,
                "method": label,
                "tasks": len(rows),
                "pass_at_1": _mean(_score(row, 1) for row in rows),
                "pass_at_3": _mean(_score(row, 3) for row in rows),
                "pass_at_5": _mean(_score(row, 5) for row in rows),
                "selected_output_correctness": _mean(bool(row.get("passed")) for row in rows),
            })

    integrity_pass = not issues
    pooled_selected_gate = pooled_new_selected >= pooled_old_selected - 2
    pooled_pass5_gate = pooled_new_pass5 >= pooled_old_pass5 - 2
    accepted = (
        integrity_pass
        and len(model_noninferiority) == len(MODELS)
        and all(model_noninferiority.values())
        and pooled_selected_gate
        and pooled_pass5_gate
    )
    if summary:
        _write_csv(PILOT_DIR / "anchor_repilot_summary.csv", summary)
        _write_csv(PILOT_DIR / "anchor_repilot_by_task.csv", detail)
    decision = {
        "accepted": accepted,
        "paper_eligible": False,
        "integrity_pass": integrity_pass,
        "per_model_selected_noninferiority": model_noninferiority,
        "pooled_old_selected": pooled_old_selected,
        "pooled_new_selected": pooled_new_selected,
        "pooled_selected_gate": pooled_selected_gate,
        "pooled_old_pass_at_5": pooled_old_pass5,
        "pooled_new_pass_at_5": pooled_new_pass5,
        "pooled_pass_at_5_gate": pooled_pass5_gate,
        "issues": issues,
    }
    (PILOT_DIR / "anchor_repilot_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
