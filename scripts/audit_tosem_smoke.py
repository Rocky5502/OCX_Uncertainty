#!/usr/bin/env python3
"""Validate the four-family smoke campaign and report measured resources."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = {
    "without": "Standard RAG",
    "rag_repair": "RAG + Verify/Repair",
    "with": "OpenCoderX",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pricing() -> dict[str, dict[str, Any]]:
    config = yaml.safe_load((ROOT / "configs/tosem/campaign.yaml").read_text(encoding="utf-8"))
    return config["cost_controls"]["gateway_pricing"]


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, str]:
    pricing = _pricing()[model]
    amount = (
        prompt_tokens * float(pricing["input_per_million"])
        + completion_tokens * float(pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(pricing["currency"])


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("requests", "retries", "failed_attempts", "prompt_tokens", "completion_tokens", "total_tokens")
    return {key: sum(int((row.get("llm_usage") or {}).get(key) or 0) for row in rows) for key in keys}


def main() -> int:
    out_dir = ROOT / "results/tosem/smoke"
    summaries: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    issues: list[str] = []
    global_task_set: set[str] | None = None

    for model, directory in MODELS.items():
        path = out_dir / directory / "rq3.json"
        if not path.is_file():
            issues.append(f"missing result: {path.relative_to(ROOT)}")
            continue
        data = _read(path)
        method_task_sets: list[set[str]] = []
        for key, label in METHODS.items():
            rows = list(data.get(key) or [])
            tasks = {str(row.get("id")) for row in rows}
            method_task_sets.append(tasks)
            summary = (data.get("summary") or {}).get(key) or {}
            summaries.append({
                "model": model,
                "method": label,
                "tasks": len(rows),
                "pass_at_1": summary.get("pass@1"),
                "pass_at_3": summary.get("pass@3"),
                "pass_at_5": summary.get("pass@5"),
                "selected_output_correctness": (
                    sum(bool(row.get("passed")) for row in rows) / len(rows)
                    if rows else None
                ),
                "mean_uncertainty": summary.get("mean_uncertainty"),
                "mean_repair_rounds": summary.get("mean_repair_rounds"),
                "mean_latency_seconds": summary.get("mean_run_latency_s"),
                "paper_eligible": False,
            })
            usage = _usage(rows)
            amount, currency = _cost(model, usage["prompt_tokens"], usage["completion_tokens"])
            resources.append({
                "run_status": "authoritative_smoke",
                "model": model,
                "method": label,
                **usage,
                "estimated_cost": amount,
                "currency": currency,
            })
            for row in rows:
                prefix = f"{model}/{label}/{row.get('id')}"
                if "error" in row:
                    issues.append(f"{prefix}: error={row['error']}")
                if len(row.get("generated_samples") or []) != 5:
                    issues.append(f"{prefix}: candidate count is not five")
                if len(row.get("sample_correctness") or []) != 5:
                    issues.append(f"{prefix}: test-outcome count is not five")
                if (row.get("generation_integrity") or {}).get("valid") is not True:
                    issues.append(f"{prefix}: generation integrity failed")
                if row.get("correctness_mode") != "repository_tests":
                    issues.append(f"{prefix}: non-executable correctness mode")
                metadata = row.get("generation_response_metadata") or []
                if len(metadata) != 5:
                    issues.append(f"{prefix}: generation metadata count is not five")
                for item in metadata:
                    if item.get("served_model") != model or not item.get("response_id"):
                        issues.append(f"{prefix}: missing/mismatched provider provenance")
                        break
        if method_task_sets and any(tasks != method_task_sets[0] for tasks in method_task_sets[1:]):
            issues.append(f"{model}: method task sets differ")
        if method_task_sets:
            if global_task_set is None:
                global_task_set = method_task_sets[0]
            elif method_task_sets[0] != global_task_set:
                issues.append(f"{model}: task set differs across model families")

    superseded = out_dir / "gpt4o_mini/rq3_pre_metadata_fix.json"
    if superseded.is_file():
        data = _read(superseded)
        rows = [row for key in METHODS for row in data.get(key, [])]
        usage = _usage(rows)
        amount, currency = _cost("gpt-4o-mini", usage["prompt_tokens"], usage["completion_tokens"])
        resources.append({
            "run_status": "superseded_diagnostic",
            "model": "gpt-4o-mini",
            "method": "all",
            **usage,
            "estimated_cost": amount,
            "currency": currency,
        })

    for model, directory in MODELS.items():
        path = ROOT / "results/tosem/preflight" / f"{directory}.json"
        if not path.is_file():
            issues.append(f"missing preflight: {model}")
            continue
        preflight = _read(path)
        usage = preflight.get("usage") or {}
        amount, currency = _cost(
            model,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )
        resources.append({
            "run_status": "preflight",
            "model": model,
            "method": "connectivity",
            **{key: int(usage.get(key) or 0) for key in ("requests", "retries", "failed_attempts", "prompt_tokens", "completion_tokens", "total_tokens")},
            "estimated_cost": amount,
            "currency": currency,
        })

    summary_path = out_dir / "summary.csv"
    resource_path = out_dir / "resource_summary.csv"
    integrity_path = out_dir / "integrity.json"
    memo_path = out_dir / "results_memo.md"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with resource_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resources[0]))
        writer.writeheader()
        writer.writerows(resources)
    totals: dict[str, float] = {}
    for row in resources:
        totals[row["currency"]] = totals.get(row["currency"], 0.0) + float(row["estimated_cost"])
    integrity = {
        "passed": not issues,
        "models": list(MODELS),
        "methods": list(METHODS.values()),
        "matched_tasks": len(global_task_set or set()),
        "issues": issues,
        "estimated_spend_including_superseded_and_preflight": totals,
        "all_call_response_audit_note": (
            "Generation and preflight response IDs are complete. Full per-call audit logging "
            "was added after this smoke and is mandatory from the pilot onward."
        ),
    }
    integrity_path.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    memo_path.write_text(
        "# Four-Family Smoke Memo\n\n"
        f"Status: {'PASSED' if not issues else 'FAILED'}. The smoke covers {len(global_task_set or set())} "
        "frozen ExecRepoBench tasks, four gateway-served model families, and three matched methods. "
        "Every current cell contains five executable candidates and is diagnostic only; no smoke "
        "number is eligible for the manuscript.\n\n"
        "The principal boundary is visible rather than hidden: verification/repair can recover "
        "individual failures, while full OpenCoderX is not uniformly stronger on this tiny sample. "
        "The 10-task pilot must therefore evaluate method parity, all-call provenance, latency, "
        "and uncertainty calibration before protocol freeze.\n\n"
        f"Estimated cumulative spend, including the superseded GPT diagnostic and preflights: {totals}.\n",
        encoding="utf-8",
    )
    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
