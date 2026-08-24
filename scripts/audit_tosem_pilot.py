#!/usr/bin/env python3
"""Audit and summarize the frozen four-family TOSEM pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "results/tosem/pilot"
PILOT_MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_pilot10_v1.jsonl"
FROZEN_INDEX = "data/indexes/execrepobench_120_repository_knowledge_v1.jsonl"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
METHODS = {
    "direct": "Direct Generation",
    "without": "Standard RAG",
    "rag_repair": "RAG + Verify/Repair",
    "with": "OpenCoderX",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _bootstrap_ci(
    comparator: Sequence[float],
    opencoder: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 20260810,
) -> tuple[float, float]:
    if not comparator or len(comparator) != len(opencoder):
        raise ValueError("paired bootstrap inputs must be non-empty and matched")
    rng = random.Random(seed)
    n = len(comparator)
    deltas = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(_mean(opencoder[i] - comparator[i] for i in indices))
    deltas.sort()
    return (
        deltas[int(0.025 * (iterations - 1))],
        deltas[int(0.975 * (iterations - 1))],
    )


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "requests",
        "retries",
        "failed_attempts",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    return {
        key: sum(int((row.get("llm_usage") or {}).get(key) or 0) for row in rows)
        for key in keys
    }


def _pricing() -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(
        (ROOT / "configs/tosem/campaign.yaml").read_text(encoding="utf-8")
    )
    return config["cost_controls"]["gateway_pricing"]


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, str]:
    pricing = _pricing()[model]
    amount = (
        prompt_tokens * float(pricing["input_per_million"])
        + completion_tokens * float(pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(pricing["currency"])


def _score(row: dict[str, Any], metric: str) -> float:
    outcomes = list(
        row.get("effective_sample_correctness")
        or row.get("sample_correctness")
        or []
    )
    k = min(int(metric.split("@", 1)[1]), len(outcomes))
    correct = sum(bool(value) for value in outcomes)
    if not outcomes or not correct or k <= 0:
        return 0.0
    if len(outcomes) - correct < k:
        return 1.0
    return 1.0 - math.comb(len(outcomes) - correct, k) / math.comb(len(outcomes), k)


def _generation_integrity_valid(row: dict[str, Any]) -> bool:
    samples = list(row.get("generated_samples") or [])
    raw = list(row.get("generation_raw_responses") or [])
    metadata = list(row.get("generation_response_metadata") or [])
    if not samples or len(samples) != len(raw) or len(samples) != len(metadata):
        return False
    for sample, response, item in zip(samples, raw, metadata):
        limited = item.get("finish_reason") in {"length", "max_tokens"}
        unclosed_fence = response.lstrip().startswith("```") and response.count("```") < 2
        if not sample.strip() and not limited:
            return False
        if unclosed_fence and not limited:
            return False
    return True


def main() -> int:
    manifest_rows = [
        json.loads(line)
        for line in PILOT_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_ids = [str(row["task_id"]) for row in manifest_rows]
    expected_set = set(expected_ids)
    summaries: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    paired_passk: list[dict[str, Any]] = []
    paired_selected: list[dict[str, Any]] = []
    selection_discordances: list[dict[str, Any]] = []
    result_files: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    response_ids: set[str] = set()
    reclassified_length_limited_cells: list[str] = []

    for model, directory in MODELS.items():
        path = PILOT_DIR / directory / "rq3.json"
        direct_path = PILOT_DIR / directory / "direct.json"
        if not path.is_file():
            issues.append(f"missing result: {path.relative_to(ROOT)}")
            continue
        if not direct_path.is_file():
            issues.append(f"missing result: {direct_path.relative_to(ROOT)}")
            continue
        data = _read(path)
        direct_data = _read(direct_path)
        data["direct"] = list(direct_data.get("direct") or [])
        result_files[model] = {
            "matched_methods": {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "created_at": (data.get("metadata") or {}).get("created_at"),
            },
            "direct": {
                "path": str(direct_path.relative_to(ROOT)),
                "sha256": _sha256(direct_path),
                "created_at": (direct_data.get("metadata") or {}).get("created_at"),
            },
        }
        metadata = data.get("metadata") or {}
        if metadata.get("dataset_path") != str(PILOT_MANIFEST.relative_to(ROOT)):
            issues.append(f"{model}: wrong pilot manifest in metadata")
        if metadata.get("frozen_retrieval_index") != FROZEN_INDEX:
            issues.append(f"{model}: wrong frozen retrieval index")
        if metadata.get("temperature") != (None if model == "claude-sonnet-5" else 0.7):
            issues.append(f"{model}: unexpected temperature")
        if metadata.get("initial_candidate_budget") != 5:
            issues.append(f"{model}: candidate budget is not five")
        if metadata.get("max_repair_rounds") != 2:
            issues.append(f"{model}: max repair rounds is not two")
        direct_metadata = direct_data.get("metadata") or {}
        if direct_metadata.get("dataset_path") != str(PILOT_MANIFEST.relative_to(ROOT)):
            issues.append(f"{model}/Direct Generation: wrong pilot manifest")
        direct_flags = (direct_metadata.get("condition_feature_flags") or {}).get("direct") or {}
        if any(bool(value) for value in direct_flags.values()):
            issues.append(f"{model}/Direct Generation: mitigation feature enabled")

        by_method: dict[str, dict[str, dict[str, Any]]] = {}
        for method_key, method_label in METHODS.items():
            rows = list(data.get(method_key) or [])
            ids = [str(row.get("id")) for row in rows]
            if len(ids) != len(set(ids)):
                issues.append(f"{model}/{method_label}: duplicate task IDs")
            if set(ids) != expected_set:
                issues.append(f"{model}/{method_label}: task set differs from pilot manifest")
            by_method[method_key] = {str(row.get("id")): row for row in rows}
            usage = _usage(rows)
            amount, currency = _cost(
                model, usage["prompt_tokens"], usage["completion_tokens"]
            )
            summaries.append({
                "model": model,
                "method": method_label,
                "tasks": len(rows),
                "pass_at_1": _mean(_score(row, "pass@1") for row in rows),
                "pass_at_3": _mean(_score(row, "pass@3") for row in rows),
                "pass_at_5": _mean(_score(row, "pass@5") for row in rows),
                "selected_output_correctness": _mean(
                    float(bool(row.get("passed"))) for row in rows
                ),
                "mean_uncertainty": _mean(
                    float((row.get("u") or {}).get("aggregate")) for row in rows
                ),
                "mean_repair_rounds": _mean(
                    float(row.get("repair_rounds") or 0) for row in rows
                ),
                "mean_latency_seconds": _mean(
                    float(row.get("run_latency_s") or 0) for row in rows
                ),
                "paper_eligible": False,
            })
            resources.append({
                "model": model,
                "method": method_label,
                **usage,
                "estimated_cost": amount,
                "currency": currency,
            })

            for row in rows:
                prefix = f"{model}/{method_label}/{row.get('id')}"
                if "error" in row:
                    issues.append(f"{prefix}: run error={row['error']}")
                for field in (
                    "generated_samples",
                    "sample_correctness",
                    "effective_sample_correctness",
                ):
                    if len(row.get(field) or []) != 5:
                        issues.append(f"{prefix}: {field} count is not five")
                derived_integrity = _generation_integrity_valid(row)
                if not derived_integrity:
                    issues.append(f"{prefix}: generation integrity failed")
                elif (row.get("generation_integrity") or {}).get("valid") is not True:
                    reclassified_length_limited_cells.append(prefix)
                if row.get("correctness_mode") != "repository_tests":
                    issues.append(f"{prefix}: non-executable correctness mode")
                if method_key == "direct":
                    if row.get("fused_evidence_ids") or str(row.get("fused_evidence") or "").strip():
                        issues.append(f"{prefix}: Direct condition contains retrieved evidence")
                    if int(row.get("repair_rounds") or 0) != 0:
                        issues.append(f"{prefix}: Direct condition used repair")
                if row.get("passed") is not bool((row.get("test_report") or {}).get("passed")):
                    issues.append(f"{prefix}: selected test outcome mismatch")
                if len(row.get("repair_history") or []) > 2:
                    issues.append(f"{prefix}: more than two repair rounds")
                generation_metadata = row.get("generation_response_metadata") or []
                if len(generation_metadata) != 5:
                    issues.append(f"{prefix}: generation metadata count is not five")
                for item in generation_metadata:
                    if item.get("served_model") != model or not item.get("response_id"):
                        issues.append(f"{prefix}: invalid generation response provenance")
                        break
                audit = row.get("provider_response_audit") or []
                requests = int((row.get("llm_usage") or {}).get("requests") or 0)
                if len(audit) != requests:
                    issues.append(f"{prefix}: all-call audit count differs from requests")
                for item in audit:
                    response_id = str(item.get("response_id") or "")
                    if not response_id:
                        issues.append(f"{prefix}: missing all-call response ID")
                    elif response_id in response_ids:
                        issues.append(f"{prefix}: duplicate response ID {response_id}")
                    else:
                        response_ids.add(response_id)
                    if item.get("requested_model") != model or item.get("served_model") != model:
                        issues.append(f"{prefix}: requested/served model mismatch")

        opencoder = by_method.get("with") or {}
        for comparator_key in ("without", "rag_repair"):
            comparator = by_method.get(comparator_key) or {}
            if set(comparator) != set(opencoder):
                continue
            comparator_label = METHODS[comparator_key]
            for metric in ("pass@1", "pass@3", "pass@5"):
                comp = [_score(comparator[task_id], metric) for task_id in expected_ids]
                op = [_score(opencoder[task_id], metric) for task_id in expected_ids]
                low, high = _bootstrap_ci(comp, op)
                paired_passk.append({
                    "model": model,
                    "comparator": comparator_label,
                    "metric": metric,
                    "comparator_value": _mean(comp),
                    "opencoder_value": _mean(op),
                    "absolute_difference": _mean(op) - _mean(comp),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "opencoder_wins": sum(o > c for c, o in zip(comp, op)),
                    "opencoder_losses": sum(o < c for c, o in zip(comp, op)),
                    "ties": sum(o == c for c, o in zip(comp, op)),
                    "matched_tasks": len(expected_ids),
                    "paper_eligible": False,
                })

            comp_selected = [
                float(bool(comparator[task_id].get("passed")))
                for task_id in expected_ids
            ]
            op_selected = [
                float(bool(opencoder[task_id].get("passed")))
                for task_id in expected_ids
            ]
            wins = sum(o > c for c, o in zip(comp_selected, op_selected))
            losses = sum(o < c for c, o in zip(comp_selected, op_selected))
            low, high = _bootstrap_ci(comp_selected, op_selected)
            paired_selected.append({
                "model": model,
                "comparator": comparator_label,
                "comparator_correctness": _mean(comp_selected),
                "opencoder_correctness": _mean(op_selected),
                "absolute_difference": _mean(op_selected) - _mean(comp_selected),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "opencoder_wins": wins,
                "opencoder_losses": losses,
                "ties": len(expected_ids) - wins - losses,
                "mcnemar_exact_p": _mcnemar_exact(wins, losses),
                "matched_tasks": len(expected_ids),
                "paper_eligible": False,
            })
            for task_id, comp_value, op_value in zip(
                expected_ids, comp_selected, op_selected
            ):
                if comp_value == op_value:
                    continue
                comp_row = comparator[task_id]
                op_row = opencoder[task_id]
                selection_discordances.append({
                    "model": model,
                    "comparator": comparator_label,
                    "task_id": task_id,
                    "opencoder_outcome": "win" if op_value > comp_value else "loss",
                    "comparator_selected_correct": bool(comp_value),
                    "opencoder_selected_correct": bool(op_value),
                    "comparator_pass_at_5": _score(comp_row, "pass@5"),
                    "opencoder_pass_at_5": _score(op_row, "pass@5"),
                    "comparator_uncertainty": (comp_row.get("u") or {}).get("aggregate"),
                    "opencoder_uncertainty": (op_row.get("u") or {}).get("aggregate"),
                    "comparator_repair_rounds": comp_row.get("repair_rounds"),
                    "opencoder_repair_rounds": op_row.get("repair_rounds"),
                })

    _write_csv(PILOT_DIR / "summary.csv", summaries)
    _write_csv(PILOT_DIR / "resource_summary.csv", resources)
    _write_csv(PILOT_DIR / "paired_passk.csv", paired_passk)
    _write_csv(PILOT_DIR / "paired_selected_output.csv", paired_selected)
    _write_csv(PILOT_DIR / "selection_discordances.csv", selection_discordances)

    totals: dict[str, float] = {}
    for row in resources:
        currency = str(row["currency"])
        totals[currency] = totals.get(currency, 0.0) + float(row["estimated_cost"])
    integrity = {
        "passed": not issues,
        "paper_eligible": False,
        "purpose": "protocol and feasibility pilot; not confirmatory evidence",
        "pilot_manifest": str(PILOT_MANIFEST.relative_to(ROOT)),
        "pilot_manifest_sha256": _sha256(PILOT_MANIFEST),
        "task_count": len(expected_ids),
        "task_ids": expected_ids,
        "models": list(MODELS),
        "methods": list(METHODS.values()),
        "total_task_method_cells": sum(row["tasks"] for row in summaries),
        "audited_provider_responses": len(response_ids),
        "reclassified_length_limited_cells": reclassified_length_limited_cells,
        "estimated_pilot_spend": totals,
        "result_files": result_files,
        "issues": issues,
    }
    (PILOT_DIR / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Four-Family Pilot Memo",
        "",
        f"Status: **{'PASSED' if not issues else 'FAILED'}**.",
        "",
        "This is a frozen 10-task protocol/feasibility pilot. It is not eligible for paper-level effectiveness claims or parameter tuning.",
        "",
        "## Diagnostic Results",
        "",
        "| Model | Method | Pass@1 | Pass@3 | Pass@5 | Selected | Mean latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model']} | {row['method']} | {100*row['pass_at_1']:.1f} | "
            f"{100*row['pass_at_3']:.1f} | {100*row['pass_at_5']:.1f} | "
            f"{100*row['selected_output_correctness']:.1f} | {row['mean_latency_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "OpenCoderX is not uniformly stronger in this pilot. Several cells favor the ordinary verify/repair control, and all selected-output McNemar comparisons are underpowered at N=10. The pilot therefore supports pipeline feasibility and exposes selection/filtering cases for pre-registered diagnostic inspection; it does not support a superiority claim.",
        "",
        f"All {len(response_ids)} provider calls have unique response IDs and matched requested/served model identities. Estimated pilot spend is {totals}.",
        "",
    ])
    (PILOT_DIR / "results_memo.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
