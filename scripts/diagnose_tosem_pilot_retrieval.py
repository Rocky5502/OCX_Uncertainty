#!/usr/bin/env python3
"""Diagnose retrieval differences in the non-confirmatory ExecRepoBench pilot."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "results/tosem/pilot"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}
COMPARATORS = {
    "without": "Standard RAG",
    "rag_repair": "RAG + Verify/Repair",
}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pass_at_five(row: dict[str, Any]) -> float:
    values = list(row.get("effective_sample_correctness") or row.get("sample_correctness") or [])
    return float(any(bool(value) for value in values))


def _evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("fused_evidence_ids") or [])


def _evidence_ids(row: dict[str, Any]) -> set[str]:
    return {str(item.get("id")) for item in _evidence(row) if item.get("id")}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _source_counts(row: dict[str, Any]) -> Counter[str]:
    return Counter(str(item.get("source") or "unknown") for item in _evidence(row))


def _score_summary(row: dict[str, Any]) -> tuple[float, float, float]:
    values = [float(item.get("score") or 0.0) for item in _evidence(row)]
    return (
        min(values) if values else 0.0,
        _mean(values),
        max(values) if values else 0.0,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for model, directory in MODELS.items():
        data = json.loads((PILOT_DIR / directory / "rq3.json").read_text(encoding="utf-8"))
        opencoder = {str(row["id"]): row for row in data["with"]}
        for comparator_key, comparator_label in COMPARATORS.items():
            comparator = {str(row["id"]): row for row in data[comparator_key]}
            if set(opencoder) != set(comparator):
                raise RuntimeError(f"unmatched task sets for {model}/{comparator_label}")
            for task_id in sorted(opencoder):
                op = opencoder[task_id]
                comp = comparator[task_id]
                op_selected = bool(op.get("passed"))
                comp_selected = bool(comp.get("passed"))
                op_pass5 = _pass_at_five(op)
                comp_pass5 = _pass_at_five(comp)
                if op_selected and not comp_selected:
                    selected_outcome = "win"
                elif comp_selected and not op_selected:
                    selected_outcome = "loss"
                else:
                    selected_outcome = "tie"
                if op_pass5 < comp_pass5:
                    failure_mode = "candidate_availability_loss"
                elif not op_selected and comp_selected and op_pass5:
                    failure_mode = "selection_loss"
                elif op_selected and not comp_selected:
                    failure_mode = "opencoder_selected_win"
                else:
                    failure_mode = "none"
                op_counts = _source_counts(op)
                comp_counts = _source_counts(comp)
                op_min, op_mean, op_max = _score_summary(op)
                comp_min, comp_mean, comp_max = _score_summary(comp)
                rows.append({
                    "model": model,
                    "comparator": comparator_label,
                    "task_id": task_id,
                    "selected_outcome": selected_outcome,
                    "failure_mode": failure_mode,
                    "opencoder_selected_correct": op_selected,
                    "comparator_selected_correct": comp_selected,
                    "opencoder_pass_at_5": op_pass5,
                    "comparator_pass_at_5": comp_pass5,
                    "opencoder_steps": len(op.get("per_step") or []),
                    "comparator_steps": len(comp.get("per_step") or []),
                    "evidence_jaccard": _jaccard(_evidence_ids(op), _evidence_ids(comp)),
                    "opencoder_api_items": op_counts.get("api", 0),
                    "opencoder_context_items": op_counts.get("context", 0),
                    "opencoder_similar_items": op_counts.get("similar_code", 0),
                    "comparator_api_items": comp_counts.get("api", 0),
                    "comparator_context_items": comp_counts.get("context", 0),
                    "comparator_similar_items": comp_counts.get("similar_code", 0),
                    "opencoder_min_fusion_score": op_min,
                    "opencoder_mean_fusion_score": op_mean,
                    "opencoder_max_fusion_score": op_max,
                    "comparator_min_fusion_score": comp_min,
                    "comparator_mean_fusion_score": comp_mean,
                    "comparator_max_fusion_score": comp_max,
                    "opencoder_uncertainty": float((op.get("u") or {}).get("aggregate") or 0.0),
                    "comparator_uncertainty": float((comp.get("u") or {}).get("aggregate") or 0.0),
                    "opencoder_repair_rounds": int(op.get("repair_rounds") or 0),
                    "comparator_repair_rounds": int(comp.get("repair_rounds") or 0),
                })

    out_csv = PILOT_DIR / "retrieval_diagnostics.csv"
    _write_csv(out_csv, rows)
    discordant = [row for row in rows if row["selected_outcome"] != "tie"]
    availability_losses = [row for row in rows if row["failure_mode"] == "candidate_availability_loss"]
    selection_losses = [row for row in rows if row["failure_mode"] == "selection_loss"]
    wins = [row for row in rows if row["selected_outcome"] == "win"]
    losses = [row for row in rows if row["selected_outcome"] == "loss"]
    memo = [
        "# Pilot Retrieval Diagnostic",
        "",
        "This diagnostic uses only the frozen 10-task development pilot. It is not a confirmatory result and must not be reported as such.",
        "",
        f"- Matched OpenCoderX comparisons: {len(rows)}",
        f"- Selected-output discordances: {len(discordant)} ({len(wins)} wins, {len(losses)} losses)",
        f"- Candidate-availability losses: {len(availability_losses)}",
        f"- Selection-only losses: {len(selection_losses)}",
        f"- Mean evidence Jaccard on discordances: {_mean(float(row['evidence_jaccard']) for row in discordant):.3f}",
        f"- Mean OpenCoderX API share on discordances: {_mean(float(row['opencoder_api_items']) / 10.0 for row in discordant):.3f}",
        f"- Mean comparator API share on discordances: {_mean(float(row['comparator_api_items']) / 10.0 for row in discordant):.3f}",
        "",
        "## Interpretation",
        "",
        "A selected-output loss accompanied by lower Pass@5 is a candidate-availability failure, not merely a candidate-selection error. The pilot losses are concentrated in this category. OpenCoderX uses multiple narrow step queries, while the matched comparators retain whole-task retrieval. On affected tasks, the resulting evidence sets have low overlap and can become dominated by one source even when its fused scores are weak.",
        "",
        "The pre-freeze correction should therefore preserve a whole-task retrieval anchor and prevent one evidence source from consuming the complete fused budget. It should not change executable tests, candidate count, generation budget, repair budget, or any confirmatory task.",
        "",
        "## Discordant Cells",
        "",
    ]
    for row in discordant:
        memo.append(
            f"- {row['model']} vs {row['comparator']}, `{row['task_id']}`: "
            f"{row['selected_outcome']}; {row['failure_mode']}; evidence Jaccard "
            f"{float(row['evidence_jaccard']):.3f}; OpenCoderX source counts "
            f"API/context/similar={row['opencoder_api_items']}/{row['opencoder_context_items']}/{row['opencoder_similar_items']}."
        )
    (PILOT_DIR / "retrieval_diagnostic_memo.md").write_text(
        "\n".join(memo) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "comparisons": len(rows),
        "discordances": len(discordant),
        "wins": len(wins),
        "losses": len(losses),
        "candidate_availability_losses": len(availability_losses),
        "selection_losses": len(selection_losses),
        "csv": str(out_csv.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
