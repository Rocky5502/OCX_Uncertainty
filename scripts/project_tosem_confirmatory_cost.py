#!/usr/bin/env python3
"""Project the frozen Tier-A cost from measured 10-task pilot usage."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "results/tosem/pilot"
OUT_DIR = ROOT / "results/tosem/cost"
MODELS = {
    "gpt-4o-mini": "gpt4o_mini",
    "gemini-2.5-flash": "gemini_2_5_flash",
    "claude-sonnet-5": "claude_sonnet_5",
    "qwen3-coder-plus": "qwen3_coder_plus",
}


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
    return {
        field: sum(int((row.get("llm_usage") or {}).get(field) or 0) for row in rows)
        for field in fields
    }


def _cost(model: str, usage: dict[str, int], pricing: dict[str, Any]) -> tuple[float, str]:
    model_pricing = pricing[model]
    amount = (
        usage["prompt_tokens"] * float(model_pricing["input_per_million"])
        + usage["completion_tokens"] * float(model_pricing["output_per_million"])
    ) / 1_000_000
    return amount, str(model_pricing["currency"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config = yaml.safe_load((ROOT / "configs/tosem/campaign.yaml").read_text(encoding="utf-8"))
    controls = config["cost_controls"]
    pricing = controls["gateway_pricing"]
    original_resources = list(csv.DictReader((PILOT_DIR / "resource_summary.csv").open()))
    measured_pilot_totals: dict[str, float] = defaultdict(float)
    for row in original_resources:
        measured_pilot_totals[row["currency"]] += float(row["estimated_cost"])
    cross_resources = list(csv.DictReader(
        (ROOT / "results/tosem/crosscodeeval_pilot/resource_summary.csv").open()
    ))
    for row in cross_resources:
        measured_pilot_totals[row["currency"]] += float(row["estimated_cost"])

    rows: list[dict[str, Any]] = []
    for model, directory in MODELS.items():
        base_rows = [row for row in original_resources if row["model"] == model]
        for row in base_rows:
            if row["method"] == "OpenCoderX":
                continue
            amount = float(row["estimated_cost"])
            rows.append({
                "model": model,
                "method": row["method"],
                "measured_tasks": 10,
                "measured_requests": int(row["requests"]),
                "measured_prompt_tokens": int(row["prompt_tokens"]),
                "measured_completion_tokens": int(row["completion_tokens"]),
                "measured_cost": amount,
                "currency": row["currency"],
                "projected_tasks": 120,
                "projected_cost": amount * 12.0,
                "projection_basis": "linear scaling of frozen 10-task pilot usage",
            })
        corrected = json.loads(
            (PILOT_DIR / directory / "opencoder_anchor_repilot.json").read_text(encoding="utf-8")
        )
        usage = _usage(list(corrected.get("with") or []))
        amount, currency = _cost(model, usage, pricing)
        measured_pilot_totals[currency] += amount
        rows.append({
            "model": model,
            "method": "OpenCoderX",
            "measured_tasks": 10,
            "measured_requests": usage["requests"],
            "measured_prompt_tokens": usage["prompt_tokens"],
            "measured_completion_tokens": usage["completion_tokens"],
            "measured_cost": amount,
            "currency": currency,
            "projected_tasks": 120,
            "projected_cost": amount * 12.0,
            "projection_basis": "linear scaling of accepted corrected 10-task re-pilot usage",
        })

    projected: dict[str, float] = defaultdict(float)
    for row in rows:
        projected[row["currency"]] += float(row["projected_cost"])
    campaign_limits = controls["campaign_limits"]
    status = {
        currency: {
            "measured_pilot_spend": amount,
            "projected_confirmatory_spend": projected.get(currency, 0.0),
            "projected_total_including_pilots": amount + projected.get(currency, 0.0),
            "campaign_limit": float(campaign_limits[currency]),
            "allowed": amount + projected.get(currency, 0.0) <= float(campaign_limits[currency]),
        }
        for currency, amount in measured_pilot_totals.items()
    }
    payload = {
        "status": "COST_BLOCKED" if not all(item["allowed"] for item in status.values()) else "ALLOWED",
        "paper_eligible": False,
        "projection": "measured frozen pilot usage scaled from 10 to 120 tasks",
        "currencies": status,
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "confirmatory_projection.csv", rows)
    (OUT_DIR / "confirmatory_projection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "ALLOWED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
