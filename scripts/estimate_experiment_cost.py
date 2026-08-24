#!/usr/bin/env python3
"""Estimate and enforce the configured cost gate before an API batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoderx.costs import estimate_cost  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tosem/campaign.yaml")
    parser.add_argument("--provider", default="zhizengzeng")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--methods", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--input-tokens-per-call", type=int, required=True)
    parser.add_argument("--output-tokens-per-call", type=int, required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    controls = config["cost_controls"]
    model_pricing = (controls.get("gateway_pricing") or {}).get(args.model) or {}
    input_price = model_pricing.get(
        "input_per_million", controls.get("gateway_input_usd_per_million")
    )
    output_price = model_pricing.get(
        "output_per_million", controls.get("gateway_output_usd_per_million")
    )
    currency = str(model_pricing.get("currency") or "USD").upper()
    estimate = estimate_cost(
        provider=args.provider,
        model=args.model,
        dataset=args.dataset,
        tasks=args.tasks,
        methods=args.methods,
        samples=args.samples,
        input_tokens_per_call=args.input_tokens_per_call,
        output_tokens_per_call=args.output_tokens_per_call,
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
        pricing_currency=currency,
    )
    payload = estimate.to_dict()
    limits = controls.get("per_run_limits") or {}
    campaign_limits = controls.get("campaign_limits") or {}
    per_run_limit = limits.get(currency)
    if per_run_limit is None and currency == "USD":
        per_run_limit = controls.get("per_run_limit_usd")
    campaign_limit = campaign_limits.get(currency)
    if campaign_limit is None and currency == "USD":
        campaign_limit = controls.get("campaign_limit_usd")
    payload["per_run_limit"] = per_run_limit
    payload["campaign_limit"] = campaign_limit
    payload["pricing_basis"] = model_pricing.get("basis", "legacy gateway rate")
    payload["pricing_source"] = model_pricing.get("source")
    payload["allowed"] = (
        estimate.estimated_cost is not None
        and per_run_limit is not None
        and estimate.estimated_cost <= float(per_run_limit)
    )
    print(json.dumps(payload, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
