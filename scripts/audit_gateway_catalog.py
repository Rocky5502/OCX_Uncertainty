#!/usr/bin/env python3
"""Audit gateway model IDs without persisting credentials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from opencoder.llm.client import _load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERMS = (
    "gpt", "gemini", "claude", "deepseek", "grok", "kimi", "moonshot",
    "llama", "meta", "qwen", "doubao", "ernie", "wenxin", "perplexity",
    "manus",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/tosem/gateway_catalog_20260821.json",
    )
    parser.add_argument(
        "--matches-out",
        type=Path,
        default=ROOT / "results/tosem/gateway_catalog_matches_20260821.csv",
    )
    parser.add_argument("--terms", nargs="*", default=list(DEFAULT_TERMS))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    _load_dotenv(str(ROOT / ".env"))
    base = os.environ.get(
        "OPENCODER_LLM_BASE_URL", "https://api.zhizengzeng.com/v1"
    ).rstrip("/")
    key = os.environ.get("OPENCODER_LLM_API_KEY", "")
    if not key:
        raise RuntimeError("OPENCODER_LLM_API_KEY is missing")
    response = requests.get(
        base + "/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=args.timeout,
        verify=os.environ.get("OPENCODER_LLM_VERIFY_SSL", "1").lower()
        not in {"0", "false", "no"},
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    ids = sorted(
        {
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        },
        key=str.lower,
    )
    terms = [term.strip().lower() for term in args.terms if term.strip()]
    matches = [
        {"term": term, "model_id": model_id}
        for term in terms
        for model_id in ids
        if re.search(re.escape(term), model_id, flags=re.IGNORECASE)
    ]

    host = urlparse(base).hostname or "unknown"
    artifact = {
        "audit_type": "authenticated_gateway_catalog",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "gateway_host_sha256": hashlib.sha256(host.encode("utf-8")).hexdigest(),
        "model_count": len(ids),
        "model_ids": ids,
        "search_terms": terms,
        "credential_persisted": False,
        "note": (
            "Catalog presence does not prove successful invocation or upstream "
            "product equivalence; bounded preflight calls are required."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.matches_out.parent.mkdir(parents=True, exist_ok=True)
    with args.matches_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["term", "model_id"])
        writer.writeheader()
        writer.writerows(matches)
    counts = {term: sum(row["term"] == term for row in matches) for term in terms}
    print(json.dumps({"model_count": len(ids), "match_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
