#!/usr/bin/env python3
"""Run one bounded gateway preflight per frozen model family."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.llm.client import LLMClient, _load_dotenv  # noqa: E402


HERE = ROOT / "human_study/gateway_agent_v1"
OUT = ROOT / "results/agent_gateway_v1/preflight"


def redacted(value: str) -> str:
    value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-...REDACTED", value)
    return value.replace(str(ROOT), "<WORKSPACE>")


def attempt(model_id: str, temperature: float | None, timeout: int) -> dict[str, Any]:
    client = LLMClient(
        backend="zhizengzeng",
        model=model_id,
        temperature=temperature,
        max_tokens=32,
        timeout=timeout,
    )
    started = time.perf_counter()
    response = client.complete_one(
        "Reply with exactly: OpenCoder agent ready.",
        system="You are a concise connectivity checker.",
        return_logprobs=False,
    )
    latency = time.perf_counter() - started
    metadata = (response.raw or {}).get("_response_metadata") or {}
    return {
        "status": "COMPLETED",
        "requested_model": model_id,
        "served_model": metadata.get("served_model"),
        "response_id": metadata.get("response_id"),
        "response_matches": response.text.strip() == "OpenCoder agent ready.",
        "latency_seconds": latency,
        "usage": client.usage_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    manifest = json.loads((HERE / "model_manifest.json").read_text(encoding="utf-8"))
    if not args.execute_paid:
        print(json.dumps({"status": "DRY_RUN", "planned_calls": len(manifest["models"])}, indent=2))
        return 0
    _load_dotenv(str(ROOT / ".env"))
    OUT.mkdir(parents=True, exist_ok=True)
    resolved: list[dict[str, Any]] = []
    for item in manifest["models"]:
        result: dict[str, Any] | None = None
        failures: list[dict[str, str]] = []
        for candidate in (item["model_id"], item["alternate_model_id"]):
            try:
                result = attempt(candidate, item["temperature"], args.timeout)
                break
            except Exception as exc:
                failures.append({"model_id": candidate, "error": redacted(f"{type(exc).__name__}: {exc}")[:1000]})
        row = dict(item)
        row["resolved_model_id"] = None if result is None else result["requested_model"]
        row["preflight_status"] = "FAILED" if result is None else "COMPLETED"
        row["preflight_failures"] = failures
        row["preflight"] = result
        row["preflight_at_utc"] = datetime.now(timezone.utc).isoformat()
        resolved.append(row)
        path = OUT / f"{item['agent_id']}.json"
        path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(f"{item['agent_id']} {item['family']}: {row['preflight_status']} {row['resolved_model_id']}", flush=True)
    payload = {
        "protocol": manifest["protocol"],
        "study_mode": manifest["study_mode"],
        "models": resolved,
        "all_models_ready": all(row["preflight_status"] == "COMPLETED" for row in resolved),
        "warning": manifest["warning"],
    }
    output = ROOT / "results/agent_gateway_v1/resolved_model_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_models_ready": payload["all_models_ready"], "output": str(output.relative_to(ROOT))}, indent=2))
    return 0 if payload["all_models_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
