#!/usr/bin/env python3
"""Run the frozen Tier-A campaign in resumable, cost-guarded batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data/manifests/execrepobench_opencoderx_120_batches_v1"
RESULT_DIR = ROOT / "results/tosem/confirmatory"
SUPERSEDED_DIR = ROOT / "results/tosem/confirmatory_superseded_pre_repair_budget"
PROTOCOL = ROOT / "results/tosem/protocol_freeze.json"
CONFIG = ROOT / "configs/tosem/campaign.yaml"
MODELS = {
    "gpt-4o-mini": ("gpt4o_mini", "configs/tosem/models/gpt4o_mini.yaml"),
    "gemini-2.5-flash": ("gemini_2_5_flash", "configs/tosem/models/gemini_2_5_flash.yaml"),
    "claude-sonnet-5": ("claude_sonnet_5", "configs/tosem/models/claude_sonnet_5.yaml"),
    "qwen3-coder-plus": ("qwen3_coder_plus", "configs/tosem/models/qwen3_coder_plus.yaml"),
}
METHODS = {
    "direct": "direct",
    "baseline_rag": "without",
    "rag_verify_repair": "rag_repair",
    "opencoder": "with",
}
PRINT_LOCK = threading.Lock()
COST_LOCK = threading.Lock()
MAX_JOB_ATTEMPTS = 5
RETRY_DELAYS_S = (30, 60, 120, 180)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_protocol() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_APPROVED":
        raise RuntimeError("unexpected scientific protocol status")
    for relative, expected in (protocol.get("file_sha256") or {}).items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen scientific file changed: {relative}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_ids(batch_path: Path) -> set[str]:
    return {
        str(json.loads(line)["task_id"])
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _complete(path: Path, result_key: str, expected: set[str]) -> bool:
    if not path.is_file():
        return False
    rows = list(_read_json(path).get(result_key) or [])
    ids = {str(row.get("id")) for row in rows if not row.get("error")}
    return ids == expected and len(rows) == len(expected)


def _usage_from_result(path: Path) -> tuple[str | None, dict[str, int]]:
    data = _read_json(path)
    model = (data.get("metadata") or {}).get("model")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {field: 0 for field in fields}
    for result_key in METHODS.values():
        for row in data.get(result_key) or []:
            row_usage = row.get("llm_usage") or {}
            for field in fields:
                usage[field] += int(row_usage.get(field) or 0)
    return model, usage


def _campaign_spend() -> dict[str, float]:
    projection = _read_json(ROOT / "results/tosem/cost/confirmatory_projection.json")
    totals = {
        currency: float(values["measured_pilot_spend"])
        for currency, values in projection["currencies"].items()
    }
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pricing = config["cost_controls"]["gateway_pricing"]
    result_paths = [
        *RESULT_DIR.glob("*/**/batch_*.json"),
        *SUPERSEDED_DIR.glob("*/**/batch_*.json"),
    ]
    for path in result_paths:
        model, usage = _usage_from_result(path)
        if model not in pricing:
            continue
        rates = pricing[model]
        currency = str(rates["currency"])
        amount = (
            usage["prompt_tokens"] * float(rates["input_per_million"])
            + usage["completion_tokens"] * float(rates["output_per_million"])
        ) / 1_000_000
        totals[currency] = totals.get(currency, 0.0) + amount
    return totals


def _enforce_campaign_cap() -> None:
    with COST_LOCK:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        limits = config["cost_controls"]["campaign_limits"]
        totals = _campaign_spend()
        for currency, amount in totals.items():
            if amount > float(limits[currency]):
                raise RuntimeError(
                    f"CAMPAIGN_COST_LIMIT_EXCEEDED:{currency}:{amount:.6f}>{limits[currency]}"
                )


def _run_job(
    model: str,
    directory: str,
    config_path: str,
    method: str,
    batch_path: Path,
) -> str:
    result_key = METHODS[method]
    batch_name = batch_path.stem
    output = RESULT_DIR / directory / method / f"{batch_name}.json"
    log = RESULT_DIR / "logs" / directory / method / f"{batch_name}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_ids(batch_path)
    if _complete(output, result_key, expected):
        return f"{model}/{method}/{batch_name}: cached-complete"
    _enforce_campaign_cap()
    command = [
        sys.executable,
        "experiments/run_rq3.py",
        "--config", config_path,
        "--benchmark", "execrepobench",
        "--benchmark-path", str(batch_path.relative_to(ROOT)),
        "--method", method,
        "--out", str(output.relative_to(ROOT)),
        "--frozen-index", "data/indexes/execrepobench_120_repository_knowledge_v1.jsonl",
        "--cost-config", "configs/tosem/campaign.yaml",
    ]
    for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"ATTEMPT {attempt}/{MAX_JOB_ATTEMPTS}\n"
                + "COMMAND " + " ".join(command) + "\n"
            )
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 3:
            raise RuntimeError(
                f"per-run cost limit reached: {model}/{method}/{batch_name}; "
                f"see {log.relative_to(ROOT)}"
            )
        if completed.returncode == 0 and _complete(output, result_key, expected):
            _enforce_campaign_cap()
            return f"{model}/{method}/{batch_name}: completed"
        if attempt < MAX_JOB_ATTEMPTS:
            delay = RETRY_DELAYS_S[attempt - 1]
            with PRINT_LOCK:
                print(
                    f"{model}/{method}/{batch_name}: incomplete attempt {attempt}; "
                    f"retrying in {delay}s",
                    flush=True,
                )
            time.sleep(delay)
    raise RuntimeError(
        f"job incomplete after {MAX_JOB_ATTEMPTS} attempts: "
        f"{model}/{method}/{batch_name}; see {log.relative_to(ROOT)}"
    )


def _run_model(model: str) -> list[str]:
    directory, config_path = MODELS[model]
    messages = []
    for batch_path in sorted(BATCH_DIR.glob("batch_*.jsonl")):
        for method in METHODS:
            message = _run_job(model, directory, config_path, method, batch_path)
            messages.append(message)
            with PRINT_LOCK:
                print(message, flush=True)
    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    _verify_protocol()
    batches = sorted(BATCH_DIR.glob("batch_*.jsonl"))
    if len(batches) != 12:
        raise SystemExit("expected 12 frozen confirmatory batches")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = RESULT_DIR / "campaign_status.json"
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_model, model): model for model in args.models}
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append({"model": model, "error": str(exc)})
                with PRINT_LOCK:
                    print(f"{model}: FAILED: {exc}", flush=True)
    payload = {
        "status": "COMPLETED" if not failures else "FAILED_OR_PARTIAL",
        "models": args.models,
        "workers": args.workers,
        "campaign_spend": _campaign_spend(),
        "failures": failures,
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
