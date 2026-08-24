#!/usr/bin/env python3
"""Run the frozen CrossCodeEval-100 campaign with retries and campaign caps."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data/manifests/crosscodeeval_opencoderx_100_batches_v1"
RESULT_DIR = ROOT / "results/tosem/crosscodeeval_confirmatory"
PROTOCOL = RESULT_DIR / "protocol_freeze.json"
CAMPAIGN_CONFIG = ROOT / "configs/tosem/campaign.yaml"
BASE_STATUS = ROOT / "results/tosem/confirmatory/campaign_status.json"
MODELS = {
    "gpt-4o-mini": ("gpt4o_mini", "configs/tosem/models/gpt4o_mini.yaml"),
    "gemini-2.5-flash": ("gemini_2_5_flash", "configs/tosem/models/gemini_2_5_flash.yaml"),
    "claude-sonnet-5": ("claude_sonnet_5", "configs/tosem/models/claude_sonnet_5.yaml"),
    "qwen3-coder-plus": ("qwen3_coder_plus", "configs/tosem/models/qwen3_coder_plus.yaml"),
}
MAX_ATTEMPTS = 5
RETRY_DELAYS = (30, 60, 120, 180)
PRINT_LOCK = threading.Lock()
COST_LOCK = threading.Lock()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_ids(path: Path) -> set[str]:
    return {str(json.loads(line)["task_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _complete(path: Path, expected_ids: set[str]) -> bool:
    if not path.is_file():
        return False
    rows = list(_read(path).get("rows") or [])
    expected = {(task_id, method) for task_id in expected_ids for method in ("direct", "context_rag")}
    actual = {
        (str(row.get("task_id")), str(row.get("method")))
        for row in rows
        if row.get("status") == "COMPLETED_NATIVE_METRICS" and not row.get("error")
    }
    return actual == expected and len(rows) == len(expected)


def _verify_protocol() -> None:
    protocol = _read(PROTOCOL)
    if protocol.get("status") != "FROZEN_APPROVED" or protocol.get("paper_eligible") is not True:
        raise RuntimeError("CrossCodeEval confirmatory protocol is not approved")
    import hashlib
    for relative, expected in (protocol.get("file_sha256") or {}).items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen CrossCodeEval file changed: {relative}")


def _result_spend() -> dict[str, float]:
    config = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    pricing = config["cost_controls"]["gateway_pricing"]
    totals: dict[str, float] = defaultdict(float)
    for path in RESULT_DIR.glob("*/batch_*.json"):
        data = _read(path)
        model = str((data.get("metadata") or {}).get("model") or "")
        if model not in pricing:
            continue
        prompt = completion = 0
        for row in data.get("rows") or []:
            usage = row.get("llm_usage") or {}
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
        rates = pricing[model]
        currency = str(rates["currency"])
        totals[currency] += (
            prompt * float(rates["input_per_million"])
            + completion * float(rates["output_per_million"])
        ) / 1_000_000
    return dict(totals)


def _campaign_spend() -> dict[str, float]:
    base = _read(BASE_STATUS)["campaign_spend"]
    increment = _result_spend()
    return {
        currency: float(base.get(currency, 0.0)) + float(increment.get(currency, 0.0))
        for currency in set(base) | set(increment)
    }


def _enforce_cap() -> None:
    with COST_LOCK:
        config = yaml.safe_load(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
        limits = config["cost_controls"]["campaign_limits"]
        for currency, amount in _campaign_spend().items():
            if amount > float(limits[currency]):
                raise RuntimeError(f"CAMPAIGN_COST_LIMIT_EXCEEDED:{currency}:{amount}>{limits[currency]}")


def _run_batch(model: str, directory: str, config: str, batch: Path) -> str:
    output = RESULT_DIR / directory / f"{batch.stem}.json"
    log = RESULT_DIR / "logs" / directory / f"{batch.stem}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_ids(batch)
    if _complete(output, expected):
        return f"{model}/{batch.stem}: cached-complete"
    command = [
        sys.executable,
        "experiments/run_crosscodeeval.py",
        "--config", config,
        "--manifest", str(batch.relative_to(ROOT)),
        "--method", "all",
        "--out", str(output.relative_to(ROOT)),
        "--cost-config", "configs/tosem/campaign.yaml",
        "--experiment-version", "crosscodeeval_native_confirmatory_v1",
        "--dataset-version", "crosscodeeval_opencoderx_100_v1",
        "--paper-eligible",
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _enforce_cap()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"ATTEMPT {attempt}/{MAX_ATTEMPTS}\nCOMMAND {' '.join(command)}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0 and _complete(output, expected):
            _enforce_cap()
            return f"{model}/{batch.stem}: completed"
        if completed.returncode == 3:
            raise RuntimeError(f"per-run cost cap reached: {model}/{batch.stem}")
        if attempt < MAX_ATTEMPTS:
            delay = RETRY_DELAYS[attempt - 1]
            with PRINT_LOCK:
                print(f"{model}/{batch.stem}: incomplete attempt {attempt}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"incomplete after {MAX_ATTEMPTS} attempts: {model}/{batch.stem}")


def _run_model(model: str) -> list[str]:
    directory, config = MODELS[model]
    messages = []
    for batch in sorted(BATCH_DIR.glob("batch_*.jsonl")):
        message = _run_batch(model, directory, config, batch)
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
    if len(batches) != 4:
        raise SystemExit("expected four frozen CrossCodeEval batches")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_model, model): model for model in args.models}
        for future in as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append({"model": model, "error": str(exc)})
                print(f"{model}: FAILED: {exc}", flush=True)
    status = {
        "status": "COMPLETED" if not failures else "FAILED_OR_PARTIAL",
        "models": args.models,
        "workers": args.workers,
        "crosscodeeval_spend": _result_spend(),
        "campaign_spend": _campaign_spend(),
        "failures": failures,
    }
    (RESULT_DIR / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
