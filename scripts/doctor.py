#!/usr/bin/env python3
"""Readiness audit for the OpenCoderX TOSEM campaign."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.llm.client import _load_dotenv  # noqa: E402
from opencoderx.providers import FROZEN_MODELS  # noqa: E402


def check(
    name: str,
    passed: bool,
    detail: str,
    *,
    required_for_smoke: bool = False,
    required_for_campaign: bool = True,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "required_for_smoke": required_for_smoke,
        "required_for_campaign": required_for_campaign,
    }


def gateway_catalog() -> tuple[set[str], str]:
    _load_dotenv(str(ROOT / ".env"))
    base = os.environ.get("OPENCODER_LLM_BASE_URL", "https://api.zhizengzeng.com/v1").rstrip("/")
    key = os.environ.get("OPENCODER_LLM_API_KEY", "")
    if not key:
        return set(), "OPENCODER_LLM_API_KEY is missing"
    response = requests.get(
        base + "/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        verify=os.environ.get("OPENCODER_LLM_VERIFY_SSL", "1").lower() not in {"0", "false", "no"},
    )
    response.raise_for_status()
    models = {
        str(item.get("id"))
        for item in response.json().get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    return models, f"catalog at {urlparse(base).hostname} returned {len(models)} models"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    parser.add_argument("--offline", action="store_true", help="skip the gateway catalog request")
    args = parser.parse_args()
    checks: list[dict[str, object]] = []

    models: set[str] = set()
    catalog_detail = "gateway check skipped"
    if not args.offline:
        try:
            models, catalog_detail = gateway_catalog()
        except Exception as exc:  # readiness command must report rather than crash
            catalog_detail = f"{type(exc).__name__}: {exc}"
    for label, key in (("GPT API", "gpt"), ("Gemini API", "gemini"), ("Claude API", "claude"), ("Qwen API", "qwen")):
        model = FROZEN_MODELS[key].model_id
        checks.append(check(
            label,
            model in models,
            f"{model}; {catalog_detail}",
            required_for_smoke=True,
        ))

    assets = {
        "RepoExec": ROOT / "input/repoexec_python_string_utils_inline20.jsonl",
        "CoderEval": ROOT / "input/codereval_neo4j_executable19.jsonl",
        "ExecRepoBench-120": ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl",
        "CrossCodeEval-100": ROOT / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl",
        "Multi-SWE-bench-35": ROOT / "data/manifests/multi_swe_bench_flash_opencoderx_35_v1.jsonl",
        "repository snapshots": ROOT / ".benchmarks/execrepobench/repos",
        "retrieval indexes": ROOT / "data/indexes/execrepobench_120_repository_knowledge_v1.jsonl",
        "test harnesses": ROOT / "opencoder/phase5_verify",
        "result directory": ROOT / "results",
        "manuscript": ROOT / "tosem/main.tex",
    }
    smoke_assets = {
        "ExecRepoBench-120",
        "repository snapshots",
        "retrieval indexes",
        "test harnesses",
        "result directory",
    }
    for name, path in assets.items():
        checks.append(check(
            name,
            path.exists(),
            str(path.relative_to(ROOT)),
            required_for_smoke=name in smoke_assets,
        ))

    checks.append(check(
        "Docker",
        shutil.which("docker") is not None,
        shutil.which("docker") or "not installed; required for Multi-SWE-bench only",
    ))
    checks.append(check(
        "execution environments",
        (ROOT / ".venv/bin/python").exists(),
        ".venv plus isolated ExecRepoBench repository environments",
        required_for_smoke=True,
    ))
    free = shutil.disk_usage(ROOT).free
    checks.append(check(
        "disk capacity",
        free >= 20 * 1024**3,
        f"{free / 1024**3:.1f} GiB free",
        required_for_smoke=True,
    ))
    checks.append(check(
        "GPU",
        shutil.which("nvidia-smi") is not None,
        "NVIDIA GPU not visible; not required for gateway-hosted model calls",
        required_for_campaign=False,
    ))

    ready_for_smoke = all(
        bool(item["passed"])
        for item in checks
        if item["required_for_smoke"]
    )
    ready_for_campaign = all(
        bool(item["passed"])
        for item in checks
        if item["required_for_campaign"]
    )
    for item in checks:
        marker = "x" if item["passed"] else " "
        print(f"[{marker}] {item['name']}: {item['detail']}")
    print("READY FOR SMOKE" if ready_for_smoke else "NOT READY FOR SMOKE")
    print("READY FOR FULL CAMPAIGN" if ready_for_campaign else "NOT READY FOR FULL CAMPAIGN")

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ready_for_smoke": ready_for_smoke,
            "ready_for_campaign": ready_for_campaign,
            "checks": checks,
        }, indent=2) + "\n", encoding="utf-8")
    return 0 if ready_for_smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
