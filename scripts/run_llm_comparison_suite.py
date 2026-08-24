"""Run OpenCoderGPT and OpenCoderGemini, then build the Overleaf table."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_dotenv_into_env(env: dict) -> dict:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return env
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value
    return env


def _run(label: str, cmd: List[str], env: dict) -> None:
    print(f"\n==> {label}", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--describe-limit", type=int, default=100)
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--api-preflight-retries", type=int, default=60)
    ap.add_argument("--api-preflight-sleep", type=int, default=60)
    ap.add_argument("--api-preflight-timeout", type=int, default=20)
    ap.add_argument("--skip-gpt", action="store_true")
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--rq1-only", action="store_true")
    ap.add_argument("--rq2-only", action="store_true")
    ap.add_argument("--allow-offline-table", action="store_true")
    args = ap.parse_args()

    if args.skip_gpt and args.skip_gemini:
        raise SystemExit("At least one backend must run.")
    if args.rq1_only and args.rq2_only:
        raise SystemExit("Use at most one of --rq1-only or --rq2-only.")

    py = sys.executable
    env = _load_dotenv_into_env(os.environ.copy())
    out_root = Path(args.out_root) if args.out_root else ROOT / "results" / f"llm_comparison_{_timestamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--dataset",
        args.dataset,
        "--limit",
        str(args.limit),
        "--describe-limit",
        str(args.describe_limit),
        "--api-preflight-retries",
        str(args.api_preflight_retries),
        "--api-preflight-sleep",
        str(args.api_preflight_sleep),
        "--api-preflight-timeout",
        str(args.api_preflight_timeout),
    ]
    if args.dataset_path:
        common.extend(["--dataset-path", args.dataset_path])
    if args.repo_root:
        common.extend(["--repo-root", args.repo_root])
    if args.rq1_only:
        common.append("--rq1-only")
    if args.rq2_only:
        common.append("--rq2-only")

    gpt_summary = out_root / "opencoder_gpt" / "report_summary.json"
    gemini_summary = out_root / "opencoder_gemini" / "report_summary.json"

    if not args.skip_gpt:
        _run(
            "Run OpenCoderGPT",
            [
                py,
                "scripts/run_experiment_suite.py",
                *common,
                "--backend",
                "openai",
                "--model",
                args.gpt_model,
                "--out-dir",
                str(out_root / "opencoder_gpt"),
            ],
            env,
        )
    if not args.skip_gemini:
        _run(
            "Run OpenCoderGemini",
            [
                py,
                "scripts/run_experiment_suite.py",
                *common,
                "--backend",
                "gemini",
                "--model",
                args.gemini_model,
                "--out-dir",
                str(out_root / "opencoder_gemini"),
            ],
            env,
        )

    table_cmd = [
        py,
        "scripts/make_overleaf_results.py",
        "--out-dir",
        str(out_root / "overleaf"),
    ]
    if gpt_summary.exists():
        table_cmd.extend(["--gpt-summary", str(gpt_summary)])
    if gemini_summary.exists():
        table_cmd.extend(["--gemini-summary", str(gemini_summary)])
    if args.allow_offline_table:
        table_cmd.append("--allow-offline")
    _run("Build Overleaf comparison table", table_cmd, env)
    print(f"\nDone. LLM comparison directory: {out_root}", flush=True)


if __name__ == "__main__":
    main()
