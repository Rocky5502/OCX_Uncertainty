"""Run the AAAI-facing test-backed OpenCoder experiment suite.

This runner deliberately stops if API preflight fails; it should not produce
paper-facing LLM result files from a broken gateway.
"""
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


def _backend_model(backend: str, gpt_model: str, gemini_model: str) -> str:
    return gemini_model if backend == "gemini" else gpt_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="input/execrepobench_testbacked.jsonl")
    ap.add_argument("--config", default="configs/real_api_testbacked.yaml")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--describe-limit", type=int, default=3)
    ap.add_argument("--conditions", default="all,no_api,no_context,no_similar_code,no_retrieval")
    ap.add_argument("--backends", default="openai",
                    help="Comma-separated backends: openai,gemini")
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--api-preflight-retries", type=int, default=2)
    ap.add_argument("--api-preflight-sleep", type=int, default=3)
    ap.add_argument("--api-preflight-timeout", type=int, default=30)
    ap.add_argument("--rq1-only", action="store_true")
    ap.add_argument("--rq2-only", action="store_true")
    args = ap.parse_args()

    if args.rq1_only and args.rq2_only:
        raise SystemExit("Use at most one of --rq1-only or --rq2-only.")

    env_base = _load_dotenv_into_env(os.environ.copy())
    py = sys.executable
    out_root = Path(args.out_root) if args.out_root else ROOT / "results" / f"aaai_testbacked_{_timestamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    for backend in backends:
        model = _backend_model(backend, args.gpt_model, args.gemini_model)
        label = "gpt" if backend in {"openai", "chatgpt", "gpt"} else backend
        out_dir = out_root / f"opencoder_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(env_base)
        env["OPENCODER_LLM_BACKEND"] = backend
        env["OPENCODER_LLM_MODEL"] = model

        _run(
            f"API preflight ({backend}/{model})",
            [
                py,
                "scripts/preflight_api.py",
                "--backend",
                backend,
                "--model",
                model,
                "--timeout",
                str(args.api_preflight_timeout),
                "--retries",
                str(args.api_preflight_retries),
                "--sleep",
                str(args.api_preflight_sleep),
            ],
            env,
        )

        rq1_path = out_dir / "rq1.json"
        rq2_path = out_dir / "rq2.json"
        if not args.rq2_only:
            _run(
                f"RQ1 focused source ablation ({backend})",
                [
                    py,
                    "scripts/ablation_rq1.py",
                    "--config",
                    args.config,
                    "--dataset",
                    "execrepobench",
                    "--dataset-path",
                    args.dataset_path,
                    "--limit",
                    str(args.limit),
                    "--describe-limit",
                    str(args.describe_limit),
                    "--conditions",
                    args.conditions,
                    "--out",
                    str(rq1_path),
                ],
                env,
            )
        if not args.rq1_only:
            _run(
                f"RQ2 baseline vs OpenCoder ({backend})",
                [
                    py,
                    "scripts/ablation_rq2.py",
                    "--config",
                    args.config,
                    "--dataset",
                    "execrepobench",
                    "--dataset-path",
                    args.dataset_path,
                    "--limit",
                    str(args.limit),
                    "--describe-limit",
                    str(args.describe_limit),
                    "--out",
                    str(rq2_path),
                ],
                env,
            )

        report_cmd = [py, "scripts/report_results.py", "--out-dir", str(out_dir)]
        if rq1_path.exists():
            report_cmd.extend(["--rq1", str(rq1_path)])
        if rq2_path.exists():
            report_cmd.extend(["--rq2", str(rq2_path)])
        _run(f"Report assets ({backend})", report_cmd, env)

        summary = out_dir / "report_summary.json"
        if summary.exists():
            _run(
                f"Paper figures ({backend})",
                [
                    py,
                    "scripts/plot_paper_figures.py",
                    "--summary",
                    str(summary),
                    "--out-dir",
                    str(out_dir / "paper_figures"),
                ],
                env,
            )

    print(f"\nDone. Test-backed AAAI suite directory: {out_root}", flush=True)


if __name__ == "__main__":
    main()
