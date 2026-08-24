"""Run OpenCoder RQ1/RQ2 experiments and generate paper-facing outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(label: str, cmd: List[str], env: dict) -> None:
    display = " ".join(cmd)
    print(f"\n==> {label}", flush=True)
    print(display, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _add_common_args(cmd: List[str], args: argparse.Namespace) -> List[str]:
    cmd.extend(["--dataset", args.dataset])
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.repo_root:
        cmd.extend(["--repo-root", args.repo_root])
    cmd.extend(["--limit", str(args.limit), "--describe-limit", str(args.describe_limit)])
    if args.config:
        cmd.extend(["--config", args.config])
    return cmd


def _default_model(backend: str) -> str:
    if backend == "offline":
        return "offline-heuristic"
    if backend == "gemini":
        return "gemini-2.5-flash"
    return "gpt-4o-mini"


def main() -> None:
    _load_dotenv_into_env(os.environ)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--describe-limit", type=int, default=50)
    ap.add_argument(
        "--backend",
        default=os.environ.get("OPENCODER_LLM_BACKEND", "openai"),
        help="offline, openai/chatgpt, gemini, or zhizengzeng.",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=os.environ.get("OPENCODER_LLM_BASE_URL"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--api-preflight-retries", type=int, default=1)
    ap.add_argument("--api-preflight-sleep", type=int, default=60)
    ap.add_argument("--api-preflight-timeout", type=int, default=20)
    ap.add_argument("--rq1-only", action="store_true")
    ap.add_argument("--rq2-only", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-paper-figures", action="store_true")
    args = ap.parse_args()

    if args.rq1_only and args.rq2_only:
        raise SystemExit("Use at most one of --rq1-only or --rq2-only.")

    backend = args.backend.lower()
    model = args.model or os.environ.get("OPENCODER_LLM_MODEL") or _default_model(backend)
    run_name = f"{args.dataset}_{backend}_{model}_{_timestamp()}".replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENCODER_LLM_BACKEND"] = backend
    env["OPENCODER_LLM_MODEL"] = model
    if args.base_url and backend != "offline":
        env["OPENCODER_LLM_BASE_URL"] = args.base_url

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "repo_root": args.repo_root,
        "limit": args.limit,
        "describe_limit": args.describe_limit,
        "backend": backend,
        "model": model,
        "base_url": args.base_url if backend != "offline" else None,
        "rq1": not args.rq2_only,
        "rq2": not args.rq1_only,
        "correctness_note": (
            "If active dataset rows lack executable tests, pass@k and variance are "
            "computed from normalized reference exact-match only."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Output directory: {out_dir}", flush=True)

    py = sys.executable
    if not args.skip_audit:
        _run("Audit code/data readiness", [py, "scripts/audit_project.py"], env)

    if backend != "offline" and not args.skip_smoke:
        _run(
            "API preflight",
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
        cmd = _add_common_args([py, "scripts/ablation_rq1.py"], args)
        cmd.extend(["--out", str(rq1_path)])
        _run("RQ1 retrieval-source ablation", cmd, env)

    if not args.rq1_only:
        cmd = _add_common_args([py, "scripts/ablation_rq2.py"], args)
        cmd.extend(["--out", str(rq2_path)])
        _run("RQ2 uncertainty-aware ablation", cmd, env)

    if not args.no_report:
        cmd = [py, "scripts/report_results.py", "--out-dir", str(out_dir)]
        if rq1_path.exists():
            cmd.extend(["--rq1", str(rq1_path)])
        if rq2_path.exists():
            cmd.extend(["--rq2", str(rq2_path)])
        _run("Generate academic report assets", cmd, env)
        if not args.no_paper_figures:
            summary_path = out_dir / "report_summary.json"
            if summary_path.exists():
                _run(
                    "Generate publication PDF/PNG figures",
                    [
                        py,
                        "scripts/plot_paper_figures.py",
                        "--summary",
                        str(summary_path),
                        "--out-dir",
                        str(out_dir / "paper_figures"),
                    ],
                    env,
                )

    print(f"\nDone. Report directory: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
