#!/usr/bin/env python3
"""Create isolated environments for selected ExecRepoBench repositories."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path


DEFAULT_REPOS = (
    "cacheout",
    "csv-diff",
    "dict2css",
    "dpath-python",
    "html-to-json",
    "lxml_html_clean",
    "markdown-it-py",
    "objprint",
    "tabulator-py",
    "textdistance",
    "transitions",
    "untangle",
)


def _run(command: list[str], *, cwd: Path, timeout: int) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"},
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": (result.stdout or "")[-4000:],
            "stderr_tail": (result.stderr or "")[-4000:],
            "latency_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": "installation timeout",
            "latency_seconds": time.monotonic() - started,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos-root", default=".benchmarks/execrepobench/repos")
    parser.add_argument("--env-root", default=".benchmarks/execrepobench/envs")
    parser.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--log", default="results/data_quality/execrepobench_environment_setup.jsonl")
    args = parser.parse_args()

    repos_root = Path(args.repos_root).resolve()
    env_root = Path(args.env_root).resolve()
    log_path = Path(args.log)
    env_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for position, repo_name in enumerate(args.repos, start=1):
        source = repos_root / repo_name
        environment = env_root / repo_name
        if not source.is_dir():
            records.append({"repo_name": repo_name, "status": "missing_repository"})
            continue
        if args.clean and environment.exists():
            shutil.rmtree(environment)
        if not (environment / "bin/python").is_file():
            venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / "bin/python"
        bootstrap = _run(
            [
                os.fspath(python), "-m", "pip", "install",
                "setuptools<80", "wheel", "pytest", "pytest-cov", "pytest-benchmark",
                "coincidence", "cssutils", "docutils", "openpyxl", "hypothesis",
                "pycodestyle", "mock",
            ],
            cwd=source,
            timeout=args.timeout,
        )
        requirements_file = source / "requirements.txt"
        requirements = (
            _run(
                [os.fspath(python), "-m", "pip", "install", "-r", os.fspath(requirements_file)],
                cwd=source,
                timeout=args.timeout,
            )
            if requirements_file.is_file()
            else {"command": [], "returncode": 0, "stdout_tail": "", "stderr_tail": "", "latency_seconds": 0.0}
        )
        install = _run(
            [os.fspath(python), "-m", "pip", "install", "--no-build-isolation", "."],
            cwd=source,
            timeout=args.timeout,
        )
        record = {
            "repo_name": repo_name,
            "python": os.fspath(python),
            "python_version": subprocess.run(
                [os.fspath(python), "--version"], capture_output=True, text=True
            ).stdout.strip(),
            "status": (
                "ready"
                if (
                    bootstrap["returncode"] == 0
                    and requirements["returncode"] == 0
                    and install["returncode"] == 0
                )
                else "install_failed"
            ),
            "bootstrap": bootstrap,
            "requirements": requirements,
            "install": install,
        }
        records.append(record)
        print(f"[{position}/{len(args.repos)}] {repo_name}: {record['status']}", flush=True)

    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    ready = sum(record.get("status") == "ready" for record in records)
    print(json.dumps({"requested": len(records), "ready": ready, "log": os.fspath(log_path)}, indent=2))
    return 0 if ready == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
