#!/usr/bin/env python3
"""Validate adapted ExecRepoBench functions with official repository tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPOS_ARCHIVE_SHA256 = "46de2f73d6f3f2c18758c826bf6a07f42250ba455bb39179b34791b581a921bb"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_target_path(repos_root: Path, file_name: str) -> Path | None:
    relative = Path(file_name.lstrip("/"))
    if ".." in relative.parts or relative.is_absolute():
        return None
    target = (repos_root / relative).resolve()
    try:
        target.relative_to(repos_root.resolve())
    except ValueError:
        return None
    return target


def _reference_source(candidate: dict[str, Any]) -> str:
    return "".join(
        str(candidate.get(key) or "")
        for key in ("official_prefix_code", "official_middle_code", "official_suffix_code")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "not_provided_by_upstream_archive"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unresolvable_upstream_git_metadata"


def _dependency_failure(stderr: str, stdout: str) -> bool:
    combined = f"{stdout}\n{stderr}"
    markers = (
        "ModuleNotFoundError",
        "ImportError",
        "DistributionNotFound",
        "PackageNotFoundError",
        "No module named",
    )
    return any(marker in combined for marker in markers)


def run_official_evaluator(
    source_repo: Path,
    *,
    timeout: int,
    python_executable: Path | None = None,
    injected_file: str | None = None,
    injected_source: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="opencoderx-execrepo-") as directory:
        checkout = Path(directory) / source_repo.name
        shutil.copytree(
            source_repo,
            checkout,
            ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc", ".coverage"),
        )
        if injected_file is not None:
            relative = Path(injected_file.lstrip("/"))
            if relative.parts and relative.parts[0] == source_repo.name:
                relative = Path(*relative.parts[1:])
            target = (checkout / relative).resolve()
            try:
                target.relative_to(checkout.resolve())
            except ValueError:
                return {
                    "passed": False,
                    "returncode": 2,
                    "stdout_head": "",
                    "stderr_head": "unsafe injected target path",
                    "dependency_complete": False,
                    "latency_seconds": time.monotonic() - started,
                }
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(injected_source or "", encoding="utf-8")
        evaluator = checkout / "evaluate_repo.py"
        if not evaluator.is_file():
            return {
                "passed": False,
                "returncode": 2,
                "stdout_head": "",
                "stderr_head": "missing official evaluate_repo.py",
                "dependency_complete": False,
                "latency_seconds": time.monotonic() - started,
            }
        env = os.environ.copy()
        env["PYTHONPATH"] = os.fspath(checkout) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                [os.fspath(python_executable or Path(sys.executable)), "evaluate_repo.py"],
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout_head": stdout[-2000:],
                "stderr_head": stderr[-2000:],
                "dependency_complete": not _dependency_failure(stderr, stdout),
                "latency_seconds": time.monotonic() - started,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": -1,
                "stdout_head": str(exc.stdout or "")[-2000:],
                "stderr_head": "official evaluator timeout",
                "dependency_complete": True,
                "latency_seconds": time.monotonic() - started,
            }


def deterministic_select(
    rows: Iterable[dict[str, Any]],
    *,
    target_size: int,
    max_per_repo: int,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    eligible = sorted(
        (row for row in rows if row.get("reference_tests_pass") is True),
        key=lambda row: str(row["artifact_hash"]),
    )
    for row in eligible:
        repo = str(row["repo_name"])
        if counts[repo] >= max_per_repo:
            continue
        selected.append(str(row["task_id"]))
        counts[repo] += 1
        if len(selected) == target_size:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/manifests/execrepobench_opencoderx_candidates_v1.jsonl")
    parser.add_argument("--repos-root", default=".benchmarks/execrepobench/repos")
    parser.add_argument("--env-root", default=".benchmarks/execrepobench/envs")
    parser.add_argument("--audit-out", default="results/data_quality/execrepobench_120_validation.csv")
    parser.add_argument("--runs-out", default="results/data_quality/execrepobench_reference_runs.jsonl")
    parser.add_argument("--validated-out", default="data/manifests/execrepobench_opencoderx_validated_v1.jsonl")
    parser.add_argument("--frozen-out", default="data/manifests/execrepobench_opencoderx_120_v1.jsonl")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--target-size", type=int, default=120)
    parser.add_argument("--max-per-repo", type=int, default=4)
    parser.add_argument("--run-reference-tests", action="store_true")
    args = parser.parse_args()

    candidates = _read_jsonl(Path(args.candidates))
    repos_root = Path(args.repos_root).resolve()
    env_root = Path(args.env_root).resolve()
    archive = repos_root.parent / "repos.zip"
    if not repos_root.is_dir():
        raise SystemExit(f"missing repository archive extraction: {repos_root}")
    if not archive.is_file() or _sha256_file(archive) != REPOS_ARCHIVE_SHA256:
        raise SystemExit("repository archive checksum mismatch")

    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        repo_name = str(candidate.get("repo_name") or "")
        repo_path = repos_root / repo_name
        target = _safe_target_path(repos_root, str(candidate.get("file_name") or ""))
        expected = _reference_source(candidate)
        source_matches = bool(
            target
            and target.is_file()
            and target.read_text(encoding="utf-8", errors="replace") == expected
        )
        row = {
            **candidate,
            "repository_available": repo_path.is_dir(),
            "official_evaluator_available": (repo_path / "evaluate_repo.py").is_file(),
            "archive_source_matches_reference": source_matches,
            "upstream_commit": _git_commit(repo_path) if repo_path.is_dir() else "repository_missing",
            "test_command": f"{sys.executable} evaluate_repo.py",
            "environment": f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "repository_archive_sha256": REPOS_ARCHIVE_SHA256,
            "reference_tests_pass": False,
            "dependency_complete": False,
            "validation_status": "not_run",
            "selected": False,
        }
        rows.append(row)
        by_repo[repo_name].append(row)

    run_records: list[dict[str, Any]] = []
    if args.run_reference_tests:
        for position, (repo_name, repo_rows) in enumerate(sorted(by_repo.items()), start=1):
            repo_path = repos_root / repo_name
            isolated_python = env_root / repo_name / "bin" / "python"
            python_executable = isolated_python if isolated_python.is_file() else Path(sys.executable)
            exact_rows = [row for row in repo_rows if row["archive_source_matches_reference"]]
            if exact_rows and (repo_path / "evaluate_repo.py").is_file():
                result = run_official_evaluator(
                    repo_path,
                    timeout=args.timeout,
                    python_executable=python_executable,
                )
                run_records.append({
                    "repo_name": repo_name,
                    "scope": "archived_repository",
                    "python_executable": os.fspath(python_executable),
                    **result,
                })
                for row in exact_rows:
                    row["reference_tests_pass"] = bool(result["passed"])
                    row["dependency_complete"] = bool(result["dependency_complete"])
                    row["validation_status"] = "repository_reference_pass" if result["passed"] else "repository_reference_fail"
            for row in repo_rows:
                if row["archive_source_matches_reference"]:
                    continue
                if not row["repository_available"] or not row["official_evaluator_available"]:
                    row["validation_status"] = "missing_repository_or_evaluator"
                    continue
                result = run_official_evaluator(
                    repo_path,
                    timeout=args.timeout,
                    python_executable=python_executable,
                    injected_file=str(row["file_name"]),
                    injected_source=_reference_source(row),
                )
                run_records.append({
                    "repo_name": repo_name,
                    "task_id": row["task_id"],
                    "scope": "task_specific_injection",
                    "python_executable": os.fspath(python_executable),
                    **result,
                })
                row["reference_tests_pass"] = bool(result["passed"])
                row["dependency_complete"] = bool(result["dependency_complete"])
                row["validation_status"] = "task_reference_pass" if result["passed"] else "task_reference_fail"
            print(f"[{position}/{len(by_repo)}] {repo_name}: {sum(r['reference_tests_pass'] for r in repo_rows)}/{len(repo_rows)}", flush=True)

    selected_ids = set(deterministic_select(rows, target_size=args.target_size, max_per_repo=args.max_per_repo))
    for row in rows:
        row["selected"] = row["task_id"] in selected_ids
        if row["selected"]:
            row["validation_status"] = "selected_reference_validated"

    audit_path = Path(args.audit_out)
    runs_path = Path(args.runs_out)
    validated_path = Path(args.validated_out)
    frozen_path = Path(args.frozen_out)
    for path in (audit_path, runs_path, validated_path, frozen_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    compact_rows = []
    omitted = {
        "context_code", "current_file", "execution_prefix_code", "execution_suffix_code",
        "instruction", "official_input", "official_middle_code", "official_prefix_code",
        "official_suffix_code", "solution", "target_function_prompt",
    }
    for row in rows:
        compact_rows.append({key: value for key, value in row.items() if key not in omitted})
    fieldnames = sorted({key for row in compact_rows for key in row})
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact_rows)
    with runs_path.open("w", encoding="utf-8") as handle:
        for record in run_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with validated_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["reference_tests_pass"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    freeze_ready = len(selected_ids) == args.target_size and args.run_reference_tests
    if freeze_ready:
        with frozen_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row["selected"]:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elif frozen_path.exists():
        raise SystemExit(f"refusing to leave stale frozen manifest: {frozen_path}")

    summary = {
        "candidates": len(rows),
        "repositories": len(by_repo),
        "archive_source_matches": sum(bool(row["archive_source_matches_reference"]) for row in rows),
        "reference_validated": sum(bool(row["reference_tests_pass"]) for row in rows),
        "selected": len(selected_ids),
        "freeze_ready": freeze_ready,
    }
    print(json.dumps(summary, indent=2))
    return 0 if (not args.run_reference_tests or freeze_ready) else 2


if __name__ == "__main__":
    raise SystemExit(main())
