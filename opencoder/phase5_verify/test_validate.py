"""Phase V, Step 12: Test & Validate.

Executes the generated code against any provided unit tests in a
subprocess with a wall-clock timeout. Returns pass/fail + stdout/stderr.
"""
from __future__ import annotations

import os
import ast
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestReport:
    passed: Optional[bool]
    stdout: str
    stderr: str
    returncode: int


def is_opencoderx_execrepobench_record(raw: dict) -> bool:
    return str(raw.get("manifest_version") or "").startswith(
        "execrepobench_opencoderx_"
    )


def normalize_execrepobench_function(
    generated_code: str,
    reference_code: str,
) -> str:
    """Normalize one generated function to the frozen target indentation."""
    candidate = textwrap.dedent(generated_code).strip("\n")
    tree = ast.parse(candidate)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1 or len(tree.body) != 1:
        raise ValueError("candidate must contain exactly one complete target function")
    reference_lines = reference_code.splitlines()
    first = next((line for line in reference_lines if line.strip()), None)
    if first is None:
        raise ValueError("missing ExecRepoBench reference function")
    indent = first[: len(first) - len(first.lstrip(" "))]
    return "\n".join(
        indent + line if line.strip() else ""
        for line in candidate.splitlines()
    ) + "\n"


def run_execrepobench_function_tests(
    generated_code: str,
    raw: dict,
    timeout: int = 240,
) -> TestReport:
    """Inject a complete function and run the official repository evaluator."""
    if not is_opencoderx_execrepobench_record(raw):
        return TestReport(passed=None, stdout="", stderr="(not an OpenCoderX ExecRepoBench record)", returncode=0)

    prefix = raw.get("execution_prefix_code")
    suffix = raw.get("execution_suffix_code")
    reference = raw.get("solution")
    repo_name = str(raw.get("repo_name") or "")
    file_name = str(raw.get("file_name") or "")
    if not all(isinstance(value, str) for value in (prefix, suffix, reference)):
        return TestReport(passed=False, stdout="", stderr="invalid frozen function metadata", returncode=2)

    default_repos = Path(__file__).resolve().parents[2] / ".benchmarks" / "execrepobench" / "repos"
    source_repo = Path(
        str(raw.get("_runtime_repo_root") or default_repos / repo_name)
    ).expanduser().resolve()
    evaluator = source_repo / "evaluate_repo.py"
    if not source_repo.is_dir() or not evaluator.is_file():
        return TestReport(passed=False, stdout="", stderr="missing official repository evaluator", returncode=2)

    relative = Path(file_name.lstrip("/"))
    if relative.parts and relative.parts[0] == repo_name:
        relative = Path(*relative.parts[1:])
    try:
        candidate = normalize_execrepobench_function(generated_code, reference)
    except (SyntaxError, ValueError) as exc:
        return TestReport(passed=False, stdout="", stderr=str(exc), returncode=1)
    full_source = f"{prefix}{candidate}{suffix}"

    with tempfile.TemporaryDirectory(prefix="opencoderx-execrepo-") as directory:
        checkout = Path(directory) / source_repo.name
        shutil.copytree(
            source_repo,
            checkout,
            ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc", ".coverage"),
        )
        target = (checkout / relative).resolve()
        try:
            target.relative_to(checkout.resolve())
        except ValueError:
            return TestReport(passed=False, stdout="", stderr="unsafe target path", returncode=2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(full_source, encoding="utf-8")

        default_python = (
            Path(__file__).resolve().parents[2]
            / ".benchmarks" / "execrepobench" / "envs" / repo_name / "bin" / "python"
        )
        python = Path(str(raw.get("_runtime_python_executable") or default_python))
        if not python.is_file():
            python = Path(sys.executable)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.fspath(checkout) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                [os.fspath(python), "evaluate_repo.py"],
                cwd=checkout,
                capture_output=True,
                timeout=timeout,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return TestReport(passed=False, stdout="", stderr="official evaluator timeout", returncode=-1)
        return TestReport(
            passed=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )


def run_tests(
    generated_code: str,
    test_code: str | None,
    timeout: int = 30,
    extra_pythonpath: str | None = None,
) -> TestReport:
    if not test_code:
        return TestReport(passed=None, stdout="", stderr="(no tests provided)", returncode=0)
    with tempfile.TemporaryDirectory() as d:
        gen_path = os.path.join(d, "solution.py")
        test_path = os.path.join(d, "test_solution.py")
        with open(gen_path, "w") as f:
            f.write(generated_code)
        with open(test_path, "w") as f:
            f.write("import sys\nsys.path.insert(0, '.')\nfrom solution import *\n\n" + test_code)
        try:
            env = os.environ.copy()
            if extra_pythonpath:
                extra = os.path.abspath(extra_pythonpath)
                env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=d,
                capture_output=True,
                timeout=timeout,
                text=True,
                env=env if extra_pythonpath else None,
            )
            return TestReport(
                passed=(r.returncode == 0),
                stdout=r.stdout,
                stderr=r.stderr,
                returncode=r.returncode,
            )
        except subprocess.TimeoutExpired:
            return TestReport(passed=False, stdout="", stderr="timeout", returncode=-1)
        except FileNotFoundError:
            # pytest not installed; fall back to plain exec.
            try:
                ns: dict = {}
                exec(compile(generated_code + "\n" + test_code, "gen", "exec"), ns)
                return TestReport(passed=True, stdout="exec-ok", stderr="", returncode=0)
            except Exception as e:
                return TestReport(passed=False, stdout="", stderr=repr(e), returncode=-1)


def _safe_repo_path(root: str, raw_path: str) -> str:
    rel = raw_path.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p and p not in {".", ".."}]
    return os.path.join(root, *parts)


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in f"/{normalized.strip('/')}/"
    )


def run_repo_completion_tests(
    generated_code: str,
    raw: dict,
    timeout: int = 60,
) -> TestReport:
    """Run tests for repository-completion records.

    ExecRepoBench-style rows provide a target file as prefix + missing middle +
    suffix, plus repository/test files in ``context_code``. This reconstructs a
    temporary checkout, inserts ``generated_code``, and runs the included tests.
    """
    prefix = raw.get("prefix_code")
    suffix = raw.get("suffix_code")
    file_name = raw.get("file_name")
    context_code = raw.get("context_code")
    if not isinstance(prefix, str) and not isinstance(suffix, str):
        return TestReport(passed=None, stdout="", stderr="(not a completion record)", returncode=0)
    if not isinstance(file_name, str) or not file_name.strip():
        return TestReport(passed=None, stdout="", stderr="(missing target file path)", returncode=0)
    if not isinstance(context_code, list):
        return TestReport(passed=None, stdout="", stderr="(no repository context files)", returncode=0)

    with tempfile.TemporaryDirectory() as d:
        test_paths: list[str] = []
        for i, entry in enumerate(context_code):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                path, text = str(entry[0]), str(entry[1])
            elif isinstance(entry, dict):
                path = str(entry.get("file_path") or entry.get("path") or f"context_{i}.py")
                text = str(entry.get("code") or entry.get("text") or "")
            else:
                continue
            out_path = _safe_repo_path(d, path)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(text, encoding="utf-8", errors="ignore")
            if _is_test_path(path):
                test_paths.append(out_path)

        target_path = _safe_repo_path(d, file_name)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        full_code = f"{prefix or ''}{generated_code}{suffix or ''}"
        Path(target_path).write_text(full_code, encoding="utf-8", errors="ignore")

        if not test_paths:
            return TestReport(passed=None, stdout="", stderr="(no test files in context_code)", returncode=0)

        repo_name = str(raw.get("repo_name") or "").strip().strip("/")
        repo_root = os.path.join(d, repo_name) if repo_name else d
        cwd = repo_root if os.path.isdir(repo_root) else d
        rel_tests = [os.path.relpath(p, cwd) for p in test_paths]
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *rel_tests],
                cwd=cwd,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return TestReport(
                passed=(r.returncode == 0),
                stdout=r.stdout,
                stderr=r.stderr,
                returncode=r.returncode,
            )
        except subprocess.TimeoutExpired:
            return TestReport(passed=False, stdout="", stderr="timeout", returncode=-1)


def _codereval_candidate_source(generated_code: str, reference_code: str) -> str:
    """Normalize a generated function to the indentation of its reference."""
    reference_lines = reference_code.splitlines()
    if not reference_lines:
        raise ValueError("missing CoderEval reference function")
    reference_indent = len(reference_lines[0]) - len(reference_lines[0].lstrip(" "))
    candidate = textwrap.dedent(generated_code).strip("\n")
    tree = ast.parse(candidate)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1 or len(tree.body) != 1:
        raise ValueError("candidate must contain exactly one target function")
    prefix = " " * reference_indent
    return "\n".join(prefix + line if line.strip() else "" for line in candidate.splitlines()) + "\n"


def run_codereval_project_tests(
    generated_code: str,
    raw: dict,
    timeout: int = 60,
) -> TestReport:
    """Inject one CoderEval function and run its audited native project tests."""
    if not raw.get("codereval_project_tests"):
        return TestReport(passed=None, stdout="", stderr="(not a CoderEval project-test record)", returncode=0)
    project_root = Path(str(raw.get("codereval_project_root") or "")).expanduser().resolve()
    target_file = str(raw.get("codereval_target_file") or "")
    reference_code = str(raw.get("codereval_reference_code") or "")
    selectors = raw.get("codereval_test_selectors") or []
    if not project_root.is_dir() or not target_file or not reference_code or not selectors:
        return TestReport(passed=False, stdout="", stderr="invalid CoderEval project-test metadata", returncode=2)
    source_path = project_root / target_file
    if not source_path.is_file():
        return TestReport(passed=False, stdout="", stderr=f"missing target file: {source_path}", returncode=2)
    original = source_path.read_text(encoding="utf-8")
    if original.count(reference_code) != 1:
        return TestReport(
            passed=False,
            stdout="",
            stderr="reference function is not unique in the audited snapshot",
            returncode=2,
        )
    try:
        candidate = _codereval_candidate_source(generated_code, reference_code)
    except (SyntaxError, ValueError) as exc:
        return TestReport(passed=False, stdout="", stderr=str(exc), returncode=1)

    with tempfile.TemporaryDirectory() as d:
        checkout = Path(d) / "project"
        shutil.copytree(
            project_root,
            checkout,
            ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.pyc", ".coverage"),
        )
        injected_path = checkout / target_file
        injected_path.write_text(original.replace(reference_code, candidate, 1), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.fspath(checkout)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *map(str, selectors)],
                cwd=checkout,
                capture_output=True,
                timeout=timeout,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return TestReport(passed=False, stdout="", stderr="CoderEval project tests timed out", returncode=-1)
        return TestReport(
            passed=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )


TestReport.__test__ = False
