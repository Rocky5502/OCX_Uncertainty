from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_execrepobench_120.py"
SPEC = importlib.util.spec_from_file_location("validate_execrepobench_120", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_safe_target_path_rejects_traversal(tmp_path: Path) -> None:
    assert MODULE._safe_target_path(tmp_path, "/repo/file.py") == (tmp_path / "repo/file.py").resolve()
    assert MODULE._safe_target_path(tmp_path, "../../etc/passwd") is None


def test_deterministic_selection_enforces_repository_cap() -> None:
    rows = [
        {"task_id": "a1", "repo_name": "a", "artifact_hash": "1", "reference_tests_pass": True},
        {"task_id": "a2", "repo_name": "a", "artifact_hash": "2", "reference_tests_pass": True},
        {"task_id": "b1", "repo_name": "b", "artifact_hash": "3", "reference_tests_pass": True},
        {"task_id": "c1", "repo_name": "c", "artifact_hash": "4", "reference_tests_pass": False},
    ]

    assert MODULE.deterministic_select(rows, target_size=3, max_per_repo=1) == ["a1", "b1"]
