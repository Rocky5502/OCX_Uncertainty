from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CROSS = _load("prepare_crosscodeeval_subset", "scripts/prepare_crosscodeeval_subset.py")
MULTI = _load("prepare_multi_swe_subset", "scripts/prepare_multi_swe_subset.py")


def test_crosscodeeval_selection_respects_repository_cap() -> None:
    rows = [
        {"artifact_hash": "1", "repository": "a"},
        {"artifact_hash": "2", "repository": "a"},
        {"artifact_hash": "3", "repository": "b"},
    ]
    assert CROSS.select_rows(rows, target=2, max_per_repository=1) == [rows[0], rows[2]]


def test_multi_swe_eligibility_requires_failure_then_repair() -> None:
    valid = {
        "base": {"sha": "abc"},
        "fix_patch": "diff --git a/a b/a",
        "run_result": {"failed_count": 1},
        "test_patch_result": {"failed_count": 2},
        "fix_patch_result": {"failed_count": 1},
    }
    assert MULTI.eligible(valid)[0] is True
    invalid = {**valid, "test_patch_result": {"failed_count": 1}}
    assert MULTI.eligible(invalid)[0] is False
