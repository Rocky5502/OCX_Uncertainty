from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_execrepobench_120.py"
SPEC = importlib.util.spec_from_file_location("freeze_execrepobench_120", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extracts_signature_and_calls() -> None:
    source = "async def render(item: str, *, size=2) -> bytes:\n    return codec.encode(item, size)\n"
    assert MODULE._function_signature(source) == "async def render(item: str, *, size=2) -> bytes:"
    assert MODULE._call_symbols(source) == ["codec.encode"]


def test_test_path_detection() -> None:
    assert MODULE._is_test_path("tests/test_api.py")
    assert MODULE._is_test_path("package/module_test.py")
    assert not MODULE._is_test_path("package/testing_utils.py")
