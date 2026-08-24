from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_execrepobench_120.py"
SPEC = importlib.util.spec_from_file_location("prepare_execrepobench_120", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(prefix: str, middle: str, suffix: str) -> dict:
    return {
        "repo_name": "demo",
        "file_name": "/demo/example.py",
        "fill_type": "grammar-based: block",
        "prefix_code": prefix,
        "middle_code": middle,
        "suffix_code": suffix,
        "context_code": [["/demo/helper.py", "VALUE = 1\n"]],
    }


def test_adapts_body_mask_without_reference_leakage() -> None:
    record = _record("def add(a, b):\n    ", "return a + b", "\n")
    audit, candidate = MODULE.adapt_record(7, record)

    assert audit["structural_status"] == "adaptable"
    assert candidate is not None
    assert candidate["solution"] == "def add(a, b):\n    return a + b"
    assert "return a + b" not in candidate["current_file"]
    assert candidate["target_function_prompt"] == "def add(a, b):\n    pass\n"


def test_rejects_signature_overlap() -> None:
    record = _record("", "def add(a, b):", "\n    return a + b\n")
    audit, candidate = MODULE.adapt_record(1, record)

    assert candidate is None
    assert audit["exclusion_reason"] == "mask_overlaps_signature_or_decorator"


def test_selects_smallest_nested_function() -> None:
    prefix = "def outer():\n    def inner():\n        "
    record = _record(prefix, "return 1", "\n    return inner()\n")
    audit, candidate = MODULE.adapt_record(2, record)

    assert audit["function_name"] == "inner"
    assert candidate is not None
    assert candidate["solution"].startswith("    def inner()")


def test_preserves_docstring_in_stub() -> None:
    prefix = 'def f(x):\n    """Explain f."""\n    '
    record = _record(prefix, "return x", "\n")
    _, candidate = MODULE.adapt_record(3, record)

    assert candidate is not None
    assert '"""Explain f."""' in candidate["target_function_prompt"]
    assert "return x" not in candidate["target_function_prompt"]


def test_removes_test_context_from_prompt() -> None:
    record = _record("def f():\n    ", "return 1", "\n")
    record["context_code"].append(["/demo/tests/test_example.py", "assert f() == 1\n"])
    audit, candidate = MODULE.adapt_record(4, record)

    assert candidate is not None
    assert audit["context_test_files_removed"] == 1
    assert all("test_example.py" not in item[0] for item in candidate["context_code"])


def test_rejects_reference_duplicated_in_context() -> None:
    record = _record("def f():\n    ", "return 1", "\n")
    record["context_code"].append(["/demo/duplicate.py", "def f():\n    return 1\n"])
    audit, candidate = MODULE.adapt_record(5, record)

    assert candidate is None
    assert audit["exclusion_reason"] == "reference_leakage_in_prompt"


def test_decorated_method_includes_at_sign_and_is_parseable() -> None:
    prefix = "class Demo:\n    @staticmethod\n    def f():\n        "
    record = _record(prefix, "return 1", "\n")
    audit, candidate = MODULE.adapt_record(6, record)

    assert audit["structural_status"] == "adaptable"
    assert candidate is not None
    assert candidate["solution"].startswith("    @staticmethod\n    def f()")
    assert MODULE.ast.parse(MODULE.textwrap.dedent(candidate["solution"]))
