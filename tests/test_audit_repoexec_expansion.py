from scripts.audit_repoexec_expansion import (
    _artifact_hash,
    _dependency_complete_execution_prefix,
    _normalize_prefix,
    _prompt_isolated,
    _self_contained_tests,
)


def test_self_contained_test_filter_matches_frozen_protocol():
    tests = [
        "def test_ok():\n    assert True",
        "def test_pickle():\n    open('/output/value.pkl', 'rb')",
        "",
    ]
    assert _self_contained_tests(tests) == [
        "def test_ok():\n    assert True",
    ]


def test_relative_repository_imports_are_normalized():
    prefix = "from .manipulation import roman_encode\n"
    assert _normalize_prefix(prefix) == (
        "from string_utils.manipulation import roman_encode\n"
    )


def test_execution_prefix_removes_only_target_implementation():
    source = (
        "from .helpers import helper\n\n"
        "def target(x):\n"
        "    return helper(x)\n\n"
        "def dependency(x):\n"
        "    return x + 1\n"
    )
    prefix = _dependency_complete_execution_prefix(source, "target")
    assert "from string_utils.helpers import helper" in prefix
    assert "def target(x):\n    pass" in prefix
    assert "def dependency(x):\n    return x + 1" in prefix
    assert "return helper(x)" not in prefix


def test_prompt_isolation_rejects_reference_or_test_content():
    record = {
        "instruction": "def target():",
        "current_file": "",
        "solution": "def target():\n    return 1",
    }
    assert _prompt_isolated(record, ["def test_target():\n    assert target() == 1"])
    record["current_file"] = record["solution"]
    assert not _prompt_isolated(record, [])


def test_artifact_hash_is_stable_and_sensitive_to_tests():
    record = {
        "task_id": "repo/1",
        "project": "repo",
        "test": "assert True",
    }
    first = _artifact_hash(record)
    assert first == _artifact_hash(dict(record))
    record["test"] = "assert False"
    assert first != _artifact_hash(record)
