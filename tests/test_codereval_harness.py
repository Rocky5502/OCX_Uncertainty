from __future__ import annotations

from opencoder.data.loaders import Example
from opencoder.phase5_verify.test_validate import run_codereval_project_tests
from opencoder.pipeline import Pipeline, PipelineConfig


def test_codereval_project_harness_detects_bad_candidate(tmp_path):
    (tmp_path / "tests").mkdir()
    reference = "def target(value):\n    return value + 1\n"
    (tmp_path / "target.py").write_text(reference, encoding="utf-8")
    (tmp_path / "tests" / "test_target.py").write_text(
        "from target import target\n\ndef test_target():\n    assert target(2) == 3\n",
        encoding="utf-8",
    )
    raw = {
        "codereval_project_tests": True,
        "codereval_project_root": str(tmp_path),
        "codereval_target_file": "target.py",
        "codereval_reference_code": reference,
        "codereval_test_selectors": ["tests/test_target.py"],
    }
    good = run_codereval_project_tests(reference, raw)
    bad = run_codereval_project_tests("def target(value):\n    return value - 1\n", raw)
    assert good.passed is True
    assert bad.passed is False


def test_codereval_project_harness_rejects_extra_top_level_code(tmp_path):
    reference = "def target(value):\n    return value\n"
    (tmp_path / "target.py").write_text(reference, encoding="utf-8")
    raw = {
        "codereval_project_tests": True,
        "codereval_project_root": str(tmp_path),
        "codereval_target_file": "target.py",
        "codereval_reference_code": reference,
        "codereval_test_selectors": ["tests/test_target.py"],
    }
    report = run_codereval_project_tests("import os\n" + reference, raw)
    assert report.passed is False
    assert "exactly one" in report.stderr


def test_pipeline_allows_indented_codereval_class_method(tmp_path):
    (tmp_path / "tests").mkdir()
    reference = "    def target(self, value):\n        return value + 1\n"
    (tmp_path / "target.py").write_text("class Subject:\n" + reference, encoding="utf-8")
    (tmp_path / "tests" / "test_target.py").write_text(
        "from target import Subject\n\ndef test_target():\n    assert Subject().target(2) == 3\n",
        encoding="utf-8",
    )
    raw = {
        "codereval_project_tests": True,
        "codereval_project_root": str(tmp_path),
        "codereval_target_file": "target.py",
        "codereval_reference_code": reference,
        "codereval_test_selectors": ["tests/test_target.py"],
    }
    example = Example("task", "query", str(tmp_path), reference, None, raw)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.cfg = PipelineConfig(llm_backend="offline", embedding_model="")
    static_report, test_report = pipeline._validate_code(reference, example)
    assert static_report.ok is True
    assert test_report.passed is True
