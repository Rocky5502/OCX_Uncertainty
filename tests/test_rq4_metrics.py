import pytest

from experiments.run_rq4 import (
    _api_item_uncertainty_rows,
    auroc,
    ece_binary,
    extract_call_names,
    ground_truth_from_codereval_record,
    normalize_api_name,
    resolve_repository_call,
    set_metrics,
    summarize_metrics,
)
from opencoder.data.loaders import Example
from opencoder.phase1_repo_knowledge.extract import RepoFunction, extract_source_knowledge
from opencoder.phase3_retrieval.api_refine import refine_api_hit_records


def test_set_metrics_handles_empty_exact_sets():
    metrics = set_metrics([], [])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["exact_set_match"] == 1.0
    assert metrics["count_outcome"] == "Exact"


def test_set_metrics_marks_under_retrieval():
    metrics = set_metrics(["is_string"], ["is_string", "InvalidInputError"])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 3) == 0.667
    assert metrics["count_outcome"] == "Under"


def test_extract_call_names_from_missing_region_block():
    calls = extract_call_names(
        """
if not is_string(value):
    raise InvalidInputError(value)
return regex.sub("-", value).lower()
"""
    )
    assert "is_string" in calls
    assert "InvalidInputError" in calls
    assert "regex.sub" in calls


def test_normalize_api_name_prefers_simple_identifier():
    assert normalize_api_name("DateTime.from_native(value)") == "from_native"
    assert normalize_api_name("class FixedOffset") == "fixedoffset"


def test_target_aware_api_refinement_removes_target_and_constructor_noise():
    hits = [
        {"api_name": "reverse", "final_score": 0.9, "kept": True},
        {"api_name": "InvalidInputError.__init__", "final_score": 0.8, "kept": True},
        {"api_name": "InvalidInputError", "final_score": 0.7, "kept": True},
        {"api_name": "__StringFormatter.__uppercase_first_char", "final_score": 0.6, "kept": True},
    ]
    refined = refine_api_hit_records(hits, target_norms=["reverse"])
    assert refined == ["InvalidInputError"]


def test_extract_source_knowledge_indexes_relative_imports_only():
    items = extract_source_knowledge(
        "module.py",
        "from ..time import Time as RepoTime\nfrom datetime import datetime\n",
    )
    assert [(item.qualname, item.kind) for item in items] == [
        ("RepoTime", "imported_api")
    ]


def test_repository_call_resolution_rejects_unrelated_attribute_receiver():
    item = RepoFunction(
        file_path="chronyk.py",
        qualname="Chronyk.timestamp",
        signature="timestamp(self)",
        docstring=None,
        body="",
        start_line=1,
        end_line=1,
        kind="method",
    )
    assert resolve_repository_call("dt.timestamp", [item]) is None
    assert resolve_repository_call("Chronyk.timestamp", [item]) == (
        "timestamp",
        "Chronyk.timestamp",
    )


def test_codereval_ground_truth_keeps_only_repository_scoped_symbols():
    example = Example(
        id="task",
        query="implement",
        repo_root=None,
        reference_code="",
        test_code=None,
        raw={"target_function_prompt": "def target():\n    pass"},
    )
    oracle = {
        "file_path": "pkg/module.py",
        "file_content": "from ..time import Time\n\ndef helper():\n    pass\n",
        "oracle_context": '{ "apis" : "[\'helper\', \'len\']", "classes" : "[\'Time\', \'ValueError\']" }',
    }
    gt = ground_truth_from_codereval_record(example, oracle)
    assert gt.api_norms == ["helper", "time"]
    assert gt.extraction_method == "official_oracle_context_repository_scoped"


def test_quality_summary_excludes_empty_ground_truth_from_api_f1():
    base = {
        "benchmark": "Bench",
        "backend": "GPT",
        "method": "OpenCoder",
        "task_passed": True,
        "uncertainty": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "jaccard": 0.0,
        "exact_set_match": 0.0,
        "count_outcome": "Under",
    }
    empty = {
        **base,
        "gt_count": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "jaccard": 1.0,
        "exact_set_match": 1.0,
        "count_outcome": "Exact",
    }
    api_task = {**base, "gt_count": 1}
    _, quality, _, _ = summarize_metrics([empty, api_task])
    assert quality[0]["n_api_tasks"] == 1
    assert quality[0]["f1"] == 0.0


def test_api_item_uncertainty_decreases_with_cross_step_consensus():
    record = {
        "benchmark": "Bench",
        "backend_label": "GPT",
        "method": "OpenCoder",
        "task_id": "task",
        "ground_truth_api_norms": ["stable"],
        "final_apis": ["stable", "single"],
        "api_hits": [
            {"api_name": "stable", "kept": True, "step_index": 1, "final_score": 0.8},
            {"api_name": "stable", "kept": True, "step_index": 2, "final_score": 0.7},
            {"api_name": "single", "kept": True, "step_index": 1, "final_score": 0.9},
        ],
    }
    rows = {row["api_norm"]: row for row in _api_item_uncertainty_rows([record])}
    assert rows["stable"]["raw_api_uncertainty"] == 0.5
    assert rows["single"]["raw_api_uncertainty"] == 1.0
    assert rows["stable"]["is_incorrect_api"] == 0
    assert rows["single"]["is_incorrect_api"] == 1


def test_uncertainty_metrics_treat_incorrect_api_as_positive_class():
    labels = [0, 0, 1, 1]
    correctly_oriented_risk = [0.1, 0.2, 0.8, 0.9]
    reversed_risk = [1.0 - score for score in correctly_oriented_risk]

    assert auroc(labels, correctly_oriented_risk) == 1.0
    assert auroc(labels, reversed_risk) == 0.0
    assert ece_binary(labels, correctly_oriented_risk, bins=2) == pytest.approx(0.15)
