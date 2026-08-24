from collections import Counter
from types import SimpleNamespace

from experiments.run_rq3 import _pipeline_config_for
from opencoder.phase3_retrieval._base import Hit
from opencoder.phase3_retrieval.score_filter import (
    Candidate,
    merge_step_candidates_balanced,
)
from opencoder.pipeline import PipelineConfig


def _candidate(source: str, score: float) -> Candidate:
    item = SimpleNamespace(source=source, score=score)
    return Candidate(Hit(item=item, score=score, source=source), score)


def test_balanced_merge_limits_dominant_source_when_alternatives_exist() -> None:
    candidates = [
        *[_candidate("api", 1.0 - index / 100) for index in range(10)],
        *[_candidate("similar_code", 0.7 - index / 100) for index in range(4)],
        _candidate("context", 0.6),
    ]

    merged = merge_step_candidates_balanced(
        [candidates], 10, max_source_fraction=0.5
    )

    counts = Counter(candidate.hit.source for candidate in merged)
    assert len(merged) == 10
    assert counts["api"] == 5
    assert counts["context"] == 1
    assert counts["similar_code"] == 4


def test_balanced_merge_fills_budget_when_only_one_source_exists() -> None:
    candidates = [_candidate("api", 1.0 - index / 100) for index in range(10)]

    merged = merge_step_candidates_balanced(
        [candidates], 10, max_source_fraction=0.5
    )

    assert len(merged) == 10
    assert all(candidate.hit.source == "api" for candidate in merged)


def test_whole_task_anchor_and_balance_are_opencoder_only() -> None:
    base = PipelineConfig()
    direct = _pipeline_config_for(base, "direct", False)
    baseline = _pipeline_config_for(base, "without", False)
    repair = _pipeline_config_for(base, "rag_repair", False)
    opencoder = _pipeline_config_for(base, "with", True)

    for config in (direct, baseline, repair):
        assert config.feature_enabled("whole_task_retrieval_anchor") is False
        assert config.feature_enabled("source_balanced_fusion") is False
    assert opencoder.feature_enabled("whole_task_retrieval_anchor") is True
    assert opencoder.feature_enabled("source_balanced_fusion") is True
