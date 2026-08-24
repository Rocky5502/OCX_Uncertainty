import pytest

from scripts.build_expanded_repoexec_results import (
    _bootstrap_ci,
    _effective_outcomes,
    _fallback_evidence_ids,
    _mcnemar_exact,
    _task_scores,
)


def test_effective_stream_replaces_primary_only_for_phase_five_methods():
    row = {
        "sample_correctness": [False, True, False, False, False],
        "passed": True,
    }
    assert _effective_outcomes(row, "without") == [
        False, True, False, False, False
    ]
    assert _effective_outcomes(row, "with") == [
        True, True, False, False, False
    ]


def test_task_scores_use_five_candidate_estimator():
    scores = _task_scores([True, False, False, False, False])
    assert scores["Pass@1"] == pytest.approx(0.2)
    assert scores["Pass@5"] == 1.0


def test_exact_mcnemar_and_bootstrap_are_paired():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(6, 0) == 0.03125
    low, high = _bootstrap_ci([0, 0, 0], [1, 1, 1], iterations=100)
    assert low == high == 1.0


def test_legacy_evidence_ids_are_stable_content_hashes():
    fused = "### Evidence: API\n- `foo()` (a.py)\n  docs"
    first = _fallback_evidence_ids(fused)
    assert first == _fallback_evidence_ids(fused)
    assert first[0]["id"].startswith("sha256:")
