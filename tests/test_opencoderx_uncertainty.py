from opencoderx.uncertainty import compute_uncertainty


def test_provider_independent_uncertainty_is_bounded():
    result = compute_uncertainty(
        candidates=["def f():\n    return 1", "def f():\n    return 2"],
        candidate_test_outcomes=[True, False],
        source_scores={"api": [0.9, 0.4], "context": [0.3], "similar_code": []},
        verifier_outcomes=[True, False],
    )
    assert result.execution_uncertainty == 1.0
    assert result.verifier_disagreement == 1.0
    assert result.evidence_incompleteness == 1 / 3
    assert all(0.0 <= value <= 1.0 for value in result.to_dict().values())


def test_identical_candidates_have_zero_disagreement():
    result = compute_uncertainty(candidates=["x = 1", "x=1"])
    assert result.candidate_disagreement == 0.0
    assert result.semantic_self_consistency == 1.0
