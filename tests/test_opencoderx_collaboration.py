from opencoderx.collaboration import (
    CollaborationDecision,
    CollaborationPolicy,
    RiskTrace,
    allocate_review_budget,
    review_allocation_metrics,
    policy_score,
)


def test_decision_boundaries_and_tie_breaking():
    policy = CollaborationPolicy(review_threshold=0.35, abstain_threshold=0.70)
    autonomous = policy.decide(task_id="a", model="m", trace=RiskTrace(api=0.1))
    review = policy.decide(
        task_id="b", model="m",
        trace=RiskTrace(api=0.35, context=0.35, similar_code=0.35,
                        generation=0.35, verification=0.35, repair=0.35),
    )
    abstain = policy.decide(
        task_id="c", model="m",
        trace=RiskTrace(api=0.7, context=0.7, similar_code=0.7,
                        generation=0.7, verification=0.7, repair=0.7),
    )
    assert autonomous.decision is CollaborationDecision.AUTONOMOUS
    assert review.decision is CollaborationDecision.REQUEST_REVIEW
    assert abstain.decision is CollaborationDecision.ABSTAIN


def test_review_budget_is_deterministic_and_budgeted():
    rows = [
        {"task_id": f"t{i}", "aggregate_risk": i / 10, "selected_output_correct": i % 2 == 0}
        for i in range(10)
    ]
    selected = allocate_review_budget(
        rows, policy="aggregate_uncertainty_deferral", review_budget=0.2
    )
    assert selected == {"t8", "t9"}
    metrics = review_allocation_metrics(rows, selected, reviewer_success=1.0, seed=7)
    assert metrics["autonomous_coverage"] == 0.8


def test_policy_scores_use_distinct_pre_outcome_signals():
    row = {
        "task_id": "t0",
        "selected_output_correct": False,
        "pre_repair_test_failed": False,
        "generation_uncertainty": 0.2,
        "token_entropy": 0.8,
        "candidate_disagreement": 0.6,
        "aggregate_risk": 0.4,
    }
    assert policy_score(row, "test_failure_deferral") == 0.0
    assert policy_score(row, "confidence_deferral") == 0.2
    assert policy_score(row, "entropy_deferral") == 0.8
    assert policy_score(row, "self_consistency_deferral") == 0.6
    assert policy_score(row, "oracle_deferral") == 1.0


def test_always_autonomous_ignores_nominal_review_budget():
    rows = [{"task_id": "a"}, {"task_id": "b"}]
    assert allocate_review_budget(
        rows, policy="always_autonomous", review_budget=1.0
    ) == set()
