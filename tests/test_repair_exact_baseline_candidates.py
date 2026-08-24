from experiments.repair_exact_baseline_candidates import (
    _add_usage,
    _is_completed_exact_repair,
)


def test_add_usage_combines_generation_and_repair_costs():
    baseline = {
        "llm_usage": {
            "requests": 5,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
    }
    repair = {
        "requests": 2,
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }

    assert _add_usage(baseline, repair) == {
        "requests": 7,
        "retries": 0,
        "failed_attempts": 0,
        "prompt_tokens": 130,
        "completion_tokens": 30,
        "total_tokens": 160,
    }


def test_completed_exact_repair_requires_same_candidate_stream():
    baseline = {"generated_samples": ["a", "b", "c", "d", "e"]}
    existing = {
        "generated_samples": ["a", "b", "c", "d", "e"],
        "repair_rounds": 2,
        "rag_verify_repair_provenance": {
            "candidate_source": "exact_matched_baseline_rag_candidates",
            "repair_required": True,
        },
    }

    assert _is_completed_exact_repair(baseline, existing)
    existing["generated_samples"] = ["different"]
    assert not _is_completed_exact_repair(baseline, existing)
