from __future__ import annotations

import itertools

import pytest

from experiments.run_rq3 import (
    _completed_ids,
    _feature_flags,
    _pipeline_config_for,
    _upsert_row,
)
from opencoder.pipeline import PipelineConfig
from scripts.ablation_rq1 import SOURCES, _summarize as summarize_rq1
from scripts.ablation_rq2 import (
    CONDITIONS,
    _config_for,
    _risk_metrics,
    _summarize as summarize_rq2,
)
from scripts.build_rq3_results import _multiseed_rows


def _factorial_rows():
    rows = []
    for task_index in range(3):
        for bits in itertools.product((0, 1), repeat=3):
            api, context, similar = bits
            uncertainty = (
                0.2
                + 0.20 * api
                - 0.10 * context
                + 0.05 * similar
                + 0.03 * api * context
                + 0.01 * task_index
            )
            rows.append({
                "example_id": f"task-{task_index}",
                "condition": "synthetic",
                "enabled": [source for source, enabled in zip(SOURCES, bits) if enabled],
                "u": {"aggregate": uncertainty},
                "passed": uncertainty < 0.4,
                "pass_at_k": {
                    "pass@1": float(uncertainty < 0.4),
                    "pass@3": float(uncertainty < 0.45),
                    "pass@5": float(uncertainty < 0.5),
                },
                "pass_rate_variance": 0.0,
            })
    return rows


def test_rq1_factorial_effects_are_task_paired():
    summary = summarize_rq1(_factorial_rows())
    effects = summary["factorial"]["uncertainty"]
    assert effects["main_effects"]["api"]["effect_present_minus_absent"] == pytest.approx(0.215)
    assert effects["main_effects"]["context"]["effect_present_minus_absent"] == pytest.approx(-0.085)
    assert effects["main_effects"]["similar_code"]["effect_present_minus_absent"] == pytest.approx(0.05)
    assert effects["two_way_interactions"]["api:context"]["effect"] == pytest.approx(0.03)
    assert effects["main_effects"]["api"]["n_tasks"] == 3
    assert summary["api"]["delta_u_p_holm"] is not None


def test_rq2_risk_metrics_reward_ordered_failure_uncertainty():
    metrics = _risk_metrics([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert metrics["failure_auroc"] == pytest.approx(1.0)
    assert metrics["failure_auprc"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.05
    assert metrics["aurc"] < 0.5


def test_rq2_component_configs_are_explicit():
    base = PipelineConfig(llm_backend="offline", uncertainty_aware=True)
    baseline = _config_for(base, CONDITIONS["without"])
    full = _config_for(base, CONDITIONS["with"])
    assert not any(baseline.feature_enabled(name) for name in CONDITIONS_COMPONENTS)
    assert all(full.feature_enabled(name) for name in CONDITIONS_COMPONENTS)


def test_rq2_summary_does_not_extrapolate_pass_at_five_from_three_samples():
    row = {
        "id": "task-1",
        "passed": True,
        "u": {"aggregate": 0.2},
        "initial_test_report": {"passed": True},
        "sample_correctness": [True, False, False],
        "repair_rounds": 0,
    }
    summary = summarize_rq2({"without": [row]})["without"]
    assert "pass@1" in summary
    assert "pass@3" in summary
    assert "pass@5" not in summary


def test_rq3_resume_retries_and_replaces_error_rows():
    rows = [{"id": "task-1", "error": "timeout"}]
    assert "task-1" not in _completed_ids(rows)
    _upsert_row(rows, {"id": "task-1", "passed": True})
    assert rows == [{"id": "task-1", "passed": True}]
    assert "task-1" in _completed_ids(rows)


def test_rq3_rag_verify_repair_keeps_only_phase5_controls():
    base = PipelineConfig(
        uncertainty_aware=True,
        api_top_k=8,
        context_top_k=8,
        similar_code_top_k=8,
        fused_top_k=10,
        n_samples_for_uncertainty=5,
        max_repair_rounds=2,
    )
    config = _pipeline_config_for(base, "rag_repair", False)

    assert _feature_flags(config) == {
        "uncertainty_decomposition": False,
        "uncertainty_filtering": False,
        "uncertainty_guided_generation": False,
        "uncertainty_verified_selection": True,
        "uncertainty_triggered_repair": True,
        "whole_task_retrieval_anchor": False,
        "source_balanced_fusion": False,
    }
    assert config.api_top_k == base.api_top_k
    assert config.context_top_k == base.context_top_k
    assert config.similar_code_top_k == base.similar_code_top_k
    assert config.fused_top_k == base.fused_top_k
    assert config.n_samples_for_uncertainty == 5
    assert config.max_repair_rounds == 2


def test_rq3_multiseed_summary_keeps_repeated_task_ids_separate():
    rows = []
    for seed, score in ((11, 0.2), (12, 0.6)):
        for task_id in ("task-1", "task-2"):
            rows.append({
                "backend": "GPT",
                "method": "OpenCoder",
                "benchmark": "ExecRepoBench",
                "seed": seed,
                "task_id": task_id,
                "Pass@1": score,
                "Pass@3": score,
                "Pass@5": score,
            })

    summary = _multiseed_rows(rows)
    pass1 = next(row for row in summary if row["Metric"] == "Pass@1")
    assert pass1["Mean"] == pytest.approx(40.0)
    assert pass1["NSeeds"] == 2
    assert pass1["TasksPerSeedMin"] == 2
    assert pass1["TasksPerSeedMax"] == 2


CONDITIONS_COMPONENTS = (
    "uncertainty_decomposition",
    "uncertainty_filtering",
    "uncertainty_guided_generation",
    "uncertainty_verified_selection",
    "uncertainty_triggered_repair",
)
