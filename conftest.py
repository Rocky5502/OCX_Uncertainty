"""Skip only tests whose restricted source-bearing fixtures are not released."""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent

REQUIREMENTS = {
    "test_crosscodeeval_gold_self_match_for_all_languages": (
        "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl",
    ),
    "test_frozen_stimuli_are_balanced_and_hide_outcomes": (
        "human_study/frozen/stimuli_public.jsonl",
    ),
    "test_human_schedule_is_within_subject_and_counterbalanced": (
        "human_study/frozen/human_randomization_schedule.csv",
    ),
    "test_invitation_pool_matches_schedule": (
        "human_study/frozen/invitation_codes.csv",
    ),
    "test_condition_prompts_only_expose_the_randomized_intervention": (
        "human_study/frozen/stimuli_public.jsonl",
    ),
    "test_empirical_server_fails_closed_without_ethics_file": (
        "human_study/frozen/human_randomization_schedule.csv",
    ),
    "test_recruitment_closes_after_completion_target": (
        "human_study/frozen/human_randomization_schedule.csv",
    ),
    "test_synthetic_participants_conform_to_record_schema": (
        "human_study/dry_run/participants.jsonl",
    ),
    "test_dry_run_artifacts_are_explicitly_non_empirical": (
        "human_study/dry_run/SIMULATION_MARKER.json",
    ),
    "test_bundled_dataset_loader_shapes": ("input/input.jsonl",),
    "test_opencoderx_execrepobench_loader_uses_function_protocol": (
        "data/manifests/execrepobench_opencoderx_120_v1.jsonl",
    ),
    "test_config_and_standard_rag_baseline": ("input/input.jsonl",),
    "test_test_context_is_excluded_by_default": ("input/execrepobench_data.jsonl",),
    "test_execrepobench_reference_uses_repository_tests": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_pipeline_marks_execrepobench_repository_tests": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_repository_test_failure_is_repairable": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_completion_validation_normalizes_missing_region_indent": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_completion_validation_repairs_common_hanging_indent": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_uncertainty_aware_selection_prefers_verified_sample": (
        "input/execrepobench_testbacked.jsonl",
    ),
    "test_completion_indent_after_function_header": ("input/execrepobench_stable7.jsonl",),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        required = REQUIREMENTS.get(item.name)
        if not required:
            continue
        missing = [relative for relative in required if not (ROOT / relative).is_file()]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason="restricted source-bearing fixture is not distributed: " + ", ".join(missing)
                )
            )
