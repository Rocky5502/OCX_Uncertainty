import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from human_study.run_agent_replication import condition_prompt
from human_study.serve_study import Handler, StudyState


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
CONFIG = json.loads((STUDY / "study_config.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_frozen_stimuli_are_balanced_and_hide_outcomes():
    public = read_jsonl(FROZEN / "stimuli_public.jsonl")
    private = read_jsonl(FROZEN / "stimuli_private.jsonl")
    assert len(public) == len(private) == 12
    assert len({row["repository"] for row in public}) == 12
    assert Counter(row["signal_category"] for row in private) == Counter({
        "api": 2, "context": 2, "similar_code": 2,
        "generation": 2, "correct_control": 4,
    })
    assert sum(row["initial_correct"] for row in private) == 4
    forbidden = {"solution", "reference_code", "initial_correct", "passed", "test_report"}
    assert all(not (forbidden & set(row)) for row in public)


def test_human_schedule_is_within_subject_and_counterbalanced():
    with (FROZEN / "human_randomization_schedule.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    task_condition = Counter()
    for row in rows:
        grouped[row["participant_code"]].append(row)
        task_condition[(row["task_id"], row["condition"])] += 1
    invitations = int(CONFIG["invitation_codes_planned"])
    assert len(rows) == invitations * int(CONFIG["tasks_per_participant"])
    assert len(grouped) == invitations
    for participant_rows in grouped.values():
        assert len(participant_rows) == 12
        assert len({row["task_id"] for row in participant_rows}) == 12
        assert Counter(row["condition"] for row in participant_rows) == Counter({
            "generic_review": 4, "uncertainty_display": 4, "targeted_guidance": 4,
        })
    for task_id in {row["task_id"] for row in rows}:
        counts = [task_condition[(task_id, condition)] for condition in (
            "generic_review", "uncertainty_display", "targeted_guidance"
        )]
        assert max(counts) - min(counts) <= 1


def test_invitation_pool_matches_schedule():
    with (FROZEN / "invitation_codes.csv").open(newline="", encoding="utf-8") as handle:
        codes = [row["participant_code"] for row in csv.DictReader(handle)]
    assert len(codes) == int(CONFIG["invitation_codes_planned"])
    assert len(codes) == len(set(codes))
    assert codes[0] == "H001"
    assert codes[-1] == "H100"


def test_condition_prompts_only_expose_the_randomized_intervention():
    stimulus = read_jsonl(FROZEN / "stimuli_public.jsonl")[0]
    generic = condition_prompt(stimulus, "generic_review")
    uncertainty = condition_prompt(stimulus, "uncertainty_display")
    targeted = condition_prompt(stimulus, "targeted_guidance")
    assert "# Uncertainty trace" not in generic
    assert "# Recommended review action" not in generic
    assert "# Uncertainty trace" in uncertainty
    assert "# Recommended review action" not in uncertainty
    assert "# Uncertainty trace" in targeted
    assert "# Recommended review action" in targeted
    for prompt in (generic, uncertainty, targeted):
        assert str(stimulus["starting_code"]) in prompt


def test_empirical_server_fails_closed_without_ethics_file(tmp_path):
    with pytest.raises(RuntimeError, match="ethics"):
        StudyState(tmp_path, "empirical", None)
    dry_run = StudyState(tmp_path, "dry-run", None)
    assert dry_run.mode_label == "SIMULATED_DRY_RUN"


def test_request_handler_does_not_override_http_connection_finish():
    assert "finish" not in Handler.__dict__
    assert "finish_study" in Handler.__dict__


def test_recruitment_closes_after_completion_target(tmp_path):
    target = int(CONFIG["participants_planned"])
    rows = [
        {"participant_code": f"H{number:03d}", "study_mode": "SIMULATED_DRY_RUN"}
        for number in range(1, target + 1)
    ]
    (tmp_path / "poststudy.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    state = StudyState(tmp_path, "dry-run", None)
    assert len(state.completed_participant_codes()) == target
    assert state.recruitment_closed() is True


def test_all_json_schemas_are_valid_draft_2020_12():
    for path in sorted((STUDY / "schemas").glob("*.schema.json")):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_synthetic_participants_conform_to_record_schema():
    schema = json.loads((STUDY / "schemas/participant.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for row in read_jsonl(STUDY / "dry_run/participants.jsonl"):
        validator.validate(row)


def test_agent_runner_record_contract_accepts_expected_provenance():
    schema = json.loads((STUDY / "schemas/agent_episode.schema.json").read_text(encoding="utf-8"))
    row = {
        "record_type": "agent_episode", "study_mode": "AGENT_EXPLORATORY",
        "agent_session_id": "A001", "model": "gpt-4o-mini", "seed": 1,
        "episode_index": 1, "task_id": "task", "condition": "generic_review",
        "signal_category": "api", "initial_correct": False,
        "starting_judgment_correct": False, "starting_confidence": 80,
        "final_code": "def target():\n    return 1", "final_correct": True,
        "evaluator_status": "ok", "raw_response": "{}", "response_metadata": {},
        "usage": {"total_tokens": 10}, "latency_seconds": 0.1, "error": None,
    }
    Draft202012Validator(schema).validate(row)


def test_dry_run_artifacts_are_explicitly_non_empirical():
    marker = json.loads((STUDY / "dry_run/SIMULATION_MARKER.json").read_text(encoding="utf-8"))
    integrity = json.loads((STUDY / "dry_run/analysis/integrity.json").read_text(encoding="utf-8"))
    assert marker["status"] == "SIMULATED_DRY_RUN_NOT_EMPIRICAL"
    assert integrity["study_mode"] == "SIMULATED_DRY_RUN"
    assert integrity["paper_eligible"] is False
    result_text = (STUDY / "dry_run/analysis/latex/human_study_results.tex").read_text(encoding="utf-8")
    assert "SIMULATED DRY RUN--NOT EMPIRICAL EVIDENCE" in result_text
