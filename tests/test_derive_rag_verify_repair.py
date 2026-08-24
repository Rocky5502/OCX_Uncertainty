from __future__ import annotations

from experiments.derive_rag_verify_repair import _derive_row


class Report:
    def __init__(self, passed):
        self.passed = passed
        self.stdout = ""
        self.stderr = ""
        self.returncode = 0 if passed else 1


class StaticReport:
    def __init__(self):
        self.ok = True


class FakeConfig:
    n_samples_for_uncertainty = 5


class FakePipeline:
    cfg = FakeConfig()

    @staticmethod
    def _normalize_completion_code(sample, _example):
        return sample

    @staticmethod
    def _validate_code(code, _example):
        passed = code == "passing"
        return StaticReport(), Report(passed)


class Example:
    id = "task-1"


def baseline(samples):
    return {
        "id": "task-1",
        "generated_samples": samples,
        "sample_correctness": [sample == "passing" for sample in samples],
        "test_report": {"passed": False},
        "llm_usage": {"total_tokens": 100},
    }


def test_derivation_selects_exact_passing_baseline_candidate():
    row, status = _derive_row(
        baseline(["bad", "passing", "bad", "bad", "bad"]),
        Example(),
        FakePipeline(),
    )
    assert row is not None
    assert row["code"] == "passing"
    assert row["selected_sample_index"] == 1
    assert row["repair_rounds"] == 0
    assert row["effective_sample_correctness"][0] is True
    assert row["llm_usage"]["total_tokens"] == 100
    assert status["reason"] == "verified_candidate_selected"


def test_derivation_leaves_all_failing_candidate_set_pending():
    row, status = _derive_row(
        baseline(["bad"] * 5),
        Example(),
        FakePipeline(),
    )
    assert row is None
    assert status["reason"] == "repair_required"
