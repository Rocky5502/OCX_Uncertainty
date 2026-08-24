"""Transparent autonomy, review, abstention, and review-budget policies."""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil
from typing import Iterable, Mapping, Sequence


class CollaborationDecision(Enum):
    AUTONOMOUS = "autonomous"
    REQUEST_REVIEW = "request_review"
    ABSTAIN = "abstain"


INTERVENTIONS = {
    "api": "inspect or correct repository API evidence",
    "context": "provide the missing repository file or dependency",
    "similar_code": "replace incompatible similar-code evidence",
    "generation": "review an alternative candidate or validation trace",
    "verification": "run or inspect additional executable checks",
    "repair": "review validation-guided repair",
}


@dataclass(frozen=True)
class RiskTrace:
    api: float = 0.0
    context: float = 0.0
    similar_code: float = 0.0
    generation: float = 0.0
    verification: float = 0.0
    repair: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    def sources(self) -> dict[str, float]:
        return {
            key: max(0.0, min(1.0, float(value)))
            for key, value in asdict(self).items()
            if key != "evidence_ids"
        }

    def aggregate(self, weights: Mapping[str, float] | None = None) -> float:
        values = self.sources()
        weights = dict(weights or {key: 1.0 for key in values})
        denominator = sum(max(0.0, weights.get(key, 0.0)) for key in values)
        if denominator <= 0.0:
            raise ValueError("risk weights must contain at least one positive value")
        return sum(
            values[key] * max(0.0, weights.get(key, 0.0)) for key in values
        ) / denominator

    def highest_source(self) -> str:
        order = ("api", "context", "similar_code", "generation", "verification", "repair")
        values = self.sources()
        return max(order, key=lambda key: (values[key], -order.index(key)))


@dataclass(frozen=True)
class DecisionRecord:
    task_id: str
    model: str
    risk_score: float
    decision: CollaborationDecision
    uncertainty_sources: Mapping[str, float]
    highest_risk_source: str
    evidence_ids: tuple[str, ...]
    reason: str
    recommended_intervention: str
    policy: str
    threshold: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


@dataclass(frozen=True)
class CollaborationPolicy:
    name: str = "source_specific_opencoderx_deferral"
    review_threshold: float = 0.35
    abstain_threshold: float = 0.70
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold < self.abstain_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= review < abstain <= 1")

    def decide(
        self,
        *,
        task_id: str,
        model: str,
        trace: RiskTrace,
        irrecoverable: bool = False,
    ) -> DecisionRecord:
        risk = trace.aggregate(self.weights or None)
        source = trace.highest_source()
        if irrecoverable or risk >= self.abstain_threshold:
            decision = CollaborationDecision.ABSTAIN
            reason = "irrecoverable validation failure" if irrecoverable else "risk meets abstention threshold"
        elif risk >= self.review_threshold:
            decision = CollaborationDecision.REQUEST_REVIEW
            reason = "risk meets developer-review threshold"
        else:
            decision = CollaborationDecision.AUTONOMOUS
            reason = "risk is below developer-review threshold"
        return DecisionRecord(
            task_id=task_id,
            model=model,
            risk_score=risk,
            decision=decision,
            uncertainty_sources=trace.sources(),
            highest_risk_source=source,
            evidence_ids=trace.evidence_ids,
            reason=reason,
            recommended_intervention=INTERVENTIONS[source],
            policy=self.name,
            threshold={
                "review": self.review_threshold,
                "abstain": self.abstain_threshold,
            },
        )


def _stable_random(task_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{task_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random()


def policy_score(
    record: Mapping[str, object],
    policy: str,
    *,
    seed: int = 20260809,
) -> float:
    if policy == "always_autonomous":
        return 0.0
    if policy == "random_deferral":
        return _stable_random(str(record["task_id"]), seed)
    if policy == "test_failure_deferral":
        # This must be an observable pre-repair signal. Using final selected
        # correctness here would leak the outcome the policy is evaluated on.
        return 1.0 if record.get("pre_repair_test_failed") is True else 0.0
    if policy == "confidence_deferral":
        return float(record.get("generation_uncertainty") or 0.0)
    if policy == "entropy_deferral":
        return float(record.get("token_entropy") or 0.0)
    if policy == "self_consistency_deferral":
        return float(record.get("candidate_disagreement") or 0.0)
    if policy == "aggregate_uncertainty_deferral":
        return float(record.get("aggregate_risk") or 0.0)
    if policy == "source_specific_opencoderx_deferral":
        source_values = record.get("uncertainty_sources") or {}
        if isinstance(source_values, Mapping) and source_values:
            return max(float(value) for value in source_values.values())
        return float(record.get("aggregate_risk") or 0.0)
    if policy == "oracle_deferral":
        return 1.0 if record.get("selected_output_correct") is False else 0.0
    raise ValueError(f"unknown deferral policy: {policy}")


def allocate_review_budget(
    records: Sequence[Mapping[str, object]],
    *,
    policy: str,
    review_budget: float,
    seed: int = 20260809,
) -> set[str]:
    """Return deterministically ranked task IDs selected for review."""
    if not 0.0 <= review_budget <= 1.0:
        raise ValueError("review_budget must lie in [0, 1]")
    if policy == "always_autonomous":
        return set()
    count = min(len(records), ceil(len(records) * review_budget))
    ranked = sorted(
        records,
        key=lambda record: (
            -policy_score(record, policy, seed=seed),
            str(record["task_id"]),
        ),
    )
    return {str(record["task_id"]) for record in ranked[:count]}


def review_allocation_metrics(
    records: Sequence[Mapping[str, object]],
    reviewed_task_ids: Iterable[str],
    *,
    reviewer_success: float,
    seed: int,
) -> dict[str, float]:
    """Simulate reviewer success without describing it as observed human data."""
    reviewed = set(reviewed_task_ids)
    if not 0.0 <= reviewer_success <= 1.0:
        raise ValueError("reviewer_success must lie in [0, 1]")
    failures = [record for record in records if record.get("selected_output_correct") is False]
    prevented = 0
    for record in failures:
        task_id = str(record["task_id"])
        # Namespace the reviewer draw so it is independent of any random
        # deferral ranking produced with the same task and seed.
        if task_id in reviewed and _stable_random(f"reviewer|{task_id}", seed) < reviewer_success:
            prevented += 1
    autonomous = [record for record in records if str(record["task_id"]) not in reviewed]
    autonomous_correct = sum(record.get("selected_output_correct") is True for record in autonomous)
    reviewed_failures = sum(str(record["task_id"]) in reviewed for record in failures)
    unnecessary = sum(
        str(record["task_id"]) in reviewed and record.get("selected_output_correct") is True
        for record in records
    )
    total = len(records)
    n_reviewed = len(reviewed)
    baseline_correct = sum(record.get("selected_output_correct") is True for record in records)
    return {
        "team_success_rate": (baseline_correct + prevented) / total if total else 0.0,
        "selective_accuracy": autonomous_correct / len(autonomous) if autonomous else 1.0,
        "autonomous_coverage": len(autonomous) / total if total else 0.0,
        "autonomous_failure_rate": (
            (len(autonomous) - autonomous_correct) / len(autonomous) if autonomous else 0.0
        ),
        "failure_capture_rate": reviewed_failures / len(failures) if failures else 1.0,
        "deferral_precision": reviewed_failures / n_reviewed if n_reviewed else 0.0,
        "unnecessary_review_rate": unnecessary / n_reviewed if n_reviewed else 0.0,
        "errors_prevented_per_review": prevented / n_reviewed if n_reviewed else 0.0,
        "reviews_per_prevented_failure": n_reviewed / prevented if prevented else float("inf"),
    }
