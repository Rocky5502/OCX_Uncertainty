"""Provider-independent uncertainty signals used by the TOSEM campaign."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_code(code: str) -> str:
    text = re.sub(r"```(?:[A-Za-z0-9_+#.-]+)?", "", str(code or ""))
    text = text.replace("```", "").strip()
    try:
        return ast.dump(ast.parse(text), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return " ".join(text.split())


def _binary_entropy(values: Sequence[bool | None]) -> float:
    observed = [bool(value) for value in values if value is not None]
    if not observed:
        return 1.0
    probability = sum(observed) / len(observed)
    if probability in {0.0, 1.0}:
        return 0.0
    return _clip(
        -(probability * math.log2(probability)
          + (1.0 - probability) * math.log2(1.0 - probability))
    )


def candidate_disagreement(candidates: Sequence[str]) -> float:
    """Return one minus mean pairwise normalized-code similarity."""
    normalized = [_normalize_code(candidate) for candidate in candidates]
    if len(normalized) < 2:
        return 0.0
    similarities = [
        SequenceMatcher(None, normalized[left], normalized[right]).ratio()
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    return _clip(1.0 - mean(similarities))


def retrieval_score_dispersion(scores: Sequence[float]) -> float:
    """Scale score dispersion by the observed score range."""
    values = [float(value) for value in scores]
    if len(values) < 2:
        return 0.0
    observed_range = max(values) - min(values)
    if observed_range <= 0.0:
        return 0.0
    return _clip(pstdev(values) / observed_range)


def retrieval_source_disagreement(
    source_scores: Mapping[str, Sequence[float]],
) -> float:
    source_means = [mean(values) for values in source_scores.values() if values]
    if len(source_means) < 2:
        return 0.0
    return _clip(max(source_means) - min(source_means))


def evidence_completeness(
    source_scores: Mapping[str, Sequence[float]],
    expected_sources: Sequence[str] = ("api", "context", "similar_code"),
) -> float:
    """Return uncertainty induced by absent evidence sources."""
    missing = sum(not source_scores.get(source) for source in expected_sources)
    return missing / len(expected_sources) if expected_sources else 0.0


def context_sensitivity(candidate_sets: Sequence[Sequence[str]]) -> float:
    """Measure output variation across context perturbations."""
    representatives = ["\n".join(group) for group in candidate_sets if group]
    return candidate_disagreement(representatives)


def repair_instability(repair_candidates: Sequence[str]) -> float:
    return candidate_disagreement(repair_candidates)


@dataclass(frozen=True)
class ProviderIndependentUncertainty:
    semantic_self_consistency: float
    candidate_disagreement: float
    execution_uncertainty: float
    retrieval_score_dispersion: float
    retrieval_source_disagreement: float
    evidence_incompleteness: float
    context_sensitivity: float
    verifier_disagreement: float
    repair_instability: float
    response_variation: float
    aggregate_risk: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_uncertainty(
    *,
    candidates: Sequence[str],
    candidate_test_outcomes: Sequence[bool | None] = (),
    source_scores: Mapping[str, Sequence[float]] | None = None,
    context_candidate_sets: Sequence[Sequence[str]] = (),
    verifier_outcomes: Sequence[bool | None] = (),
    repair_candidates: Sequence[str] = (),
    repeated_responses: Sequence[str] = (),
) -> ProviderIndependentUncertainty:
    """Compute bounded signals without relying on provider log probabilities."""
    source_scores = source_scores or {}
    disagreement = candidate_disagreement(candidates)
    signals = {
        "candidate_disagreement": disagreement,
        "execution_uncertainty": _binary_entropy(candidate_test_outcomes),
        "retrieval_score_dispersion": mean(
            [retrieval_score_dispersion(values) for values in source_scores.values()]
            or [0.0]
        ),
        "retrieval_source_disagreement": retrieval_source_disagreement(source_scores),
        "evidence_incompleteness": evidence_completeness(source_scores),
        "context_sensitivity": context_sensitivity(context_candidate_sets),
        "verifier_disagreement": _binary_entropy(verifier_outcomes),
        "repair_instability": repair_instability(repair_candidates),
        "response_variation": candidate_disagreement(repeated_responses),
    }
    aggregate = mean(signals.values())
    return ProviderIndependentUncertainty(
        semantic_self_consistency=_clip(1.0 - disagreement),
        aggregate_risk=_clip(aggregate),
        **{key: _clip(value) for key, value in signals.items()},
    )
