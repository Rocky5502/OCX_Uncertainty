"""Aggregate the three uncertainty signals into a single trace.

The trace is what Phase IV ("Uncertainty-Guided Code Generation") consumes
and what RQ1 correlates against retrieval source quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class UncertaintyTrace:
    token_entropy: float
    self_consistency: float
    semantic_variance: float
    aggregate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def aggregate_uncertainty(
    token_entropy: float,
    self_consistency: float,
    semantic_variance: float,
    weights: Dict[str, float] | None = None,
) -> UncertaintyTrace:
    w = weights or {
        "token_entropy_weight": 0.4,
        "self_consistency_weight": 0.4,
        "semantic_variance_weight": 0.2,
    }
    agg = (
        w["token_entropy_weight"] * token_entropy
        + w["self_consistency_weight"] * self_consistency
        + w["semantic_variance_weight"] * semantic_variance
    )
    return UncertaintyTrace(
        token_entropy=token_entropy,
        self_consistency=self_consistency,
        semantic_variance=semantic_variance,
        aggregate=float(agg),
    )
