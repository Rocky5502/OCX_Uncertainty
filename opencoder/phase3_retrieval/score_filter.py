"""Phase III, Step 8: Score & Filter Candidates by Uncertainty.

We combine three signals per candidate:
  - retrieval_score (cosine similarity): higher = better
  - knowledge_uncertainty (Phase I, Step 3): higher = riskier
  - source_weight (Phase II, Step 6 intent): higher = more relevant source

Composite score:
    final = retrieval_score * source_weight * (1 - alpha * knowledge_uncertainty)

Then we keep only candidates whose `final` is in the top
`uncertainty_filter_quantile` of their source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Sequence

import numpy as np

from ._base import Hit


@dataclass
class Candidate:
    hit: Hit
    final_score: float
    components: dict = field(default_factory=dict)


def score_and_filter(
    hits_by_source: Dict[str, List[Hit]],
    source_weights: Dict[str, float],
    *,
    knowledge_uncertainty_alpha: float = 0.5,
    keep_quantile: float = 0.7,
) -> List[Candidate]:
    out: List[Candidate] = []
    for source, hits in hits_by_source.items():
        sw = float(source_weights.get(source, 0.0))
        if not hits or sw <= 0:
            continue
        scored: List[Candidate] = []
        for h in hits:
            ku = float(h.metadata.get("knowledge_uncertainty") or 0.0)
            final = h.score * sw * max(0.0, 1.0 - knowledge_uncertainty_alpha * ku)
            scored.append(
                Candidate(
                    hit=h,
                    final_score=final,
                    components={
                        "retrieval_score": h.score,
                        "source_weight": sw,
                        "knowledge_uncertainty": ku,
                    },
                )
            )
        # Filter to top quantile per-source.
        if len(scored) > 1:
            thr = float(np.quantile([c.final_score for c in scored], 1.0 - keep_quantile))
            scored = [c for c in scored if c.final_score >= thr]
        out.extend(scored)
    out.sort(key=lambda c: -c.final_score)
    return out


def merge_step_candidates(per_step: Sequence[List[Candidate]], fused_top_k: int) -> List[Candidate]:
    """Merge candidates across all implementation steps, deduping by item identity."""
    seen = {}
    for cands in per_step:
        for c in cands:
            key = id(c.hit.item)
            if key not in seen or c.final_score > seen[key].final_score:
                seen[key] = c
    merged = sorted(seen.values(), key=lambda c: -c.final_score)
    return merged[:fused_top_k]


def merge_step_candidates_balanced(
    per_step: Sequence[List[Candidate]],
    fused_top_k: int,
    *,
    max_source_fraction: float = 0.5,
) -> List[Candidate]:
    """Merge candidates while preserving coverage and limiting source dominance.

    The cap applies only when alternatives from other sources exist. If the
    available pool contains fewer sources, the remaining slots are filled by
    score so the final evidence budget is not artificially reduced.
    """
    if fused_top_k <= 0:
        return []
    if not 0.0 < max_source_fraction <= 1.0:
        raise ValueError("max_source_fraction must be in (0, 1]")

    seen: dict[int, Candidate] = {}
    for candidates in per_step:
        for candidate in candidates:
            key = id(candidate.hit.item)
            if key not in seen or candidate.final_score > seen[key].final_score:
                seen[key] = candidate
    merged = sorted(seen.values(), key=lambda candidate: -candidate.final_score)
    if not merged:
        return []

    max_per_source = max(1, math.ceil(fused_top_k * max_source_fraction))
    by_source: dict[str, list[Candidate]] = {}
    for candidate in merged:
        by_source.setdefault(candidate.hit.source, []).append(candidate)

    selected: list[Candidate] = []
    selected_ids: set[int] = set()
    source_counts: dict[str, int] = {}

    # Reserve the best available item from every source before filling by score.
    for source_candidates in sorted(
        by_source.values(), key=lambda values: -values[0].final_score
    ):
        if len(selected) >= fused_top_k:
            break
        candidate = source_candidates[0]
        selected.append(candidate)
        selected_ids.add(id(candidate))
        source_counts[candidate.hit.source] = 1

    for candidate in merged:
        if len(selected) >= fused_top_k:
            break
        if id(candidate) in selected_ids:
            continue
        source = candidate.hit.source
        if source_counts.get(source, 0) >= max_per_source:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        source_counts[source] = source_counts.get(source, 0) + 1

    # Preserve the declared final budget when the available source mix cannot
    # satisfy the cap (for example, a repository with API evidence only).
    for candidate in merged:
        if len(selected) >= fused_top_k:
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))

    return sorted(selected, key=lambda candidate: -candidate.final_score)
