"""Semantic-cluster-variance uncertainty signal.

Embed each of N sampled completions, compute pairwise cosine similarity,
and define uncertainty as 1 - mean_pairwise_similarity. This captures
disagreement that token-level metrics miss (e.g. two completions that
are syntactically different but semantically the same will cluster
together in embedding space).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def semantic_variance_score(samples: Sequence[str], encoder) -> float:
    if len(samples) < 2:
        return 0.0
    vecs = encoder.encode(list(samples))  # already L2-normalized
    sim = vecs @ vecs.T
    # Mean of upper-triangular (excluding diagonal).
    n = len(samples)
    iu = np.triu_indices(n, k=1)
    mean_sim = float(np.mean(sim[iu])) if iu[0].size else 1.0
    return float(max(0.0, min(1.0, 1.0 - mean_sim)))
