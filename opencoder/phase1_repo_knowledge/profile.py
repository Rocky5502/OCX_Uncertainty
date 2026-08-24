"""Phase I, Step 3: Profile Uncertainty for each knowledge item.

We attach a per-item uncertainty score derived from the description
generation step's token logprobs. Items the model was less sure about
(complex / underspecified code) get higher scores; downstream retrieval
uses these to weight evidence (Phase III, Step 8).
"""
from __future__ import annotations

from typing import List

from ..uncertainty.token_entropy import token_entropy_score
from .extract import RepoFunction


def profile_knowledge_uncertainty(items: List[RepoFunction]) -> List[RepoFunction]:
    for it in items:
        lps = it.metadata.get("describe_logprobs")
        it.uncertainty = token_entropy_score(lps)
    return items
