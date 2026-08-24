"""Phase III: Similar-Code Retriever.

Indexes function bodies and retrieves the most similar implementations
to the query (or to a draft generation). This is the source that, per
our RQ1 analysis, has the strongest correlation with reduced output
uncertainty when relevant — and the strongest correlation with
*hallucination* when irrelevant. Hence uncertainty-aware filtering in
Step 8 is critical for this source.
"""
from __future__ import annotations

from typing import List, Sequence

from ..phase1_repo_knowledge.extract import RepoFunction
from ._base import Hit, VectorIndex


class SimilarCodeRetriever:
    def __init__(self, encoder):
        self.index = VectorIndex(encoder)

    def build(self, items: Sequence[RepoFunction]) -> None:
        texts = [it.body for it in items]
        self.index.build(items, texts)

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        hits = self.index.search(query, top_k)
        for h in hits:
            h.source = "similar_code"
            h.metadata["knowledge_uncertainty"] = h.item.uncertainty
        return hits
