"""Phase III: API Retriever.

Indexes (signature + NL description) for each repo function. Returns
the top-k API knowledge items semantically matching the query.
"""
from __future__ import annotations

from typing import List, Sequence

from ..phase1_repo_knowledge.extract import RepoFunction
from ._base import Hit, VectorIndex


class APIRetriever:
    def __init__(self, encoder):
        self.index = VectorIndex(encoder)

    def build(self, items: Sequence[RepoFunction]) -> None:
        texts = [
            f"{it.signature}\n{it.description or it.docstring or ''}"
            for it in items
        ]
        self.index.build(items, texts)

    @property
    def items(self) -> Sequence[RepoFunction]:
        """Return the indexed API candidate universe for audit/evaluation."""
        return tuple(self.index._items)

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        hits = self.index.search(query, top_k)
        for h in hits:
            h.source = "api"
            h.metadata["knowledge_uncertainty"] = h.item.uncertainty
        return hits
