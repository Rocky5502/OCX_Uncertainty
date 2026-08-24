"""Base vector-store retriever (cosine similarity over L2-normalized embeddings)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np


@dataclass
class Hit:
    item: Any
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class VectorIndex:
    def __init__(self, encoder):
        self.encoder = encoder
        self._matrix: np.ndarray | None = None
        self._items: List[Any] = []
        self._texts: List[str] = []

    def build(self, items: Sequence[Any], texts: Sequence[str]) -> None:
        assert len(items) == len(texts)
        self._items = list(items)
        self._texts = list(texts)
        if not texts:
            self._matrix = np.zeros((0, self.encoder.dim), dtype=np.float32)
            return
        self._matrix = self.encoder.encode(self._texts)

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = self.encoder.encode([query])[0]
        sims = self._matrix @ q
        idx = np.argsort(-sims)[: top_k]
        return [Hit(item=self._items[i], score=float(sims[i]), source="base") for i in idx]
