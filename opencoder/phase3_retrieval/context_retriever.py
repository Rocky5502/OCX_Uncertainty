"""Phase III: Context Retriever.

Indexes whole-file or surrounding-context snippets so the generator
sees how a function is *used* in the repo, not just its definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from ._base import Hit, VectorIndex


@dataclass
class ContextChunk:
    file_path: str
    text: str
    metadata: dict = field(default_factory=dict)


class ContextRetriever:
    def __init__(self, encoder, chunk_chars: int = 1200, overlap: int = 200):
        self.encoder = encoder
        self.index = VectorIndex(encoder)
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    @staticmethod
    def chunk_file(path: str, text: str, size: int, overlap: int) -> List[ContextChunk]:
        out: List[ContextChunk] = []
        i = 0
        while i < len(text):
            out.append(ContextChunk(file_path=path, text=text[i : i + size]))
            i += max(1, size - overlap)
        return out

    def build_from_files(self, files: Sequence[tuple]) -> None:
        """files: iterable of (relative_path, file_contents)."""
        chunks: List[ContextChunk] = []
        for path, txt in files:
            chunks.extend(self.chunk_file(path, txt, self.chunk_chars, self.overlap))
        self.index.build(chunks, [c.text for c in chunks])

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        hits = self.index.search(query, top_k)
        for h in hits:
            h.source = "context"
        return hits
