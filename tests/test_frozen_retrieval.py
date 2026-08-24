from __future__ import annotations

import json

from opencoder.embeddings import Encoder
from opencoderx.frozen_retrieval import build_frozen_retrievers


def test_frozen_retrievers_preserve_document_ids(tmp_path) -> None:
    rows = [
        {"document_id": "api-a", "repository": "r", "path": "a.py", "qualified_name": "f", "line": 1, "source_type": "api", "content": "def f(x):\nAdd one."},
        {"document_id": "similar-a", "repository": "r", "path": "a.py", "qualified_name": "f", "line": 1, "source_type": "similar_code", "content": "def f(x):\n    return x + 1"},
        {"document_id": "context-a", "repository": "r", "path": "b.py", "source_type": "context", "content": "from a import f\nvalue = f(1)"},
    ]
    path = tmp_path / "index.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _, retrievers = build_frozen_retrievers(path, "r", Encoder())
    assert retrievers["api"].search("add", 1)[0].item.metadata["document_id"] == "api-a"
    assert retrievers["similar_code"].search("return", 1)[0].item.metadata["document_id"] == "similar-a"
    assert retrievers["context"].search("value", 1)[0].item.metadata["document_id"] == "context-a"
