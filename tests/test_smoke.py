"""Smoke test that exercises the pipeline plumbing without hitting the LLM.

Verifies: AST extraction, embedding fallback, uncertainty math, static
checks. The LLM-dependent paths are covered in integration tests.
"""
from __future__ import annotations

import os
import tempfile

from opencoder.embeddings import Encoder
from opencoder.phase1_repo_knowledge.extract import extract_repo_knowledge
from opencoder.phase3_retrieval._base import VectorIndex
from opencoder.phase5_verify.static_checks import static_check
from opencoder.uncertainty import (
    aggregate_uncertainty,
    self_consistency_score,
    semantic_variance_score,
    token_entropy_score,
)


def test_extract_and_index():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "m.py"), "w") as f:
            f.write("def add(a, b):\n    '''add two ints'''\n    return a + b\n\nclass C: pass\n")
        items = extract_repo_knowledge(d)
        assert any(it.qualname == "add" for it in items)
        assert any(it.qualname == "C" for it in items)
        enc = Encoder()
        idx = VectorIndex(enc)
        idx.build(items, [it.body for it in items])
        hits = idx.search("add numbers", top_k=2)
        assert hits and hits[0].score >= -1.0


def test_uncertainty_aggregate():
    te = token_entropy_score([-0.1, -0.2, -0.1])
    sc = self_consistency_score(["x", "x", "y"])
    sv = semantic_variance_score(["x", "y", "z"], Encoder())
    tr = aggregate_uncertainty(te, sc, sv)
    assert 0.0 <= tr.aggregate <= 1.0


def test_static_check():
    ok = static_check("def f(x):\n    return x + 1\n")
    assert ok.ok and not ok.syntax_error
    bad = static_check("def f(:\n    return 1\n")
    assert not bad.ok


if __name__ == "__main__":
    test_extract_and_index()
    test_uncertainty_aggregate()
    test_static_check()
    print("smoke tests OK")
