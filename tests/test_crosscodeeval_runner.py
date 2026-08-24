from experiments.run_crosscodeeval import _context_evidence, _messages


def _row():
    return {
        "task_id": "t",
        "language": "python",
        "prompt": "prefix",
        "right_context": "SECRET_FUTURE_SUFFIX",
        "crossfile_context": {
            "list": [{"filename": "other.py", "retrieved_chunk": "helper()", "score": 0.8}]
        },
    }


def test_crosscodeeval_direct_omits_retrieval_and_future_suffix() -> None:
    messages, evidence = _messages(_row(), "direct")
    prompt = messages[-1]["content"]
    assert evidence == []
    assert "helper()" not in prompt
    assert "SECRET_FUTURE_SUFFIX" not in prompt


def test_crosscodeeval_context_rag_uses_frozen_evidence() -> None:
    messages, evidence = _messages(_row(), "context_rag")
    assert "helper()" in messages[-1]["content"]
    assert len(evidence) == 1
    assert evidence[0]["score"] == 0.8
    assert evidence[0]["document_id"].startswith("sha256:")
