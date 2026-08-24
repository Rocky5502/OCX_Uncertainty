import json

import pytest

from opencoderx.provenance import AppendOnlyResultStore, ResponseCache, RunRecord


def _record(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id, dataset="d", dataset_version="v", task_id="t",
        repository="r", repository_commit="c", language="Python",
        model_provider="gateway", model_family="GPT", model_id="m",
        model_revision="rev", method="OpenCoderX", prompt_hash="p",
        retrieval_hash="q", status="COMPLETED", functional_correctness=True,
    )


def test_append_only_store_rejects_duplicate_run_ids(tmp_path):
    store = AppendOnlyResultStore(tmp_path / "raw.jsonl")
    store.append(_record("one"))
    with pytest.raises(ValueError, match="duplicate"):
        store.append(_record("one"))
    assert json.loads((tmp_path / "raw.jsonl").read_text())["run_id"] == "one"


def test_response_cache_requires_full_parameter_identity(tmp_path):
    cache = ResponseCache(tmp_path)
    parameters = {
        "provider": "g", "model": "m", "prompt_hash": "p", "task_id": "t",
        "method": "x", "temperature": 0.7, "seed": 1, "context_hash": "c",
        "retrieval_hash": "r", "experiment_version": "v1",
    }
    path = cache.put(parameters, {"text": "answer"})
    assert path.is_file()
    assert cache.get(parameters) == {"text": "answer"}
    with pytest.raises(ValueError, match="non-identical"):
        cache.put(parameters, {"text": "different"})
