from __future__ import annotations

from opencoder.data.loaders import load_dataset
from opencoder.evaluation.metrics import (
    estimate_pass_at_k,
    pass_at_k_from_samples,
    pass_rate_variance,
)
from opencoder.llm import LLMClient
from opencoder.pipeline import Pipeline, PipelineConfig
from opencoder.phase1_repo_knowledge.extract import extract_sources_knowledge
from opencoder.phase5_verify.static_checks import static_check
from opencoder.phase5_verify.test_validate import run_repo_completion_tests
from opencoder.phase5_verify.test_validate import run_execrepobench_function_tests
from opencoder.uncertainty import aggregate_uncertainty, self_consistency_score


def _stable_testbacked_example():
    return next(
        ex
        for ex in load_dataset("execrepobench", "input/execrepobench_testbacked.jsonl")
        if ex.id == "shortuuid/0"
    )


def _stable_workdays_example():
    return next(
        ex
        for ex in load_dataset("execrepobench", "input/execrepobench_stable7.jsonl")
        if ex.id == "workdays/0"
    )


def test_in_memory_extraction():
    items = extract_sources_knowledge([
        ("pkg/mod.py", "class C:\n    def add(self, a, b):\n        return a + b\n"),
    ])
    assert any(it.qualname == "C" for it in items)
    assert any(it.qualname == "C.add" and it.kind == "method" for it in items)


def test_description_limit_does_not_truncate_api_index():
    source = "\n\n".join(
        f"def api_{i}():\n    return {i}"
        for i in range(4)
    )
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    items, retrievers = pipe.index_sources(
        [("pkg/mod.py", source)],
        describe_limit=1,
    )
    assert len(items) == 4
    assert len(retrievers["api"].items) == 4
    assert all(
        item.metadata.get("description_fallback") == "description_limit"
        for item in items[1:]
    )


def test_bundled_dataset_loader_shapes():
    sample = next(load_dataset("sample", limit=1))
    assert sample.query
    assert sample.repo_root
    assert sample.reference_code

    execrepobench = next(load_dataset("execrepobench", limit=1))
    assert "Prefix Code" in execrepobench.query
    assert execrepobench.reference_code


def test_opencoderx_execrepobench_loader_uses_function_protocol():
    example = next(load_dataset(
        "execrepobench",
        "data/manifests/execrepobench_opencoderx_120_v1.jsonl",
        limit=1,
    ))
    assert "complete target Python function" in example.query
    assert example.reference_code == example.raw["solution"]
    assert example.repo_root
    assert example.raw["reference_tests_pass"] is True


def test_opencoderx_execrepobench_official_evaluator(tmp_path):
    repo = tmp_path / "toy"
    repo.mkdir()
    (repo / "module.py").write_text(
        "VALUE = 1\n\ndef target(x):\n    return x + 1\n",
        encoding="utf-8",
    )
    (repo / "evaluate_repo.py").write_text(
        "from module import target\nassert target(2) == 3\n",
        encoding="utf-8",
    )
    raw = {
        "manifest_version": "execrepobench_opencoderx_test",
        "repo_name": "toy",
        "file_name": "/toy/module.py",
        "execution_prefix_code": "VALUE = 1\n\n",
        "execution_suffix_code": "",
        "solution": "def target(x):\n    return x + 1\n",
        "_runtime_repo_root": str(repo),
        "_runtime_python_executable": __import__("sys").executable,
    }
    assert run_execrepobench_function_tests(raw["solution"], raw).passed is True
    assert run_execrepobench_function_tests(
        "def target(x):\n    return x - 1\n", raw
    ).passed is False


def test_uncertainty_and_pass_metrics():
    tr = aggregate_uncertainty(0.2, self_consistency_score(["x", "x", "y"]), 0.4)
    assert 0.0 <= tr.aggregate <= 1.0
    assert estimate_pass_at_k(n=5, c=1, k=5) == 1.0
    assert pass_at_k_from_samples([[False, True, False, False, False]], 1) > 0.0
    assert pass_rate_variance([[False, True, False, False]]) > 0.0


def test_static_check():
    assert static_check("def f(x):\n    return x + 1\n").ok
    assert not static_check("def f(:\n    return 1\n").ok


def test_offline_llm_backend_needs_no_api_key():
    client = LLMClient(backend="offline")
    resp = client.complete_one(
        "def reverse(input_string: str) -> str:\n    \"\"\"Reverse a string.\"\"\"\n",
        system="You are a senior Python engineer. Return code.",
    )
    assert "def reverse" in resp.text
    assert resp.logprobs


def test_compatible_backend_uses_neutral_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODER_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENCODER_LLM_API_KEY", "test-only-key")

    client = LLMClient(backend="zhizengzeng", model="gpt-4o-mini")

    assert client.api_key == "test-only-key"
    assert client.endpoint == "https://example.invalid/v1/chat/completions"


def test_config_and_standard_rag_baseline():
    cfg = PipelineConfig.from_yaml("configs/default.yaml")
    assert cfg.llm_temperature == 0.2
    assert cfg.knowledge_uncertainty_alpha == 0.5

    rq3_cfg = PipelineConfig.from_yaml("configs/rq3/gpt4o_mini.yaml")
    assert rq3_cfg.llm_seed == 20260704

    example = next(load_dataset("sample", limit=1))
    pipe = Pipeline(PipelineConfig(
        llm_backend="offline",
        uncertainty_aware=False,
        n_samples_for_uncertainty=2,
    ))
    _, retrievers = pipe.index_example(example, describe_limit=10)
    run = pipe.run(example, retrievers)
    assert len(run.per_step) == 1
    assert run.per_step[0]["step_uncertainty"] == 0.0
    assert run.repair_rounds == 0
    assert run.correctness_mode == "reference_exact_match"


def test_llm_client_sends_requested_seed():
    client = LLMClient.__new__(LLMClient)
    client.backend = "openai"
    client.model = "test-model"
    client.temperature = 0.7
    client.max_tokens = 100
    client.seed = 1234
    client.supports_logprobs = False
    client.response_audit = []
    captured = {}

    def fake_post(body):
        captured.update(body)
        client.response_audit.append({
            "response_id": "resp-test",
            "requested_model": client.model,
        })
        return {
            "id": "resp-test",
            "model": "served-test-model",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "choices": [{"message": {"content": "ok"}}],
        }

    client._post_chat_with_fallbacks = fake_post
    response = client.complete([{"role": "user", "content": "test"}])

    assert response[0].text == "ok"
    assert captured["seed"] == 1234
    assert response[0].raw["_response_metadata"]["response_id"] == "resp-test"
    assert response[0].raw["_response_metadata"]["served_model"] == "served-test-model"
    assert client.response_audit_snapshot()[0]["response_id"] == "resp-test"
    assert client.response_audit_snapshot()[0]["requested_model"] == "test-model"


def test_generation_preserves_provider_response_metadata():
    from opencoder.embeddings import Encoder
    from opencoder.phase4_generation.generate import generate_target_code

    class StubLLM:
        def complete(self, messages, **kwargs):
            from opencoder.llm.client import LLMResponse
            assert kwargs["temperature"] is None
            return [LLMResponse(
                text="```python\ndef f():\n    return 1\n```",
                raw={
                    "finish_reason": "stop",
                    "index": 0,
                    "_response_metadata": {
                        "response_id": "resp-1",
                        "served_model": "claude-sonnet-5",
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                    },
                },
            )]

    result = generate_target_code(
        "implement f", "", {}, StubLLM(), Encoder(),
        n_samples=1, sampling_temperature=None,
    )
    assert result.response_metadata[0]["response_id"] == "resp-1"
    assert result.response_metadata[0]["served_model"] == "claude-sonnet-5"


def test_generation_treats_provider_length_limit_as_scored_failure():
    from opencoder.embeddings import Encoder
    from opencoder.phase4_generation.generate import generate_target_code

    class StubLLM:
        def complete(self, messages, **kwargs):
            from opencoder.llm.client import LLMResponse
            return [LLMResponse(text="", raw={"finish_reason": "max_tokens", "index": 0})]

    result = generate_target_code(
        "implement f", "", {}, StubLLM(), Encoder(), n_samples=1,
    )
    assert result.generation_integrity["valid"] is True
    assert result.generation_integrity["n_length_limited_candidates"] == 1
    assert result.generation_integrity["n_empty_candidates"] == 1
    assert result.generation_integrity["n_unexplained_empty_candidates"] == 0


def test_generation_rejects_unexplained_empty_response():
    from opencoder.embeddings import Encoder
    from opencoder.phase4_generation.generate import generate_target_code

    class StubLLM:
        def complete(self, messages, **kwargs):
            from opencoder.llm.client import LLMResponse
            return [LLMResponse(text="", raw={"finish_reason": "stop", "index": 0})]

    result = generate_target_code(
        "implement f", "", {}, StubLLM(), Encoder(), n_samples=1,
    )
    assert result.generation_integrity["valid"] is False
    assert result.generation_integrity["n_unexplained_empty_candidates"] == 1


def test_test_context_is_excluded_by_default():
    example = next(load_dataset("execrepobench", limit=1))
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    paths = [path for path, _ in pipe._sources_from_example(example)]
    assert paths
    assert not any(pipe._is_test_path(path) for path in paths)


def test_execrepobench_reference_uses_repository_tests():
    example = _stable_testbacked_example()
    report = run_repo_completion_tests(example.reference_code or "", example.raw)
    assert report.passed is True
    assert "passed" in report.stdout


def test_pipeline_marks_execrepobench_repository_tests():
    example = _stable_testbacked_example()
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    assert pipe._correctness_mode(example) == "repository_tests"
    _, report = pipe._validate_code(example.reference_code or "", example)
    assert report.passed is True


def test_repository_test_failure_is_repairable():
    example = _stable_testbacked_example()
    pipe = Pipeline(PipelineConfig(llm_backend="offline", uncertainty_aware=True))
    static_report, test_report = pipe._validate_code("    return ''\n", example)
    assert static_report.ok is True
    assert test_report.passed is False
    assert pipe._should_repair(example, static_report, test_report) is True
    assert "Repository Test File" in pipe._repair_test_context(example)


def test_completion_validation_normalizes_missing_region_indent():
    example = _stable_testbacked_example()
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    unindented = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in (example.reference_code or "").splitlines()
    )
    _, report = pipe._validate_code(unindented, example)
    assert report.passed is True


def test_completion_validation_repairs_common_hanging_indent():
    example = _stable_testbacked_example()
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    code = """    if padding is not None:
            base = len(alphabet)
            return alphabet[0] * padding

        base = len(alphabet)
        return alphabet[0]
"""
    normalized = pipe._normalize_completion_code(code, example)
    static_report, _ = pipe._validate_code(normalized, example)
    assert static_report.ok is True


def test_uncertainty_aware_selection_prefers_verified_sample():
    example = _stable_testbacked_example()
    pipe = Pipeline(PipelineConfig(llm_backend="offline", uncertainty_aware=True))
    chosen, _, report = pipe._select_verified_sample(
        ["    return ''\n", example.reference_code or ""],
        example,
    )
    assert chosen is not None
    assert report is not None
    assert report.passed is True


def test_completion_indent_after_function_header():
    example = _stable_workdays_example()
    assert Pipeline._expected_completion_indent(example) == "    "
    pipe = Pipeline(PipelineConfig(llm_backend="offline"))
    code = "return 0\n"
    normalized = pipe._normalize_completion_code(code, example)
    assert normalized.startswith("    return 0")
