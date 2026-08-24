from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from opencoder.baselines.alliancecoder import (
    AllianceCoderAdapter,
    DependencyPrediction,
    parse_dependency_descriptions,
)
from opencoder.data.loaders import Example
from opencoder.llm.client import LLMClient
from opencoder.phase1_repo_knowledge.extract import extract_sources_knowledge
from opencoder.pipeline import Pipeline, PipelineConfig


@dataclass
class Item:
    file_path: str
    qualname: str
    signature: str
    description: str
    body: str = "pass"


class FixedEncoder:
    dim = 2

    def encode(self, texts):
        vectors = {
            "target behavior": [1.0, 0.0],
            "parse configuration": [0.9, 0.1],
            "write result": [0.0, 1.0],
        }
        return np.asarray([vectors.get(text, [0.5, 0.5]) for text in texts], dtype=float)


class Response:
    def __init__(self, text):
        self.text = text


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete_one(self, *args, **kwargs):
        return Response(next(self.responses))


def test_dependency_parser_preserves_order_and_deduplicates():
    parsed = parse_dependency_descriptions(
        "1. `parse_config`: parse configuration\n- write result\n"
        "3) `parse_config`: parse configuration\nNone"
    )
    assert [item.description for item in parsed] == ["parse configuration", "write result"]
    assert [item.ordinal for item in parsed] == [1, 2]


def test_retrieval_selects_one_non_target_api_per_dependency():
    adapter = AllianceCoderAdapter(llm=None, encoder=FixedEncoder())
    adapter.build_index(
        [
            Item("target.py", "Target.run", "run()", "target behavior"),
            Item("config.py", "Config.parse", "parse(value)", "parse configuration"),
            Item("io.py", "write", "write(value)", "write result"),
        ]
    )
    hits = adapter.retrieve(
        [DependencyPrediction("parse configuration", 1), "write result"],
        target_qualname="Target.run",
    )
    assert [hit.item.qualname for hit in hits] == ["Config.parse", "write"]
    assert len(hits) == 2
    assert all(hit.item.qualname != "Target.run" for hit in hits)


def test_prediction_trace_retrieves_from_extension_stage_only():
    adapter = AllianceCoderAdapter(
        llm=SequenceLLM(
            [
                "- inspect the configuration",
                "1. `old_api`: old dependency",
                "1. `parse_config`: parse configuration",
            ]
        ),
        encoder=FixedEncoder(),
    )
    example = Example("task", "Implement target().", None)
    trace = adapter.predict_dependencies_with_trace(example)
    assert [item.description for item in trace.predictions] == ["parse configuration"]
    assert "old dependency" not in [item.description for item in trace.predictions]
    assert set(trace.prompt_hashes) == {"decomposition", "dependency", "extension"}


def test_generation_prompt_excludes_reference_and_tests():
    example = Example(
        id="task",
        query="Implement target().",
        repo_root=None,
        reference_code="REFERENCE_SECRET",
        test_code="TEST_SECRET",
        raw={
            "current_file": "def target():\n    pass",
            "target_function_prompt": "def target():",
            "codereval_test_selectors": ["SECRET_SELECTOR"],
        },
    )
    adapter = AllianceCoderAdapter(llm=None, encoder=FixedEncoder())
    prompt = adapter.build_generation_prompt(example, [])
    assert "Target request:\ndef target():" in prompt
    assert "def target()" in prompt
    assert "REFERENCE_SECRET" not in prompt
    assert "TEST_SECRET" not in prompt
    assert "SECRET_SELECTOR" not in prompt
    assert prompt.count("def target():\n    pass") == 1


def test_completion_prompt_does_not_duplicate_prefix_or_suffix():
    example = Example(
        id="completion",
        query="normalized loader query with embedded context",
        repo_root=None,
        raw={
            "file_name": "pkg/module.py",
            "fill_type": "function block",
            "prefix_code": "def target():\n",
            "suffix_code": "\nresult = target()",
        },
    )
    adapter = AllianceCoderAdapter(llm=None, encoder=FixedEncoder())
    generation = adapter.build_generation_prompt(example, [])
    decomposition = adapter.build_decomposition_prompt(example)
    for prompt in (generation, decomposition):
        assert prompt.count("def target():") == 1
        assert prompt.count("result = target()") == 1
    assert "Complete the missing function block in pkg/module.py." in generation


def test_completion_source_reconstruction_keeps_empty_function_parseable():
    example = Example(
        id="completion",
        query="complete the function",
        repo_root=None,
        raw={
            "file_name": "pkg/module.py",
            "prefix_code": "def target(value):\n",
            "suffix_code": "\ndef helper():\n    return 1\n",
        },
    )
    pipe = object.__new__(Pipeline)
    pipe.cfg = SimpleNamespace(include_test_context=False)
    sources = pipe._sources_from_example(example)
    assert "def target(value):\n    pass" in sources[-1][1]
    items = extract_sources_knowledge(sources)
    assert {item.qualname for item in items} >= {"target", "helper"}


def test_openai_compatible_request_includes_reasoning_effort():
    client = object.__new__(LLMClient)
    client.backend = "gemini"
    client.model = "gemini-2.5-flash"
    client.temperature = 0.7
    client.max_tokens = 768
    client.seed = None
    client.reasoning_effort = "none"
    client.thinking_budget = None
    client.supports_logprobs = False
    captured = {}

    def respond(body):
        captured.update(body)
        return {"choices": [{"message": {"content": "OK"}}]}

    client._post_chat_with_fallbacks = respond
    responses = client.complete(
        [{"role": "user", "content": "Reply OK"}],
        return_logprobs=False,
    )
    assert responses[0].text == "OK"
    assert captured["reasoning_effort"] == "none"


def test_gemini_thinking_budget_uses_provider_extra_body():
    client = object.__new__(LLMClient)
    client.backend = "gemini"
    client.model = "gemini-2.5-flash"
    client.temperature = 0.7
    client.max_tokens = 768
    client.seed = None
    client.reasoning_effort = "none"
    client.thinking_budget = 0
    client.supports_logprobs = False
    captured = {}

    def respond(body):
        captured.update(body)
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"content": "OK"},
                }
            ]
        }

    client._post_chat_with_fallbacks = respond
    responses = client.complete(
        [{"role": "user", "content": "Reply OK"}],
        return_logprobs=False,
    )
    assert responses[0].text == "OK"
    assert "reasoning_effort" not in captured
    assert captured["extra_body"] == {
        "google": {
            "thinking_config": {
                "thinking_budget": 0,
                "include_thoughts": False,
            }
        }
    }


def test_length_limited_candidate_is_a_scored_failure_not_protocol_corruption():
    from experiments.run_alliancecoder_baseline import (
        _row_generation_integrity_valid,
    )

    row = {
        "candidates": ["```python\ndef target():\n    return"],
        "candidate_raw_responses": ["```python\ndef target():\n    return"],
        "candidate_response_metadata": [{"finish_reason": "length", "index": 0}],
        "generation_integrity": {"valid": False},
    }
    assert _row_generation_integrity_valid(row)


def test_unexplained_unclosed_candidate_remains_protocol_invalid():
    from experiments.run_alliancecoder_baseline import (
        _row_generation_integrity_valid,
    )

    row = {
        "candidates": ["```python\ndef target():\n    return"],
        "candidate_raw_responses": ["```python\ndef target():\n    return"],
        "candidate_response_metadata": [{"finish_reason": "stop", "index": 0}],
        "generation_integrity": {"valid": False},
    }
    assert not _row_generation_integrity_valid(row)


def test_rq3_description_limit_accepts_full_index_mode():
    from experiments.run_rq3 import _parse_description_limit

    assert _parse_description_limit("all") is None
    assert _parse_description_limit("full") is None
    assert _parse_description_limit("5") == 5


def test_rq3_direct_condition_disables_retrieval_and_mitigation():
    from experiments.run_rq3 import _pipeline_config_for

    cfg = _pipeline_config_for(PipelineConfig(), "direct", False)
    assert cfg.enable_sources == ()
    assert cfg.uncertainty_aware is False
    assert cfg.feature_enabled("uncertainty_decomposition") is False
    assert cfg.feature_enabled("uncertainty_verified_selection") is False
    assert cfg.feature_enabled("uncertainty_triggered_repair") is False


def test_gemini_rq3_config_records_explicit_request_timeout():
    cfg = PipelineConfig.from_yaml("configs/rq3/gemini_2_5_flash.yaml")
    assert cfg.llm_timeout == 180


def test_repository_index_excludes_target_implementation(tmp_path):
    source = """
def target(value):
    leaked_secret = value + 41
    return leaked_secret

def helper(value):
    return value * 2
""".strip()
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    example = Example(
        id="target-exclusion",
        query="Implement target.",
        repo_root=str(tmp_path),
        raw={"entry_point": "target"},
    )
    pipe = Pipeline(
        PipelineConfig(
            llm_backend="offline",
            embedding_model="hash",
        )
    )
    items, retrievers = pipe.index_example(example, describe_limit=1)
    assert all(item.qualname != "target" for item in items)
    assert {item.qualname for item in items} >= {"helper"}
    context_text = "\n".join(
        chunk.text for chunk in retrievers["context"].index._items
    )
    assert "leaked_secret" not in context_text
    assert "def target(value):\n    pass" in context_text
