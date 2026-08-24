from opencoderx.providers import FROZEN_MODELS


def test_frozen_gateway_model_ids():
    assert FROZEN_MODELS["gpt"].model_id == "gpt-4o-mini"
    assert FROZEN_MODELS["gemini"].model_id == "gemini-2.5-flash"
    assert FROZEN_MODELS["claude"].model_id == "claude-sonnet-5"
    assert FROZEN_MODELS["qwen"].model_id == "qwen3-coder-plus"
    assert FROZEN_MODELS["claude"].default_temperature is None


def test_gateway_logprobs_are_disabled_until_transport_is_verified():
    assert all(not spec.capability.logprobs for spec in FROZEN_MODELS.values())
