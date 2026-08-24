"""Frozen model-family specifications for the shared research gateway."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import urlparse

from opencoder.llm.client import LLMClient, LLMResponse


@dataclass(frozen=True)
class ModelCapability:
    logprobs: bool
    deterministic_seed: bool
    temperature: bool
    top_p: bool
    structured_output: bool
    token_usage: bool
    cached_input: bool
    max_context: int
    native_reasoning_control: str
    raw_response_id: bool


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_id: str
    gateway_backend: str
    capability: ModelCapability
    default_temperature: float | None
    model_revision: str


FROZEN_MODELS: Mapping[str, ModelSpec] = {
    "gpt": ModelSpec(
        family="GPT", model_id="gpt-4o-mini", gateway_backend="zhizengzeng",
        default_temperature=0.7, model_revision="gateway-catalog-2026-08-09",
        capability=ModelCapability(False, True, True, True, True, True, True, 128000, "none", True),
    ),
    "gemini": ModelSpec(
        family="Gemini", model_id="gemini-2.5-flash", gateway_backend="zhizengzeng",
        default_temperature=0.7, model_revision="gateway-catalog-2026-08-09",
        capability=ModelCapability(False, False, True, True, True, True, True, 1048576, "thinking budget", True),
    ),
    "claude": ModelSpec(
        family="Claude", model_id="claude-sonnet-5", gateway_backend="zhizengzeng",
        default_temperature=None, model_revision="claude-sonnet-5-fixed-id",
        capability=ModelCapability(False, False, False, False, True, True, True, 1000000, "adaptive effort", True),
    ),
    "qwen": ModelSpec(
        family="Qwen", model_id="qwen3-coder-plus", gateway_backend="zhizengzeng",
        default_temperature=0.7, model_revision="qwen3-coder-plus-2025-07-22-equivalent",
        capability=ModelCapability(False, True, True, True, False, True, True, 1000000, "provider-specific", True),
    ),
}


class GatewayModelClient:
    """Use one OpenAI-compatible gateway while retaining upstream-family provenance."""

    def __init__(self, family: str, *, max_tokens: int = 2048, timeout: int = 120):
        key = family.strip().lower()
        if key not in FROZEN_MODELS:
            raise ValueError(f"unknown frozen family: {family}")
        self.spec = FROZEN_MODELS[key]
        self.client = LLMClient(
            backend=self.spec.gateway_backend,
            model=self.spec.model_id,
            temperature=self.spec.default_temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    @property
    def gateway_fingerprint(self) -> str:
        endpoint = getattr(self.client, "logical_endpoint", self.client.endpoint)
        host = urlparse(endpoint).hostname or "unknown"
        return "sha256:" + hashlib.sha256(host.encode("utf-8")).hexdigest()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        n: int = 1,
        max_tokens: int | None = None,
    ) -> list[LLMResponse]:
        return self.client.complete(
            [dict(message) for message in messages],
            n=n,
            max_tokens=max_tokens,
            return_logprobs=self.spec.capability.logprobs,
        )

    def provenance(self) -> dict[str, object]:
        return {
            "model_provider": "ZhiZengZeng gateway",
            "model_family": self.spec.family,
            "requested_model_id": self.spec.model_id,
            "model_revision": self.spec.model_revision,
            "gateway_fingerprint": self.gateway_fingerprint,
            "sampling_temperature": self.spec.default_temperature,
            "gateway_catalog_verified": "2026-08-09",
        }
