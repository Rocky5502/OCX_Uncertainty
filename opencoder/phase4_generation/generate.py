"""Phase IV, Step 10: Uncertainty-Guided Code Generation.

Receives: user query, fused evidence, and the uncertainty trace from
Phase II. Samples N candidate completions; computes the three
uncertainty signals over those samples; selects the best by
self-consistency cluster.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..uncertainty import (
    UncertaintyTrace,
    aggregate_uncertainty,
    normalize_code,
    self_consistency_score,
    semantic_variance_score,
    token_entropy_score,
)


_SYSTEM = (
    "You are a senior Python engineer. Use the provided evidence to implement the user's "
    "task. Return ONLY the Python code block — no commentary, no markdown outside one code "
    "fence. If the task asks for a missing region between prefix and suffix code, return "
    "only that missing region and do not repeat the prefix or suffix."
)


@dataclass
class GenerationResult:
    code: str
    samples: List[str]
    sample_logprobs: List[List[float]]
    trace: UncertaintyTrace
    components: dict = field(default_factory=dict)
    raw_responses: List[str] = field(default_factory=list)
    response_metadata: List[Dict[str, Any]] = field(default_factory=list)
    generation_integrity: Dict[str, Any] = field(default_factory=dict)


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _pick_consensus(samples: List[str]) -> str:
    from collections import Counter
    norm = [normalize_code(s) for s in samples]
    counts = Counter(norm)
    winner_norm, _ = counts.most_common(1)[0]
    for s in samples:
        if normalize_code(s) == winner_norm:
            return s
    return samples[0]


def generate_target_code(
    query: str,
    fused_evidence: str,
    query_uncertainty_summary: Dict[str, float],
    llm: LLMClient,
    encoder,
    n_samples: int = 5,
    uncertainty_weights: Dict[str, float] | None = None,
    use_uncertainty_guidance: bool = True,
    completion_mode: bool = False,
    expected_indent: str = "",
    sampling_temperature: float | None = 0.7,
) -> GenerationResult:
    parts = [f"# Task\n{query}"]
    if use_uncertainty_guidance:
        parts.append(f"# Query Uncertainty Trace\n{query_uncertainty_summary}")
    parts.append(f"# Retrieved Evidence\n{fused_evidence}")
    instruction = "Implement the task using the retrieved evidence."
    if completion_mode or ("# Prefix Code" in query and "# Suffix Code" in query):
        indent_desc = f"{len(expected_indent)} spaces" if expected_indent else "the prefix-implied indentation"
        instruction += (
            " This is a repository completion task: return only the code that belongs "
            f"inside the missing region. Top-level statements in the missing region must "
            f"use exactly {indent_desc}; nested blocks add four more spaces. "
            "Do not include imports, wrappers, tests, repeated prefix code, or repeated suffix code "
            "unless they are literally part of the missing region."
        )
    elif "# Function Stub" in query:
        instruction += (
            " This is a function-completion task: return the complete target function "
            "definition, preserving the original signature, indentation, decorators, "
            "and docstring shown in the stub. Do not return only the function body."
        )
    if use_uncertainty_guidance:
        instruction += (
            " Treat high-uncertainty evidence with caution; prefer evidence "
            "with lower knowledge uncertainty when there is conflict."
        )
    parts.append(f"# Instruction\n{instruction}")
    prompt = "\n\n".join(parts)
    resps = llm.complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        n=n_samples,
        temperature=sampling_temperature,
        return_logprobs=True,
    )
    codes = [_extract_code(r.text) for r in resps]
    logprobs = [r.logprobs or [] for r in resps]
    raw_responses = [r.text for r in resps]
    response_metadata = [
        {
            "finish_reason": r.raw.get("finish_reason"),
            "index": r.raw.get("index"),
            **dict(r.raw.get("_response_metadata") or {}),
        }
        for r in resps
    ]
    unclosed_fences = [
        bool(re.match(r"^\s*```(?:python)?", raw))
        and _FENCE.search(raw) is None
        for raw in raw_responses
    ]
    length_limited_flags = [
        metadata.get("finish_reason") in {"length", "max_tokens"}
        for metadata in response_metadata
    ]
    unexplained_truncations = sum(
        is_unclosed and not length_limited_flags[index]
        for index, is_unclosed in enumerate(unclosed_fences)
    )
    empty_flags = [not sample.strip() for sample in codes]
    unexplained_empty_samples = sum(
        is_empty and not length_limited_flags[index]
        for index, is_empty in enumerate(empty_flags)
    )

    # Per-sample token entropies, averaged.
    te = sum(token_entropy_score(lp) for lp in logprobs) / max(1, len(logprobs))
    sc = self_consistency_score(codes)
    sv = semantic_variance_score(codes, encoder)
    trace = aggregate_uncertainty(te, sc, sv, weights=uncertainty_weights)

    chosen = _pick_consensus(codes)
    return GenerationResult(
        code=chosen,
        samples=codes,
        sample_logprobs=logprobs,
        trace=trace,
        components={"token_entropy": te, "self_consistency": sc, "semantic_variance": sv},
        raw_responses=raw_responses,
        response_metadata=response_metadata,
        generation_integrity={
            "valid": unexplained_empty_samples == 0 and unexplained_truncations == 0,
            "n_truncated_fences": sum(unclosed_fences),
            "n_length_limited_candidates": sum(length_limited_flags),
            "n_unexplained_truncations": unexplained_truncations,
            "n_empty_candidates": sum(empty_flags),
            "n_unexplained_empty_candidates": unexplained_empty_samples,
        },
    )
