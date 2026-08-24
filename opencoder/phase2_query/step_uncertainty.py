"""Phase II, Step 5: Estimate per-step uncertainty.

For each implementation step, we ask the LLM to produce a short
"specificity rationale" and use the token-logprob entropy of that
rationale as the step-level uncertainty. This is the per-step signal
that drives uncertainty-aware retrieval scoring in Phase III.
"""
from __future__ import annotations

from typing import List

from ..llm.client import LLMClient
from ..uncertainty.token_entropy import token_entropy_score
from .decompose import ImplementationStep


_SYSTEM = (
    "You are reasoning about how concretely a single implementation step is specified. "
    "Reply with one short sentence describing what is unambiguous about this step and what is "
    "open-ended. Be brief; do not write code."
)


def estimate_step_uncertainty(steps: List[ImplementationStep], llm: LLMClient) -> List[ImplementationStep]:
    for s in steps:
        try:
            resp = llm.complete_one(s.description, system=_SYSTEM, max_tokens=80, return_logprobs=True)
            s.uncertainty = token_entropy_score(resp.logprobs)
        except Exception:
            s.uncertainty = 0.5
    return steps
