"""Token-logprob-based uncertainty signal.

We use -mean(logprob_of_chosen_token) as a bounded proxy for predictive
entropy. Higher value = the model was less confident on each emitted
token, on average. This is a standard signal in calibration literature
for autoregressive LMs and is robust to length differences across
generations because of the per-token averaging.
"""
from __future__ import annotations

from typing import Optional, Sequence


def token_entropy_score(logprobs: Optional[Sequence[float]]) -> float:
    if not logprobs:
        return 1.0  # no info → treat as max uncertainty
    mean_lp = sum(logprobs) / len(logprobs)
    # Squash to [0, 1] via 1 - exp(mean_lp); exp(mean_lp) is the geometric
    # mean of per-token probabilities, in [0, 1].
    import math

    return float(1.0 - math.exp(max(mean_lp, -50.0)))
