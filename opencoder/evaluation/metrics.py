"""Evaluation metrics.

- exact_match / edit_similarity: lexical baselines.
- pass_at_k: execution-based, requires per-example test_code.
- uncertainty_calibration_ece: Expected Calibration Error of the
  aggregate uncertainty score against actual correctness. Central to
  RQ2 — measures whether the uncertainty signal is *useful*, not just
  reported.
"""
from __future__ import annotations

import difflib
import math
from typing import Mapping, Sequence

import numpy as np


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


def edit_similarity(pred: str, ref: str) -> float:
    return float(difflib.SequenceMatcher(None, pred, ref).ratio())


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Codex/HumanEval.

    ``n`` is the number of generated samples and ``c`` is how many passed.
    """
    if n <= 0 or c <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))


def pass_at_k(correct_per_example: Sequence[bool], k: int) -> float:
    """Compatibility helper for one sample per example."""
    if not correct_per_example:
        return 0.0
    return float(np.mean([1.0 if c else 0.0 for c in correct_per_example]))


def pass_at_k_from_samples(outcomes_by_example: Sequence[Sequence[bool]], k: int) -> float:
    if not outcomes_by_example:
        return 0.0
    vals = [estimate_pass_at_k(len(outcomes), sum(bool(x) for x in outcomes), k)
            for outcomes in outcomes_by_example]
    return float(np.mean(vals)) if vals else 0.0


def pass_at_ks_from_samples(
    outcomes_by_example: Sequence[Sequence[bool]],
    ks: Sequence[int] = (1, 3, 5),
) -> Mapping[str, float]:
    return {f"pass@{k}": pass_at_k_from_samples(outcomes_by_example, k) for k in ks}


def pass_rate_variance(outcomes_by_example: Sequence[Sequence[bool]]) -> float:
    """Mean Bernoulli variance p(1-p) of sample pass rates per example."""
    if not outcomes_by_example:
        return 0.0
    variances = []
    for outcomes in outcomes_by_example:
        if not outcomes:
            continue
        p = sum(bool(x) for x in outcomes) / len(outcomes)
        variances.append(p * (1.0 - p))
    return float(np.mean(variances)) if variances else 0.0


def uncertainty_calibration_ece(
    uncertainties: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE over (1 - uncertainty) as the model's confidence."""
    assert len(uncertainties) == len(correctness)
    if not uncertainties:
        return 0.0
    confidences = np.clip(1.0 - np.asarray(uncertainties, dtype=float), 0.0, 1.0)
    correct = np.asarray(correctness, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
