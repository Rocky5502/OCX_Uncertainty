"""Self-consistency uncertainty signal.

Sample N completions at temperature > 0; measure agreement among them.
Low agreement → high uncertainty. We normalize whitespace and a few
trivial syntactic variations before comparing, then use the size of the
largest equivalence class over N samples.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_WS_RE = re.compile(r"\s+")
_COMMENT_PY = re.compile(r"(?m)#.*$")


def normalize_code(s: str) -> str:
    s = _COMMENT_PY.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def self_consistency_score(samples: Sequence[str]) -> float:
    """Returns uncertainty in [0, 1]. 0 = all samples identical."""
    if not samples:
        return 1.0
    norm = [normalize_code(s) for s in samples]
    counts = Counter(norm)
    most_common = counts.most_common(1)[0][1]
    agreement = most_common / len(norm)
    return float(1.0 - agreement)
