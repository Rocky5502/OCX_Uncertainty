"""Target-aware API evidence refinement.

Repository-level completion often places the target function/class in the same
file context that is indexed for retrieval. Returning that target as "API
evidence" inflates over-retrieval and does not help the generator implement the
missing body. This module provides a small, deterministic refinement stage that
keeps supporting APIs while removing target self-retrieval and collapsing
constructor internals to the class-level API.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence


def normalize_api_name(value: str) -> str:
    """Normalize an API identifier for set comparison and de-duplication."""
    value = str(value or "").strip().strip("`")
    if not value:
        return ""
    if "(" in value and not value.lstrip().startswith("class "):
        value = value.split("(", 1)[0]
    value = value.replace("class ", "").strip()
    value = value.rsplit(".", 1)[-1]
    return re.sub(r"[^0-9a-zA-Z_]+", "", value).lower()


def canonical_api_name(value: str, *, drop_private_methods: bool = True) -> str:
    """Map implementation internals to the public API item they represent."""
    value = str(value or "").strip()
    if "." not in value:
        return value
    parent, child = value.rsplit(".", 1)
    if child == "__init__":
        return parent
    if child.startswith("__") and child.endswith("__"):
        return ""
    if drop_private_methods and child.startswith("__"):
        return ""
    return value


def refine_api_hit_records(
    hit_records: Sequence[Mapping[str, object]],
    *,
    target_norms: Iterable[str] = (),
    kept_only: bool = True,
    drop_private_methods: bool = True,
) -> list[str]:
    """Return refined API names ordered by best available final score.

    The refinement is intentionally label-free: it uses only the current target
    identifier and retrieved-candidate metadata, never ground-truth API sets.
    """
    targets = {normalize_api_name(x) for x in target_norms if normalize_api_name(x)}
    best: dict[str, tuple[float, str]] = {}
    for record in hit_records:
        if kept_only and not bool(record.get("kept")):
            continue
        raw_name = str(record.get("api_name") or "")
        name = canonical_api_name(raw_name, drop_private_methods=drop_private_methods)
        norm = normalize_api_name(name)
        if not norm or norm in targets:
            continue
        try:
            score = float(record.get("final_score") or 0.0)
        except Exception:
            score = 0.0
        if norm not in best or score > best[norm][0]:
            best[norm] = (score, name)
    return [name for _, name in sorted(best.values(), key=lambda item: (-item[0], item[1]))]
