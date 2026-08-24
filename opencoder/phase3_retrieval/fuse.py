"""Phase III, Step 9: Fuse Evidence.

Formats the selected candidates into a single evidence block ready to
be injected into the generation prompt. Groups by source so the
generator can attend to each kind of evidence distinctly.
"""
from __future__ import annotations

from typing import List

from .score_filter import Candidate


def fuse_evidence(candidates: List[Candidate]) -> str:
    by_source: dict = {}
    for c in candidates:
        by_source.setdefault(c.hit.source, []).append(c)

    parts: List[str] = []
    for source in ("api", "context", "similar_code"):
        cs = by_source.get(source, [])
        if not cs:
            continue
        parts.append(f"### Evidence: {source.replace('_', ' ').title()}")
        for c in cs:
            it = c.hit.item
            if source == "api":
                parts.append(
                    f"- `{it.signature}` ({it.file_path})\n"
                    f"  {it.description or it.docstring or ''}".rstrip()
                )
            elif source == "context":
                parts.append(f"- From `{it.file_path}`:\n```python\n{it.text[:800]}\n```")
            else:  # similar_code
                parts.append(
                    f"- `{it.qualname}` in `{it.file_path}`:\n```python\n{it.body[:800]}\n```"
                )
        parts.append("")
    return "\n".join(parts).strip()
