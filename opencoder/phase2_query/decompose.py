"""Phase II, Step 4: Generate Implementation Steps from a user query."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from ..llm.client import LLMClient

_SYSTEM = (
    "You are a senior Python engineer. Given a coding task, decompose it into 3-6 "
    "ordered implementation steps. Output STRICT JSON: "
    '{"steps": [{"index": 1, "description": "..."}, ...]}'
)


@dataclass
class ImplementationStep:
    index: int
    description: str
    uncertainty: float = 0.0
    intent: dict = field(default_factory=dict)


def decompose_into_steps(query: str, llm: LLMClient) -> List[ImplementationStep]:
    resp = llm.complete_one(query, system=_SYSTEM, max_tokens=600, return_logprobs=False, temperature=0.2)
    txt = resp.text.strip()
    # Strip code fences if present.
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.MULTILINE).strip()
    try:
        data = json.loads(txt)
        steps = data.get("steps", [])
    except Exception:
        # Fallback: split lines starting with a digit.
        steps = [
            {"index": i + 1, "description": ln.strip(" -*0123456789.")}
            for i, ln in enumerate([line for line in txt.splitlines() if line.strip()])
        ]
    return [ImplementationStep(index=int(s.get("index", i + 1)), description=str(s["description"]))
            for i, s in enumerate(steps) if s.get("description")]
