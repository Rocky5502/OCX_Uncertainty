"""Phase II, Step 6: Predict Retrieval Intent per step.

For each step we predict (a) which retrieval source is most useful
(api / context / similar_code), and (b) a short search query for that
source. The prediction is conditioned on the step text plus its
estimated uncertainty so that high-uncertainty steps lean more on
similar-code evidence (which we found empirically anchors generation).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from ..llm.client import LLMClient
from .decompose import ImplementationStep


SOURCES = ("api", "context", "similar_code")


@dataclass
class RetrievalIntent:
    step_index: int
    source_weights: dict   # {"api": float, "context": float, "similar_code": float}
    queries: dict          # {"api": str, ...}


_SYSTEM = (
    "Given a single implementation step, decide which evidence types help most: "
    "(a) API/function knowledge from the repo, (b) related context code, "
    "(c) similar code snippets from elsewhere. Output STRICT JSON: "
    '{"weights": {"api": 0..1, "context": 0..1, "similar_code": 0..1}, '
    '"queries": {"api": "...", "context": "...", "similar_code": "..."}}'
)


def predict_retrieval_intent(steps: List[ImplementationStep], llm: LLMClient) -> List[RetrievalIntent]:
    intents: List[RetrievalIntent] = []
    for s in steps:
        prompt = f"Step: {s.description}\nEstimated uncertainty: {s.uncertainty:.2f}"
        try:
            resp = llm.complete_one(prompt, system=_SYSTEM, max_tokens=250, return_logprobs=False)
            txt = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
            data = json.loads(txt)
            weights = {k: float(data.get("weights", {}).get(k, 0.33)) for k in SOURCES}
            queries = {k: str(data.get("queries", {}).get(k, s.description)) for k in SOURCES}
        except Exception:
            weights = {k: 1.0 / 3 for k in SOURCES}
            queries = {k: s.description for k in SOURCES}
        # Renormalize.
        z = sum(weights.values()) or 1.0
        weights = {k: v / z for k, v in weights.items()}
        s.intent = {"weights": weights, "queries": queries}
        intents.append(RetrievalIntent(step_index=s.index, source_weights=weights, queries=queries))
    return intents
