"""Clean-room implementation of AllianceCoder's published retrieval policy.

This module does not import the authors' unlicensed source. It implements the
method-level behavior needed for a controlled reproduction: predict API
dependency descriptions, then retrieve one non-target repository API for each
description using normalized embedding similarity.
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import numpy as np

from opencoder.data.loaders import Example


@dataclass(frozen=True)
class DependencyPrediction:
    description: str
    ordinal: int


@dataclass(frozen=True)
class DependencyTrace:
    predictions: List[DependencyPrediction]
    decomposition_prompt: str
    decomposition_response: str
    dependency_prompt: str
    dependency_response: str
    extension_prompt: str
    extension_response: str

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return {
            "decomposition": hashlib.sha256(self.decomposition_prompt.encode()).hexdigest(),
            "dependency": hashlib.sha256(self.dependency_prompt.encode()).hexdigest(),
            "extension": hashlib.sha256(self.extension_prompt.encode()).hexdigest(),
        }


@dataclass(frozen=True)
class RetrievedAPI:
    dependency: str
    item: Any
    score: float
    rank: int

    @property
    def api_id(self) -> str:
        path = str(getattr(self.item, "file_path", ""))
        name = str(getattr(self.item, "qualname", ""))
        return f"{path}::{name}"


_LIST_PREFIX = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)")
_NAMED_DESCRIPTION = re.compile(r"^(?:\*\*)?`[^`]+`(?:\*\*)?\s*:\s*(.+)$")


def parse_dependency_descriptions(text: str) -> List[DependencyPrediction]:
    """Parse an LLM list while preserving order and removing exact repeats."""
    predictions: List[DependencyPrediction] = []
    seen: set[str] = set()
    for line in text.splitlines():
        value = _LIST_PREFIX.sub("", line).strip()
        value = value.strip("` ")
        named = _NAMED_DESCRIPTION.match(_LIST_PREFIX.sub("", line).strip())
        if named:
            value = named.group(1).strip()
        if not value or value.lower() in {"none", "n/a", "no dependencies"}:
            continue
        normalized = re.sub(r"\s+", " ", value).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        predictions.append(DependencyPrediction(value, len(predictions) + 1))
    return predictions


def _item_text(item: Any) -> str:
    return str(
        getattr(item, "description", None)
        or getattr(item, "docstring", None)
        or getattr(item, "signature", None)
        or getattr(item, "qualname", "")
    )


def _simple_name(value: str) -> str:
    return value.rsplit(".", 1)[-1].strip().casefold()


def _current_file(example: Example) -> str:
    raw = example.raw or {}
    if isinstance(raw.get("current_file"), str):
        return raw["current_file"]
    prefix = raw.get("prefix_code") or ""
    suffix = raw.get("suffix_code") or ""
    return f"{prefix}\n# <MISSING CODE>\n{suffix}" if prefix or suffix else ""


def _target_request(example: Example) -> str:
    raw = example.raw or {}
    if isinstance(raw.get("prefix_code"), str) or isinstance(raw.get("suffix_code"), str):
        file_name = str(raw.get("file_name") or "target file")
        fill_type = str(raw.get("fill_type") or "code region")
        return f"Complete the missing {fill_type} in {file_name}."
    return str(
        raw.get("target_function_prompt")
        or raw.get("instruction")
        or raw.get("prompt")
        or example.query
    )


class AllianceCoderAdapter:
    """Policy adapter with injected OpenCoder LLM and encoder interfaces."""

    def __init__(self, llm: Any, encoder: Any):
        self.llm = llm
        self.encoder = encoder
        self._items: List[Any] = []
        self._matrix: np.ndarray | None = None

    def build_index(self, items: Sequence[Any]) -> None:
        self._items = list(items)
        texts = [_item_text(item) for item in self._items]
        self._matrix = self.encoder.encode(texts) if texts else np.zeros((0, 0))
        if self._matrix.size:
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True).clip(min=1e-9)
            self._matrix = self._matrix / norms

    def predict_dependencies(self, example: Example) -> List[DependencyPrediction]:
        return self.predict_dependencies_with_trace(example).predictions

    def predict_dependencies_with_trace(self, example: Example) -> DependencyTrace:
        decomposition_prompt = self.build_decomposition_prompt(example)
        body = self.llm.complete_one(
            decomposition_prompt,
            system="Analyze the requested repository function without writing its final code.",
            return_logprobs=False,
        ).text
        dependency_prompt = self.build_dependency_prompt(example, body)
        initial = self.llm.complete_one(
            dependency_prompt,
            system="Return a numbered list of concise descriptions of repository APIs likely needed.",
            return_logprobs=False,
        ).text
        extension_prompt = self.build_extension_prompt(example, body, initial)
        extended = self.llm.complete_one(
            extension_prompt,
            system="Return only additional likely repository API descriptions as a numbered list, or None.",
            return_logprobs=False,
        ).text
        # The released pipeline conditions the extension call on the initial
        # list, then retrieves from the extension-stage output.
        return DependencyTrace(
            predictions=parse_dependency_descriptions(extended),
            decomposition_prompt=decomposition_prompt,
            decomposition_response=body,
            dependency_prompt=dependency_prompt,
            dependency_response=initial,
            extension_prompt=extension_prompt,
            extension_response=extended,
        )

    def retrieve(
        self,
        dependencies: Sequence[DependencyPrediction | str],
        *,
        target_qualname: str,
    ) -> List[RetrievedAPI]:
        if self._matrix is None:
            raise RuntimeError("build_index must be called before retrieve")
        if not dependencies or not self._items:
            return []
        descriptions = [
            dependency.description if isinstance(dependency, DependencyPrediction) else str(dependency)
            for dependency in dependencies
        ]
        queries = self.encoder.encode(descriptions)
        norms = np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-9)
        similarities = (queries / norms) @ self._matrix.T
        target = _simple_name(target_qualname)
        selected: List[RetrievedAPI] = []
        for dependency, scores in zip(descriptions, similarities):
            order = np.argsort(scores)[::-1]
            for rank, index in enumerate(order, start=1):
                item = self._items[int(index)]
                if _simple_name(str(getattr(item, "qualname", ""))) == target:
                    continue
                selected.append(
                    RetrievedAPI(dependency, item, float(scores[index]), rank)
                )
                break
        return selected

    @staticmethod
    def build_decomposition_prompt(example: Example) -> str:
        prompt = (
            "Describe a plausible implementation plan for the target function. "
            "Focus on behavior and repository interactions.\n\n"
            f"Target request:\n{_target_request(example)}"
        )
        raw = example.raw or {}
        if isinstance(raw.get("prefix_code"), str) or isinstance(raw.get("suffix_code"), str):
            prompt += f"\n\nCurrent file with missing region:\n{_current_file(example)}"
        return prompt

    @staticmethod
    def build_dependency_prompt(example: Example, possible_body: str) -> str:
        return (
            "List the repository functions or methods that the target implementation is likely to call. "
            "Describe each dependency semantically; do not generate code.\n\n"
            f"Target request:\n{_target_request(example)}\n\nImplementation plan:\n{possible_body}"
        )

    @staticmethod
    def build_extension_prompt(example: Example, possible_body: str, initial: str) -> str:
        return (
            "Identify any likely repository API dependencies missing from the current list.\n\n"
            f"Target request:\n{_target_request(example)}\n\nImplementation plan:\n{possible_body}\n\n"
            f"Current dependency list:\n{initial}"
        )

    @staticmethod
    def build_generation_prompt(
        example: Example,
        retrieved: Iterable[RetrievedAPI],
    ) -> str:
        evidence = []
        for hit in retrieved:
            item = hit.item
            evidence.append(
                "\n".join(
                    [
                        f"API: {getattr(item, 'qualname', '')}",
                        f"Signature: {getattr(item, 'signature', '')}",
                        f"Description: {_item_text(item)}",
                        f"Source:\n{getattr(item, 'body', '')}",
                    ]
                )
            )
        evidence_text = "\n\n".join(evidence) if evidence else "(none)"
        return (
            "Implement the target using the repository APIs when appropriate. "
            "Return only the completed target code.\n\n"
            f"Target request:\n{_target_request(example)}\n\n"
            f"Current file:\n{_current_file(example)}\n\n"
            f"Retrieved repository APIs:\n{evidence_text}"
        )

    def generate_candidates(
        self,
        example: Example,
        retrieved: Sequence[RetrievedAPI],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> list[Any]:
        prompt = self.build_generation_prompt(example, retrieved)
        return self.llm.complete(
            [{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
