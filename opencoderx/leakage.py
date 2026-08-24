"""Retrieval-index leakage checks for target implementations."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


def _normalized(value: str) -> str:
    try:
        return ast.dump(ast.parse(value), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return " ".join(str(value).split())


@dataclass(frozen=True)
class LeakageFinding:
    document_id: str
    suspected_leak: bool
    reason: str


def detect_target_leakage(
    target_implementation: str,
    documents: Iterable[Mapping[str, str]],
) -> list[LeakageFinding]:
    target = _normalized(target_implementation)
    target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
    findings = []
    for index, document in enumerate(documents):
        identifier = str(document.get("id") or index)
        content = str(document.get("content") or "")
        normalized = _normalized(content)
        exact = target and target in normalized
        digest_match = hashlib.sha256(normalized.encode("utf-8")).hexdigest() == target_hash
        suspected = bool(exact or digest_match)
        findings.append(LeakageFinding(
            document_id=identifier,
            suspected_leak=suspected,
            reason="normalized target implementation present" if suspected else "",
        ))
    return findings
