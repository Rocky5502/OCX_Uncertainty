"""Append-only provenance records and parameter-exact response caching."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FAILURE_TYPES = {
    "API_RATE_LIMIT", "API_TIMEOUT", "API_BAD_REQUEST", "MODEL_REFUSAL",
    "MODEL_EMPTY_RESPONSE", "OUTPUT_PARSE_FAILURE", "REPO_SETUP_FAILURE",
    "DEPENDENCY_FAILURE", "COMPILATION_FAILURE", "TEST_FAILURE", "TEST_TIMEOUT",
    "DOCKER_FAILURE", "RETRIEVAL_FAILURE", "UNKNOWN",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.replace(str(Path.home()), "<HOME>")
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "<REDACTED_API_KEY>", text)
    return text


@dataclass
class RunRecord:
    run_id: str
    dataset: str
    dataset_version: str
    task_id: str
    repository: str
    repository_commit: str
    language: str
    model_provider: str
    model_family: str
    model_id: str
    model_revision: str
    method: str
    prompt_hash: str
    retrieval_hash: str
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = "UNAVAILABLE"
    dirty_worktree: bool | None = None
    ablation: str = "none"
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    candidate_count: int = 1
    retrieval_budget: Mapping[str, int] = field(default_factory=dict)
    context_budget: int | None = None
    verification_budget: int = 0
    repair_budget: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int = 0
    latency: float | None = None
    estimated_cost: float | None = None
    functional_correctness: bool | None = None
    compilation_result: bool | None = None
    test_result: bool | None = None
    api_uncertainty: float | None = None
    context_uncertainty: float | None = None
    similar_code_uncertainty: float | None = None
    generation_uncertainty: float | None = None
    aggregate_risk: float | None = None
    deferral_policy: str | None = None
    review_budget: float | None = None
    collaboration_decision: str | None = None
    error_type: str | None = None
    provider_gateway: str | None = None
    response_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_id or not self.task_id:
            raise ValueError("run_id and task_id are required")
        if self.error_type is not None and self.error_type not in FAILURE_TYPES:
            raise ValueError(f"unknown failure type: {self.error_type}")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.status == "COMPLETED" and self.functional_correctness is None:
            raise ValueError("completed records require a correctness outcome")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _sanitize(asdict(self))


class AppendOnlyResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: RunRecord) -> None:
        payload = record.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("run_id") == record.run_id:
                    raise ValueError(f"duplicate run_id: {record.run_id}")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)


class ResponseCache:
    REQUIRED_KEY_FIELDS = (
        "provider", "model", "prompt_hash", "task_id", "method", "temperature",
        "seed", "context_hash", "retrieval_hash", "experiment_version",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def key(self, parameters: Mapping[str, Any]) -> str:
        missing = [field for field in self.REQUIRED_KEY_FIELDS if field not in parameters]
        if missing:
            raise ValueError(f"cache key missing fields: {', '.join(missing)}")
        return canonical_hash(dict(parameters)).split(":", 1)[1]

    def get(self, parameters: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.root / f"{self.key(parameters)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def put(self, parameters: Mapping[str, Any], response: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{self.key(parameters)}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != _sanitize(dict(response)):
                raise ValueError("refusing to replace a non-identical cached response")
            return path
        payload = json.dumps(_sanitize(dict(response)), ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile("w", dir=self.root, delete=False, encoding="utf-8") as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(path)
        return path
