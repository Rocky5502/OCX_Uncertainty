"""Matched-protocol runner for the clean-room AllianceCoder reproduction.

The runner never imports AllianceCoder source or published results. It uses
OpenCoder's manifests, LLM client, candidate budget, executable validators,
and Pass@k implementation while preserving AllianceCoder's dependency-to-API
retrieval policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.baselines import AllianceCoderAdapter  # noqa: E402
from opencoder.data.loaders import Example, load_dataset  # noqa: E402
from opencoder.evaluation.metrics import pass_at_ks_from_samples  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
DEFAULT_MANIFESTS = {
    "repoexec": "input/repoexec_python_string_utils_inline14.jsonl",
    "codereval": "input/codereval_neo4j_executable19.jsonl",
    "execrepobench": "input/execrepobench_testbacked.jsonl",
}


def _extract_code(text: str) -> str:
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def _target_name(example: Example) -> str | None:
    raw = example.raw or {}
    meta = raw.get("metadata") or {}
    explicit = raw.get("entry_point") or meta.get("function_name") or raw.get("target_function_name")
    if explicit:
        return str(explicit)
    definitions = re.findall(
        r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(",
        str(raw.get("prefix_code") or ""),
        re.MULTILINE,
    )
    return definitions[-1] if definitions else None


def _repo_root(benchmark: str, example: Example, override: str | None) -> str | None:
    if override:
        return override
    if example.repo_root and os.path.isdir(example.repo_root):
        return example.repo_root
    if benchmark == "repoexec" and (ROOT / "input" / "string_utils").is_dir():
        return str(ROOT / "input" / "string_utils")
    return None


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "__dict__"):
        return dict(report.__dict__)
    return {"value": str(report)}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _row_generation_integrity_valid(row: dict) -> bool:
    recorded = row.get("generation_integrity") or {}
    if recorded.get("valid") is True:
        return True
    candidates = list(row.get("candidates") or [])
    raw_responses = list(row.get("candidate_raw_responses") or [])
    response_metadata = list(row.get("candidate_response_metadata") or [])
    if not candidates or len(response_metadata) != len(candidates):
        return False
    length_limited = [
        item.get("finish_reason") in {"length", "max_tokens"}
        for item in response_metadata
    ]
    if any(
        not candidate.strip() and not length_limited[index]
        for index, candidate in enumerate(candidates)
    ):
        return False
    unclosed = [
        bool(re.match(r"^\s*```(?:python)?", response))
        and _FENCE.search(response) is None
        for response in raw_responses
    ]
    if not any(unclosed):
        return True
    if len(response_metadata) != len(raw_responses):
        return False
    return all(
        not is_unclosed or length_limited[index]
        for index, is_unclosed in enumerate(unclosed)
    )


def _row_protocol_valid(row: dict) -> bool:
    if "error" in row or not row.get("faithful_full_api_descriptions"):
        return False
    index = row.get("index") or {}
    description_sources = set((index.get("description_sources") or {}).keys())
    return (
        int(index.get("n_items") or 0) > 0
        and int(index.get("n_description_errors") or 0) == 0
        and description_sources.issubset({"llm"})
        and _row_generation_integrity_valid(row)
    )


def _normalize_recorded_protocol(row: dict) -> dict:
    if "error" in row:
        return row
    raw_responses = list(row.get("candidate_raw_responses") or [])
    response_metadata = list(row.get("candidate_response_metadata") or [])
    unclosed = [
        bool(re.match(r"^\s*```(?:python)?", response))
        and _FENCE.search(response) is None
        for response in raw_responses
    ]
    if response_metadata and len(response_metadata) == len(raw_responses):
        length_limited_flags = [
            item.get("finish_reason") in {"length", "max_tokens"}
            for item in response_metadata
        ]
        unexplained = sum(
            is_unclosed and not length_limited_flags[index]
            for index, is_unclosed in enumerate(unclosed)
        )
        empty_flags = [
            not candidate.strip() for candidate in row.get("candidates") or []
        ]
        unexplained_empty = sum(
            is_empty and not length_limited_flags[index]
            for index, is_empty in enumerate(empty_flags)
        )
        row["generation_integrity"] = {
            "valid": unexplained_empty == 0 and unexplained == 0,
            "n_truncated_fences": sum(unclosed),
            "n_length_limited_candidates": sum(length_limited_flags),
            "n_unexplained_truncations": unexplained,
            "n_empty_candidates": sum(empty_flags),
            "n_unexplained_empty_candidates": unexplained_empty,
        }
        row["protocol_valid"] = _row_protocol_valid(row)
    return row


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        _normalize_recorded_protocol(row)
        for row in list(payload.get("rows") or [])
    ]


def _write(path: Path, metadata: dict, rows: list[dict]) -> None:
    outcomes = [row["sample_correctness"] for row in rows if row.get("sample_correctness")]
    summary = {
        "n_tasks": len(rows),
        "n_successful": sum("error" not in row for row in rows),
        "n_errors": sum("error" in row for row in rows),
    }
    if outcomes:
        summary.update(pass_at_ks_from_samples(outcomes, ks=(1, 3, 5)))
    payload = {"metadata": metadata, "summary": summary, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_task(
    pipe: Pipeline,
    adapter: AllianceCoderAdapter,
    example: Example,
    *,
    benchmark: str,
    repo_root: str | None,
    describe_limit: int | None,
    n_candidates: int,
) -> dict:
    started = time.perf_counter()
    usage_start = pipe.llm.usage_snapshot()
    index_started = time.perf_counter()
    items, _ = pipe.index_example(
        example,
        fallback_repo_root=repo_root,
        describe_limit=describe_limit,
    )
    index_latency_s = time.perf_counter() - index_started
    usage_after_index = pipe.llm.usage_snapshot()
    adapter.build_index(items)
    description_sources = Counter(
        str(getattr(item, "metadata", {}).get("description_source") or "unknown")
        for item in items
    )
    description_errors = sum(
        bool(getattr(item, "metadata", {}).get("describe_error")) for item in items
    )
    target = _target_name(example)
    if not target:
        raise ValueError("target function name is unavailable")

    trace = adapter.predict_dependencies_with_trace(example)
    usage_after_prediction = pipe.llm.usage_snapshot()
    retrieved = adapter.retrieve(trace.predictions, target_qualname=target)
    generation_prompt = adapter.build_generation_prompt(example, retrieved)
    responses = adapter.generate_candidates(
        example,
        retrieved,
        n=n_candidates,
        temperature=pipe.cfg.llm_temperature,
        max_tokens=pipe.cfg.llm_max_tokens,
        seed=pipe.cfg.llm_seed,
    )
    usage_after_generation = pipe.llm.usage_snapshot()
    candidates = [_extract_code(response.text) for response in responses]
    unclosed_fences = [
        bool(re.match(r"^\s*```(?:python)?", response.text))
        and _FENCE.search(response.text) is None
        for response in responses
    ]
    empty_candidates = sum(not candidate.strip() for candidate in candidates)
    finish_reasons = [response.raw.get("finish_reason") for response in responses]
    length_limited_candidates = sum(
        is_unclosed and finish_reasons[index] == "length"
        for index, is_unclosed in enumerate(unclosed_fences)
    )
    unexplained_truncations = sum(unclosed_fences) - length_limited_candidates
    generation_integrity = (
        unexplained_truncations == 0 and empty_candidates == 0
    )
    validations = []
    correctness = []
    for code in candidates:
        static_report, test_report = pipe._validate_code(code, example)
        passed = test_report.passed
        validations.append(
            {
                "static": _report_dict(static_report),
                "test": _report_dict(test_report),
                "passed": passed,
            }
        )
        correctness.append(bool(passed) if passed is not None else False)

    return {
        "id": example.id,
        "benchmark": benchmark,
        "target_name": target,
        "correctness_mode": pipe._correctness_mode(example),
        "faithful_full_api_descriptions": describe_limit is None,
        "protocol_valid": (
            describe_limit is None
            and description_errors == 0
            and bool(items)
            and set(description_sources).issubset({"llm"})
            and generation_integrity
        ),
        "index": {
            "repo_root": repo_root,
            "n_items": len(items),
            "n_cached_descriptions": sum(bool(getattr(item, "metadata", {}).get("describe_cached")) for item in items),
            "n_fallback_descriptions": sum(bool(getattr(item, "metadata", {}).get("description_fallback")) for item in items),
            "n_description_errors": description_errors,
            "description_sources": dict(description_sources),
            "item_ids": [f"{getattr(item, 'file_path', '')}::{getattr(item, 'qualname', '')}" for item in items],
            "latency_s": index_latency_s,
        },
        "dependency_trace": {
            "predictions": [prediction.__dict__ for prediction in trace.predictions],
            "prompts": {
                "decomposition": trace.decomposition_prompt,
                "dependency": trace.dependency_prompt,
                "extension": trace.extension_prompt,
            },
            "prompt_hashes": trace.prompt_hashes,
            "responses": {
                "decomposition": trace.decomposition_response,
                "dependency": trace.dependency_response,
                "extension": trace.extension_response,
            },
        },
        "retrieved_apis": [
            {
                "dependency": hit.dependency,
                "api_id": hit.api_id,
                "score": hit.score,
                "rank_after_target_filter": hit.rank,
            }
            for hit in retrieved
        ],
        "generation_prompt": generation_prompt,
        "generation_prompt_hash": _hash(generation_prompt),
        "generation_prompt_chars": len(generation_prompt),
        "candidates": candidates,
        "candidate_raw_responses": [response.text for response in responses],
        "candidate_response_metadata": [
            {
                "finish_reason": response.raw.get("finish_reason"),
                "index": response.raw.get("index"),
            }
            for response in responses
        ],
        "generation_integrity": {
            "valid": generation_integrity,
            "n_truncated_fences": sum(unclosed_fences),
            "n_length_limited_candidates": length_limited_candidates,
            "n_unexplained_truncations": unexplained_truncations,
            "n_empty_candidates": empty_candidates,
        },
        "validations": validations,
        "sample_correctness": correctness,
        "pass_at_k": dict(pass_at_ks_from_samples([correctness], ks=(1, 3, 5))),
        "llm_usage": {
            "index": _usage_delta(usage_start, usage_after_index),
            "dependency_prediction": _usage_delta(usage_after_index, usage_after_prediction),
            "generation": _usage_delta(usage_after_prediction, usage_after_generation),
            "total": _usage_delta(usage_start, usage_after_generation),
        },
        "latency_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=sorted(DEFAULT_MANIFESTS), required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        default=None,
        help="Run one exact manifest task ID for a targeted diagnostic.",
    )
    parser.add_argument("--describe-limit", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default="cache/external_baseline/alliancecoder",
        help="Isolated cache root for provenance-bearing baseline API descriptions.",
    )
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.backend:
        cfg.llm_backend = args.backend
    if args.model:
        cfg.llm_model = args.model
    cfg.cache_dir = args.cache_dir
    pipe = Pipeline(cfg)
    adapter = AllianceCoderAdapter(pipe.llm, pipe.encoder)
    manifest = args.manifest or DEFAULT_MANIFESTS[args.benchmark]
    examples = list(
        load_dataset(
            args.benchmark,
            manifest,
            limit=None if args.task_id else args.limit,
        )
    )
    if args.task_id:
        examples = [example for example in examples if example.id == args.task_id]
        if not examples:
            raise SystemExit(
                f"Task ID {args.task_id!r} is not present in {manifest}."
            )
    output = Path(args.out)
    rows = [] if args.force else _load_rows(output)
    complete = {
        str(row.get("id"))
        for row in rows
        if (
            _row_protocol_valid(row)
        )
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "AllianceCoder clean-room reproduction",
        "not_official_source_execution": True,
        "prompt_policy_version": "opencoder-alliancecoder-cleanroom-v1",
        "benchmark": args.benchmark,
        "manifest": manifest,
        "backend": cfg.llm_backend,
        "model": pipe.llm.model,
        "temperature": cfg.llm_temperature,
        "max_generation_tokens_per_candidate": cfg.llm_max_tokens,
        "request_timeout_s": cfg.llm_timeout,
        "n_candidates": cfg.n_samples_for_uncertainty,
        "seed": cfg.llm_seed,
        "embedding_model": cfg.embedding_model,
        "reasoning_effort": cfg.llm_reasoning_effort,
        "thinking_budget": cfg.llm_thinking_budget,
        "cache_dir": cfg.cache_dir,
        "describe_limit": args.describe_limit,
        "task_id_filter": args.task_id,
        "paper_eligible": args.describe_limit is None and cfg.llm_backend != "offline",
        "protocol_note": "A reduced description limit or offline backend is smoke evidence only.",
        "integrity_policy": "length-limited candidates are scored as failures; unexplained truncations invalidate the task",
        "expected_n_tasks": len(examples),
        "expected_task_ids": [example.id for example in examples],
    }
    for example in examples:
        if example.id in complete:
            print(f"{example.id}: skipped", flush=True)
            continue
        rows = [row for row in rows if str(row.get("id")) != example.id]
        try:
            root = _repo_root(args.benchmark, example, args.repo_root)
            row = run_task(
                pipe,
                adapter,
                example,
                benchmark=args.benchmark,
                repo_root=root,
                describe_limit=args.describe_limit,
                n_candidates=cfg.n_samples_for_uncertainty,
            )
            print(f"{example.id}: {row['pass_at_k']}", flush=True)
        except Exception as exc:
            row = {
                "id": example.id,
                "benchmark": args.benchmark,
                "error": f"{type(exc).__name__}: {exc}",
                "llm_usage_at_failure": pipe.llm.usage_snapshot(),
            }
            print(f"{example.id}: ERROR {exc}", flush=True)
        rows.append(row)
        _write(output, metadata, rows)
    _write(output, metadata, rows)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
