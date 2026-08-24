#!/usr/bin/env python3
"""Run native multilingual CrossCodeEval evaluation through the shared gateway."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.evaluation.crosscodeeval import CrossCodeEvalEvaluator  # noqa: E402
from opencoder.llm.client import LLMClient, _load_dotenv  # noqa: E402
from opencoder.pipeline import PipelineConfig  # noqa: E402
from opencoderx.provenance import ResponseCache, canonical_hash  # noqa: E402
from opencoderx.uncertainty import candidate_disagreement, retrieval_score_dispersion  # noqa: E402


METHODS = {
    "direct": "Direct Generation",
    "context_rag": "Cross-file Context RAG",
}
PREFIX_CHAR_BUDGET = 6000
CONTEXT_CHAR_BUDGET = 2048
OUTPUT_TOKEN_BUDGET = 128
DEFAULT_EXPERIMENT_VERSION = "crosscodeeval_native_pilot_v1"
DEFAULT_DATASET_VERSION = "crosscodeeval_opencoderx_pilot8_v1"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _context_evidence(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    evidence = []
    for index, item in enumerate((row.get("crossfile_context") or {}).get("list") or []):
        filename = str(item.get("filename") or f"context-{index}")
        chunk = str(item.get("retrieved_chunk") or "")
        score = float(item.get("score") or 0.0)
        document_id = canonical_hash({
            "task_id": row.get("task_id"),
            "filename": filename,
            "chunk": chunk,
            "rank": index,
        })
        blocks.append(f"// File: {filename}\n{chunk}")
        evidence.append({
            "document_id": document_id,
            "filename": filename,
            "score": score,
            "rank": index + 1,
        })
    return "\n\n".join(blocks)[:CONTEXT_CHAR_BUDGET], evidence


def _messages(row: dict[str, Any], method: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    language = str(row["language"])
    prefix = str(row.get("prompt") or "")[-PREFIX_CHAR_BUDGET:]
    context, evidence = _context_evidence(row)
    parts = [
        f"Complete the code at the cursor in {language}.",
        "Return only the missing continuation. Do not use Markdown or repeat the prefix.",
    ]
    if method == "context_rag":
        parts.extend([
            "\n# Retrieved cross-file context",
            context or "(no cross-file context retrieved)",
        ])
    else:
        evidence = []
    parts.extend(["\n# Code prefix", prefix, "\n# Missing continuation"])
    return [
        {
            "role": "system",
            "content": "You are a precise multilingual repository code-completion model.",
        },
        {"role": "user", "content": "\n".join(parts)},
    ], evidence


def _integrity(candidates: list[str], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    limited = [
        item.get("finish_reason") in {"length", "max_tokens"}
        for item in metadata
    ]
    empty = [not candidate.strip() for candidate in candidates]
    unexplained_empty = sum(
        is_empty and not limited[index]
        for index, is_empty in enumerate(empty)
    )
    return {
        "valid": len(candidates) == 5 and unexplained_empty == 0,
        "n_candidates": len(candidates),
        "n_length_limited_candidates": sum(limited),
        "n_empty_candidates": sum(empty),
        "n_unexplained_empty_candidates": unexplained_empty,
    }


def _load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _cost_gate(config_path: str | None, model: str, client: LLMClient) -> dict[str, Any] | None:
    if not config_path:
        return None
    controls = (_load_yaml(config_path).get("cost_controls") or {})
    pricing = (controls.get("gateway_pricing") or {}).get(model)
    if not pricing:
        raise RuntimeError(f"COST_GATE_UNPRICED_MODEL:{model}")
    usage = client.usage_snapshot()
    currency = str(pricing.get("currency") or "USD").upper()
    limit = (controls.get("per_run_limits") or {}).get(currency)
    if limit is None:
        raise RuntimeError(f"COST_GATE_MISSING_LIMIT:{currency}")
    amount = (
        usage["prompt_tokens"] * float(pricing["input_per_million"])
        + usage["completion_tokens"] * float(pricing["output_per_million"])
    ) / 1_000_000
    return {**usage, "amount": amount, "currency": currency, "limit": float(limit), "allowed": amount <= float(limit)}


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, label in METHODS.items():
        known = [row for row in rows if row.get("method") == method and "error" not in row]
        selected = [row["selected_metrics"] for row in known]
        candidates = [metric for row in known for metric in row["candidate_metrics"]]
        output[method] = {
            "label": label,
            "tasks": len(known),
            "errors": sum(row.get("method") == method and "error" in row for row in rows),
            "selected_exact_match": _mean(float(item["exact_match"]) for item in selected),
            "selected_edit_similarity": _mean(float(item["edit_similarity"]) for item in selected),
            "selected_identifier_f1": _mean(float(item["identifier_f1"]) for item in selected),
            "candidate_exact_match": _mean(float(item["exact_match"]) for item in candidates),
            "candidate_edit_similarity": _mean(float(item["edit_similarity"]) for item in candidates),
            "candidate_identifier_f1": _mean(float(item["identifier_f1"]) for item in candidates),
            "mean_aggregate_risk": _mean(float(row["uncertainty"]["aggregate_risk"]) for row in known),
            "mean_latency_seconds": _mean(float(row["latency_seconds"]) for row in known),
        }
    return output


def _write(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": metadata,
        "summary": _summary(rows),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cost-config")
    parser.add_argument("--experiment-version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument(
        "--paper-eligible",
        action="store_true",
        help="Mark a frozen non-pilot campaign as eligible after integrity audit.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    client = LLMClient(
        backend=cfg.llm_backend,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=OUTPUT_TOKEN_BUDGET,
        timeout=cfg.llm_timeout,
        seed=cfg.llm_seed,
        reasoning_effort=cfg.llm_reasoning_effort,
        thinking_budget=cfg.llm_thinking_budget,
    )
    evaluator = CrossCodeEvalEvaluator(ROOT / ".benchmarks/crosscodeeval")
    cache = ResponseCache(ROOT / "cache/tosem/crosscodeeval")
    rows: list[dict[str, Any]] = []
    out_path = Path(args.out)
    if out_path.is_file() and not args.force:
        rows = list(json.loads(out_path.read_text(encoding="utf-8")).get("rows") or [])
    methods = list(METHODS) if args.method == "all" else [args.method]
    tasks = _read_jsonl(args.manifest)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CrossCodeEval",
        "dataset_version": args.dataset_version,
        "manifest": args.manifest,
        "model": cfg.llm_model,
        "backend": cfg.llm_backend,
        "temperature": cfg.llm_temperature,
        "candidate_count": 5,
        "max_output_tokens": OUTPUT_TOKEN_BUDGET,
        "prefix_char_budget": PREFIX_CHAR_BUDGET,
        "crossfile_context_char_budget": CONTEXT_CHAR_BUDGET,
        "functional_execution": False,
        "native_metrics": ["exact_match", "edit_similarity", "identifier_f1"],
        "methods": METHODS,
        "paper_eligible": bool(args.paper_eligible),
        "experiment_version": args.experiment_version,
        "cost_config": args.cost_config,
    }

    for task in tasks:
        for method in methods:
            task_id = str(task["task_id"])
            if any(row.get("task_id") == task_id and row.get("method") == method and "error" not in row for row in rows):
                print(f"  {task_id:<32} {method:<12} skipped", flush=True)
                continue
            messages, evidence = _messages(task, method)
            prompt_hash = canonical_hash(messages)
            retrieval_hash = canonical_hash(evidence)
            cache_parameters = {
                "provider": cfg.llm_backend,
                "model": cfg.llm_model,
                "prompt_hash": prompt_hash,
                "task_id": task_id,
                "method": method,
                "temperature": cfg.llm_temperature,
                "seed": cfg.llm_seed,
                "context_hash": canonical_hash(messages[-1]["content"]),
                "retrieval_hash": retrieval_hash,
                "experiment_version": args.experiment_version,
            }
            started = time.perf_counter()
            usage_before = client.usage_snapshot()
            audit_before = len(client.response_audit_snapshot())
            try:
                cached = cache.get(cache_parameters)
                if cached:
                    candidates = list(cached["candidates"])
                    response_metadata = list(cached["response_metadata"])
                    provider_audit = list(cached["provider_response_audit"])
                    task_usage = dict(cached["llm_usage"])
                    cache_hit = True
                else:
                    responses = client.complete(messages, n=5, max_tokens=OUTPUT_TOKEN_BUDGET, return_logprobs=False)
                    candidates = [response.text for response in responses]
                    response_metadata = [
                        {
                            "finish_reason": response.raw.get("finish_reason"),
                            "index": response.raw.get("index"),
                            **dict(response.raw.get("_response_metadata") or {}),
                        }
                        for response in responses
                    ]
                    usage_after = client.usage_snapshot()
                    task_usage = _usage_delta(usage_before, usage_after)
                    provider_audit = client.response_audit_snapshot()[audit_before:]
                    cache.put(cache_parameters, {
                        "candidates": candidates,
                        "response_metadata": response_metadata,
                        "provider_response_audit": provider_audit,
                        "llm_usage": task_usage,
                    })
                    cache_hit = False
                candidate_metrics = [
                    evaluator.evaluate(
                        prompt=str(task.get("prompt") or ""),
                        prediction=candidate,
                        target=str(task.get("reference_completion") or ""),
                        language=str(task["language"]),
                    ).to_dict()
                    for candidate in candidates
                ]
                processed = [item["postprocessed_prediction"] for item in candidate_metrics]
                scores = [float(item["score"]) for item in evidence]
                disagreement = candidate_disagreement(processed)
                dispersion = retrieval_score_dispersion(scores)
                missing_context = 0.0 if evidence else 1.0
                aggregate_risk = statistics.mean([disagreement, dispersion, missing_context])
                row = {
                    "task_id": task_id,
                    "repository": task.get("repository"),
                    "language": task.get("language"),
                    "file": task.get("file"),
                    "method": method,
                    "method_label": METHODS[method],
                    "candidates": candidates,
                    "candidate_metrics": candidate_metrics,
                    "selected_candidate_index": 0,
                    "selected_output": processed[0],
                    "selected_metrics": candidate_metrics[0],
                    "retrieval_evidence": evidence,
                    "prompt_hash": prompt_hash,
                    "retrieval_hash": retrieval_hash,
                    "uncertainty": {
                        "candidate_disagreement": disagreement,
                        "retrieval_score_dispersion": dispersion,
                        "context_evidence_missing": missing_context,
                        "aggregate_risk": aggregate_risk,
                    },
                    "generation_integrity": _integrity(candidates, response_metadata),
                    "response_metadata": response_metadata,
                    "provider_response_audit": provider_audit,
                    "llm_usage": task_usage,
                    "latency_seconds": time.perf_counter() - started,
                    "cache_hit": cache_hit,
                    "functional_correctness": None,
                    "status": "COMPLETED_NATIVE_METRICS",
                }
            except Exception as exc:
                row = {
                    "task_id": task_id,
                    "repository": task.get("repository"),
                    "language": task.get("language"),
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "FAILED_INFRASTRUCTURE",
                }
            rows = [item for item in rows if not (item.get("task_id") == task_id and item.get("method") == method)]
            rows.append(row)
            _write(out_path, metadata, rows)
            print(f"  {task_id:<32} {method:<12} {row['status']}", flush=True)
            gate = _cost_gate(args.cost_config, cfg.llm_model, client)
            if gate is not None:
                metadata["runtime_cost_gate"] = gate
                _write(out_path, metadata, rows)
                if not gate["allowed"]:
                    print(json.dumps(gate, indent=2), flush=True)
                    return 3
    _write(out_path, metadata, rows)
    print(json.dumps(_summary(rows), indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
