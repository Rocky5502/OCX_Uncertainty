#!/usr/bin/env python3
"""Run the frozen, gateway-mediated exploratory AI-agent study."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.llm.client import LLMClient, _load_dotenv  # noqa: E402
from opencoder.phase5_verify.test_validate import run_execrepobench_function_tests  # noqa: E402


HERE = ROOT / "human_study/gateway_agent_v1"
RESOLVED = ROOT / "results/agent_gateway_v1/resolved_model_manifest.json"
RAW = ROOT / "results/agent_gateway_v1/raw_results.jsonl"
BENCHMARK = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
PUBLIC = ROOT / "human_study/frozen/stimuli_public.jsonl"
PRIVATE = ROOT / "human_study/frozen/stimuli_private.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sanitize(text: str) -> str:
    text = str(text or "")
    text = text.replace(str(ROOT), "<WORKSPACE>")
    text = re.sub(r"/private/(?:tmp|var)/[^\s:'\"]+", "<TEMP_PATH>", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-...REDACTED", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer ...REDACTED", text)
    return text


def strip_fence(code: str) -> str:
    value = code.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else value).strip()


def parse_response(text: str) -> dict[str, Any]:
    errors: list[str] = []
    start_match = re.search(r"^STARTING_CORRECT:\s*(true|false)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    start_conf_match = re.search(r"^STARTING_CONFIDENCE:\s*(\d{1,3})\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    final_conf_match = re.search(r"^FINAL_CONFIDENCE:\s*(\d{1,3})\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    code_match = re.search(r"FINAL_CODE_BEGIN\s*(.*?)\s*FINAL_CODE_END", text, flags=re.DOTALL | re.IGNORECASE)
    parse_mode = "strict_markers"
    code = strip_fence(code_match.group(1)) if code_match else ""
    if not code:
        fences = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        functions = [strip_fence(candidate) for candidate in fences if re.search(r"^\s*(?:async\s+)?def\s+", candidate, flags=re.MULTILINE)]
        if functions:
            code = max(functions, key=len)
            parse_mode = "fallback_code_fence"
        else:
            errors.append("missing_final_code")
    starting_correct = None if not start_match else start_match.group(1).lower() == "true"
    if start_match is None:
        errors.append("missing_starting_correct")

    def confidence(match: re.Match[str] | None, name: str) -> int | None:
        if match is None:
            errors.append(f"missing_{name}")
            return None
        value = int(match.group(1))
        if not 0 <= value <= 100:
            errors.append(f"invalid_{name}")
            return None
        return value

    return {
        "starting_correct": starting_correct,
        "starting_confidence": confidence(start_conf_match, "starting_confidence"),
        "final_confidence": confidence(final_conf_match, "final_confidence"),
        "final_code": code,
        "parse_mode": parse_mode,
        "parse_errors": errors,
    }


def canonical_function(code: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.dump(node, annotate_fields=True, include_attributes=False)
    return None


def run_episode(
    row: dict[str, str],
    model: dict[str, Any],
    stimulus: dict[str, Any],
    private: dict[str, Any],
    benchmark: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    prompt_path = ROOT / row["prompt_path"]
    prompt = prompt_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if digest != row["prompt_sha256"]:
        raise RuntimeError(f"prompt hash mismatch: {row['prompt_path']}")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    raw_text = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None
    usage: dict[str, int] = {}
    response_metadata: dict[str, Any] = {}
    try:
        client = LLMClient(
            backend="zhizengzeng",
            model=model["resolved_model_id"],
            temperature=model["temperature"],
            max_tokens=2048,
            timeout=timeout,
        )
        response = client.complete_one(
            prompt,
            system="You are an exploratory repository code-review agent. Follow the response contract exactly.",
            return_logprobs=False,
        )
        raw_text = response.text
        parsed = parse_response(raw_text)
        usage = client.usage_snapshot()
        metadata = (response.raw or {}).get("_response_metadata") or {}
        response_metadata = {
            "response_id": metadata.get("response_id"),
            "served_model": metadata.get("served_model"),
            "created": metadata.get("created"),
            "finish_reason": (response.raw or {}).get("finish_reason"),
        }
    except Exception as exc:
        error = sanitize(f"{type(exc).__name__}: {exc}")[:1200]
    latency = time.perf_counter() - started

    report = None
    evaluator_status = "not_scored"
    final_correct: bool | None = None
    if parsed is not None and parsed["final_code"]:
        report = run_execrepobench_function_tests(parsed["final_code"], benchmark, timeout=240)
        evaluator_status = "error" if report.returncode in {2, -1} else "ok"
        final_correct = None if evaluator_status == "error" else bool(report.passed)

    function_name = str(stimulus["function_name"])
    starting_function = canonical_function(str(stimulus["starting_code"]), function_name)
    final_function = None if parsed is None else canonical_function(parsed["final_code"], function_name)
    code_changed = final_function is None or starting_function != final_function
    initial_correct = bool(private["initial_correct"])
    starting_judgment = None if parsed is None else parsed["starting_correct"]

    return {
        "record_type": "gateway_agent_episode",
        "study_mode": "AGENT_EXPLORATORY",
        "protocol": "gateway_agent_v1",
        "agent_id": row["agent_id"],
        "family": row["family"],
        "requested_model": model["resolved_model_id"],
        "served_model": response_metadata.get("served_model"),
        "response_id": response_metadata.get("response_id"),
        "finish_reason": response_metadata.get("finish_reason"),
        "assignment_group": int(row["assignment_group"]),
        "episode_index": int(row["episode_index"]),
        "task_id": row["task_id"],
        "condition": row["condition"],
        "signal_category": private["signal_category"],
        "initial_correct": initial_correct,
        "starting_judgment_correct": starting_judgment,
        "starting_confidence": None if parsed is None else parsed["starting_confidence"],
        "final_confidence": None if parsed is None else parsed["final_confidence"],
        "failure_detection_accurate": None if starting_judgment is None else starting_judgment == initial_correct,
        "final_code": "" if parsed is None else parsed["final_code"],
        "final_code_sha256": None if parsed is None else hashlib.sha256(parsed["final_code"].encode("utf-8")).hexdigest(),
        "code_changed": code_changed,
        "final_correct": final_correct,
        "repair_opportunity": not initial_correct,
        "repair_success": bool(not initial_correct and final_correct is True),
        "unnecessary_edit": bool(initial_correct and code_changed),
        "parse_mode": None if parsed is None else parsed["parse_mode"],
        "parse_errors": ["api_error"] if parsed is None else parsed["parse_errors"],
        "evaluator_status": evaluator_status,
        "evaluator_returncode": None if report is None else report.returncode,
        "evaluator_stdout": "" if report is None else sanitize(report.stdout)[:2000],
        "evaluator_stderr": "" if report is None else sanitize(report.stderr)[:2000],
        "raw_response": sanitize(raw_text),
        "raw_response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else None,
        "prompt_path": row["prompt_path"],
        "prompt_sha256": digest,
        "usage": usage,
        "latency_seconds": latency,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit-per-model", type=int)
    parser.add_argument("--agent-id", action="append", dest="agent_ids")
    args = parser.parse_args()
    schedule = read_csv(HERE / "schedule.csv")
    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
    if not resolved.get("all_models_ready"):
        raise RuntimeError("all frozen model preflights must pass before study execution")
    models = {row["agent_id"]: row for row in resolved["models"]}
    public = {row["task_id"]: row for row in read_jsonl(PUBLIC)}
    private = {row["task_id"]: row for row in read_jsonl(PRIVATE)}
    benchmark = {row["task_id"]: row for row in read_jsonl(BENCHMARK)}
    existing_rows = read_jsonl(RAW)
    existing = {(row["agent_id"], int(row["episode_index"])) for row in existing_rows}
    pending = [row for row in schedule if (row["agent_id"], int(row["episode_index"])) not in existing]
    if args.agent_ids:
        selected_agents = set(args.agent_ids)
        unknown = selected_agents - set(models)
        if unknown:
            raise RuntimeError(f"unknown agent IDs: {sorted(unknown)}")
        pending = [row for row in pending if row["agent_id"] in selected_agents]
    if args.limit_per_model is not None:
        limited: list[dict[str, str]] = []
        existing_counts = Counter(row["agent_id"] for row in existing_rows)
        selected_counts: Counter[str] = Counter()
        for row in pending:
            agent_id = row["agent_id"]
            if existing_counts[agent_id] + selected_counts[agent_id] < max(0, args.limit_per_model):
                limited.append(row)
                selected_counts[agent_id] += 1
        pending = limited
    if not args.execute_paid:
        print(json.dumps({"status": "DRY_RUN", "existing": len(existing), "pending_selected": len(pending), "planned_total": len(schedule)}, indent=2))
        return 0
    _load_dotenv(str(ROOT / ".env"))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    completed = len(existing)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_episode,
                row,
                models[row["agent_id"]],
                public[row["task_id"]],
                private[row["task_id"]],
                benchmark[row["task_id"]],
                args.timeout,
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "record_type": "gateway_agent_episode",
                    "study_mode": "AGENT_EXPLORATORY",
                    "protocol": "gateway_agent_v1",
                    "agent_id": row["agent_id"],
                    "family": row["family"],
                    "episode_index": int(row["episode_index"]),
                    "task_id": row["task_id"],
                    "condition": row["condition"],
                    "final_correct": None,
                    "evaluator_status": "not_scored",
                    "error": sanitize(f"worker_error:{type(exc).__name__}:{exc}")[:1200],
                }
            with RAW.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            completed += 1
            print(
                f"{completed:03d}/144 {row['agent_id']} e{int(row['episode_index']):02d} "
                f"{row['condition']} correct={result.get('final_correct')} status={result.get('evaluator_status')}",
                flush=True,
            )
    records = read_jsonl(RAW)
    status = {
        "status": "COMPLETE" if len(records) == len(schedule) else "RESUMABLE",
        "records": len(records),
        "planned": len(schedule),
        "evaluable": sum(row.get("evaluator_status") == "ok" for row in records),
        "missing_or_error": sum(row.get("evaluator_status") != "ok" for row in records),
        "output": str(RAW.relative_to(ROOT)),
        "warning": "Gateway agents are not human participants or consumer-product evaluations.",
    }
    (RAW.parent / "campaign_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
