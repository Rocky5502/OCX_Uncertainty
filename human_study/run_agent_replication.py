#!/usr/bin/env python3
"""Run the separate, exploratory ten-session LLM review replication."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.llm.client import LLMClient, _load_dotenv  # noqa: E402
from opencoder.phase5_verify.test_validate import run_execrepobench_function_tests  # noqa: E402


STUDY = ROOT / "human_study"
FROZEN = STUDY / "frozen"
MANIFEST = ROOT / "data/manifests/execrepobench_opencoderx_120_v1.jsonl"
DEFAULT_CONFIG = ROOT / "configs/tosem/models/gpt4o_mini.yaml"
CAMPAIGN = ROOT / "configs/tosem/campaign.yaml"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_schedule(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def condition_prompt(stimulus: dict[str, Any], condition: str) -> str:
    parts = [
        "# Repository review task",
        str(stimulus["task_text"]),
        "\n# Repository context",
        str(stimulus["repository_context"]),
        "\n# Retrieved evidence",
        str(stimulus["retrieved_evidence"]),
        "\n# Starting implementation",
        str(stimulus["starting_code"]),
    ]
    if condition in {"uncertainty_display", "targeted_guidance"}:
        risks = stimulus["source_risks"]
        parts.extend([
            "\n# Uncertainty trace",
            "\n".join(f"{key}: {100*float(risks[key]):.1f}% risk" for key in ("api", "context", "similar_code", "generation")),
            f"aggregate review risk: {100*float(stimulus['aggregate_risk']):.1f}%",
        ])
    if condition == "targeted_guidance":
        parts.extend(["\n# Recommended review action", str(stimulus["targeted_guidance"])])
    parts.extend([
        "\nJudge the starting implementation, then return a corrected complete target function.",
        "Return strict JSON with keys starting_correct (boolean), confidence (integer 0-100), and final_code (string).",
        "Do not return Markdown. Do not describe hidden tests or invent repository files.",
    ])
    return "\n".join(parts)


def parse_response(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    payload = json.loads(candidate)
    if not isinstance(payload.get("starting_correct"), bool):
        raise ValueError("starting_correct must be boolean")
    confidence = int(payload.get("confidence"))
    if not 0 <= confidence <= 100:
        raise ValueError("confidence must be between 0 and 100")
    final_code = str(payload.get("final_code") or "")
    if not final_code.strip():
        raise ValueError("final_code is empty")
    return {"starting_correct": payload["starting_correct"], "confidence": confidence, "final_code": final_code}


def usage_cost(usage: dict[str, int], model: str) -> float:
    campaign = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    pricing = campaign["cost_controls"]["gateway_pricing"][model]
    if str(pricing["currency"]).upper() != "USD":
        raise RuntimeError("agent runner currently requires USD-denominated pricing")
    return (
        int(usage.get("prompt_tokens", 0)) * float(pricing["input_per_million"])
        + int(usage.get("completion_tokens", 0)) * float(pricing["output_per_million"])
    ) / 1_000_000


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "results/human_study/agent_raw.jsonl")
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    schedule = read_schedule(FROZEN / "agent_randomization_schedule.csv")
    if args.limit is not None:
        schedule = schedule[: max(0, args.limit)]
    if not args.execute_paid:
        print(json.dumps({
            "status": "DRY_RUN_ONLY",
            "planned_agent_sessions": len({row["agent_session_id"] for row in schedule}),
            "planned_episodes": len(schedule),
            "model": schedule[0]["model"] if schedule else None,
            "note": "No API call made. Re-run with --execute-paid after protocol review and budget approval.",
        }, indent=2))
        return 0

    model_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["llm"]
    model = str(model_config["model"])
    if {row["model"] for row in schedule} != {model}:
        raise RuntimeError("agent schedule model does not match model configuration")
    client = LLMClient(
        backend=str(model_config["backend"]), model=model,
        temperature=model_config.get("temperature"), max_tokens=2048,
        timeout=int(model_config.get("timeout", 180)),
    )
    stimuli = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_public.jsonl")}
    private = {row["task_id"]: row for row in read_jsonl(FROZEN / "stimuli_private.jsonl")}
    manifest = {row["task_id"]: row for row in read_jsonl(MANIFEST)}
    existing = {(row["agent_session_id"], row["task_id"]) for row in read_jsonl(args.out)} if args.out.is_file() else set()

    for item in schedule:
        key = (item["agent_session_id"], item["task_id"])
        if key in existing:
            continue
        before = client.usage_snapshot()
        started = time.perf_counter()
        raw_text = ""
        parsed: dict[str, Any] | None = None
        error: str | None = None
        try:
            response = client.complete_one(
                condition_prompt(stimuli[item["task_id"]], item["condition"]),
                system="You are an exploratory AI code-review agent. Follow the requested strict JSON schema.",
                seed=int(item["seed"]), return_logprobs=False,
            )
            raw_text = response.text
            parsed = parse_response(raw_text)
        except Exception as exc:
            error = f"{type(exc).__name__}:{str(exc)[:300]}"
        latency = time.perf_counter() - started
        after = client.usage_snapshot()
        delta = {name: int(after[name]) - int(before[name]) for name in after}
        spend = usage_cost(after, model)
        if spend > args.max_cost_usd:
            raise RuntimeError(f"agent cost cap exceeded: USD {spend:.4f} > {args.max_cost_usd:.4f}")
        report = None
        if parsed is not None:
            report = run_execrepobench_function_tests(parsed["final_code"], manifest[item["task_id"]], timeout=240)
        row = {
            "record_type": "agent_episode", "study_mode": "AGENT_EXPLORATORY",
            "agent_session_id": item["agent_session_id"], "model": model,
            "seed": int(item["seed"]), "episode_index": int(item["episode_index"]),
            "task_id": item["task_id"], "condition": item["condition"],
            "signal_category": private[item["task_id"]]["signal_category"],
            "initial_correct": private[item["task_id"]]["initial_correct"],
            "starting_judgment_correct": None if parsed is None else parsed["starting_correct"],
            "starting_confidence": None if parsed is None else parsed["confidence"],
            "final_code": "" if parsed is None else parsed["final_code"],
            "final_correct": None if report is None or report.returncode in {2, -1} else bool(report.passed),
            "evaluator_status": "not_scored" if report is None else ("error" if report.returncode in {2, -1} else "ok"),
            "raw_response": raw_text, "response_metadata": {} if parsed is None else response.raw,
            "usage": delta, "latency_seconds": latency, "error": error,
        }
        append_jsonl(args.out, row)
        existing.add(key)
        print(f"{item['agent_session_id']} {item['task_id']} {item['condition']} final={row['final_correct']} cost_usd={spend:.4f}")

    final_usage = client.usage_snapshot()
    summary = {
        "status": "COMPLETED_OR_RESUMABLE",
        "study_mode": "AGENT_EXPLORATORY",
        "records": len(read_jsonl(args.out)),
        "usage": final_usage,
        "cost_usd_this_process": usage_cost(final_usage, model),
        "output": str(args.out.relative_to(ROOT)),
        "warning": "Agent sessions are exploratory and must not be pooled with human participants.",
    }
    args.out.with_suffix(".status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
