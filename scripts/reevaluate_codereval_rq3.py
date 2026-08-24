"""Re-evaluate saved RQ3 CoderEval runs after evaluator fixes.

This script does not call an LLM. It reloads generated candidates from an
existing RQ3 JSON file and recomputes correctness with the mutation-audited
native CoderEval project harness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_rq3 import _summarize  # noqa: E402
from opencoder.data.loaders import Example, load_dataset  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402


def _load_examples(dataset: str, path: str) -> Dict[str, Example]:
    return {example.id: example for example in load_dataset(dataset, path)}


def _make_validator() -> Pipeline:
    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = PipelineConfig(llm_backend="offline", embedding_model="")
    return pipe


def _candidate_stream(row: Dict[str, Any]) -> List[str]:
    samples = row.get("generated_samples") or []
    if samples:
        return [str(sample) for sample in samples]
    code = row.get("code")
    return [str(code)] if code is not None else []


def _evaluate_samples(pipe: Pipeline, example: Example, samples: Iterable[str]) -> List[bool]:
    out: List[bool] = []
    for sample in samples:
        _, report = pipe._validate_code(sample, example)
        if report.passed is not None:
            out.append(bool(report.passed))
    return out


def _evaluate_code(pipe: Pipeline, example: Example, code: str) -> tuple[bool | None, Dict[str, Any]]:
    _, report = pipe._validate_code(code, example)
    return report.passed, report.__dict__


def reevaluate_run(
    run_path: Path,
    out_path: Path,
    dataset_path_override: Path | None = None,
) -> Dict[str, Any]:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    metadata = deepcopy(data.get("metadata") or {})
    dataset = str(metadata.get("dataset") or "codereval")
    dataset_path = str(dataset_path_override or metadata.get("dataset_path") or "")
    if dataset.lower() != "codereval":
        raise ValueError(f"Expected a CoderEval run, got dataset={dataset!r}")
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Cannot load CoderEval dataset path from metadata: {dataset_path!r}")

    examples = _load_examples(dataset, dataset_path)
    pipe = _make_validator()
    out = {k: deepcopy(data.get(k) or []) for k in ("with", "without")}

    for method_key in ("without", "with"):
        for row in out[method_key]:
            example = examples.get(str(row.get("id")))
            if example is None:
                row["reevaluation_error"] = "missing example"
                continue
            samples = _candidate_stream(row)
            sample_correctness = _evaluate_samples(pipe, example, samples)
            passed, test_report = _evaluate_code(pipe, example, str(row.get("code") or ""))
            effective = list(sample_correctness)
            if method_key == "with" and sample_correctness and passed is not None:
                effective = [bool(passed), *sample_correctness[1:]]
            row["sample_correctness"] = sample_correctness
            row["effective_sample_correctness"] = effective
            row["passed"] = passed
            row["test_report"] = test_report
            row["correctness_mode"] = pipe._correctness_mode(example)
            row["reevaluated_from"] = os.fspath(run_path)

    metadata["reevaluated_from"] = os.fspath(run_path)
    metadata["dataset_path"] = dataset_path
    metadata["reevaluation_note"] = (
        "Recomputed with the mutation-audited native CoderEval project harness. "
        "Saved generations and uncertainty traces are unchanged; no LLM calls were made."
    )
    payload: Dict[str, Any] = {
        "with": out["with"],
        "without": out["without"],
        "summary": _summarize(out),
        "metadata": metadata,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-path", default=None)
    args = parser.parse_args()
    payload = reevaluate_run(
        Path(args.run),
        Path(args.out),
        Path(args.dataset_path) if args.dataset_path else None,
    )
    print(json.dumps(payload.get("summary", {}), indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
