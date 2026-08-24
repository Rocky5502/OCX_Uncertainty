from __future__ import annotations

import json
from pathlib import Path

from opencoder.evaluation.crosscodeeval import CrossCodeEvalEvaluator


ROOT = Path(__file__).resolve().parents[1]


def test_crosscodeeval_gold_self_match_for_all_languages() -> None:
    evaluator = CrossCodeEvalEvaluator(ROOT / ".benchmarks/crosscodeeval")
    rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/crosscodeeval_opencoderx_100_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for language in ("python", "java", "typescript", "csharp"):
        row = next(item for item in rows if item["language"] == language)
        metrics = evaluator.evaluate(
            prompt=row["prompt"],
            prediction=row["reference_completion"],
            target=row["reference_completion"],
            language=language,
        )
        assert metrics.exact_match == 1.0
        assert metrics.edit_similarity == 1.0
        assert metrics.identifier_exact_match == 1.0
        assert metrics.identifier_f1 == 1.0


def test_crosscodeeval_extracts_fenced_completion() -> None:
    evaluator = CrossCodeEvalEvaluator(ROOT / ".benchmarks/crosscodeeval")
    assert evaluator.extract_completion("```java\nreturn value;\n```") == "return value;"
