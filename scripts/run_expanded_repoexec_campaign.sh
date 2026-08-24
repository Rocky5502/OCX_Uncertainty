#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MANIFEST="results/rq3_expanded_repoexec/new_tasks_manifest.jsonl"
GPT_RUN="results/rq3_expanded_repoexec/runs/gpt_new/rq3.json"
GEMINI_RUN="results/rq3_expanded_repoexec/runs/gemini_new/rq3.json"

"$PYTHON" experiments/run_rq3.py \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --method paired \
  --limit 18 \
  --describe-limit all \
  --repo-root input/string_utils \
  --out "$GPT_RUN"

"$PYTHON" experiments/run_rq3.py \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --method paired \
  --limit 18 \
  --describe-limit all \
  --repo-root input/string_utils \
  --out "$GEMINI_RUN"

"$PYTHON" scripts/check_expanded_rq3_runs.py \
  --manifest "$MANIFEST" \
  --run "$GPT_RUN" \
  --run "$GEMINI_RUN" \
  --conditions without with

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run "$GPT_RUN" \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --limit 18 \
  --apply

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run "$GEMINI_RUN" \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --limit 18 \
  --apply

"$PYTHON" experiments/repair_exact_baseline_candidates.py \
  --run "$GPT_RUN" \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --limit 18

"$PYTHON" experiments/repair_exact_baseline_candidates.py \
  --run "$GEMINI_RUN" \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark repoexec \
  --benchmark-path "$MANIFEST" \
  --limit 18

"$PYTHON" scripts/check_expanded_rq3_runs.py \
  --manifest "$MANIFEST" \
  --run "$GPT_RUN" \
  --run "$GEMINI_RUN" \
  --conditions without rag_repair with

echo "Expanded RepoExec campaign completed for both backends."
