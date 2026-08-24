#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run results/rq3/runs/matched_external/gpt_repoexec_full/rq3.json \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark repoexec \
  --benchmark-path input/repoexec_python_string_utils_inline14.jsonl \
  --limit 14 \
  --apply

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run results/rq3/runs/matched_external/gemini_repoexec_full/rq3.json \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark repoexec \
  --benchmark-path input/repoexec_python_string_utils_inline14.jsonl \
  --limit 14 \
  --apply

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run results/rq3/runs/matched_external/gpt_execrepobench_full/rq3.json \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark execrepobench \
  --benchmark-path input/execrepobench_testbacked.jsonl \
  --limit 10 \
  --apply

"$PYTHON" experiments/derive_rag_verify_repair.py \
  --run results/rq3/runs/matched_external/gemini_execrepobench_full/rq3.json \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark execrepobench \
  --benchmark-path input/execrepobench_testbacked.jsonl \
  --limit 10 \
  --apply

"$PYTHON" experiments/repair_exact_baseline_candidates.py \
  --run results/rq3/runs/matched_external/gpt_repoexec_full/rq3.json \
  --config configs/rq3/gpt4o_mini.yaml \
  --benchmark repoexec \
  --benchmark-path input/repoexec_python_string_utils_inline14.jsonl \
  --limit 14

"$PYTHON" experiments/repair_exact_baseline_candidates.py \
  --run results/rq3/runs/matched_external/gemini_repoexec_full/rq3.json \
  --config configs/rq3/gemini_2_5_flash.yaml \
  --benchmark repoexec \
  --benchmark-path input/repoexec_python_string_utils_inline14.jsonl \
  --limit 14

echo "Exact-candidate RQ3 repairs completed for GPT and Gemini."
