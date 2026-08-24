# OpenCoder Setup

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

The default encoder falls back to a deterministic hash encoder when
`torch`/`transformers` are unavailable, so smoke tests do not require a
large model download.

## Configure LLM

OpenCoder currently supports the two backends needed for this project:

```bash
# OpenAI / ChatGPT
export OPENCODER_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...

# Gemini
export OPENCODER_LLM_BACKEND=gemini
export GEMINI_API_KEY=...
export OPENCODER_LLM_MODEL=gemini-2.5-flash

# ZhiZengZeng gateway, shared key for ChatGPT/Gemini-compatible models
export OPENCODER_LLM_BASE_URL=https://api.zhizengzeng.com/v1
export ZHIZENGZENG_API_KEY=...
```

OpenAI logprobs feed token-entropy uncertainty. Gemini uses the same
multi-sample path, and when token logprobs are unavailable the aggregate
uncertainty still uses self-consistency and semantic variance.

## Bundled Data

The loaders know these local files:

- `sample`: `input/input.jsonl`
- `execrepobench`: `input/execrepobench_data.jsonl`

You can still pass `--dataset-path` for external CoderEval, RepoExec, or
alternate ExecRepoBench JSONL files.

## Run

```bash
python -m opencoder.cli run --dataset sample --limit 1
python -m opencoder.cli run --dataset execrepobench --limit 1
```

## Experiments

```bash
python scripts/ablation_rq1.py --dataset execrepobench --limit 10 --out results/rq1.json
python scripts/ablation_rq2.py --dataset execrepobench --limit 10 --out results/rq2.json
```

RQ1 records per-source enabled conditions, aggregate uncertainty,
pass/fail when a reference or tests are available, sample correctness,
pass@1/3/5, and pass-rate variance. RQ2 compares uncertainty-aware
retrieval/scoring against the baseline with uncertainty filtering turned
off.

## Verify

```bash
python -m pytest -q
python -m compileall -q opencoder scripts tests
```

## Current File Map

```text
configs/default.yaml
opencoder/pipeline.py
opencoder/phase1_repo_knowledge/
opencoder/phase2_query/
opencoder/phase3_retrieval/
opencoder/phase4_generation/
opencoder/phase5_verify/
opencoder/uncertainty/
opencoder/evaluation/
opencoder/llm/client.py
opencoder/data/loaders.py
scripts/ablation_rq1.py
scripts/ablation_rq2.py
tests/
```
