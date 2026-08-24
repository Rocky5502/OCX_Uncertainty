# OpenCoder

**Beyond 'What to Retrieve': Uncertainty in Retrieval-Augmented Code Generation**

OpenCoder is a research framework for studying how different retrieved
information sources (similar code, repo context, API knowledge) influence
predictive uncertainty in repository-level code generation, and how
uncertainty signals can be used to drive better retrieval, generation,
and verification.

## 5-Phase / 13-Step Framework

```
Phase 1: Repository Knowledge & Uncertainty Profiling
  1. Parse repo -> AST/symbol graph
  2. Generate API descriptions (LLM)
  3. Build UniXcoder embeddings + vector index

Phase 2: Query Uncertainty Decomposition
  4. Decompose query into sub-intents
  5. Per-intent uncertainty estimation

Phase 3: Uncertainty-Aware Multi-Source Retrieval
  6. Similar code retrieval
  7. Repo context retrieval
  8. API knowledge retrieval
  9. Uncertainty-weighted fusion & reranking

Phase 4: Uncertainty-Guided Generation
 10. Generate N candidates with logprobs
 11. Token entropy + self-consistency + semantic variance

Phase 5: Verification & Mitigation
 12. Static/exec verification
 13. Uncertainty-triggered repair loop
```

## Backends supported
- OpenAI / ChatGPT (`OPENAI_API_KEY`)
- Google Gemini (`GEMINI_API_KEY`)

Configure in `configs/default.yaml`.

## Datasets
- ExecRepoBench
- CoderEval
- RepoExec

See `opencoder/data/` for loaders.

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# or GEMINI_API_KEY / ANTHROPIC_API_KEY

# Run the smoke pipeline on the bundled sample repo
python -m opencoder.cli run --config configs/default.yaml --dataset sample --limit 1

# RQ1 ablation (effect of retrieval source on uncertainty)
python scripts/ablation_rq1.py --config configs/default.yaml --limit 50

# RQ2 ablation (uncertainty-aware vs baseline)
python scripts/ablation_rq2.py --config configs/default.yaml --limit 50
```

## Repo layout
```
opencoder/
  pipeline.py            # 13-step orchestrator
  knowledge/             # Phase 1: parsing, API desc, embeddings
  retrieval/             # Phase 3: similar/context/api retrievers
  uncertainty/           # Phase 2 + 4: entropy, self-consistency, semvar
  generation/            # Phase 4: uncertainty-aware prompting
  verification/          # Phase 5: static + exec checks, repair loop
  llm/                   # backend clients (openai/chatgpt, gemini)
  data/                  # dataset loaders
configs/                 # YAML configs
scripts/                 # ablation harnesses
tests/                   # pytest unit tests
```
