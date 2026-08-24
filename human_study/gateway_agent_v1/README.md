# Gateway-Mediated AI-Agent Replication

This directory contains the frozen protocol and reproducible analysis for an
exploratory multi-model robustness study. It is separate from the human study:
the twelve entries are gateway-mediated model configurations, not human
participants or consumer chat products.

## Design

- 12 model configurations and 12 frozen ExecRepoBench review tasks.
- 144 planned independent episodes, balanced at 48 per condition.
- Conditions: generic review, aggregate uncertainty display, and
  source-specific targeted guidance.
- No conversation history, tools, hidden tests, or reference implementation is
  supplied to a model.
- Each response is constrained to one complete target function and evaluated
  by the same local executable tests.
- Model-side missing output is excluded from evaluable correctness and retained
  in the planned-denominator end-to-end sensitivity measure.

The immutable assignment is in `schedule.csv`; configuration details and
hashes are in `protocol_freeze.json`. `model_manifest.json` records primary and
predeclared alternate gateway IDs.

## Reproduction

The paid campaign has already completed. Reanalyze it without making API calls:

```bash
MPLCONFIGDIR=/tmp/opencoder-mpl \
  .venv/bin/python human_study/gateway_agent_v1/analyze_results.py
```

The public task-level file is
`results/agent_gateway_v1/raw_results_public.jsonl`. Its SHA-256 is recorded
in `results/agent_gateway_v1/analysis/integrity.json`. Technical substitutions
are documented in `model_replacement_audit.json` and
`gemini_protocol_amendment.json`; superseded records remain archived and are
not included in the final 144-episode analysis. Generated code, raw provider
text, provider response identifiers, and evaluator logs are redacted; the
retained outcome fields reproduce the released aggregates.

## Reporting Boundary

Use this campaign only as exploratory robustness evidence. It does not measure
developer usability, cannot replace genuine participant data, and must not be
reported as direct evaluation of ChatGPT, Claude.ai, Gemini, Manus,
Perplexity, or any other consumer product.
