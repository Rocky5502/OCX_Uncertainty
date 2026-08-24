# Exploratory AI-Agent Replication

## Purpose

The agent replication asks whether the same uncertainty displays alter an
LLM reviewer's decisions on the frozen human-study stimuli. It is exploratory
and does not simulate people, validate human usability, or increase the human
sample size.

## Design

- Population: ten independent GPT-4o-mini review sessions (`A001`--`A010`).
- Episodes: twelve frozen tasks per session, one observation per task.
- Conditions: generic review, uncertainty display, and targeted guidance.
- Inputs: identical task text, repository context, retrieved evidence, and
  starting implementation used by the human interface.
- Outcome: executable correctness of the submitted complete function.
- Separation: agent records use `AGENT_EXPLORATORY` and are never pooled with
  participant records or included in human inferential tests.

The ten-session schedule is not perfectly divisible across three assignment
groups. Report exact episode denominators by condition and treat comparisons as
descriptive. A balanced 12-session extension would require a new protocol
version and must not replace the frozen plan after outcomes are observed.

## Execution Guard

The runner makes no API request unless `--execute-paid` is supplied and stops
when its explicit cost cap is reached. Run a zero-cost protocol check first:

```bash
PYTHONPATH=. .venv/bin/python human_study/run_agent_replication.py
```

Authorized execution:

```bash
PYTHONPATH=. .venv/bin/python human_study/run_agent_replication.py \
  --execute-paid --max-cost-usd 1.00
```

Report model identifier, gateway/provider, schedule hash, errors, tokens,
latency, and exact task-level outcomes. A missing or evaluator-error response
is missing, never correct or incorrect.
