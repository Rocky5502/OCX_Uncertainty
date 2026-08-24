# Confirmatory Protocol Amendment: Repair Prompt Budget

Date: 2026-08-10
Status: `FROZEN_BEFORE_RESUME`

## Trigger

During GPT RAG + Verify/Repair batch 07, task
`execrepobench-0868-5875950a21` produced a 384,260-token repair request. The
gateway rejected it because GPT-4o-mini supports a 128k context. The affected
cell is an infrastructure failure and is not scored as model incorrect.

## Root Cause

Generation used the declared prompt/evidence budget, but the repair loop
appended unbounded executable-test stdout/stderr and the full original task to
the failed code. A repository test report can be much larger than the initial
generation prompt. This violated the intended matched prompt-budget policy.

## Frozen Amendment

Every repair call in both RAG + Verify/Repair and OpenCoderX now uses the same
deterministic head/tail character budgets:

- original task: 24,000 characters;
- failed code: 32,000 characters;
- diagnostics: 32,000 characters, preserving 80% of the retained diagnostic
  budget from the tail where assertion traces and failure summaries occur;
- repair output: unchanged at 1,200 tokens;
- repair rounds: unchanged at at most two.

The maximum constructed user prompt is below 90,000 characters. Every repair
history records original/retained sizes, truncation flags, policy identifier,
and the resulting prompt size.

## Reproducibility Action

Direct and Standard RAG are unaffected. All pre-amendment RAG + Verify/Repair
and OpenCoderX confirmatory files are preserved as superseded infrastructure
runs and excluded from final analysis. Those two methods are rerun from their
frozen batch manifests for every model family. No completed Direct or Standard
RAG output is regenerated. Superseded API usage remains included in campaign
cost accounting.

This amendment fixes a provider-context incompatibility. It does not change
task inclusion, retrieval, candidate count, evidence budget, tests, selection,
uncertainty thresholds, or statistical analysis.
