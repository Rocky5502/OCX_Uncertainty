# OpenCoderX Confirmatory Protocol

Status: `FROZEN_APPROVED`. Dataset inclusion, model families, methods,
metrics, thresholds, and statistical tests are frozen. The bounded smoke and
two pilots passed their integrity audits, the one-shot retrieval correction was
accepted under its predeclared rule, and gateway pricing is recorded. The
measured full-matrix projection was approved under revised operational caps of
USD 80 and CNY 65. No outcome from a confirmatory cell may be used for tuning.

## Primary Study

- Dataset: ExecRepoBench-120 manifest
  `data/manifests/execrepobench_opencoderx_120_v1.jsonl`.
- Manifest SHA-256:
  `4b14e4a648e80b11c7a7011b6c9f878fcae5ec243f5c1b260e905e693c3f459d`.
- Models: `gpt-4o-mini`, `gemini-2.5-flash`, `claude-sonnet-5`, and
  `qwen3-coder-plus`, all requested through the same ZhiZengZeng gateway.
- Methods: Direct, Standard RAG, RAG + Verification + Repair, and full
  OpenCoderX.
- Sampling: five candidates, temperature 0.7 where accepted, 2,048 maximum
  output tokens, and at most two repair rounds. Claude omits unsupported
  non-default temperature/top-p parameters; this difference is recorded.
- Retrieval: the same frozen candidate pools, source budgets, prompt budget,
  and evidence ordering for matched methods. The target implementation and
  evaluator tests are excluded from prompt-visible evidence.
- Correctness: official repository `evaluate_repo.py` execution in an isolated
  copy using the audited per-repository Python environment.

## Outcomes and Analysis

Primary technical outcomes are Pass@1, selected-output correctness, test
success, and repair success. Uncertainty outcomes are AUROC, AUPRC, Brier score,
ECE, calibration slope/intercept, AURC, and risk-coverage. Paired correctness
uses exact McNemar tests and paired bootstrap 95% confidence intervals. Holm
correction is applied within each declared family of confirmatory comparisons.
Statistical non-significance is not interpreted as equivalence.

CrossCodeEval-100 is evaluated only with its native exact-match, edit-similarity,
and identifier-F1 metrics; these are never called functional correctness.
Multi-SWE-bench-Flash-35 is a stress benchmark and remains `BLOCKED` until its
official Docker evaluator can run. Existing RepoExec and CoderEval artifacts are
reused only under the provenance boundaries in `EXISTING_RESULTS_AUDIT.md`.

## Collaboration Protocol

Selective decisions are `AUTONOMOUS`, `REQUEST_REVIEW`, and `ABSTAIN` using the
predeclared review threshold 0.35 and abstention threshold 0.70. Review budgets
are 0%, 5%, 10%, 20%, 30%, 50%, and 100%. Reviewer-success simulations use
60%, 75%, 90%, and 100% with seeds 20260809--20260813. The 100% condition is an
oracle/reference-review upper bound, not observed human performance.

## Execution Gates

1. Full regression passes.
2. All four model IDs remain present in the gateway catalog.
3. A two-to-three-task smoke run completes without infrastructure failures.
4. Actual gateway usage metadata and response IDs are preserved.
5. Model-specific pricing and estimated cost are below configured run and
   campaign caps.
6. Raw records pass completeness, duplicate, task-set, leakage, and credential
   audits.

Gateway prices were documented on 2026-08-10, opening the bounded smoke gate.
The smoke and pilot gates completed on 2026-08-10. The executable pilot used 10
frozen ExecRepoBench tasks, and the multilingual pilot used two tasks per
CrossCodeEval language. Both are explicitly marked `paper_eligible=false`.

## Pre-Freeze Diagnostic Boundary

The executable pilot found that full OpenCoderX did not consistently outperform
the matched RAG + Verify/Repair baseline. An offline discordance audit traced
several losses to step-specific retrieval queries displacing stronger
whole-task matches and to source-imbalanced final fusion. This diagnosis was
made exclusively on the non-confirmatory pilot. Before freeze, either:

1. apply a provider-independent whole-task retrieval anchor and source-balanced
   fusion, rerun only the bounded pilot cells affected by that change, and
   accept or reject the change using a predeclared integrity/parity rule; or
2. preserve the current method unchanged and state this pilot-observed boundary
   in the protocol.

The one-shot re-pilot passed all integrity and non-inferiority gates. Pooled
selected-output correctness and task-level Pass@5 each changed from 16/40 to
17/40. The whole-task anchor and source-balanced fusion are therefore accepted
and frozen. The re-pilot is development evidence only and cannot enter the
journal results.

## Measured Cost Gate

Scaling the frozen 10-task resource records, including the accepted corrected
OpenCoderX implementation, to 120 tasks gives USD 62.63 plus CNY 48.24 for the
full four-model, four-method Tier-A matrix. All measured executable,
multilingual, and correction pilots bring the projected totals to USD 69.98 and
CNY 54.08. On 2026-08-10, the campaign caps were explicitly revised to USD 80
and CNY 65 before confirmatory analysis. The complete matrix is retained.

## Repair-Prompt Amendment

One GPT RAG + Verify/Repair request exposed an unbounded diagnostic payload and
was rejected by the provider before generation. Before resuming, a shared,
deterministic repair-prompt budget was frozen for RAG + Verify/Repair and
OpenCoderX. All pre-amendment outputs from those two methods are preserved as
superseded infrastructure runs, excluded from analysis, and rerun from batch
00. Direct and Standard RAG are unaffected. Full details are recorded in
`PROTOCOL_AMENDMENT_REPAIR_BUDGET.md`.
