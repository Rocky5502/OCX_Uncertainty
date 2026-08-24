# Pre-Freeze Retrieval Correction

Status: `ACCEPTED_AND_FROZEN` on 2026-08-10.

## Scope

The non-confirmatory 10-task ExecRepoBench pilot exposed four OpenCoderX
selected-output losses. All four were candidate-availability losses: no correct
candidate remained at Pass@5. Across the five selected-output discordances,
the mean evidence-set Jaccard overlap was 0.022, and OpenCoderX allocated 68%
of fused evidence to APIs compared with 24% for the matched controls. One loss
used ten API items; another produced no fused evidence.

The task-level source is
`results/tosem/pilot/retrieval_diagnostics.csv`; the derivation is implemented
in `scripts/diagnose_tosem_pilot_retrieval.py`.

## Correction

Two provider-independent retrieval controls are added to full OpenCoderX only:

1. **Whole-task anchor.** In addition to step-specific queries, retrieve from
   the unchanged complete task query using the same frozen per-source indexes
   and per-query top-k values.
2. **Source-balanced fusion.** Preserve at least the best available item from
   each source and cap a source at 50% of the final evidence budget whenever
   alternatives exist. If fewer sources are available, fill the declared final
   budget by score.

The final evidence budget remains 10. Candidate count remains five, generation
and prompt budgets are unchanged, tests are unchanged, and at most two repair
rounds remain available. Direct, Standard RAG, and RAG + Verify/Repair are
unchanged. No confirmatory task or output was inspected.

## One-Shot Development Decision Rule

Run only the corrected OpenCoderX condition on the frozen pilot-10 manifest for
all four model families. Preserve the original pilot files and write separate
raw outputs. Accept the correction and freeze it when all conditions hold:

1. all 40 task-model cells complete with five candidates and valid provider
   provenance;
2. all cells retain 10 fused evidence items, record a whole-task anchor, and
   no source exceeds five items when multiple sources are available;
3. no model loses more than one selected-output success relative to its
   original OpenCoderX pilot;
4. pooled selected-output correctness decreases by no more than 2/40 tasks
   (five percentage points);
5. pooled task-level Pass@5 decreases by no more than 2/40 tasks.

These are integrity and non-inferiority gates, not a requirement to outperform
the matched baseline. The correction is evaluated once. If it fails, revert to
the original implementation for confirmatory work and report the pilot
boundary; do not introduce a second outcome-driven correction.

The re-pilot remains development evidence and is never promoted to a journal
result.

## Decision

The one-shot audit passed with no integrity issues. All four model families met
the per-family selected-output non-inferiority gate. Pooled selected-output
correctness and task-level Pass@5 each changed from 16/40 under the original
pilot implementation to 17/40 after correction. The accepted decision and
task-level derivation are stored in:

- `results/tosem/pilot/anchor_repilot_decision.json`;
- `results/tosem/pilot/anchor_repilot_summary.csv`;
- `results/tosem/pilot/anchor_repilot_by_task.csv`.

The corrected implementation is the frozen OpenCoderX method for confirmatory
execution. No further method tuning is permitted from this point.
