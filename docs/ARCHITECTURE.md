# Architecture

OpenCoderX is organized as five linked phases.

## I. Repository Knowledge And Uncertainty Profiling

Repository APIs, contextual code, and similar implementations are extracted,
described, embedded, and assigned source-specific uncertainty metadata.

## II. Query Uncertainty Decomposition

The requested implementation is decomposed into steps. Step-level uncertainty
and retrieval intent determine which evidence sources should be queried.

## III. Uncertainty-Aware Multi-Source Retrieval

API, context, and similar-code retrievers produce candidates. Evidence is
scored by relevance, source uncertainty, agreement, conflict, and redundancy,
then fused under per-source and total retrieval budgets.

## IV. Uncertainty-Guided Generation

The generator receives the user query, selected evidence, and an uncertainty
trace. Five candidates are sampled in the frozen campaign. Candidate Pass@k is
computed only from this original set.

## V. Verification And Mitigation

Static checks and executable tests validate candidates. Verified selection and
at most two targeted repairs produce a separate selected-output outcome.
Repaired outputs are never relabeled as ordinary sampling gains.

## Collaboration Policy

`opencoderx.collaboration` maps pre-decision risk to:

- **ACT**: generate and verify when evidence is sufficiently reliable.
- **ASK**: request a source-specific item when localized uncertainty is high.
- **STOP**: abstain or escalate when evidence remains conflicting or
  insufficient.

Executable outcomes are not used to construct the uncertainty-only routing
score. Lifecycle-gated analyses are reported separately.
