# Experiments

## ExecRepoBench-120

The confirmatory campaign crosses four model families with four methods:

1. Direct Generation
2. Standard RAG
3. RAG + Verify/Repair
4. OpenCoderX

Each task-method-model cell uses five candidates and the same executable tests.
Pass@1/3/5 is estimated from the original candidates. Selected-output
correctness is reported separately after each method's declared selection and
repair policy. Comparisons use paired task bootstrap intervals, exact McNemar
tests for selected correctness, and Holm correction within declared families.

## CrossCodeEval-100

Direct Generation and Cross-file Context RAG are compared using the benchmark's
native exact match, edit similarity, and identifier F1. These records do not
support functional-correctness claims.

## Source And Uncertainty Analyses

The focused factorial analysis crosses API, context, and similar-code evidence
on ten tasks for two backends. Uncertainty discrimination and calibration are
also evaluated over all 480 OpenCoderX ExecRepoBench model-task records and
transferred to CrossCodeEval.

## Collaboration Analyses

Fixed uncertainty thresholds route tasks to autonomous action, review, or
abstention. Reviewer-budget outcomes are simulations, not observed developer
performance. Matched method transitions provide observed technical
interventions; source-specific human interventions are not treated as run when
the corresponding records are absent.

## Models

The technical campaign identifies the served model using frozen IDs:
`gpt-4o-mini`, `gemini-2.5-flash`, `claude-sonnet-5`, and
`qwen3-coder-plus`. Future provider aliases are not automatically equivalent.
