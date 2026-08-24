# Results

## Technical Campaign

`tosem/` contains analysis-ready task-level records, summaries, paired
statistics, resource use, calibration, integrity reports, LaTeX tables, and
publication figures for ExecRepoBench-120 and CrossCodeEval-100.

The public release omits raw generated code, provider text, response IDs,
repository source, reference implementations, and evaluator logs. Candidate
outcomes and selected-output correctness remain available at task level.

## API Grounding

`api_grounding/` contains the reconciled RepoExec/CoderEval API-set audit and
its authoritative aggregation records.

## Automated Reviewers

`agent_gateway_v1/` contains 144 redacted task records and the associated
analysis. Missing outputs remain explicit; they are not silently imputed.

## Regeneration

```bash
make tables
make figures
make agent-analysis
make verify-public
```

The manuscript is not part of this repository.
