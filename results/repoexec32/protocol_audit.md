# Expanded RepoExec-inline Protocol Audit

- Remaining tasks audited before generation: 25
- New tasks accepted: 18
- Tasks excluded by the reference-validation gate: 7
- Complete expanded task count: 32
- Raw task-backend-method records: 192
- Methods: Baseline RAG, RAG + Verify/Repair, OpenCoder
- Backends: gpt-4o-mini, gemini-2.5-flash
- Candidate budget: five
- Temperature: 0.7
- Maximum repair rounds: two
- Retrieval budgets: API/context/similar-code = 8/8/8; fused = 10

## Checks

- expected_raw_records: PASS
- five_candidates_per_record: PASS
- five_outcomes_per_record: PASS
- no_missing_selected_test_result: PASS
- no_generation_integrity_failures: PASS
- no_recorded_failures: PASS
- analytical_raw_record_count_match: PASS
- Baseline RAG and RAG + Verify/Repair exact candidate reuse: PASS
- Baseline RAG and RAG + Verify/Repair exact evidence reuse: PASS
- Original/new task sets are disjoint: PASS
- Missing API responses excluded rather than scored: PASS

New-task retry telemetry is recorded directly. Legacy 14-task records predate retry telemetry and are marked unavailable rather than imputed.
