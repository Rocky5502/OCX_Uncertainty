# ExecRepoBench-120 Adaptation Protocol

Protocol status: DRAFT, not frozen. No confirmatory output may be inspected
before the task manifest passes every gate below.

## Candidate Construction

For each official ExecRepoBench candidate:

1. Reconstruct the reference target file as prefix + middle + suffix.
2. Parse the complete file and locate the smallest enclosing function or async
   function containing the masked region.
3. Reject masks that overlap a function signature, decorator, or cannot be
   mapped uniquely to one function.
4. Convert the prediction unit to the complete enclosing function while
   preserving the original repository and context available at inference.
5. Remove the complete target function from every retrieval document and
   index; retain only a signature/docstring stub in the generation prompt.
6. Reconstruct the repository at its recorded commit in an isolated worktree.
7. Restore the reference function and execute the official public tests.
8. Reject dependency, setup, timeout, compilation, or reference-test failures.
9. Check that prompts and retrieval indexes contain no reference function,
   future solution commit, hidden test oracle, duplicate target, or gold patch.
10. Record repository/license/provenance, adaptation rationale, test command,
    environment hash, reference status, and exclusion reason.

## Deterministic Selection

Eligible tasks are sorted by a pre-output hash of dataset revision, repository,
commit, file, function, and original row index. The first 120 passing tasks are
selected subject to a predeclared repository cap that prevents one repository
from dominating the benchmark. Selection never uses model outputs.

## Freeze Artifacts

- `data/manifests/execrepobench_opencoderx_120_v1.jsonl`
- `results/data_quality/execrepobench_120_validation.csv`
- `results/data_quality/retrieval_leakage_report.csv`
- manifest SHA-256, environment hashes, task IDs, exclusions, and freeze date

After freezing, invalid tasks are reported and excluded; they are not silently
replaced after confirmatory generation starts.
