# Datasets And Reconstruction

The repository publishes source-free task indexes rather than third-party
source code. This preserves the frozen selection while respecting upstream
licenses and avoiding leakage of references or tests into prompts.

## Released Indexes

| File | Records | Purpose |
|---|---:|---|
| `execrepobench_120_public.jsonl` | 120 | Executable confirmatory task identities and validation provenance |
| `crosscodeeval_100_public.jsonl` | 100 | Native-metric multilingual task identities and selection metadata |
| `task_catalog.csv` | 220 | Compact cross-benchmark lookup table |

## Official Sources

- ExecRepoBench: <https://huggingface.co/datasets/CSJianYang/ExecRepoBench>
- CrossCodeEval: <https://github.com/amazon-science/cceval>
- RepoExec: <https://github.com/FSoft-AI4Code/RepoExec>
- CoderEval: <https://github.com/CoderEval/CoderEval>

CrossCodeEval's official repository is Apache-2.0 licensed and RepoExec's is
MIT licensed. Dataset records may embed material from multiple repositories;
their original licenses still apply. ExecRepoBench and CoderEval users must
consult their official distributions and every source-repository notice.

## ExecRepoBench Reconstruction

1. Download the official dataset to a local benchmark directory.
2. Run `prepare_execrepobench_120.py` to adapt grammar-block samples to
   function-level tasks without inspecting model outputs.
3. Reconstruct repository environments and run
   `validate_execrepobench_120.py`.
4. Include a task only when dependencies are complete and its reference passes
   the executable tests.
5. Run `freeze_execrepobench_120.py` to apply the predeclared repository cap,
   leakage checks, and deterministic hash ordering.

The expected upstream dataset hash is embedded in the preparation script.

## CrossCodeEval Reconstruction

Download the official archive and run `prepare_crosscodeeval_subset.py`. The
selection takes 25 tasks per language across Python, Java, TypeScript, and C#,
with at most two tasks per repository per language. Selection uses a
pre-output artifact hash and does not inspect model results.

## Leakage Controls

- Tests and reference implementations are excluded from generation prompts.
- Exact reference overlap in retrieved chunks is audited before generation.
- Inclusion is fixed before method outputs are inspected.
- All methods and model backends use identical task identities within each
  matched campaign.

## Public Redaction

Public manifests exclude source, prompts, references, tests, local paths, and
repository archives. `tools/build_public_records.py` documents the structured
redaction and emits SHA-256 checksums for the released records.
