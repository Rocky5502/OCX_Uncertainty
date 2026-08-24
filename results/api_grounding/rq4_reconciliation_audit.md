# RQ4 Reconciliation Audit

Audit date: 2026-07-27 (Asia/Shanghai)

## Scope and decision

This audit used only existing local artifacts. It made no LLM/API calls and did
not modify the AAAI manuscript or its PDF.

The values `37.8/51.0/48.6`, `16.7/50.0/47.6`, and
`0.120/0.463/0.179/0.382` belong to an **older RQ4 campaign**. They are not the
source of truth for the current paper. The corrected campaign under
`results/rq4/` reproduces all current paper anchors and verifies the missing
OpenCoder-NoAPIRefine Exact values:

- GPT OpenCoder-NoAPIRefine Exact: **1/26 = 3.846153846% = 3.8%**.
- Gemini OpenCoder-NoAPIRefine Exact: **0/26 = 0.000000000% = 0.0%**.

The gate required by the audit is satisfied:

| Backend | Method | Recomputed macro F1 | Recomputed Exact |
|---|---|---:|---:|
| GPT | Baseline RAG | 43.4 | 0.0 |
| GPT | OpenCoder-NoAPIRefine | 45.1 | 3.8 |
| GPT | OpenCoder | 64.8 | 57.7 |
| Gemini | Baseline RAG | 43.4 | 0.0 |
| Gemini | OpenCoder-NoAPIRefine | 39.9 | 0.0 |
| Gemini | OpenCoder | 59.1 | 50.0 |

## Paper anchors

The audited paper is:

`AAAI_27_Project_1__Final__Technical_Track_Version_2 (1).pdf`

- SHA-256:
  `25099bbc0858e812cc22fe50c548725cf317fe5501f05982291f19dff06c18fd`
- Modified: 2026-07-27 00:29:10 +0800
- Table 4 location: PDF page 7

The same current anchors occur in:

- `latextRQ4.txt`
- `results/rq4/table_api_quality.tex`
- `results/rq4/table_api_item_uncertainty.tex`
- `results/rq4/api_quality_summary.csv`
- `results/rq4/api_item_uncertainty_summary.csv`
- `results/camera_ready_results_7page.tex`

## Authoritative corrected campaign

### Command and evaluator

The corrected RQ4 outputs were produced by replaying cached RQ3 records:

```bash
.venv/bin/python -m experiments.run_rq4 --out-dir results/rq4
```

No generation calls are made by this replay.

- Evaluator: `experiments/run_rq4.py`
- Evaluator SHA-256:
  `60ac2f82bd94fbe0a6e7f1bb836011011e219ead0aa615f1a3d2205a0b78e332`
- API normalizer/refiner:
  `opencoder/phase3_retrieval/api_refine.py`
- Audit recomputation:
  `scripts/audit_rq4_reconciliation.py`

The independent audit recomputed all 516 task-backend-method records through
`_record_to_metrics`, compared them field-by-field with
`per_task_api_metrics.csv`, and aborted unless all current paper anchors
matched after one-decimal rounding.

### Corrected result files

| Artifact | SHA-256 |
|---|---|
| `results/rq4/raw_api_predictions.jsonl` | `9130a107227510ff7cc6d089012a27cd113f5284b1ee194c7af73d07a2c906d5` |
| `results/rq4/ground_truth_api_sets.jsonl` | `6850b1ddc55b78ae60dab604a11d6c7b969f731fa25dbc8c672210d51683a943` |
| `results/rq4/per_task_api_metrics.csv` | `142c10b27ddd9577d310f290eac5fa3cb85f221dbc261433aff9acba20885934` |
| `results/rq4/api_quality_summary.csv` | `ac0ef939dc8a6987040e2492333a244f173c615979c68121d1141448c793c07f` |
| `results/rq4/api_item_uncertainty.csv` | `ff5b0f61be1bcaf3ad0c668b22a66b2c9c1f71db6a7f43ce6a2c146f9b537958` |
| `results/rq4/api_item_uncertainty_summary.csv` | `3f85327d03eab18f9ce9b54b0f264b0359984a8bccefb307859d183a571a34de` |
| `results/rq4/table_api_quality.tex` | `391b6b00ce090aff9550c3e173580413dffb82024c4cdfcccc0e4d19c86f9008` |
| `results/rq4/table_api_item_uncertainty.tex` | `fe08c2c8a3ea7605eb47dec83b4c2be506c0a3d60a5ad3aea9329ff1cc296bc6` |

The raw records contain timestamps from
`2026-07-14T15:54:45.593604+00:00` through
`2026-07-14T15:55:00.801116+00:00`.

### Upstream RQ3 run IDs

The following immutable file hashes serve as the available run IDs:

| Benchmark | Backend | Source run | Created at (UTC) | SHA-256 |
|---|---|---|---|---|
| RepoExec | GPT | `results/rq3/runs/repoexec_inline14_gpt/rq3.json` | 2026-07-04 08:29:36 | `b16fdc7ef6db5527762a16379ab6b8ec846c32bff2d783660bb799a77b776e0f` |
| RepoExec | Gemini | `results/rq3/runs/repoexec_inline14_gemini/rq3.json` | 2026-07-04 08:55:25 | `c0501628ffc3b59f95c8e177d73a3c6f380601ef18d73f14d14692ad40f86a57` |
| CoderEval | GPT | `results/rq3/codereval_exec19/replication_gpt/rq3.json` | 2026-07-07 17:54:50 | `ec51f99ca6ea370fec23f4fbdf2045f9e8af2204af8f95a455e5eb3bd37f88c0` |
| CoderEval | Gemini | `results/rq3/codereval_exec19/replication_gemini/rq3.json` | 2026-07-07 18:30:00 | `0c46b0ce929e4c8fb7b8285f2652ab0b895e4e124be441df19a16e4b45d24d43` |
| ExecRepoBench | GPT | `results/rq3/runs/execrepobench_testbacked10_gpt/rq3.json` | 2026-07-07 06:45:07 | `0505a28a3632dbd5690dd9978d8d09bc3d450fef5d7902f301faa4e456c1c38e` |
| ExecRepoBench | Gemini | `results/rq3/runs/execrepobench_testbacked10_gemini/rq3.json` | 2026-07-07 07:05:47 | `c5b1bf3e3d650e4cfd1dd79ff05ab77bc158ececa2d1977c067dcfc0b2644236` |

The CoderEval files are test-harness reevaluations of saved generations. Their
metadata explicitly states that no new LLM calls were made.

### Configuration provenance

The source-run metadata records `gpt-4o-mini` and `gemini-2.5-flash`,
temperature `0.7`, and five samples. The evaluator uses API/context/similar-code
top-k `8/8/8`, fused top-k `10`, uncertainty alpha `0.5`, keep quantile `0.7`,
no test context, and at most two repair rounds.

Current configuration files and audit-time hashes are:

- `configs/rq3/gpt4o_mini.yaml`:
  `61dc096dfc5777d6e4d94a7e580983aa6542bb31a35d3660eefc070376f9a0a1`
- `configs/rq3/gemini_2_5_flash.yaml`:
  `3e0de931f7d8f13762e6ed8fff5fbd1935f8b89e562b27c7cde408f583755d0a`

An at-run config hash was not embedded in the RQ3 or RQ4 records. The Gemini
YAML was modified after the corrected RQ4 campaign, so its current hash must
not be represented as the at-run hash. The reconciliation CSV records both
this limitation and the upstream run hash for every row.

## Task manifests and composition

| Benchmark | Manifest | Manifest SHA-256 | Tasks in replay | API-quality tasks |
|---|---|---|---:|---:|
| RepoExec | `input/repoexec_python_string_utils_inline14.jsonl` | `d60142da52c157fdd53930e8f7d1e2df01c5c4a2132d8494fadd6389475e361b` | 14 per backend | 13 per backend |
| CoderEval | `input/codereval_neo4j_executable19.jsonl` | `813e84f0bd1b1502f4169354d26f1333c189e83b4f3c20c4148d4d865b719318` | 19 per backend | 13 per backend |
| ExecRepoBench | `input/execrepobench_testbacked.jsonl` | `f1bce561f04f54e19d4b4495ce0489a577a433aeb07483029fddcd30b77c1eaf` | 10 per backend | 0 |

The CoderEval API oracle is
`empirical_study/API/CoderEval4Python.json`, SHA-256
`8364fc274ef2e19de2ca792c16b6d3def6e9262d827132349194288b9e7c72c9`.

### Included task IDs

RepoExec API quality (13):

`test-apps/python-string-utils/{0,1,2,4,5,6,7,8,9,14,15,16,17}`

CoderEval API quality (13):

`neo4j/{id0,id1,id2,id3,id5,id8,id9,id11,id12,id14,id15,id16,id17}`

ExecRepoBench replay (10, detector/count diagnostics only):

`Chronyk/{1,2,3}`, `shortuuid/{0,2}`, and
`workdays/{0,1,2,3,4}`.

### Exclusions

- RepoExec: `test-apps/python-string-utils/13` has an empty ground-truth API
  set and is excluded from API-quality aggregation.
- CoderEval: `neo4j/{id4,id6,id7,id10,id13,id18}` have empty scoped
  ground-truth API sets and are excluded from API-quality aggregation.
- ExecRepoBench: all 10 selected tasks have no receiver-aware resolvable
  repository-specific API invocation. They are excluded from API-set
  precision/recall/F1/Exact, but remain in count/selectivity and API-item
  false-positive detection.

The selected task IDs are identical for both backends and all methods.

## Corrected evaluator semantics

### Aggregation

For each API-bearing task, the evaluator computes set precision, recall, F1,
and exact-set match. It then:

1. macro-averages tasks within each benchmark;
2. excludes benchmark cells with no API-bearing tasks;
3. takes an unweighted macro-average across RepoExec and CoderEval.

RepoExec and CoderEval each contribute 13 API-bearing tasks, so for this
campaign the two-benchmark macro Exact is numerically identical to pooling the
26 task indicators. It is not an API-item micro-average.

### API normalization and aliases

`normalize_api_name` removes call syntax and class prefixes, keeps the final
qualified-name component, removes non-alphanumeric/underscore characters, and
lowercases the result. Canonicalization collapses `Class.__init__` to `Class`,
drops private/dunder methods, and removes target-function self-retrieval.

Ground-truth call resolution accepts direct repository calls, exact
class-qualified calls, and compatible `self`/`cls` calls. It rejects suffix
matches from unrelated receivers. Relative imports are indexed. CoderEval
oracle labels are scoped to locally indexed repository symbols; built-ins and
external APIs are removed.

### Prediction stage

- Baseline RAG: final ordinary retrieval set without uncertainty filtering.
- OpenCoder-NoAPIRefine: after uncertainty filtering, before target-aware API
  refinement.
- OpenCoder: after uncertainty filtering and target-aware API refinement.

All table metrics use each record's `final_apis` field.

### Empty sets

At the primitive set-metric level, empty prediction and empty ground truth
receive precision/recall/F1/Exact of 1.0. Such tasks are then excluded from
API-quality aggregation by the `gt_count > 0` rule. They are retained only for
count/selectivity diagnostics. This prevents empty-ground-truth
ExecRepoBench tasks from inflating Table 4a.

## Exact accounting

### Aggregate values used by Table 4a

| Backend | Method | Numerator | Denominator | Unrounded % | Rounded % |
|---|---|---:|---:|---:|---:|
| GPT | Baseline RAG | 0 | 26 | 0.000000000 | 0.0 |
| GPT | OpenCoder-NoAPIRefine | 1 | 26 | 3.846153846 | 3.8 |
| GPT | OpenCoder | 15 | 26 | 57.692307692 | 57.7 |
| Gemini | Baseline RAG | 0 | 26 | 0.000000000 | 0.0 |
| Gemini | OpenCoder-NoAPIRefine | 0 | 26 | 0.000000000 | 0.0 |
| Gemini | OpenCoder | 13 | 26 | 50.000000000 | 50.0 |

### Benchmark-level numerators

| Backend | Method | RepoExec | CoderEval |
|---|---|---:|---:|
| GPT | Baseline RAG | 0/13 (0.0000%) | 0/13 (0.0000%) |
| GPT | OpenCoder-NoAPIRefine | 0/13 (0.0000%) | 1/13 (7.6923%) |
| GPT | OpenCoder | 13/13 (100.0000%) | 2/13 (15.3846%) |
| Gemini | Baseline RAG | 0/13 (0.0000%) | 0/13 (0.0000%) |
| Gemini | OpenCoder-NoAPIRefine | 0/13 (0.0000%) | 0/13 (0.0000%) |
| Gemini | OpenCoder | 12/13 (92.3077%) | 1/13 (7.6923%) |

## F1 anchor reproduction

The corrected two-benchmark macro calculations are:

- GPT Baseline:
  `(22.53968254 + 64.27350427) / 2 = 43.40659341 -> 43.4`.
- GPT OpenCoder-NoAPIRefine:
  `(24.38990497 + 65.81196581) / 2 = 45.10093539 -> 45.1`.
- GPT OpenCoder:
  `(29.65797023 + 100.00000000) / 2 = 64.82898512 -> 64.8`.
- Gemini Baseline:
  `(22.53968254 + 64.27350427) / 2 = 43.40659341 -> 43.4`.
- Gemini OpenCoder-NoAPIRefine:
  `(19.05888983 + 60.68376068) / 2 = 39.87132526 -> 39.9`.
- Gemini OpenCoder:
  `(25.91774239 + 92.30769231) / 2 = 59.11271735 -> 59.1`.

## Corrected uncertainty detector

The paper's detector is an API-item false-positive detector, not a task-level
incomplete-retrieval detector.

- Label: `1` when a selected OpenCoder API item is absent from the normalized
  ground-truth API set.
- Raw risk: inverse number of independent retrieval steps supporting the item.
- Split: deterministic SHA-256 parity of `benchmark|task_id`.
- Calibration: backend-specific logistic regression, `random_state=0`, fitted
  on calibration tasks.
- AUROC orientation: a larger score means a greater probability that the API
  item is incorrect.
- ECE target: the binary incorrect-API label.
- ECE: 10 equal-width bins on `[0,1]`, with the final bin including 1.0.

Held-out test accounting:

| Backend | Test items | False positives | AUROC | ECE |
|---|---:|---:|---:|---:|
| GPT | 111 | 92 | 0.735411899 | 0.029606974 |
| Gemini | 94 | 77 | 0.790679908 | 0.040892188 |

These round to the paper values `0.735/0.030` and `0.791/0.041`.
Calibration contains 99 GPT items and 98 Gemini items.

## Older table provenance

The older values survive in:

- Immutable pasted artifact:
  `<PRIVATE_ATTACHMENT>/pasted-text.txt`
  (SHA-256
  `7eb04f63cd1ce534174b253fb3af200d7a2c222616bfe5ab68c366df1d59d0fc`)
- `results/aaai_testbacked_overleaf_results.tex`
- `results/camera_ready_rq1_rq4_overleaf.tex`
- `results/AAAI_TESTBACKED_RESULTS_STATUS.md`

`AAAI_TESTBACKED_RESULTS_STATUS.md` is explicitly marked “Historical snapshot.
Do not use” and points to `results/rq4/RQ4_STATUS.md` and `latextRQ4.txt` as its
replacement.

No immutable old `raw_api_predictions.jsonl`, per-task API metrics, exact
evaluator revision, command line, run ID, or at-run configuration hash was
retained for that table. The current files under `results/rq4/` are the
corrected campaign and do not reproduce the old table. Therefore details of
old API normalization, alias handling, qualified-name handling, and excluded
task IDs cannot be asserted beyond what the surviving table/status text
records.

### How 16.7, 50.0, and 47.6 were produced

The older count table shows benchmark Exact cells that algebraically reproduce
the aggregate Exact column as an unweighted three-benchmark macro:

- `16.7 = (RepoExec 0.0 + CoderEval 0.0 + ExecRepoBench 50.0) / 3
  = 16.666666667`.
- `50.0 = (GPT OpenCoder RepoExec 100.0 + CoderEval 0.0 +
  ExecRepoBench 50.0) / 3`.
- `47.6 = (Gemini OpenCoder RepoExec 92.857142857 + CoderEval 0.0 +
  ExecRepoBench 50.0) / 3 = 47.619047619`.

Thus these are benchmark-macro values, not pooled task percentages. In
particular, they allow an empty-ground-truth ExecRepoBench cell to contribute
50%. The original old task-level match indicators are no longer present, so
their task-level numerators must not be reconstructed or reported as verified.

### Old detector

The older table evaluates task-level incomplete API retrieval
(`recall < 1`) using aggregate generation uncertainty. It reports
`0.120/0.463` for GPT and `0.179/0.382` for Gemini. The corrected table changes
both the statistical unit and target: it evaluates false-positive API items
with API-specific consensus risk. The two detector tables therefore do not
measure the same event and must not be compared as reruns of one metric.

## Pipeline comparison

| Property | Older table | Corrected paper table |
|---|---|---|
| Classification | Older RQ4 campaign | Final source-of-truth RQ4 evaluation |
| Quality benchmarks | Three-benchmark macro in surviving table | RepoExec + CoderEval only |
| Quality denominator | Not preserved at task level | 13 + 13 = 26 per backend |
| ExecRepoBench | Contributed to macro Exact | Excluded from API quality |
| Aggregation | Unweighted benchmark macro | Task macro within benchmark, then two-benchmark macro |
| Empty GT | Contributed to old aggregate | Excluded from quality |
| Normalization/resolver | Exact revision unavailable | Corrected receiver-aware resolver and public-API canonicalization |
| NoAPIRefine stage | Not verifiable from raw records | Post-filter, pre-target-aware refinement |
| AUROC unit/label | Task-level incomplete retrieval | API-item false positive |
| AUROC orientation | Historical implementation not retained | Higher risk = more likely incorrect API |
| ECE target | Task incomplete-retrieval event | Incorrect API item |
| ECE bins | Historical implementation not retained | 10 equal-width bins |
| Models | GPT/Gemini labels in table | `gpt-4o-mini` / `gemini-2.5-flash` |
| Run/config provenance | Not retained | Source run hashes retained; at-run YAML hash absent |

The discrepancy is caused by mixing two different evaluation definitions:
the old table included a third, empty-ground-truth benchmark in an unweighted
macro and used a task-level uncertainty target, while the corrected table
uses only API-bearing RepoExec/CoderEval tasks, corrected API resolution and
target-aware normalization, and a held-out API-item false-positive detector.

## Numerical-anchor trace

| Anchors | Raw input | Aggregation/output |
|---|---|---|
| 43.4, 45.1, 64.8, 57.7, 39.9, 59.1, 50.0 | `raw_api_predictions.jsonl` | `_record_to_metrics` -> `summarize_metrics` -> `aggregate_quality_for_table` -> `table_api_quality.tex` |
| 0.735, 0.030, 0.791, 0.041 | `raw_api_predictions.jsonl` -> `api_item_uncertainty.csv` | `_api_item_uncertainty_rows` -> calibration -> `api_item_uncertainty_summary.csv` -> `table_api_item_uncertainty.tex` |
| 37.8, 51.0, 48.6, 16.7, 50.0, 47.6, 0.120, 0.463, 0.179, 0.382 | Old raw input unavailable | Historical attachment/LaTeX/status snapshots only |

## Verification

- `scripts/audit_rq4_reconciliation.py`: passed all anchor assertions.
- Recomputed/saved metric comparison: 516/516 rows matched.
- `tests/test_rq4_metrics.py`: 11 passed.
- Ruff audit script check: passed.
- Task-level audit output:
  `results/rq4_reconciliation_by_task.csv`.
- No paid experiment was rerun.
- No manuscript file was modified.

## Required conclusion

- **Authoritative campaign:** corrected cached-record RQ4 replay in
  `results/rq4/`, raw campaign SHA-256
  `9130a107227510ff7cc6d089012a27cd113f5284b1ee194c7af73d07a2c906d5`.
- **Authoritative evaluator:** `experiments/run_rq4.py`, SHA-256
  `60ac2f82bd94fbe0a6e7f1bb836011011e219ead0aa615f1a3d2205a0b78e332`.
- **Authoritative task denominator:** 26 API-bearing tasks per backend:
  13 RepoExec + 13 CoderEval. ExecRepoBench is excluded from API-quality
  aggregation.
- **Reason for the discrepancy:** the older campaign used a three-benchmark
  macro that allowed empty-ground-truth ExecRepoBench cells to contribute and
  used a different task-level uncertainty target; the corrected campaign uses
  receiver-aware API labels, two API-bearing benchmarks, and API-item
  false-positive calibration.
- **Verified GPT NoAPIRefine Exact:** **1/26 =
  3.846153846% -> 3.8%**.
- **Verified Gemini NoAPIRefine Exact:** **0/26 =
  0.000000000% -> 0.0%**.
- **Should Table 4 remain unchanged?** **No.** Keep all current paper anchors
  unchanged, but replace the two NoAPIRefine Exact dashes with `3.8` (GPT) and
  `0.0` (Gemini) in a later author-approved manuscript edit. This audit did
  not modify the manuscript.
