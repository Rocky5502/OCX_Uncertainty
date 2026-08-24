# Corrected RQ1/RQ2 Experiment Status

## Integrity

- GPT RQ1: 80/80 valid task-condition rows.
- Gemini RQ1: 80/80 valid task-condition rows.
- GPT RQ2: 60/60 valid task-condition rows.
- Gemini RQ2: 60/60 valid task-condition rows.
- All RQ2 uncertainty-aware conditions reuse identical Phase-II plans within each task/backend: yes.
- Correctness mode: reconstructed repository tests on 10 audited ExecRepoBench tasks.
- Raw candidates: 3 per task-condition. Pass@5 is intentionally not reported.
- All numbers in the paper artifacts are generated from the JSON result files; no assumed cells are used.

## Main Results

- GPT effective Pass@1: 10.0% -> 80.0%.
- Gemini effective Pass@1: 0.0% -> 60.0%.
- GPT sample Pass@1/3: 13.3/40.0 -> 36.7/80.0.
- Gemini sample Pass@1/3: 13.3/40.0 -> 26.7/40.0.
- Calibration is backend-dependent: GPT ECE 0.162 -> 0.222; Gemini ECE 0.191 -> 0.153.

## Camera-Ready Files

- `overleaf_rq1_rq2.tex`
- `table_source_factorial_effects.tex`
- `table_uncertainty_components.tex`
- `fig_source_factorial_effects.pdf` / `.png`
- `fig_source_interactions.pdf` / `.png`
- `fig_component_effectiveness.pdf` / `.png`

## Scope

These are complete real-API results for the local 10-task execution-backed subset and one three-candidate run per condition. Confidence intervals use tasks as the inference unit. The results support strong mitigation claims but not universal calibration or broad benchmark-level generalization.
