# RQ1-RQ2 Consolidated Results

Integrity: **PASSED**.

## RQ1 Evidence interactions

The complete 2^3 factorial remains the audited 10-task GPT/Gemini campaign. It is not relabeled as a four-family experiment.

- GPT context:similar_code: uncertainty interaction -0.044 (Holm p=0.085); Pass@1 interaction +33.3 points (Holm p=0.032).
- Gemini api:similar_code: uncertainty interaction -0.059 (Holm p=0.032); Pass@1 interaction +10.0 points (Holm p=1.000).

CrossCodeEval-100 independently shows positive context exact-match effects for GPT, Claude, and Qwen after Holm correction; Gemini is inconclusive. These are native completion metrics, not executable correctness.

## RQ2 Uncertainty generalization

- gpt-4o-mini: aggregate source+generation risk AUROC 0.632 (95% CI [0.533, 0.728]), AUPRC 0.828.
- gemini-2.5-flash: aggregate source+generation risk AUROC 0.602 (95% CI [0.497, 0.701]), AUPRC 0.763.
- claude-sonnet-5: aggregate source+generation risk AUROC 0.692 (95% CI [0.583, 0.795]), AUPRC 0.489.
- qwen3-coder-plus: aggregate source+generation risk AUROC 0.596 (95% CI [0.486, 0.702]), AUPRC 0.549.

Uncertainty is directionally informative but backend-dependent. Calibration and discrimination must be reported separately; no universal threshold claim is supported.
