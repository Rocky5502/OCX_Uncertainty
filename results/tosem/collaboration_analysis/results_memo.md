# TOSEM Collaboration and Generalizability Analysis

Integrity status: **PASSED**.

## Selective autonomy

The fixed policy uses frozen review/abstain thresholds (0.35/0.70). The primary uncertainty-only policy excludes executable outcomes; a separate lifecycle-gated policy is reported as an operational deployment analysis.

- gpt-4o-mini (uncertainty only): autonomous coverage 30.0%, autonomous accuracy 41.7%, failure capture 74.7%, and AUROC 0.632.
- gemini-2.5-flash (uncertainty only): autonomous coverage 25.8%, autonomous accuracy 45.2%, failure capture 77.6%, and AUROC 0.602.
- claude-sonnet-5 (uncertainty only): autonomous coverage 33.3%, autonomous accuracy 90.0%, failure capture 87.1%, and AUROC 0.692.
- qwen3-coder-plus (uncertainty only): autonomous coverage 37.5%, autonomous accuracy 64.4%, failure capture 66.7%, and AUROC 0.596.

When executable validation and exhausted repair enforce the delivery gate:

- gpt-4o-mini: autonomous coverage 26.7%, autonomous accuracy 100.0%, and failure capture 100.0%.
- gemini-2.5-flash: autonomous coverage 30.0%, autonomous accuracy 100.0%, and failure capture 100.0%.
- claude-sonnet-5: autonomous coverage 63.3%, autonomous accuracy 100.0%, and failure capture 100.0%.
- qwen3-coder-plus: autonomous coverage 53.3%, autonomous accuracy 100.0%, and failure capture 100.0%.

## Review allocation simulation

Reviewer corrections are offline simulations, not observed developer behavior. At a 20% nominal budget and 75% reviewer success:

- gpt-4o-mini: simulated team success 41.3%, failure capture 24.1%, unnecessary review 16.7%.
- gemini-2.5-flash: simulated team success 44.7%, failure capture 19.7%, unnecessary review 37.5%.
- claude-sonnet-5: simulated team success 77.5%, failure capture 22.6%, unnecessary review 70.8%.
- qwen3-coder-plus: simulated team success 67.0%, failure capture 27.1%, unnecessary review 45.8%.

## Observed interventions

- gpt-4o-mini: OpenCoderX versus matched RAG + Verify/Repair -1.7 points (95% CI [-5.0, 1.7]).
- gemini-2.5-flash: OpenCoderX versus matched RAG + Verify/Repair +1.7 points (95% CI [-4.2, 7.5]).
- claude-sonnet-5: OpenCoderX versus matched RAG + Verify/Repair +0.8 points (95% CI [-4.2, 5.8]).
- qwen3-coder-plus: OpenCoderX versus matched RAG + Verify/Repair -0.8 points (95% CI [-4.2, 2.5]).

API-, context-, and similar-code correction effects are explicitly NOT_RUN because no matched correction records exist.

## Generalizability

ExecRepoBench reports executable correctness. CrossCodeEval reports native exact match, edit similarity, and identifier F1 only; these endpoints are not pooled.

Cross-file context increased exact match in 13/16 model-language cells; each cell contains 25 matched tasks.

## Guardrails

- No model calls were made by this analysis.
- The test-failure policy uses post-selection, pre-repair validation, never final correctness.
- The oracle policy is an explicit upper bound and uses final outcomes only for ranking.
- Repository-level effects are descriptive because repositories contain only one to four tasks.
- No claim of universal OpenCoderX superiority is supported by the matched results.
