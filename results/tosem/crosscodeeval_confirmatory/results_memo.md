# CrossCodeEval-100 Confirmatory Results

Integrity: **PASSED**.

All values are native completion metrics. They are not executable functional correctness.

| Model | Method | EM | Edit | Identifier F1 |
|---|---|---:|---:|---:|
| gpt-4o-mini | Direct Generation | 6.0 | 58.4 | 43.9 |
| gpt-4o-mini | Cross-file Context RAG | 15.0 | 65.9 | 54.7 |
| gemini-2.5-flash | Direct Generation | 5.0 | 48.5 | 37.4 |
| gemini-2.5-flash | Cross-file Context RAG | 10.0 | 50.9 | 42.9 |
| claude-sonnet-5 | Direct Generation | 16.0 | 61.6 | 52.8 |
| claude-sonnet-5 | Cross-file Context RAG | 27.0 | 64.8 | 59.4 |
| qwen3-coder-plus | Direct Generation | 7.0 | 58.4 | 44.7 |
| qwen3-coder-plus | Cross-file Context RAG | 17.0 | 65.9 | 57.1 |

## Paired Context Effects

- gpt-4o-mini: exact-match difference +9.0 points, 95% CI [4.0, 15.0], W/L/T=9/0/91, McNemar p=0.004, Holm p=0.016.
- gemini-2.5-flash: exact-match difference +5.0 points, 95% CI [-1.0, 11.0], W/L/T=8/3/89, McNemar p=0.227, Holm p=0.227.
- claude-sonnet-5: exact-match difference +11.0 points, 95% CI [3.0, 19.0], W/L/T=14/3/83, McNemar p=0.013, Holm p=0.025.
- qwen3-coder-plus: exact-match difference +10.0 points, 95% CI [4.0, 17.0], W/L/T=11/1/88, McNemar p=0.006, Holm p=0.019.

Cross-file context effects are reported as backend- and language-dependent. A non-significant difference is not interpreted as equivalence.
