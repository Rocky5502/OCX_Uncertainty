# ExecRepoBench-120 Confirmatory Results

Integrity status: **PASSED**.

Candidate Pass@k uses only the five original samples. Selected-output correctness is a separate endpoint that includes each method's declared verified-selection and repair behavior.

## Effectiveness

| Model | Method | Pass@1 | Pass@3 | Pass@5 | Selected |
|---|---|---:|---:|---:|---:|
| gpt-4o-mini | Direct Generation | 15.7 | 18.8 | 20.0 | 16.7 |
| gpt-4o-mini | Standard RAG | 25.8 | 29.6 | 30.8 | 25.8 |
| gpt-4o-mini | RAG + Verify/Repair | 24.8 | 29.3 | 30.8 | 32.5 |
| gpt-4o-mini | OpenCoderX | 23.0 | 25.6 | 26.7 | 30.8 |
| gemini-2.5-flash | Direct Generation | 22.8 | 29.1 | 30.8 | 22.5 |
| gemini-2.5-flash | Standard RAG | 24.0 | 33.5 | 39.2 | 25.8 |
| gemini-2.5-flash | RAG + Verify/Repair | 21.8 | 29.0 | 32.5 | 35.0 |
| gemini-2.5-flash | OpenCoderX | 19.3 | 27.3 | 30.0 | 36.7 |
| claude-sonnet-5 | Direct Generation | 51.7 | 58.8 | 61.7 | 51.7 |
| claude-sonnet-5 | Standard RAG | 55.8 | 62.7 | 65.0 | 58.3 |
| claude-sonnet-5 | RAG + Verify/Repair | 55.0 | 63.3 | 66.7 | 73.3 |
| claude-sonnet-5 | OpenCoderX | 55.8 | 62.4 | 65.0 | 74.2 |
| qwen3-coder-plus | Direct Generation | 47.3 | 52.3 | 54.2 | 48.3 |
| qwen3-coder-plus | Standard RAG | 53.3 | 55.7 | 55.8 | 53.3 |
| qwen3-coder-plus | RAG + Verify/Repair | 52.0 | 55.3 | 55.8 | 60.8 |
| qwen3-coder-plus | OpenCoderX | 49.8 | 53.8 | 55.0 | 60.0 |

## Matched Control

- gpt-4o-mini: OpenCoderX -1.7 percentage points; 95% CI [-5.0, 1.7], W/L/T=1/3/116, exact McNemar p=0.625, Holm p=1.000.
- gemini-2.5-flash: OpenCoderX +1.7 percentage points; 95% CI [-4.2, 7.5], W/L/T=8/6/106, exact McNemar p=0.791, Holm p=1.000.
- claude-sonnet-5: OpenCoderX +0.8 percentage points; 95% CI [-4.2, 5.8], W/L/T=5/4/111, exact McNemar p=1.000, Holm p=1.000.
- qwen3-coder-plus: OpenCoderX -0.8 percentage points; 95% CI [-4.2, 2.5], W/L/T=2/3/115, exact McNemar p=1.000, Holm p=1.000.

## Interpretation

The matched comparison isolates uncertainty-aware decomposition, filtering, fusion, and generation from ordinary verification/repair. Results are backend dependent: OpenCoderX does not universally outperform the matched control, and significance is claimed only where the adjusted paired test supports it.

The audit covered 19,589 successful provider responses and 1,011 repair prompts. All length-limited candidates (1,520) remain failures under the frozen integrity policy rather than being silently dropped.
