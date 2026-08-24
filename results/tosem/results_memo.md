# OpenCoderX Final Results Memo

## Answer First

The expanded campaign supports uncertainty as a useful, model-dependent control
signal, but it does not support a claim that OpenCoderX universally outperforms
a retrieval-matched verification-and-repair control. The strongest reproducible
evidence is: repository evidence helps across languages; verification and repair
account for much of the executable gain; uncertainty can concentrate failures
for review; and the value of uncertainty-aware evidence control depends on the
LLM family and evidence interaction.

## Executable Effectiveness

Selected-output correctness on ExecRepoBench-120 was:

| Model | Direct | RAG | RAG + Verify/Repair | OpenCoderX |
|---|---:|---:|---:|---:|
| GPT-4o-mini | 16.7% | 25.8% | 32.5% | 30.8% |
| Gemini 2.5 Flash | 22.5% | 25.8% | 35.0% | 36.7% |
| Claude Sonnet 5 | 51.7% | 58.3% | 73.3% | 74.2% |
| Qwen3-Coder-Plus | 48.3% | 53.3% | 60.8% | 60.0% |

Against RAG + Verify/Repair, OpenCoderX changes selected correctness by -1.7
points for GPT (95% CI [-5.0, 1.7]), +1.7 for Gemini ([-4.2, 7.5]), +0.8 for
Claude ([-4.2, 5.8]), and -0.8 for Qwen ([-4.2, 2.5]). None supports a universal
superiority claim.

## Uncertainty and Collaboration

Aggregate uncertainty AUROC for failure detection is 0.632 for GPT, 0.602 for
Gemini, 0.692 for Claude, and 0.596 for Qwen. Under frozen uncertainty-only
thresholds of 0.35 and 0.70, autonomous coverage ranges from 25.8% to 37.5%,
while failure capture among deferred tasks ranges from 66.7% to 87.1%. These
figures describe selective risk concentration, not guaranteed calibration.

Reviewer-budget results are simulations, not observed human performance.
Observed intervention analysis is restricted to matched method transitions;
API-, context-, and similar-code correction interventions remain `NOT_RUN`.

## Cross-Language Transfer

On CrossCodeEval-100, context RAG increases selected exact match from 6% to 15%
for GPT, 5% to 10% for Gemini, 16% to 27% for Claude, and 7% to 17% for Qwen.
The aggregate Holm-adjusted exact-match effect is supported for GPT, Claude, and
Qwen, but not Gemini. These are native completion metrics; CrossCodeEval was not
executed as a functional benchmark.

## Audit Boundary

All 2,720 task-method-model cells are retained, all frozen hashes reproduce,
and no result placeholder remains in the compiled manuscript. Multi-SWE-bench
is excluded from quantitative claims because the official Docker evaluator was
unavailable. Campaign spend was USD 72.1299 and CNY 56.5373.

Authoritative machine-readable records are `final_integrity.json` and
`final_artifact_manifest.json` in this directory.
