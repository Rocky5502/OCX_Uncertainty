# Expanded RepoExec-inline Results Memo

The pre-output audit examined 25 remaining tasks, accepted 18, and excluded 7 before method outputs were inspected. The expanded set therefore contains 32 matched tasks.

## Complete Expanded Subset

### GPT

- Baseline RAG: Pass@1/3/5 = 58.75/68.44/71.88; selected correctness = 56.25.
- RAG + Verify/Repair: Pass@1/3/5 = 61.88/74.38/78.13; selected correctness = 78.13.
- OpenCoder: Pass@1/3/5 = 61.88/72.50/78.13; selected correctness = 78.13.

### Gemini

- Baseline RAG: Pass@1/3/5 = 66.25/70.31/71.88; selected correctness = 68.75.
- RAG + Verify/Repair: Pass@1/3/5 = 70.00/78.75/84.38; selected correctness = 84.38.
- OpenCoder: Pass@1/3/5 = 65.00/73.75/78.13; selected correctness = 78.13.

## Statistical Interpretation

- GPT, OpenCoder vs Baseline RAG: selected-output difference 21.88 points (95% CI [6.25, 40.63]), W/L/T 8/1/23, McNemar p=0.039, N=32.
- GPT, OpenCoder vs RAG + Verify/Repair: selected-output difference 0.00 points (95% CI [-9.38, 9.38]), W/L/T 1/1/30, McNemar p=1.000, N=32.
- Gemini, OpenCoder vs Baseline RAG: selected-output difference 9.38 points (95% CI [-6.25, 25.00]), W/L/T 5/2/25, McNemar p=0.453, N=32.
- Gemini, OpenCoder vs RAG + Verify/Repair: selected-output difference -6.25 points (95% CI [-15.63, 0.00]), W/L/T 0/2/30, McNemar p=0.500, N=32.

## Recommended Paper Statement

On the expanded 32-task subset, OpenCoder improves selected-output correctness over plain Baseline RAG by 21.88 percentage points with GPT (78.13% vs. 56.25%; nominal two-sided exact McNemar p=0.039, unadjusted for multiple comparisons), but ties the verification/repair control. With Gemini, neither selected-output nor Pass@k differences are statistically supported. Across both backends, all paired Pass@k confidence intervals include zero. These results support the value of validation and repair while providing only backend-specific evidence for an additional OpenCoder advantage.

The expansion satisfies the five-task decision rule and may replace the 14-task RepoExec-inline analysis. Claims must remain benchmark- and backend-specific: confidence intervals and exact tests determine whether any observed difference supports a significance statement, and the results do not establish universal OpenCoder superiority.
