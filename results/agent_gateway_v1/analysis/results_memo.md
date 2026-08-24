# Gateway AI-Agent Replication Results

## Status

- Study mode: `AGENT_EXPLORATORY`.
- Planned records: 144 (12 gateway model configurations x 12 tasks).
- Evaluable executable outputs: 134.
- Missing model outputs: 10.
- Executable-evaluator infrastructure errors: 0.
- Served-model mismatches: 0.
- This is not a human study and not an evaluation of consumer chat products.

## Main descriptive result

| Condition | Evaluable | Correct/evaluable | E2E success |
|---|---:|---:|---:|
| Generic review | 44/48 | 17/44 (38.6%) | 35.4% |
| Uncertainty display | 45/48 | 21/45 (46.7%) | 43.8% |
| Targeted guidance | 45/48 | 16/45 (35.6%) | 33.3% |


Targeted guidance minus generic review was
-2.1 percentage points for
end-to-end successful output, with a model-clustered 95% bootstrap interval of
[-12.5,
+8.3]. This interval is a
descriptive sensitivity analysis over the configured systems; it is not a
population-level confidence interval over all LLMs.

## Technical amendments and missingness

- `claude-sonnet-5` produced zero visible text in its first fixed-budget task;
  that unevaluable record is preserved and the predeclared
  `claude-sonnet-4-6` alternate was used for the complete Claude slice.
- `gemini-2.5-flash` produced 9/12 unevaluable records. Under the documented
  majority-missing technical rule, the full slice was archived and replaced
  with the predeclared `gemini-2.5-flash-lite` alternate.
- Final missing records were not counted as correct or incorrect. The E2E
  sensitivity column reports successful outputs over all planned episodes.
- DeepSeek and Kimi were not replaced because they did not cross the
  majority-missing threshold.

## Interpretation

The agent replication should be used as a robustness analysis of how frozen
uncertainty displays affect heterogeneous gateway-hosted reviewers. It cannot
substitute for genuine participant data, establish developer usability, or be
described as direct access to ChatGPT, Claude.ai, Gemini, Manus, Perplexity, or
other consumer products.
