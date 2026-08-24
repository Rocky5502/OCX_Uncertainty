# Gateway-Mediated Automated Reviewers

This exploratory robustness study evaluates whether frozen uncertainty displays
change the behavior of heterogeneous model-based code reviewers. It is not a
human study and does not evaluate consumer chat products.

## Design

- 12 gateway-mediated model configurations
- 12 frozen ExecRepoBench tasks per configuration
- 144 planned episodes, balanced across three review conditions
- no tools, browsing, hidden tests, or reference implementation
- one complete target function per response
- the same local executable evaluator for all responses

The configured families are GPT, Claude, Gemini, DeepSeek, Qwen, Kimi, Grok,
Llama, GLM, MiniMax, Doubao, and ERNIE. Predeclared technical replacements are
documented in `results/agent_gateway_v1/model_replacement_audit.json` and the
Gemini protocol amendment.

## Public Prompt Contract

The task-independent response contract is in
`human_study/gateway_agent_v1/PROMPT_TEMPLATE.md`. Task-bearing prompt payloads
are not redistributed because they include repository source. Their hashes and
balanced assignments remain in the frozen protocol.

## Results

The public task records retain conditions, model IDs, confidence, correctness,
repair outcomes, tokens, latency, failure status, and cryptographic hashes.
Generated code, raw provider text, response identifiers, evaluator logs, and
local paths are redacted.

Of 144 planned episodes, 134 yielded executable outputs and 10 were missing.
There were zero evaluator infrastructure errors and zero served-model
mismatches. Missing outputs are excluded from evaluable correctness and remain
visible in planned-denominator end-to-end sensitivity.
