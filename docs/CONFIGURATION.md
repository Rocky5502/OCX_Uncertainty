# Configuration

## Local Settings

`configs/settings.example.json` is a non-secret convenience template for local
paths and environment-variable names. Frozen scientific parameters live in
`configs/tosem/campaign.yaml` and `configs/tosem/models/*.yaml`.

## OpenAI-Compatible Gateway

```dotenv
OPENCODER_LLM_BASE_URL=https://your-compatible-endpoint.example/v1
OPENCODER_LLM_API_KEY=replace_locally
```

The frozen campaign used the compatibility variable
`ZHIZENGZENG_API_KEY`. The client accepts either name; never commit either
value. A gateway may expose a model name without serving the expected model, so
the preflight and response-metadata audits must pass before a campaign run.

## Frozen Generation Parameters

- candidate count: 5
- temperature: 0.7, except provider default for Claude
- maximum output: 2,048 tokens
- maximum repair rounds: 2
- API/context/similar-code retrieval budgets: 8 each
- fused evidence budget: 10
- whole-task anchor: enabled
- maximum source fraction: 0.5

The exact protocol and hashes are recorded in
`results/tosem/protocol_freeze.json`. Do not edit frozen parameters and then
compare new outputs with the released campaign as if they were matched.

## Cost Controls

Provider prices and model availability change. Re-audit prices before every
paid run. Use an explicit per-run and campaign cap, preserve failed requests,
and never infer cost from current list prices after the fact.
