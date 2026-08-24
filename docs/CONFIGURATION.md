# Configuration

## Local Settings

`configs/settings.example.json` is a non-secret convenience template for local
paths and environment-variable names. Frozen scientific parameters live in
`configs/tosem/campaign.yaml` and `configs/tosem/models/*.yaml`.

## LLM Backends

The frozen campaign evaluates four requested model IDs through one
OpenAI-compatible chat-completions transport:

| Family | Model ID | Official documentation |
|---|---|---|
| GPT | `gpt-4o-mini` | [OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-mini) |
| Gemini | `gemini-2.5-flash` | [Google](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash) |
| Claude | `claude-sonnet-5` | [Anthropic](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5) |
| Qwen | `qwen3-coder-plus` | [Alibaba Cloud](https://help.aliyun.com/en/model-studio/qwen3-coder-plus) |

Configure a compatible endpoint with neutral local variables:

```dotenv
OPENCODER_LLM_BASE_URL=https://your-compatible-endpoint.example/v1
OPENCODER_LLM_API_KEY=replace_locally
```

Never commit the populated `.env` file. A gateway may expose a model name
without serving the expected model, so the preflight and response-metadata
audits must pass before a campaign run. Model-family labels describe requested
models, not direct vendor connections.

Select the model through `OPENCODER_LLM_MODEL` or a file under
`configs/tosem/models/`, then run:

```bash
python scripts/preflight_api.py \
  --backend zhizengzeng \
  --model "$OPENCODER_LLM_MODEL"
```

The client also supports direct OpenAI and Gemini endpoints. Use backend
`openai` with `OPENAI_API_KEY`, or backend `gemini` with `GEMINI_API_KEY`.
Claude and Qwen in the released campaign require an endpoint that implements
the OpenAI-compatible chat-completions interface and exposes the frozen IDs.

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
