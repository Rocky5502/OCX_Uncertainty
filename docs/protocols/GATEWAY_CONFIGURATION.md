# Gateway Configuration

The OpenCoderX campaign uses one OpenAI-compatible ZhiZengZeng endpoint for all
four frozen model families:

```text
base URL environment variable: OPENCODER_LLM_BASE_URL
expected URL: https://api.zhizengzeng.com/v1
credential environment variable: OPENCODER_LLM_API_KEY
```

No direct `ANTHROPIC_API_KEY`, `DASHSCOPE_API_KEY`, `GEMINI_API_KEY`, or
`OPENAI_API_KEY` is required by the TOSEM campaign. The requested model IDs are
`gpt-4o-mini`, `gemini-2.5-flash`, `claude-sonnet-5`, and
`qwen3-coder-plus`. The authenticated `/models` catalog exposed all four IDs on
2026-08-09.

This is a gateway-mediated evaluation. Provenance must therefore record both
the requested upstream family and the gateway. It must not describe Claude or
Qwen requests as direct Anthropic or Alibaba Cloud calls. Gateway behavior can
differ from upstream behavior, so unsupported sampling parameters and native
probability fields remain disabled until verified by a bounded response audit.

The same endpoint does not imply the same model price. On 2026-08-10, the
gateway help application linked its official model-price document at
`https://doc.zhizengzeng.com/doc-3979947`. The documented rows list GPT-4o-mini
at USD 0.15/0.60, Gemini-2.5-Flash at USD 0.30/2.50, Claude-Sonnet-5 at USD
2.00/10.00, and Qwen3-Coder-Plus at CNY 4.00/16.00 per million input/output
tokens. The same page states an 8% surcharge for foreign models and at-or-below
official pricing for domestic models. The cost gate therefore applies the 8%
surcharge to GPT, Gemini, and Claude, and uses Qwen's listed official CNY price
as a conservative upper bound without inventing a currency conversion. Prices
must be re-audited before the main campaign. No credential value is written to
repository artifacts.
