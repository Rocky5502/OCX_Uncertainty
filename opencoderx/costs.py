"""Cost-gate calculations for experiment plans."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentCostEstimate:
    provider: str
    model: str
    dataset: str
    tasks: int
    methods: int
    samples: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_calls: int
    estimated_cost: float | None
    pricing_currency: str
    estimated_cost_usd: float | None
    pricing_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_cost(
    *,
    provider: str,
    model: str,
    dataset: str,
    tasks: int,
    methods: int,
    samples: int,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
    pricing_currency: str = "USD",
) -> ExperimentCostEstimate:
    calls = tasks * methods * samples
    input_tokens = calls * input_tokens_per_call
    output_tokens = calls * output_tokens_per_call
    if input_usd_per_million is None or output_usd_per_million is None:
        cost = None
        status = "UNPRICED_BLOCK_LARGE_RUN"
    else:
        cost = (
            input_tokens * input_usd_per_million
            + output_tokens * output_usd_per_million
        ) / 1_000_000
        status = "PRICED"
    currency = pricing_currency.upper()
    return ExperimentCostEstimate(
        provider=provider,
        model=model,
        dataset=dataset,
        tasks=tasks,
        methods=methods,
        samples=samples,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_calls=calls,
        estimated_cost=cost,
        pricing_currency=currency,
        estimated_cost_usd=cost if currency == "USD" else None,
        pricing_status=status,
    )
