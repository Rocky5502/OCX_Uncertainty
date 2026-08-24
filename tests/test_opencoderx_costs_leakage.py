from opencoderx.costs import estimate_cost
from opencoderx.leakage import detect_target_leakage


def test_unpriced_cost_estimate_blocks_large_run():
    result = estimate_cost(
        provider="g", model="m", dataset="d", tasks=120, methods=4, samples=5,
        input_tokens_per_call=1000, output_tokens_per_call=100,
        input_usd_per_million=None, output_usd_per_million=None,
    )
    assert result.estimated_calls == 2400
    assert result.estimated_cost_usd is None
    assert result.pricing_status == "UNPRICED_BLOCK_LARGE_RUN"


def test_non_usd_pricing_preserves_currency_without_fake_conversion():
    result = estimate_cost(
        provider="g", model="q", dataset="d", tasks=3, methods=3, samples=5,
        input_tokens_per_call=8000, output_tokens_per_call=1000,
        input_usd_per_million=4.0, output_usd_per_million=16.0,
        pricing_currency="CNY",
    )
    assert result.estimated_cost == 2.16
    assert result.pricing_currency == "CNY"
    assert result.estimated_cost_usd is None


def test_leakage_detector_normalizes_python():
    findings = detect_target_leakage(
        "def f(x):\n    return x + 1",
        [{"id": "safe", "content": "def g(x): return x"},
         {"id": "leak", "content": "def f(x): return x+1"}],
    )
    assert [finding.suspected_leak for finding in findings] == [False, True]
