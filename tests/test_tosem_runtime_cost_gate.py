from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_rq3_cost_test", ROOT / "experiments/run_rq3.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Client:
    def usage_snapshot(self):
        return {"prompt_tokens": 1_000_000, "completion_tokens": 0, "requests": 1}


class _Pipe:
    llm = _Client()


def test_runtime_cost_gate_uses_model_specific_currency(tmp_path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(
        "cost_controls:\n"
        "  per_run_limits: {CNY: 3.0}\n"
        "  gateway_pricing:\n"
        "    q:\n"
        "      currency: CNY\n"
        "      input_per_million: 4.0\n"
        "      output_per_million: 16.0\n",
        encoding="utf-8",
    )
    gate = MODULE._runtime_cost_gate(str(config), "q", [_Pipe()])
    assert gate["amount"] == 4.0
    assert gate["currency"] == "CNY"
    assert gate["allowed"] is False
