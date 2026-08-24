"""Plot combined RQ1 source-ablation deltas across backends."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _source_value(summary: dict, source_key: str) -> float:
    for row in summary.get("rq1_table", []):
        if row.get("source_key") == source_key:
            return float(row["_delta_u_value"])
    raise KeyError(f"Missing RQ1 source row for source_key={source_key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-summary", required=True)
    ap.add_argument("--gemini-summary", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    summaries = {
        "GPT": _load_summary(args.gpt_summary),
        "Gemini": _load_summary(args.gemini_summary),
    }
    sources = [
        ("api", "API"),
        ("context", "Context"),
        ("similar_code", "Similar code"),
    ]
    backends = list(summaries)
    x = np.arange(len(sources))
    width = 0.34
    colors = {"GPT": "#2563eb", "Gemini": "#16a34a"}

    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    for i, backend in enumerate(backends):
        values = [_source_value(summaries[backend], key) for key, _ in sources]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=backend, color=colors[backend])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("RQ1 Source-Wise Change in Aggregate Uncertainty", fontsize=11)
    ax.set_ylabel(r"$\Delta U$ present - absent")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in sources])
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_rq1_backend_delta_uncertainty.{ext}", dpi=300)
    plt.close(fig)
    print(f"Wrote {os.fspath(out_dir / 'fig_rq1_backend_delta_uncertainty.pdf')}")


if __name__ == "__main__":
    main()
