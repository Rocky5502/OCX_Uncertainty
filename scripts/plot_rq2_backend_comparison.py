"""Plot combined RQ2 backend results for the paper."""
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


def _row(summary: dict, method_key: str) -> dict:
    for item in summary.get("rq2_table", []):
        if item.get("method_key") == method_key:
            return item
    raise KeyError(f"Missing RQ2 row for method_key={method_key}")


def _float(item: dict, key: str) -> float:
    return float(item[key])


def _plot_pass_at_k(summaries: dict, out_dir: Path) -> None:
    backends = list(summaries)
    methods = [("without", "Baseline RAG"), ("with", "OpenCoder")]
    metrics = [("pass@1", "Pass@1"), ("pass@3", "Pass@3"), ("pass@5", "Pass@5")]
    colors = {"Baseline RAG": "#6b7280", "OpenCoder": "#2563eb"}

    fig, axes = plt.subplots(1, len(backends), figsize=(8.0, 3.2), sharey=True, constrained_layout=True)
    if len(backends) == 1:
        axes = [axes]

    x = np.arange(len(metrics))
    width = 0.34
    for ax, backend in zip(axes, backends):
        for i, (method_key, method_label) in enumerate(methods):
            values = [
                _float(_row(summaries[backend], method_key), metric_key)
                for metric_key, _ in metrics
            ]
            bars = ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                label=method_label,
                color=colors[method_label],
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(backend, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(0, 0.52)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Estimated pass@k")
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
    fig.suptitle("RQ2 Pass@k on the Stable 7-Task Execution-Backed Subset", fontsize=11)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_rq2_improved_backend_pass_at_k.{ext}", dpi=300)
    plt.close(fig)


def _plot_diagnostics(summaries: dict, out_dir: Path) -> None:
    backends = list(summaries)
    methods = [("without", "Baseline RAG"), ("with", "OpenCoder")]
    metrics = [
        ("mean_uncertainty", "Mean Aggregate Uncertainty", "Mean $U$"),
        ("ece", "Expected Calibration Error", "ECE"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    x = np.arange(len(backends))
    width = 0.34
    colors = {"Baseline RAG": "#6b7280", "OpenCoder": "#2563eb"}

    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        for i, (method_key, method_label) in enumerate(methods):
            values = [
                _float(_row(summaries[backend], method_key), metric_key)
                for backend in backends
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=method_label,
                color=colors[method_label],
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(backends)
        ax.set_ylim(0, max(0.85, ax.get_ylim()[1] * 1.12))
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.04, 1.24), ncol=2, frameon=False)
    fig.suptitle("RQ2 Backend Diagnostics on the Stable 7-Task Execution-Backed Subset", fontsize=11)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_rq2_backend_diagnostics.{ext}", dpi=300)
    plt.close(fig)


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_pass_at_k(summaries, out_dir)
    _plot_diagnostics(summaries, out_dir)
    print(f"Wrote {os.fspath(out_dir / 'fig_rq2_improved_backend_pass_at_k.pdf')}")
    print(f"Wrote {os.fspath(out_dir / 'fig_rq2_backend_diagnostics.pdf')}")


if __name__ == "__main__":
    main()
