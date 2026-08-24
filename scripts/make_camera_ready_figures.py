"""Build camera-ready paper figures from the current verified result tables.

The figure titles deliberately avoid question-number labels so the plots read like
paper figures rather than experiment bookkeeping.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "aaai_camera_ready_figures"

PATHS = {
    "rq1_gpt": ROOT / "results/aaai_testbacked_stable7_gpt/opencoder_gpt/rq1_source_table.csv",
    "rq1_gemini": ROOT / "results/aaai_testbacked_stable7_gemini/opencoder_gemini/rq1_source_table.csv",
    "rq2_gpt": ROOT / "results/aaai_improved_repair_gpt_stable7_v2/opencoder_gpt/rq2_method_table.csv",
    "rq2_gemini": ROOT / "results/aaai_improved_repair_gemini_stable7/opencoder_gemini/rq2_method_table.csv",
    "rq3_summary": ROOT / "results/rq3_testbacked10/summary.csv",
    "rq4_quality": ROOT / "results/rq4/api_quality_summary.csv",
    "rq4_metrics": ROOT / "results/rq4/per_task_api_metrics.csv",
}


COLORS = {
    "GPT": "#2563eb",
    "Gemini": "#dc2626",
    "Baseline RAG": "#8c8c8c",
    "OpenCoder": "#2ca25f",
    "No API refine": "#9ecae1",
    "No uncertainty filter": "#f59e0b",
    "Context": "#7c3aed",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _label_bars(ax: plt.Axes, bars: Iterable[Any], *, fmt: str = "{:.1f}", fontsize: int = 7) -> None:
    for bar in bars:
        height = float(bar.get_height())
        offset = 0.012 if abs(height) < 1 else 0.8
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def source_evidence_figure() -> None:
    data = {
        "GPT": _read_csv(PATHS["rq1_gpt"]),
        "Gemini": _read_csv(PATHS["rq1_gemini"]),
    }
    sources = ["API knowledge", "Context code", "Similar code"]
    short = ["API", "Context", "Similar"]
    x = np.arange(len(sources))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    for i, backend in enumerate(("GPT", "Gemini")):
        rows = {r["source"]: r for r in data[backend]}
        delta = [100.0 * _float(rows[s], "delta_u") for s in sources]
        rho = [_float(rows[s], "spearman_rho") for s in sources]
        bars = axes[0].bar(
            x + (i - 0.5) * width,
            delta,
            width,
            label=backend,
            color=COLORS[backend],
            alpha=0.9,
        )
        _label_bars(axes[0], bars, fmt="{:.1f}", fontsize=7)
        axes[1].plot(
            x,
            rho,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=backend,
            color=COLORS[backend],
        )
        for j, value in enumerate(rho):
            axes[1].text(j, value + (0.045 if value >= 0 else -0.065), f"{value:.2f}", ha="center", fontsize=7)

    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Evidence Presence and Uncertainty", fontsize=10)
    axes[0].set_ylabel(r"$\Delta$ uncertainty (points)", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, fontsize=8)
    axes[0].set_ylim(-1.5, 6.0)
    _style_axes(axes[0])

    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_title("Association With Uncertainty", fontsize=10)
    axes[1].set_ylabel("Spearman correlation", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, fontsize=8)
    axes[1].set_ylim(-0.35, 0.95)
    _style_axes(axes[1])
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    _save(fig, "fig_source_evidence_uncertainty")


def mitigation_effect_figure() -> None:
    data = {
        "GPT": _read_csv(PATHS["rq2_gpt"]),
        "Gemini": _read_csv(PATHS["rq2_gemini"]),
    }
    metrics = [("pass@1", "Pass@1"), ("pass@3", "Pass@3"), ("pass@5", "Pass@5")]
    methods = ["Baseline RAG", "OpenCoder"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.4), constrained_layout=False)
    fig.subplots_adjust(top=0.93, bottom=0.14, left=0.09, right=0.99, hspace=0.48, wspace=0.28)
    for ax, backend in zip(axes[0], ("GPT", "Gemini")):
        rows = {r["method"]: r for r in data[backend]}
        for i, method in enumerate(methods):
            vals = [100.0 * _float(rows[method], key) for key, _ in metrics]
            bars = ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=method,
                color=COLORS[method],
            )
            _label_bars(ax, bars, fmt="{:.1f}", fontsize=7)
        ax.set_title(f"{backend} Functional Correctness", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], fontsize=8)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Estimated pass rate (%)", fontsize=9)
        _style_axes(ax)

    diag = [("mean_uncertainty", "Mean uncertainty"), ("ece", "Calibration error")]
    x2 = np.arange(len(diag))
    for backend in ("GPT", "Gemini"):
        rows = {r["method"]: r for r in data[backend]}
        ax = axes[1][0] if backend == "GPT" else axes[1][1]
        for method in methods:
            i = methods.index(method)
            vals = [_float(rows[method], key) for key, _ in diag]
            bars = ax.bar(
                x2 + (i - 0.5) * width,
                vals,
                width,
                color=COLORS[method],
                label=method,
            )
            _label_bars(ax, bars, fmt="{:.2f}", fontsize=6)
        ax.set_title(f"{backend} Uncertainty Diagnostics", fontsize=10)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xticks(x2)
        ax.set_xticklabels([label for _, label in diag], fontsize=8)
        ax.set_ylim(0, 0.84)
        _style_axes(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    _save(fig, "fig_uncertainty_mitigation_effect")


def end_to_end_figure() -> None:
    rows = _read_csv(PATHS["rq3_summary"])
    lookup = {(r["Backend"], r["Method"]): r for r in rows}
    backends = ["GPT", "Gemini"]
    datasets = [
        ("RepoExec-inline", "RepoExec-inline"),
        ("CoderEval", "CoderEval exact match"),
        ("ExecRepoBench", "ExecRepoBench"),
    ]
    metrics = [("Pass1", "Pass@1"), ("Pass3", "Pass@3"), ("Pass5", "Pass@5")]
    methods = ["Baseline RAG", "OpenCoder"]

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.7), sharey=False, constrained_layout=False)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.99, hspace=0.44, wspace=0.22)
    x = np.arange(len(metrics))
    width = 0.34
    for row_idx, backend in enumerate(backends):
        for col_idx, (dataset_key, dataset_label) in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            for i, method in enumerate(methods):
                vals = [
                    _float(lookup[(backend, method)], f"{dataset_key}_{metric_key}")
                    for metric_key, _ in metrics
                ]
                bars = ax.bar(
                    x + (i - 0.5) * width,
                    vals,
                    width,
                    label=method,
                    color=COLORS[method],
                )
                _label_bars(ax, bars, fmt="{:.1f}", fontsize=6)
            ax.set_title(f"{backend} · {dataset_label}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in metrics], fontsize=7)
            ymax = 85 if dataset_key != "CoderEval" else 20
            ax.set_ylim(0, ymax)
            if col_idx == 0:
                ax.set_ylabel("Pass rate (%)", fontsize=9)
            _style_axes(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    _save(fig, "fig_end_to_end_correctness")


def api_reliability_figure() -> None:
    quality = _read_csv(PATHS["rq4_quality"])
    metrics = _read_csv(PATHS["rq4_metrics"])

    def agg(backend: str, method: str) -> float:
        vals = [
            _float(r, "f1")
            for r in quality
            if r["backend"] == backend and r["method"] == method
        ]
        return 100.0 * sum(vals) / len(vals) if vals else 0.0

    backends = ["GPT", "Gemini"]
    methods = [
        ("Baseline RAG", "Baseline"),
        ("OpenCoder-NoAPIRefine", "No API refine"),
        ("OpenCoder", "OpenCoder"),
    ]
    x = np.arange(len(backends))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.25), constrained_layout=True)

    for i, (method_key, method_label) in enumerate(methods):
        vals = [agg(backend, method_key) for backend in backends]
        bars = axes[0].bar(
            x + (i - 1) * width,
            vals,
            width,
            label=method_label,
            color=COLORS.get(method_label, COLORS["Baseline RAG"]),
        )
        _label_bars(axes[0], bars, fmt="{:.1f}", fontsize=7)
    axes[0].set_title("API-Set Quality", fontsize=10)
    axes[0].set_ylabel("API F1 (%)", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(backends, fontsize=8)
    axes[0].set_ylim(0, 62)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _style_axes(axes[0])

    markers = {"RepoExec": "o", "CoderEval": "s", "ExecRepoBench": "^"}
    for row in metrics:
        if row["method"] != "OpenCoder" or row.get("uncertainty") in ("", None):
            continue
        axes[1].scatter(
            _float(row, "uncertainty"),
            _float(row, "f1"),
            marker=markers.get(row["benchmark"], "o"),
            color=COLORS.get(row["backend"], "#333333"),
            alpha=0.68,
            s=36,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[1].set_title("Uncertainty Diagnostic", fontsize=10)
    axes[1].set_xlabel("OpenCoder uncertainty", fontsize=9)
    axes[1].set_ylabel("API F1", fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    handles: List[Any] = []
    labels: List[str] = []
    for backend in backends:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[backend], markersize=6))
        labels.append(backend)
    for benchmark, marker in markers.items():
        handles.append(plt.Line2D([0], [0], marker=marker, color="#555555", linestyle="None", markersize=6))
        labels.append(benchmark)
    axes[1].legend(handles, labels, ncol=2, frameon=False, fontsize=8, loc="center")
    _save(fig, "fig_api_retrieval_reliability")


def write_index() -> None:
    lines = [
        "# Camera-Ready Figure Set",
        "",
        "Generated from current verified result tables. The plots intentionally avoid visual question-number labels.",
        "",
        "| Purpose | PDF | PNG | Source data |",
        "|---|---|---|---|",
        "| Source evidence and uncertainty | `fig_source_evidence_uncertainty.pdf` | `fig_source_evidence_uncertainty.png` | stable-7 source ablation CSVs |",
        "| Uncertainty-aware mitigation | `fig_uncertainty_mitigation_effect.pdf` | `fig_uncertainty_mitigation_effect.png` | improved stable-7 method CSVs |",
        "| End-to-end correctness | `fig_end_to_end_correctness.pdf` | `fig_end_to_end_correctness.png` | corrected end-to-end summary CSV |",
        "| API retrieval reliability | `fig_api_retrieval_reliability.pdf` | `fig_api_retrieval_reliability.png` | refreshed API reliability CSVs |",
        "",
        "Recommended Overleaf path after copying PDFs into the paper project: `figures/<filename>.pdf`.",
        "",
    ]
    (OUT_DIR / "FIGURE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_source_evidence_uncertainty.pdf}
\caption{Influence of heterogeneous retrieved evidence on aggregate uncertainty. Bars show the change in uncertainty when each evidence type is present rather than absent; lines show Spearman association with uncertainty.}
\label{fig:source_evidence_uncertainty}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_uncertainty_mitigation_effect.pdf}
\caption{Effect of uncertainty-aware generation, verification, and repair on functional correctness and calibration across GPT and Gemini backends.}
\label{fig:uncertainty_mitigation_effect}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_end_to_end_correctness.pdf}
\caption{End-to-end Pass@k comparison between Baseline RAG and OpenCoder across repository-level benchmarks and LLM backends. CoderEval is evaluated using normalized reference exact match because the executable harness is unavailable locally.}
\label{fig:end_to_end_correctness}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_api_retrieval_reliability.pdf}
\caption{API retrieval reliability. The left panel compares aggregate API F1 across Baseline RAG, OpenCoder without API refinement, and target-aware OpenCoder; the right panel shows the relationship between OpenCoder uncertainty and per-task API F1.}
\label{fig:api_retrieval_reliability}
\end{figure*}
"""
    (OUT_DIR / "figure_snippets.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_evidence_figure()
    mitigation_effect_figure()
    end_to_end_figure()
    api_reliability_figure()
    write_index()
    print(f"Wrote camera-ready figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
