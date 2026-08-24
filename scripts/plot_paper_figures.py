"""Generate publication-quality OpenCoder result figures.

The report generator writes simple dependency-free SVGs. This script uses
matplotlib when available to produce paper-ready PDF and 300 DPI PNG figures.
It consumes ``report_summary.json`` so it can be rerun after any API-backed
experiment without touching the raw result JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_MPL_CACHE = Path("cache") / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
    "light_gray": "#D9D9D9",
}


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> Optional[float]:
    if value in (None, "--", ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _values(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> List[float]:
    return [default if _num(row.get(key)) is None else float(_num(row.get(key))) for row in rows]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["gray"],
        )


def plot_rq1_delta(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    labels = [row.get("source", "") for row in rows]
    deltas = _values(rows, "_delta_u_value")
    y = np.arange(len(labels))
    max_abs = max([abs(v) for v in deltas] + [0.01])
    fig, ax = plt.subplots(figsize=(5.2, 2.45))
    colors = [COLORS["blue"] if v >= 0 else COLORS["red"] for v in deltas]
    ax.barh(y, deltas, color=colors, height=0.52)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_xlabel("Delta aggregate uncertainty (present - absent)")
    ax.set_title("RQ1: Source Effect on Uncertainty")
    for yi, value in zip(y, deltas):
        offset = max_abs * 0.04
        ha = "left" if value >= 0 else "right"
        x = value + offset if value >= 0 else value - offset
        ax.text(x, yi, f"{value:.3f}", va="center", ha=ha, fontsize=8, color=COLORS["gray"])
    _save(fig, out_dir, "fig_rq1_delta_uncertainty")


def plot_rq1_present_absent(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    labels = [row.get("source", "") for row in rows]
    present = _values(rows, "mean_u_present")
    absent = _values(rows, "mean_u_absent")
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.4, 2.65))
    bars1 = ax.bar(x - width / 2, present, width, label="Present", color=COLORS["blue"])
    bars2 = ax.bar(x + width / 2, absent, width, label="Absent", color=COLORS["orange"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean aggregate uncertainty")
    ax.set_title("RQ1: Uncertainty With vs. Without Each Source")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ymax = max(present + absent + [0.1])
    ax.set_ylim(0, ymax * 1.25)
    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    _save(fig, out_dir, "fig_rq1_present_absent_uncertainty")


def plot_rq2_passk(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    metrics = ["pass@1", "pass@3", "pass@5"]
    methods = [row.get("method", "") for row in rows]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 2.75))
    for i, row in enumerate(rows):
        vals = [_num(row.get(metric)) or 0.0 for metric in metrics]
        offset = (i - (len(rows) - 1) / 2) * width
        color = COLORS["gray"] if row.get("method_key") == "without" else COLORS["green"]
        bars = ax.bar(x + offset, vals, width, label=methods[i], color=color)
        _annotate_bars(ax, bars)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Estimated pass@k")
    ax.set_title("RQ2: Generation Robustness")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig_rq2_pass_at_k")


def plot_rq2_diagnostics(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    metrics = [
        ("pass_rate_variance", "Pass-rate variance"),
        ("ece", "ECE"),
        ("mean_uncertainty", "Mean uncertainty"),
    ]
    methods = [row.get("method", "") for row in rows]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.4, 2.75))
    for i, row in enumerate(rows):
        vals = [_num(row.get(key)) or 0.0 for key, _ in metrics]
        offset = (i - (len(rows) - 1) / 2) * width
        color = COLORS["gray"] if row.get("method_key") == "without" else COLORS["green"]
        bars = ax.bar(x + offset, vals, width, label=methods[i], color=color)
        _annotate_bars(ax, bars)
    ax.set_xticks(x, [label for _, label in metrics], rotation=8, ha="right")
    ymax = max([_num(row.get(key)) or 0.0 for row in rows for key, _ in metrics] + [0.1])
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("Metric value")
    ax.set_title("RQ2: Uncertainty and Calibration Diagnostics")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig_rq2_uncertainty_diagnostics")


def plot_combined_panel(
    rq1_rows: List[Dict[str, Any]],
    rq2_rows: List[Dict[str, Any]],
    out_dir: Path,
    readiness: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    fig.suptitle("OpenCoder Uncertainty-Aware RAG Results", y=1.02)

    labels = [row.get("source", "") for row in rq1_rows]
    deltas = _values(rq1_rows, "_delta_u_value")
    y = np.arange(len(labels))
    max_abs = max([abs(v) for v in deltas] + [0.01])
    ax = axes[0, 0]
    ax.barh(y, deltas, color=[COLORS["blue"] if v >= 0 else COLORS["red"] for v in deltas])
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_title("RQ1 delta U")
    ax.set_xlabel("Present - absent")

    ax = axes[0, 1]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, _values(rq1_rows, "mean_u_present"), width, label="Present", color=COLORS["blue"])
    ax.bar(x + width / 2, _values(rq1_rows, "mean_u_absent"), width, label="Absent", color=COLORS["orange"])
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("RQ1 mean U")
    ax.set_ylabel("Uncertainty")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    metrics = ["pass@1", "pass@3", "pass@5"]
    x = np.arange(len(metrics))
    for i, row in enumerate(rq2_rows):
        vals = [_num(row.get(metric)) or 0.0 for metric in metrics]
        offset = (i - (len(rq2_rows) - 1) / 2) * width
        color = COLORS["gray"] if row.get("method_key") == "without" else COLORS["green"]
        ax.bar(x + offset, vals, width, label=row.get("method", ""), color=color)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_title("RQ2 pass@k")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    diag = [("pass_rate_variance", "Var"), ("ece", "ECE"), ("mean_uncertainty", "Mean U")]
    x = np.arange(len(diag))
    for i, row in enumerate(rq2_rows):
        vals = [_num(row.get(key)) or 0.0 for key, _ in diag]
        offset = (i - (len(rq2_rows) - 1) / 2) * width
        color = COLORS["gray"] if row.get("method_key") == "without" else COLORS["green"]
        ax.bar(x + offset, vals, width, label=row.get("method", ""), color=color)
    ax.set_xticks(x, [label for _, label in diag])
    ax.set_title("RQ2 diagnostics")
    ax.set_ylabel("Metric")

    if "offline" in readiness.lower() or "pilot" in readiness.lower():
        fig.text(
            0.5,
            -0.015,
            readiness,
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS["red"],
        )
    _save(fig, out_dir, "fig_main_results_panel")


def write_figure_index(out_dir: Path, readiness: str) -> None:
    lines = [
        "# Paper Figure Index",
        "",
        f"Readiness note: {readiness}",
        "",
        "| Figure | PDF | PNG | Suggested paper use |",
        "|---|---|---|---|",
        "| RQ1 delta uncertainty | `fig_rq1_delta_uncertainty.pdf` | `fig_rq1_delta_uncertainty.png` | Source contribution plot for RQ1. |",
        "| RQ1 present/absent uncertainty | `fig_rq1_present_absent_uncertainty.pdf` | `fig_rq1_present_absent_uncertainty.png` | Supporting plot for source ablation. |",
        "| RQ2 pass@k | `fig_rq2_pass_at_k.pdf` | `fig_rq2_pass_at_k.png` | Main RQ2 robustness plot. |",
        "| RQ2 uncertainty diagnostics | `fig_rq2_uncertainty_diagnostics.pdf` | `fig_rq2_uncertainty_diagnostics.png` | Calibration/uncertainty mitigation plot. |",
        "| Main results panel | `fig_main_results_panel.pdf` | `fig_main_results_panel.png` | Compact overview figure for paper or slides. |",
        "",
        "For LaTeX, prefer the PDF files. For Word or quick preview, use PNG.",
        "",
    ]
    (out_dir / "FIGURE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/dryrun_execrepobench_20/report_summary.json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    data = _load_summary(summary_path)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent / "paper_figures"
    _setup_style()

    rq1_rows = data.get("rq1_table", [])
    rq2_rows = data.get("rq2_table", [])
    readiness = data.get("paper_readiness", "Unknown readiness.")

    if rq1_rows:
        plot_rq1_delta(rq1_rows, out_dir)
        plot_rq1_present_absent(rq1_rows, out_dir)
    if rq2_rows:
        plot_rq2_passk(rq2_rows, out_dir)
        plot_rq2_diagnostics(rq2_rows, out_dir)
    if rq1_rows and rq2_rows:
        plot_combined_panel(rq1_rows, rq2_rows, out_dir, readiness)
    write_figure_index(out_dir, readiness)
    print(f"Wrote paper figures to {out_dir}")


if __name__ == "__main__":
    main()
