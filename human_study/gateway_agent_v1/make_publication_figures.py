#!/usr/bin/env python3
"""Create publication figures and a camera-ready text fragment."""
from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results/agent_gateway_v1/analysis"
OUT = ANALYSIS / "publication"
FIGURES = OUT / "figures"
LATEX = OUT / "latex"

INK = "#1D2939"
MUTED = "#667085"
RULE = "#D0D5DD"
PANEL = "#F8FAFC"
BLUE = "#1859A9"
BLUE_LIGHT = "#EFF6FF"
GOLD = "#9A6700"
GOLD_LIGHT = "#FFF8E7"
NEUTRAL_LIGHT = "#F2F4F7"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = RULE,
    linewidth: float = 0.8,
    radius: float = 0.012,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def make_outcome_figure(rows: list[dict[str, str]]) -> None:
    fig = plt.figure(figsize=(7.2, 4.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    fig.text(
        0.045,
        0.945,
        "Agent-review outcomes under three evidence displays",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.892,
        "12 gateway configurations × 12 frozen ExecRepoBench tasks; 48 planned episodes per condition",
        ha="left",
        va="top",
        fontsize=8.6,
        color=MUTED,
    )

    card_x = (0.045, 0.358, 0.671)
    card_width = 0.284
    card_y = 0.235
    card_height = 0.605
    accents = (MUTED, BLUE, GOLD)
    fills = (PANEL, BLUE_LIGHT, GOLD_LIGHT)

    for index, row in enumerate(rows):
        x = card_x[index]
        rounded_box(
            ax,
            x,
            card_y,
            card_width,
            card_height,
            facecolor=fills[index],
            edgecolor=accents[index],
            linewidth=1.0,
        )
        ax.add_patch(
            Rectangle(
                (x, card_y + card_height - 0.012),
                card_width,
                0.012,
                transform=ax.transAxes,
                color=accents[index],
                linewidth=0,
            )
        )
        fig.text(
            x + 0.018,
            0.795,
            row["condition_label"],
            fontsize=10.3,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )
        fig.text(
            x + card_width - 0.018,
            0.748,
            f"{row['evaluable']}/{row['planned']}",
            fontsize=8.0,
            fontweight="bold",
            color=accents[index],
            ha="right",
            va="top",
        )
        fig.text(
            x + 0.018,
            0.751,
            "executable / planned",
            fontsize=7.2,
            color=MUTED,
            ha="left",
            va="top",
        )

        correctness = 100 * float(row["correctness_evaluable"])
        end_to_end = 100 * float(row["end_to_end_success"])
        ci_low = 100 * float(row["end_to_end_ci95_low"])
        ci_high = 100 * float(row["end_to_end_ci95_high"])
        detection = 100 * float(row["failure_detection_accuracy"])
        repair = 100 * float(row["repair_success"])

        fig.text(
            x + 0.018,
            0.682,
            "Correctness among executable outputs",
            fontsize=7.4,
            color=MUTED,
            ha="left",
            va="top",
        )
        fig.text(
            x + 0.018,
            0.638,
            f"{correctness:.1f}%",
            fontsize=18,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )

        fig.text(
            x + 0.018,
            0.555,
            "End-to-end success",
            fontsize=7.4,
            color=MUTED,
            ha="left",
            va="top",
        )
        fig.text(
            x + 0.018,
            0.513,
            f"{end_to_end:.1f}%",
            fontsize=16,
            fontweight="bold",
            color=accents[index],
            ha="left",
            va="top",
        )
        fig.text(
            x + card_width - 0.018,
            0.511,
            f"95% CI\n[{ci_low:.1f}, {ci_high:.1f}]",
            fontsize=7.1,
            color=MUTED,
            ha="right",
            va="top",
            linespacing=1.2,
        )

        ax.plot(
            [x + 0.018, x + card_width - 0.018],
            [0.435, 0.435],
            transform=ax.transAxes,
            color=RULE,
            linewidth=0.7,
        )
        fig.text(x + 0.018, 0.397, "Failure detection", fontsize=7.4, color=MUTED, ha="left")
        fig.text(
            x + card_width - 0.018,
            0.397,
            f"{detection:.1f}%",
            fontsize=9.2,
            fontweight="bold",
            color=INK,
            ha="right",
        )
        fig.text(x + 0.018, 0.335, "Repair success", fontsize=7.4, color=MUTED, ha="left")
        fig.text(
            x + card_width - 0.018,
            0.335,
            f"{repair:.1f}%",
            fontsize=9.2,
            fontweight="bold",
            color=INK,
            ha="right",
        )

    rounded_box(ax, 0.045, 0.055, 0.910, 0.125, facecolor="white", edgecolor=RULE)
    ax.plot([0.5, 0.5], [0.072, 0.163], transform=ax.transAxes, color=RULE, linewidth=0.8)
    fig.text(0.065, 0.145, "Uncertainty display − generic review", fontsize=7.6, color=MUTED, ha="left")
    fig.text(0.065, 0.096, "+8.3 pp", fontsize=12.2, fontweight="bold", color=BLUE, ha="left")
    fig.text(0.208, 0.099, "95% CI [−2.1, +18.8]", fontsize=7.7, color=INK, ha="left")
    fig.text(0.520, 0.145, "Targeted guidance − generic review", fontsize=7.6, color=MUTED, ha="left")
    fig.text(0.520, 0.096, "−2.1 pp", fontsize=12.2, fontweight="bold", color=GOLD, ha="left")
    fig.text(0.666, 0.099, "95% CI [−12.5, +8.3]", fontsize=7.7, color=INK, ha="left")

    save(fig, "fig_agent_review_boxed")


def make_coverage_figure(models: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(7.2, 5.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    fig.text(
        0.045,
        0.955,
        "Gateway configurations and campaign integrity",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.905,
        "Resolved model identifiers used in the final 144-episode campaign",
        ha="left",
        va="top",
        fontsize=8.6,
        color=MUTED,
    )

    columns = 3
    card_width = 0.292
    card_height = 0.133
    x_positions = (0.045, 0.354, 0.663)
    y_positions = (0.735, 0.578, 0.421, 0.264)
    for index, model in enumerate(models):
        row = index // columns
        column = index % columns
        x, y = x_positions[column], y_positions[row]
        rounded_box(ax, x, y, card_width, card_height, facecolor=PANEL, edgecolor=RULE)
        badge_x, badge_y = x + 0.037, y + card_height / 2
        ax.add_patch(
            Circle(
                (badge_x, badge_y),
                0.026,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=0.9,
            )
        )
        fig.text(
            badge_x,
            badge_y,
            str(model["family"])[0],
            fontsize=8.0,
            fontweight="bold",
            color=BLUE,
            ha="center",
            va="center",
        )
        fig.text(
            x + 0.076,
            y + 0.091,
            str(model["family"]),
            fontsize=8.8,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
        )
        model_id = str(model["resolved_model_id"])
        wrapped = "\n".join(textwrap.wrap(model_id, width=25, break_long_words=True))
        fig.text(
            x + 0.076,
            y + 0.050,
            wrapped,
            fontsize=6.6,
            family="monospace",
            color=MUTED,
            ha="left",
            va="center",
            linespacing=1.05,
        )
        fig.text(
            x + card_width - 0.014,
            y + 0.100,
            str(model["agent_id"]),
            fontsize=6.2,
            color=MUTED,
            ha="right",
            va="center",
        )

    metrics = (
        ("144", "planned episodes"),
        ("134", "executable"),
        ("10", "missing outputs"),
        ("0", "evaluator errors"),
        ("0", "model mismatches"),
        ("599,210", "reported tokens"),
    )
    footer_y = 0.080
    footer_width = 0.141
    gap = 0.012
    for index, (value, label) in enumerate(metrics):
        x = 0.045 + index * (footer_width + gap)
        rounded_box(ax, x, footer_y, footer_width, 0.115, facecolor="white", edgecolor=RULE)
        fig.text(
            x + footer_width / 2,
            footer_y + 0.073,
            value,
            fontsize=11.0,
            fontweight="bold",
            color=INK if index > 1 else BLUE,
            ha="center",
            va="center",
        )
        fig.text(
            x + footer_width / 2,
            footer_y + 0.032,
            label,
            fontsize=6.5,
            color=MUTED,
            ha="center",
            va="center",
        )

    fig.text(
        0.045,
        0.030,
        "Gateway-mediated configurations; these are not consumer-product evaluations or human participants.",
        fontsize=7.1,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    save(fig, "fig_gateway_campaign_coverage")


def write_latex() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    text = r"""% Camera-ready exploratory agent robustness analysis.
% These gateway configurations must not be described as human participants or
% direct evaluations of consumer chat products.
\subsection{Multi-Model Agent Robustness Analysis}
\label{sec:multi-model-agent-analysis}

We further examine whether OpenCoder's uncertainty presentation changes the
behavior of automated code reviewers. Twelve gateway-mediated model
configurations---GPT, Claude, Gemini, DeepSeek, Qwen, Kimi, Grok, Llama, GLM,
MiniMax, Doubao, and ERNIE---each reviewed 12 frozen ExecRepoBench tasks under
a counterbalanced design. The study therefore comprised 144 independent
episodes, with 48 episodes assigned to each of generic review, aggregate
uncertainty display, and source-specific targeted guidance. Models received no
test outcomes, hidden reference code, tools, or conversational history. Every
returned target function was assessed with the same private executable tests.

Figure~\ref{fig:agent-review-summary} summarizes the aggregate outcomes. Generic
review achieved 38.6\% correctness among executable outputs and 35.4\%
end-to-end success over all planned episodes. Aggregate uncertainty display
increased these descriptive rates to 46.7\% and 43.8\%, respectively. Its
end-to-end difference from generic review was +8.3 percentage points, although
the model-clustered 95\% bootstrap interval included zero
($[-2.1,+18.8]$). Source-specific targeted guidance achieved 35.6\%
correctness among executable outputs and 33.3\% end-to-end success, a
$-2.1$-point difference from generic review ($95\%$ CI
$[-12.5,+8.3]$). Thus, this campaign does not support a claim that targeted
guidance is superior across model families.

Figure~\ref{fig:gateway-campaign-coverage} reports the resolved gateway
configurations and campaign audit. The final artifact contains 144 unique
records, including 134 executable outputs and 10 model-side missing outputs,
with no evaluator infrastructure errors or served-model mismatches. The
gateway reported 599,210 tokens across the campaign. Missing outputs were
excluded from executable-output correctness and retained in the denominator of
the end-to-end sensitivity analysis.

\noindent\textbf{Finding.}
The aggregate uncertainty display yielded the strongest descriptive agent
outcomes, but the uncertainty interval does not establish a reliable positive
effect. Source-specific guidance did not improve aggregate performance. We
therefore treat this experiment as a robustness analysis of uncertainty
presentation across gateway-mediated model configurations, not as evidence of
human usability or direct evaluation of consumer chat products.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig_agent_review_boxed.pdf}
    \caption{Executable review outcomes across three evidence-display
    conditions. Correctness uses only executable model outputs; end-to-end
    success retains all 48 planned episodes per condition. Intervals are
    model-clustered 95\% bootstrap intervals over 10,000 resamples.}
    \label{fig:agent-review-summary}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig_gateway_campaign_coverage.pdf}
    \caption{Resolved gateway configurations and integrity summary for the
    exploratory multi-model campaign. Model names denote API configurations,
    not consumer chat products.}
    \label{fig:gateway-campaign-coverage}
\end{figure*}
"""
    (LATEX / "agent_replication_camera_ready.tex").write_text(text, encoding="utf-8")


def main() -> int:
    condition_rows = read_csv(ANALYSIS / "condition_summary.csv")
    manifest = json.loads(
        (ROOT / "results/agent_gateway_v1/resolved_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if len(condition_rows) != 3:
        raise RuntimeError("Expected three frozen review conditions.")
    if len(manifest["models"]) != 12:
        raise RuntimeError("Expected twelve resolved gateway configurations.")
    make_outcome_figure(condition_rows)
    make_coverage_figure(manifest["models"])
    write_latex()
    print(f"Wrote publication artifacts to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
