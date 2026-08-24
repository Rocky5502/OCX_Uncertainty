"""Build paper-facing tables and figures from OpenCoder RQ runs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SOURCE_LABELS = {
    "api": "API knowledge",
    "context": "Context code",
    "similar_code": "Similar code",
}

METHOD_LABELS = {
    "without": "Baseline RAG",
    "with": "OpenCoder",
}


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _load_run_config(out_dir: Path, explicit_path: Optional[str]) -> Dict[str, Any]:
    path = Path(explicit_path) if explicit_path else out_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value)
    if number is None:
        return "--"
    return f"{number:.{digits}f}"


def _latex_escape(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def _write_latex_table(
    path: Path,
    rows: List[Dict[str, Any]],
    columns: List[str],
    headers: List[str],
    caption: str,
    label: str,
) -> None:
    aligns = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            values.append(_latex_escape(value if value not in (None, "") else "--"))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _correctness_modes_from_rq1(data: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not data:
        return counts
    for row in data.get("rows", []):
        mode = row.get("correctness_mode")
        if mode:
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _correctness_modes_from_rq2(data: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not data:
        return counts
    for key in ("without", "with"):
        for row in data.get(key, []):
            mode = row.get("correctness_mode")
            if mode:
                counts[mode] = counts.get(mode, 0) + 1
    return counts


def build_rq1_table(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []
    rows = []
    summary = data.get("summary", {})
    for source, item in summary.items():
        mean_present = _safe_float(item.get("mean_u_when_present"))
        mean_absent = _safe_float(item.get("mean_u_when_absent"))
        var_present = _safe_float(item.get("mean_pass_rate_variance_when_present"))
        var_absent = _safe_float(item.get("mean_pass_rate_variance_when_absent"))
        rho = (item.get("spearman_uncertainty") or {}).get("rho")
        p_value = (item.get("spearman_uncertainty") or {}).get("p")
        rows.append({
            "source": SOURCE_LABELS.get(source, source),
            "source_key": source,
            "mean_u_present": _fmt(mean_present),
            "mean_u_absent": _fmt(mean_absent),
            "delta_u": _fmt(item.get("delta_u_present_minus_absent")),
            "spearman_rho": _fmt(rho),
            "spearman_p": _fmt(p_value),
            "pass_variance_present": _fmt(var_present),
            "pass_variance_absent": _fmt(var_absent),
            "delta_pass_variance": _fmt(
                None if var_present is None or var_absent is None else var_present - var_absent
            ),
            "_delta_u_value": _safe_float(item.get("delta_u_present_minus_absent")),
        })
    return rows


def build_rq2_table(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []
    rows = []
    summary = data.get("summary", {})
    for method_key in ("without", "with"):
        item = summary.get(method_key, {})
        rows.append({
            "method": METHOD_LABELS.get(method_key, method_key),
            "method_key": method_key,
            "pass@1": _fmt(item.get("pass@1")),
            "pass@3": _fmt(item.get("pass@3")),
            "pass@5": _fmt(item.get("pass@5")),
            "pass_rate_variance": _fmt(item.get("pass_rate_variance")),
            "ece": _fmt(item.get("ece")),
            "mean_uncertainty": _fmt(item.get("mean_uncertainty")),
            "mean_repair_rounds": _fmt(item.get("mean_repair_rounds")),
            "n": item.get("n", 0),
            "n_known_correctness": item.get("n_known_correctness", 0),
            "_pass1_value": _safe_float(item.get("pass@1")),
            "_pass3_value": _safe_float(item.get("pass@3")),
            "_pass5_value": _safe_float(item.get("pass@5")),
            "_pass_var_value": _safe_float(item.get("pass_rate_variance")),
            "_ece_value": _safe_float(item.get("ece")),
            "_mean_u_value": _safe_float(item.get("mean_uncertainty")),
        })
    return rows


def _first_metadata(
    run_config: Dict[str, Any],
    rq1_data: Optional[Dict[str, Any]],
    rq2_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for source in (rq1_data, rq2_data):
        if source and isinstance(source.get("metadata"), dict):
            metadata.update(source["metadata"])
    metadata.update({k: v for k, v in run_config.items() if v is not None})
    if str(metadata.get("backend") or "").lower() == "offline":
        metadata["model"] = "offline-heuristic"
        metadata["base_url"] = None
    return metadata


def _paper_readiness(metadata: Dict[str, Any], modes: Dict[str, int]) -> str:
    backend = str(metadata.get("backend") or "").lower()
    if backend == "offline":
        return "Pipeline validation only: offline backend outputs must not be reported as model performance."
    if modes and not ({"execution_tests", "repository_tests"} & set(modes)):
        return "Pilot evidence only: correctness is reference exact-match, not execution tests."
    if not modes:
        return "Needs review: no correctness mode was recorded."
    return "Paper-ready evidence path: execution-test correctness was observed."


def _bar_svg(
    path: Path,
    *,
    title: str,
    rows: List[Dict[str, Any]],
    label_key: str,
    value_key: str,
    x_label: str,
) -> None:
    width = 920
    margin_left = 190
    margin_right = 130
    margin_top = 58
    margin_bottom = 52
    row_h = 56
    height = margin_top + margin_bottom + max(1, len(rows)) * row_h
    values = [_safe_float(row.get(value_key)) for row in rows]
    values = [v for v in values if v is not None]
    max_abs = max([abs(v) for v in values] + [1e-9])
    plot_w = width - margin_left - margin_right
    zero_x = margin_left + plot_w / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1f2937">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af" stroke-width="1"/>',
        f'<line x1="{zero_x}" y1="{margin_top - 12}" x2="{zero_x}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
    ]
    if not rows:
        parts.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#6b7280">No data</text>'
        )
    for idx, row in enumerate(rows):
        y = margin_top + idx * row_h + 18
        val = _safe_float(row.get(value_key))
        label = html.escape(str(row.get(label_key, "")))
        parts.append(
            f'<text x="{margin_left - 16}" y="{y + 6}" text-anchor="end" font-family="Arial, sans-serif" font-size="15" fill="#111827">{label}</text>'
        )
        if val is None:
            continue
        half_w = plot_w / 2
        bar_w = abs(val) / max_abs * half_w
        x = zero_x if val >= 0 else zero_x - bar_w
        color = "#2563eb" if val >= 0 else "#dc2626"
        parts.append(
            f'<rect x="{x}" y="{y - 13}" width="{bar_w}" height="24" fill="{color}" rx="3"/>'
        )
        text_x = x + bar_w + 8 if val >= 0 else x - 8
        anchor = "start" if val >= 0 else "end"
        parts.append(
            f'<text x="{text_x}" y="{y + 5}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{_fmt(val)}</text>'
        )
    parts.append(
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _grouped_pass_svg(path: Path, rows: List[Dict[str, Any]]) -> None:
    width = 920
    height = 360
    margin_left = 86
    margin_right = 42
    margin_top = 58
    margin_bottom = 78
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    metrics = [("pass@1", "_pass1_value"), ("pass@3", "_pass3_value"), ("pass@5", "_pass5_value")]
    colors = {"pass@1": "#2563eb", "pass@3": "#059669", "pass@5": "#d97706"}
    max_val = max(
        [_safe_float(row.get(key)) or 0.0 for row in rows for _, key in metrics] + [1.0]
    )
    max_val = max(1.0, max_val)
    group_w = plot_w / max(1, len(rows))
    bar_w = min(42, group_w / 5)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1f2937">RQ2 Pass@k Comparison</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = height - margin_bottom - value / max_val * plot_h
        parts.append(f'<line x1="{margin_left - 5}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{value:.1f}</text>'
        )
    for i, row in enumerate(rows):
        group_x = margin_left + i * group_w + group_w / 2
        for j, (metric, key) in enumerate(metrics):
            val = _safe_float(row.get(key))
            if val is None:
                continue
            h = val / max_val * plot_h
            x = group_x + (j - 1) * (bar_w + 8) - bar_w / 2
            y = height - margin_bottom - h
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[metric]}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{y - 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_fmt(val)}</text>'
            )
        label = html.escape(str(row.get("method", "")))
        parts.append(
            f'<text x="{group_x}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{label}</text>'
        )
    legend_x = margin_left
    for i, (metric, _) in enumerate(metrics):
        x = legend_x + i * 110
        parts.append(f'<rect x="{x}" y="{height - 34}" width="14" height="14" fill="{colors[metric]}" rx="2"/>')
        parts.append(
            f'<text x="{x + 20}" y="{height - 22}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{metric}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _grouped_source_uncertainty_svg(path: Path, rows: List[Dict[str, Any]]) -> None:
    width = 920
    height = 380
    margin_left = 82
    margin_right = 42
    margin_top = 58
    margin_bottom = 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    metrics = [
        ("Present", "mean_u_present", "#2563eb"),
        ("Absent", "mean_u_absent", "#dc2626"),
    ]
    values = [
        _safe_float(row.get(key)) or 0.0
        for row in rows
        for _, key, _ in metrics
    ]
    max_val = max(values + [1.0])
    group_w = plot_w / max(1, len(rows))
    bar_w = min(54, group_w / 4)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1f2937">RQ1 Uncertainty by Retrieval Source</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5 * max_val
        y = height - margin_bottom - value / max_val * plot_h
        parts.append(f'<line x1="{margin_left - 5}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{value:.2f}</text>'
        )
    for i, row in enumerate(rows):
        group_x = margin_left + i * group_w + group_w / 2
        for j, (_, key, color) in enumerate(metrics):
            val = _safe_float(row.get(key))
            if val is None:
                continue
            h = val / max_val * plot_h
            x = group_x + (j - 0.5) * (bar_w + 10) - bar_w / 2
            y = height - margin_bottom - h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>')
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{y - 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_fmt(val)}</text>'
            )
        label = html.escape(str(row.get("source", "")))
        parts.append(
            f'<text x="{group_x}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>'
        )
    legend_x = margin_left
    for i, (label, _, color) in enumerate(metrics):
        x = legend_x + i * 110
        parts.append(f'<rect x="{x}" y="{height - 36}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{x + 20}" y="{height - 24}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _grouped_rq2_diagnostics_svg(path: Path, rows: List[Dict[str, Any]]) -> None:
    width = 920
    height = 380
    margin_left = 86
    margin_right = 42
    margin_top = 58
    margin_bottom = 84
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    metrics = [
        ("Pass var.", "_pass_var_value", "#2563eb"),
        ("ECE", "_ece_value", "#dc2626"),
        ("Mean U", "_mean_u_value", "#059669"),
    ]
    values = [
        _safe_float(row.get(key)) or 0.0
        for row in rows
        for _, key, _ in metrics
    ]
    max_val = max(values + [1.0])
    group_w = plot_w / max(1, len(rows))
    bar_w = min(42, group_w / 5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1f2937">RQ2 Uncertainty and Calibration Diagnostics</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5 * max_val
        y = height - margin_bottom - value / max_val * plot_h
        parts.append(f'<line x1="{margin_left - 5}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{value:.2f}</text>'
        )
    for i, row in enumerate(rows):
        group_x = margin_left + i * group_w + group_w / 2
        for j, (_, key, color) in enumerate(metrics):
            val = _safe_float(row.get(key))
            if val is None:
                continue
            h = val / max_val * plot_h
            x = group_x + (j - 1) * (bar_w + 8) - bar_w / 2
            y = height - margin_bottom - h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>')
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{y - 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{_fmt(val)}</text>'
            )
        label = html.escape(str(row.get("method", "")))
        parts.append(
            f'<text x="{group_x}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{label}</text>'
        )
    legend_x = margin_left
    for i, (label, _, color) in enumerate(metrics):
        x = legend_x + i * 120
        parts.append(f'<rect x="{x}" y="{height - 36}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{x + 20}" y="{height - 24}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str], headers: List[str]) -> str:
    if not rows:
        return "_No data available._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "--")) for col in columns) + " |")
    return "\n".join(lines)


def _format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _write_paper_results_section(
    path: Path,
    *,
    metadata: Dict[str, Any],
    readiness: str,
    rq1_table: List[Dict[str, Any]],
    rq2_table: List[Dict[str, Any]],
) -> None:
    rq1_sentence = "RQ1 results were not available in this run."
    if rq1_table:
        ordered = sorted(
            rq1_table,
            key=lambda row: abs(_safe_float(row.get("_delta_u_value")) or 0.0),
            reverse=True,
        )
        top = ordered[0]
        rq1_sentence = (
            f"Among the retrieval sources, {top['source']} produced the largest "
            f"absolute uncertainty shift in this run (Delta U={top['delta_u']})."
        )

    rq2_sentence = "RQ2 results were not available in this run."
    baseline = next((row for row in rq2_table if row.get("method_key") == "without"), None)
    opencoder = next((row for row in rq2_table if row.get("method_key") == "with"), None)
    if baseline and opencoder:
        rq2_sentence = (
            "Compared with the standard RAG baseline, OpenCoder obtained "
            f"pass@1={opencoder['pass@1']} versus {baseline['pass@1']}, "
            f"pass-rate variance={opencoder['pass_rate_variance']} versus "
            f"{baseline['pass_rate_variance']}, and ECE={opencoder['ece']} "
            f"versus {baseline['ece']}."
        )

    lines = [
        "# Draft Results Section Template",
        "",
        "Use this as prose scaffolding after replacing any pilot-only values with the final API-backed execution-test run.",
        "",
        f"Run setting: dataset `{metadata.get('dataset', '--')}`, backend `{metadata.get('backend', '--')}`, model `{metadata.get('model', '--')}`, n=`{metadata.get('limit', '--')}`.",
        "",
        f"Readiness note: {readiness}",
        "",
        "## RQ1 Paragraph",
        "",
        rq1_sentence,
        "Table~\\ref{tab:rq1_source_uncertainty} reports source-wise uncertainty changes. Figure~\\ref{fig:rq1_delta_uncertainty} visualizes the direction and magnitude of uncertainty shifts when each retrieval source is present versus absent.",
        "",
        "## RQ2 Paragraph",
        "",
        rq2_sentence,
        "Table~\\ref{tab:rq2_opencoder_ablation} compares the baseline RAG configuration with the uncertainty-aware OpenCoder configuration. Figure~\\ref{fig:rq2_pass_at_k} summarizes pass@k, while Figure~\\ref{fig:rq2_uncertainty_diagnostics} summarizes uncertainty and calibration diagnostics.",
        "",
        "## LaTeX Figure Snippets",
        "",
        "```latex",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/rq1_delta_uncertainty.pdf}",
        r"\caption{Source-wise change in aggregate uncertainty for RQ1.}",
        r"\label{fig:rq1_delta_uncertainty}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/rq2_pass_at_k.pdf}",
        r"\caption{Pass@k comparison between baseline RAG and OpenCoder for RQ2.}",
        r"\label{fig:rq2_pass_at_k}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/rq2_uncertainty_diagnostics.pdf}",
        r"\caption{Uncertainty and calibration diagnostics for RQ2.}",
        r"\label{fig:rq2_uncertainty_diagnostics}",
        r"\end{figure}",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    *,
    out_dir: Path,
    rq1_path: Optional[str],
    rq2_path: Optional[str],
    rq1_data: Optional[Dict[str, Any]],
    rq2_data: Optional[Dict[str, Any]],
    run_config: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_config or {}
    rq1_table = build_rq1_table(rq1_data)
    rq2_table = build_rq2_table(rq2_data)

    rq1_columns = [
        "source",
        "mean_u_present",
        "mean_u_absent",
        "delta_u",
        "spearman_rho",
        "spearman_p",
        "pass_variance_present",
        "pass_variance_absent",
        "delta_pass_variance",
    ]
    rq1_headers = [
        "Source",
        "Mean U present",
        "Mean U absent",
        "Delta U",
        "rho",
        "p",
        "Var present",
        "Var absent",
        "Delta var",
    ]
    rq2_columns = [
        "method",
        "pass@1",
        "pass@3",
        "pass@5",
        "pass_rate_variance",
        "ece",
        "mean_uncertainty",
        "mean_repair_rounds",
        "n",
        "n_known_correctness",
    ]
    rq2_headers = [
        "Method",
        "pass@1",
        "pass@3",
        "pass@5",
        "Pass var.",
        "ECE",
        "Mean U",
        "Repairs",
        "n",
        "Known",
    ]

    if rq1_table:
        _write_csv(out_dir / "rq1_source_table.csv", rq1_table, rq1_columns)
        _write_latex_table(
            out_dir / "rq1_source_table.tex",
            rq1_table,
            rq1_columns,
            rq1_headers,
            "RQ1 source-wise relationship between retrieved evidence and uncertainty.",
            "tab:rq1_source_uncertainty",
        )
        _bar_svg(
            out_dir / "rq1_delta_uncertainty.svg",
            title="RQ1 Source Effect on Aggregate Uncertainty",
            rows=rq1_table,
            label_key="source",
            value_key="_delta_u_value",
            x_label="Delta uncertainty: present minus absent",
        )
        _grouped_source_uncertainty_svg(out_dir / "rq1_uncertainty_by_source.svg", rq1_table)
    if rq2_table:
        _write_csv(out_dir / "rq2_method_table.csv", rq2_table, rq2_columns)
        _write_latex_table(
            out_dir / "rq2_method_table.tex",
            rq2_table,
            rq2_columns,
            rq2_headers,
            "RQ2 baseline RAG versus uncertainty-aware OpenCoder.",
            "tab:rq2_opencoder_ablation",
        )
        _grouped_pass_svg(out_dir / "rq2_pass_at_k.svg", rq2_table)
        _grouped_rq2_diagnostics_svg(out_dir / "rq2_uncertainty_diagnostics.svg", rq2_table)

    modes = _correctness_modes_from_rq1(rq1_data)
    for key, value in _correctness_modes_from_rq2(rq2_data).items():
        modes[key] = modes.get(key, 0) + value
    metadata = _first_metadata(run_config, rq1_data, rq2_data)
    readiness = _paper_readiness(metadata, modes)

    report_summary = {
        "metadata": metadata,
        "paper_readiness": readiness,
        "correctness_modes": modes,
        "rq1_table": rq1_table,
        "rq2_table": rq2_table,
    }
    (out_dir / "report_summary.json").write_text(
        json.dumps(report_summary, indent=2),
        encoding="utf-8",
    )
    _write_paper_results_section(
        out_dir / "paper_results_section.md",
        metadata=metadata,
        readiness=readiness,
        rq1_table=rq1_table,
        rq2_table=rq2_table,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# OpenCoder Experiment Report",
        "",
        f"Generated: {now}",
        "",
        "Paper: *Beyond \"What to Retrieve\": Uncertainty in Retrieval-Augmented Code Generation*",
        "",
        "## Inputs",
        "",
        f"- RQ1 JSON: `{rq1_path or 'not provided'}`",
        f"- RQ2 JSON: `{rq2_path or 'not provided'}`",
        f"- Dataset: `{metadata.get('dataset', '--')}`",
        f"- Backend/model: `{metadata.get('backend', '--')}` / `{metadata.get('model', '--')}`",
        f"- Limit / describe limit: `{metadata.get('limit', '--')}` / `{metadata.get('describe_limit', '--')}`",
        f"- Correctness modes observed: {_format_counts(modes)}",
        f"- Paper readiness: {readiness}",
        "",
        "Important interpretation note: when `correctness_mode=reference_exact_match`, pass@k and pass-rate variance are based on normalized exact match against the reference code, not execution tests. AAAI-ready execution claims require repository test harnesses or normalized executable tests.",
        "",
        "## RQ1: Retrieval Source Influence",
        "",
        _markdown_table(rq1_table, rq1_columns, rq1_headers),
        "",
        "Figures: `rq1_delta_uncertainty.svg`, `rq1_uncertainty_by_source.svg`",
        "",
        "## RQ2: Uncertainty-Aware OpenCoder",
        "",
        _markdown_table(rq2_table, rq2_columns, rq2_headers),
        "",
        "Figures: `rq2_pass_at_k.svg`, `rq2_uncertainty_diagnostics.svg`",
        "",
        "## Files for Paper Draft",
        "",
        "- `rq1_source_table.csv` and `rq1_source_table.tex`",
        "- `rq2_method_table.csv` and `rq2_method_table.tex`",
        "- `rq1_delta_uncertainty.svg` and `rq1_uncertainty_by_source.svg`",
        "- `rq2_pass_at_k.svg` and `rq2_uncertainty_diagnostics.svg`",
        "- `report_summary.json`",
        "- `paper_results_section.md`",
        "",
    ]
    report_path = out_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", default=None, help="Path to RQ1 JSON output.")
    ap.add_argument("--rq2", default=None, help="Path to RQ2 JSON output.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-config", default=None, help="Optional run_config.json path.")
    args = ap.parse_args()

    rq1_data = _load_json(args.rq1)
    rq2_data = _load_json(args.rq2)
    if not rq1_data and not rq2_data:
        raise SystemExit("Provide --rq1, --rq2, or both.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.rq2:
        out_dir = Path(args.rq2).resolve().parent
    elif args.rq1:
        out_dir = Path(args.rq1).resolve().parent
    else:
        out_dir = Path("results")
    run_config = _load_run_config(out_dir, args.run_config)

    report_path = build_report(
        out_dir=out_dir,
        rq1_path=args.rq1,
        rq2_path=args.rq2,
        rq1_data=rq1_data,
        rq2_data=rq2_data,
        run_config=run_config,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
