"""Aggregate measured, paper-eligible external-baseline run artifacts."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _generation_integrity_valid(row: dict) -> bool:
    recorded = row.get("generation_integrity") or {}
    if recorded.get("valid") is True:
        return True
    candidates = list(row.get("candidates") or [])
    raw_responses = list(row.get("candidate_raw_responses") or [])
    response_metadata = list(row.get("candidate_response_metadata") or [])
    if not candidates or any(not candidate.strip() for candidate in candidates):
        return False
    unclosed = [
        bool(re.match(r"^\s*```(?:python)?", response))
        and _FENCE.search(response) is None
        for response in raw_responses
    ]
    if not any(unclosed):
        return True
    if len(response_metadata) != len(raw_responses):
        return False
    return all(
        not is_unclosed or response_metadata[index].get("finish_reason") == "length"
        for index, is_unclosed in enumerate(unclosed)
    )


def _protocol_valid(row: dict) -> bool:
    if "error" in row or not row.get("faithful_full_api_descriptions"):
        return False
    index = row.get("index") or {}
    description_sources = set((index.get("description_sources") or {}).keys())
    return (
        int(index.get("n_items") or 0) > 0
        and int(index.get("n_description_errors") or 0) == 0
        and description_sources.issubset({"llm"})
        and _generation_integrity_valid(row)
    )


def _usage(rows: list[dict], key: str) -> int:
    return sum(int((row.get("llm_usage") or {}).get("total", {}).get(key, 0)) for row in rows)


def load_run(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    if not metadata.get("paper_eligible"):
        raise ValueError(f"{path}: smoke/offline artifact is not paper eligible")
    rows = list(payload.get("rows") or [])
    expected = int(metadata.get("expected_n_tasks") or 0)
    if expected and len(rows) != expected:
        raise ValueError(f"{path}: partial campaign ({len(rows)}/{expected} tasks)")
    invalid = [
        row.get("id")
        for row in rows
        if "error" not in row
        and not _protocol_valid(row)
    ]
    if invalid:
        raise ValueError(f"{path}: protocol-invalid tasks: {invalid}")
    successful = [row for row in rows if "error" not in row]
    summary = payload.get("summary") or {}
    return {
        "backend": metadata.get("backend"),
        "model": metadata.get("model"),
        "benchmark": metadata.get("benchmark"),
        "qualification": (
            "context-limited"
            if metadata.get("benchmark") == "execrepobench"
            else "matched local subset"
        ),
        "method": metadata.get("method"),
        "n_tasks": len(rows),
        "n_successful": len(successful),
        "n_failures": len(rows) - len(successful),
        "pass@1": summary.get("pass@1"),
        "pass@3": summary.get("pass@3"),
        "pass@5": summary.get("pass@5"),
        "requests": _usage(successful, "requests"),
        "prompt_tokens": _usage(successful, "prompt_tokens"),
        "completion_tokens": _usage(successful, "completion_tokens"),
        "total_tokens": _usage(successful, "total_tokens"),
        "source": path,
    }


def _pct(value) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failures(path: Path, inputs: list[str]) -> None:
    fields = ["source", "backend", "benchmark", "task_id", "error", "requests", "total_tokens"]
    rows = []
    for source in inputs:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or {}
        for row in payload.get("rows") or []:
            if "error" not in row:
                continue
            usage = row.get("llm_usage_at_failure") or {}
            rows.append({
                "source": source,
                "backend": metadata.get("backend"),
                "benchmark": metadata.get("benchmark"),
                "task_id": row.get("id"),
                "error": row.get("error"),
                "requests": usage.get("requests", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Measured clean-room AllianceCoder reproduction under the common OpenCoder executable protocol. Values are percentages; costs report observed API usage.}",
        r"\label{tab:external_alliancecoder}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Backend & Benchmark & $N$ & Pass@1 & Pass@3 & Pass@5 & Requests & Tokens & Fail. \\",
        r"\midrule",
    ]
    for row in rows:
        backend = str(row["backend"]).replace("_", r"\_")
        benchmark_names = {
            "repoexec": "RepoExec",
            "codereval": "CoderEval",
            "execrepobench": r"ExecRepoBench$^{\dagger}$",
        }
        benchmark = benchmark_names.get(
            str(row["benchmark"]),
            str(row["benchmark"]).replace("_", r"\_"),
        )
        lines.append(
            f"{backend} & {benchmark} & {row['n_tasks']} & {_pct(row['pass@1'])} & "
            f"{_pct(row['pass@3'])} & {_pct(row['pass@5'])} & {row['requests']} & "
            f"{row['total_tokens']} & {row['n_failures']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize $^{\dagger}$Context-limited comparison because complete repository snapshots are unavailable.}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out-dir", default="results/external_baseline/summary")
    args = parser.parse_args()
    inputs = args.inputs or sorted(glob.glob("results/external_baseline/runs/**/*.json", recursive=True))
    if not inputs:
        raise SystemExit("No external-baseline run artifacts found.")
    rows = sorted((load_run(path) for path in inputs), key=lambda row: (str(row["backend"]), str(row["benchmark"])))
    out = Path(args.out_dir)
    write_csv(out / "summary.csv", rows)
    write_failures(out / "failures.csv", inputs)
    write_tex(out / "table.tex", rows)
    print(f"Aggregated {len(inputs)} measured runs into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
