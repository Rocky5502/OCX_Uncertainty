"""Filter saved RQ1/RQ2 experiment JSON files to a stable example subset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_rq1 import _summarize as summarize_rq1  # noqa: E402
from scripts.ablation_rq2 import _summarize as summarize_rq2  # noqa: E402


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _filter_rq1(src: str, dst: str, ids: set[str]) -> None:
    data = _load(src)
    rows = [r for r in data.get("rows", []) if r.get("example_id") in ids]
    metadata = dict(data.get("metadata", {}))
    metadata["filtered_ids"] = sorted(ids)
    metadata["filtered_n_examples"] = len({r.get("example_id") for r in rows})
    payload = {
        "metadata": metadata,
        "rows": rows,
        "summary": summarize_rq1(rows),
    }
    _write(dst, payload)


def _filter_rq2(src: str, dst: str, ids: set[str]) -> None:
    data = _load(src)
    out = {}
    for key, value in data.items():
        if key in {"with", "without"} and isinstance(value, list):
            out[key] = [r for r in value if r.get("id") in ids]
    metadata = dict(data.get("metadata", {}))
    metadata["filtered_ids"] = sorted(ids)
    metadata["filtered_n_examples"] = len({
        r.get("id")
        for rows in out.values()
        for r in rows
    })
    out["summary"] = summarize_rq2(out)
    out["metadata"] = metadata
    _write(dst, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="Comma-separated example ids to keep.")
    ap.add_argument("--rq1-in")
    ap.add_argument("--rq1-out")
    ap.add_argument("--rq2-in")
    ap.add_argument("--rq2-out")
    args = ap.parse_args()

    ids = {x.strip() for x in args.ids.split(",") if x.strip()}
    if args.rq1_in and args.rq1_out:
        _filter_rq1(args.rq1_in, args.rq1_out, ids)
        print(f"Wrote {args.rq1_out}")
    if args.rq2_in and args.rq2_out:
        _filter_rq2(args.rq2_in, args.rq2_out, ids)
        print(f"Wrote {args.rq2_out}")


if __name__ == "__main__":
    main()
