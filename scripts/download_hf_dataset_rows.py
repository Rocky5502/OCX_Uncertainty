#!/usr/bin/env python3
"""Download complete Hugging Face dataset rows through datasets-server."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {output}")

    completed = 0
    if temporary.exists():
        completed = sum(1 for line in temporary.read_text(encoding="utf-8").splitlines() if line.strip())

    with temporary.open("a", encoding="utf-8") as handle:
        for offset in range(completed, args.rows, args.page_size):
            length = min(args.page_size, args.rows - offset)
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response = requests.get(
                        "https://datasets-server.huggingface.co/rows",
                        params={
                            "dataset": args.dataset,
                            "config": args.config,
                            "split": args.split,
                            "offset": offset,
                            "length": length,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                    if attempt == 3:
                        raise
                    time.sleep(attempt * 2)
            else:  # pragma: no cover
                raise RuntimeError(last_error)

            rows = payload.get("rows") or []
            if len(rows) != length:
                raise RuntimeError(f"expected {length} rows at {offset}, received {len(rows)}")
            for position, wrapped in enumerate(rows, start=offset):
                if int(wrapped.get("row_idx", -1)) != position:
                    raise RuntimeError(f"row index mismatch at {position}")
                if wrapped.get("truncated_cells"):
                    raise RuntimeError(f"truncated cells at row {position}")
                handle.write(json.dumps(wrapped["row"], ensure_ascii=False) + "\n")
            handle.flush()
            print(f"downloaded {offset + length}/{args.rows}")

    observed = sum(1 for line in temporary.read_text(encoding="utf-8").splitlines() if line.strip())
    if observed != args.rows:
        raise RuntimeError(f"expected {args.rows} rows, found {observed}")
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"sha256:{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
