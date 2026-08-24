#!/usr/bin/env python3
"""Freeze a multilingual native-completion subset of CrossCodeEval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHIVE_SHA256 = "d65c0316f63df3434deac3b67ae95478cbe00c706c14ff1a91e1173619962b88"
LANGUAGES = ("python", "java", "typescript", "csharp")
TASK_FILE = "line_completion_rg1_unixcoder_cosine_sim.jsonl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    *,
    target: int,
    max_per_repository: int,
) -> list[dict[str, Any]]:
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for row in sorted(rows, key=lambda item: str(item["artifact_hash"])):
        repository = str(row["repository"])
        if counts[repository] >= max_per_repository:
            continue
        selected.append(row)
        counts[repository] += 1
        if len(selected) == target:
            return selected
    raise ValueError(f"only {len(selected)} tasks satisfy the predeclared repository cap")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".benchmarks/crosscodeeval/data")
    parser.add_argument("--per-language", type=int, default=25)
    parser.add_argument("--max-per-repository", type=int, default=2)
    parser.add_argument("--manifest", default="data/manifests/crosscodeeval_opencoderx_100_v1.jsonl")
    parser.add_argument("--audit", default="results/data_quality/crosscodeeval_subset_audit.csv")
    parser.add_argument("--report", default="results/data_quality/crosscodeeval_freeze.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    archive = data_root / "crosscodeeval_data.tar.xz"
    if not archive.is_file() or _sha256_file(archive) != ARCHIVE_SHA256:
        raise SystemExit("CrossCodeEval archive checksum mismatch")

    all_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    file_hashes: dict[str, str] = {}
    for language in LANGUAGES:
        source = data_root / language / TASK_FILE
        file_hash = _sha256_file(source)
        file_hashes[language] = file_hash
        language_rows = []
        with source.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                task_id = str(metadata.get("task_id") or f"{language}-{row_index}")
                repository = str(metadata.get("repository") or "unknown")
                identity = json.dumps(
                    {
                        "archive_sha256": ARCHIVE_SHA256,
                        "language": language,
                        "row_index": row_index,
                        "task_id": task_id,
                        "prompt": record.get("prompt"),
                        "groundtruth": record.get("groundtruth"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                artifact_hash = hashlib.sha256(identity).hexdigest()
                retrieval = record.get("crossfile_context") or {}
                retrieved_chunks = retrieval.get("list") if isinstance(retrieval, dict) else []
                groundtruth = str(record.get("groundtruth") or "")
                exact_overlap = bool(
                    len(groundtruth.strip()) >= 20
                    and any(
                        groundtruth.strip() in str(chunk.get("retrieved_chunk") or "")
                        for chunk in (retrieved_chunks or [])
                        if isinstance(chunk, dict)
                    )
                )
                row = {
                    "task_id": task_id,
                    "language": language,
                    "row_index": row_index,
                    "repository": repository,
                    "file": str(metadata.get("file") or ""),
                    "artifact_hash": artifact_hash,
                    "source_file_sha256": file_hash,
                    "long_reference_exact_retrieval_overlap": exact_overlap,
                    "selected": False,
                    "record": record,
                }
                language_rows.append(row)
                all_rows.append(row)
        selected = select_rows(
            language_rows,
            target=args.per_language,
            max_per_repository=args.max_per_repository,
        )
        selected_ids.update(str(row["artifact_hash"]) for row in selected)

    manifest_rows = []
    for row in sorted(all_rows, key=lambda item: (item["language"], item["artifact_hash"])):
        row["selected"] = row["artifact_hash"] in selected_ids
        if not row["selected"]:
            continue
        record = row["record"]
        manifest_rows.append({
            "manifest_version": "crosscodeeval_opencoderx_100_v1",
            "task_id": row["task_id"],
            "language": row["language"],
            "repository": row["repository"],
            "file": row["file"],
            "upstream_row_index": row["row_index"],
            "prompt": record.get("prompt"),
            "right_context": record.get("right_context"),
            "reference_completion": record.get("groundtruth"),
            "crossfile_context": record.get("crossfile_context"),
            "native_metrics": ["exact_match", "edit_similarity", "identifier_f1"],
            "functional_execution": False,
            "selection_rule": {
                "order": "ascending pre-output artifact_hash",
                "tasks_per_language": args.per_language,
                "maximum_tasks_per_repository_per_language": args.max_per_repository,
            },
            "provenance": {
                "source": "amazon-science/cceval",
                "archive_sha256": ARCHIVE_SHA256,
                "source_file_sha256": row["source_file_sha256"],
                "artifact_hash": row["artifact_hash"],
                "retrieval_setting": "rg1_unixcoder_cosine_sim",
            },
        })

    if len(manifest_rows) != args.per_language * len(LANGUAGES):
        raise SystemExit("failed to select the exact frozen CrossCodeEval subset")
    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit)
    report_path = Path(args.report)
    for path in (manifest_path, audit_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit_rows = [
        {key: value for key, value in row.items() if key != "record"}
        for row in all_rows
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    report = {
        "status": "FROZEN_NATIVE_COMPLETION",
        "tasks": len(manifest_rows),
        "tasks_per_language": args.per_language,
        "languages": list(LANGUAGES),
        "functional_execution": False,
        "long_reference_exact_retrieval_overlaps_selected": sum(
            bool(row["long_reference_exact_retrieval_overlap"])
            for row in all_rows if row["selected"]
        ),
        "archive_sha256": ARCHIVE_SHA256,
        "source_file_sha256": file_hashes,
        "manifest_sha256": _sha256_file(manifest_path),
        "audit_sha256": _sha256_file(audit_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
