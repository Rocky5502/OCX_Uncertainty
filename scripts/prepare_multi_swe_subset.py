#!/usr/bin/env python3
"""Freeze a 35-task multilingual Multi-SWE-bench Flash subset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_REVISION = "b0485dbebaf8a1317ebf140e80e6fc6c02d3502b"
DATASET_SHA256 = "48d6d02cc976a71a06b494cc60581d92e82c06c2793c0d412c52c63e6956bebe"
LANGUAGES = ("java", "typescript", "javascript", "go", "rust", "c", "c++")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(result: Any, key: str) -> int:
    return int((result or {}).get(key) or 0)


def eligible(record: dict[str, Any]) -> tuple[bool, str]:
    run_fail = _count(record.get("run_result"), "failed_count")
    test_fail = _count(record.get("test_patch_result"), "failed_count")
    fix_fail = _count(record.get("fix_patch_result"), "failed_count")
    if test_fail <= run_fail:
        return False, "test_patch_does_not_expose_additional_failure"
    if fix_fail > run_fail:
        return False, "gold_fix_does_not_restore_baseline_failure_count"
    if not str(record.get("fix_patch") or "").strip():
        return False, "missing_gold_fix_patch"
    if not str(record.get("base", {}).get("sha") or "").strip():
        return False, "missing_base_commit"
    return True, "eligible_official_flash_instance"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".benchmarks/multi-swe-bench/data/multi_swe_bench_flash.jsonl")
    parser.add_argument("--per-language", type=int, default=5)
    parser.add_argument("--manifest", default="data/manifests/multi_swe_bench_flash_opencoderx_35_v1.jsonl")
    parser.add_argument("--audit", default="results/data_quality/multi_swe_subset_audit.csv")
    parser.add_argument("--report", default="results/data_quality/multi_swe_freeze.json")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file() or _sha256_file(source) != DATASET_SHA256:
        raise SystemExit("Multi-SWE-bench Flash checksum mismatch")
    rows = []
    with source.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = str(record.get("instance_id") or f"row-{row_index}")
            identity = json.dumps(
                {
                    "dataset_revision": DATASET_REVISION,
                    "row_index": row_index,
                    "instance_id": instance_id,
                    "base_sha": (record.get("base") or {}).get("sha"),
                    "title": record.get("title"),
                    "body": record.get("body"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            is_eligible, reason = eligible(record)
            rows.append({
                "row_index": row_index,
                "instance_id": instance_id,
                "language": str(record.get("language") or "").lower(),
                "repository": f"{record.get('org')}/{record.get('repo')}",
                "artifact_hash": hashlib.sha256(identity).hexdigest(),
                "eligible": is_eligible,
                "reason": reason,
                "selected": False,
                "record": record,
            })

    selected_ids = set()
    for language in LANGUAGES:
        candidates = sorted(
            (row for row in rows if row["language"] == language and row["eligible"]),
            key=lambda row: row["artifact_hash"],
        )
        if len(candidates) < args.per_language:
            raise SystemExit(f"only {len(candidates)} eligible {language} instances")
        selected_ids.update(row["instance_id"] for row in candidates[: args.per_language])

    manifest_rows = []
    for row in sorted(rows, key=lambda item: (item["language"], item["artifact_hash"])):
        row["selected"] = row["instance_id"] in selected_ids
        if not row["selected"]:
            continue
        record = row["record"]
        manifest_rows.append({
            "manifest_version": "multi_swe_bench_flash_opencoderx_35_v1",
            "instance_id": row["instance_id"],
            "language": row["language"],
            "organization": record.get("org"),
            "repository": record.get("repo"),
            "pull_request_number": record.get("number"),
            "base": record.get("base"),
            "prompt": {
                "title": record.get("title"),
                "body": record.get("body"),
            },
            "difficulty": record.get("difficulty"),
            "evaluation_only": {
                "prompt_visible": False,
                "test_patch": record.get("test_patch"),
                "reference_fix_patch": record.get("fix_patch"),
                "fixed_tests": record.get("fixed_tests"),
                "p2p_tests": record.get("p2p_tests"),
                "f2p_tests": record.get("f2p_tests"),
                "s2p_tests": record.get("s2p_tests"),
                "n2p_tests": record.get("n2p_tests"),
                "run_result": record.get("run_result"),
                "test_patch_result": record.get("test_patch_result"),
                "fix_patch_result": record.get("fix_patch_result"),
            },
            "execution": {
                "harness": "multi-swe-bench",
                "requires_docker": True,
                "status": "BLOCKED_DOCKER_UNAVAILABLE_ON_CURRENT_MACHINE",
            },
            "selection_rule": {
                "order": "ascending pre-output artifact_hash within language",
                "tasks_per_language": args.per_language,
                "gold_fix_restores_baseline_failure_count": True,
                "test_patch_exposes_additional_failure": True,
            },
            "provenance": {
                "dataset": "ByteDance-Seed/Multi-SWE-bench-flash",
                "dataset_revision": DATASET_REVISION,
                "dataset_sha256": DATASET_SHA256,
                "upstream_row_index": row["row_index"],
                "artifact_hash": row["artifact_hash"],
            },
        })

    expected = args.per_language * len(LANGUAGES)
    if len(manifest_rows) != expected:
        raise SystemExit(f"expected {expected} selected tasks, found {len(manifest_rows)}")
    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit)
    report_path = Path(args.report)
    for path in (manifest_path, audit_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit_rows = [{key: value for key, value in row.items() if key != "record"} for row in rows]
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    report = {
        "status": "MANIFEST_FROZEN_EXECUTION_BLOCKED",
        "tasks": len(manifest_rows),
        "tasks_per_language": args.per_language,
        "languages": list(LANGUAGES),
        "requires_docker": True,
        "docker_available": False,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": _sha256_file(manifest_path),
        "audit_sha256": _sha256_file(audit_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
