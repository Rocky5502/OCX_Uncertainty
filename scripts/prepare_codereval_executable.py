"""Build a mutation-audited executable CoderEval subset for the Neo4j tasks."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.phase5_verify.test_validate import run_codereval_project_tests  # noqa: E402


SNAPSHOT_COMMIT = "e29e042b5038e59f2bf2d0b57ff842ff51538faf"
TEST_SELECTORS = {
    "neo4j/id0": ["tests/unit/common/codec/hydration/v1/test_temporal_hydration.py"],
    "neo4j/id1": ["tests/unit/common/codec/hydration/v1/test_temporal_dehydration.py::TestTimeDehydration"],
    "neo4j/id2": ["tests/unit/common/codec/hydration/v1/test_temporal_dehydration.py::TestTimeDehydration"],
    "neo4j/id3": ["tests/unit/common/codec/hydration/v1/test_spacial_dehydration.py"],
    "neo4j/id4": ["tests/unit/common/test_record.py"],
    "neo4j/id5": ["tests/unit/sync/io/test_class_bolt.py"],
    "neo4j/id6": ["tests/unit/sync/work/test_session.py::test_decorated_tx_function_argument_type"],
    "neo4j/id7": ["tests/unit/common/test_record.py"],
    "neo4j/id8": ["tests/unit/common/test_record.py"],
    "neo4j/id9": ["tests/unit/common/test_record.py"],
    "neo4j/id10": [
        "tests/unit/sync/io/test_class_bolt3.py",
        "tests/unit/async_/io/test_class_bolt3.py",
    ],
    "neo4j/id11": ["tests/unit/async_/io/test_class_bolt3.py"],
    "neo4j/id12": ["tests/unit/async_/io/test_class_bolt3.py"],
    "neo4j/id13": ["tests/unit/common/time/test_time.py"],
    "neo4j/id14": [
        "tests/unit/common/test_types.py",
    ],
    "neo4j/id15": ["tests/unit/common/test_api.py"],
    "neo4j/id16": ["tests/unit/common/time/test_time.py"],
    "neo4j/id17": ["tests/unit/async_/io/test_class_bolt.py"],
    "neo4j/id18": ["tests/unit/common/test_api.py"],
}


def _mutation(reference_code: str) -> str:
    tree = ast.parse(textwrap.dedent(reference_code))
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        raise ValueError("reference does not contain exactly one function")
    functions[0].body = [
        ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="AssertionError", ctx=ast.Load()),
                args=[ast.Constant(value="mutation gate")],
                keywords=[],
            ),
            cause=None,
        )
    ]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _head(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="empirical_study/API/input/input.jsonl")
    parser.add_argument("--dataset", default="empirical_study/API/CoderEval4Python.json")
    parser.add_argument("--project-root", default="external/codereval/neo4j-python-driver")
    parser.add_argument("--out", default="input/codereval_neo4j_executable19.jsonl")
    parser.add_argument("--audit", default="results/codereval_neo4j_harness_audit.json")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    head = _head(project_root)
    if head != SNAPSHOT_COMMIT:
        raise SystemExit(f"Expected CoderEval snapshot {SNAPSHOT_COMMIT}, found {head}")

    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["RECORDS"]
    records_by_id = {record["_id"]: record for record in records}
    input_rows = [
        json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prepared = []
    audit_rows = []
    for row in input_rows:
        metadata = row.get("metadata") or {}
        task_id = str(metadata.get("task_id"))
        record = records_by_id[str(metadata.get("_id"))]
        target_file = str(record["file_path"])
        reference_code = str(record["code"])
        snapshot_source = (project_root / target_file).read_text(encoding="utf-8")
        exact_snapshot = snapshot_source == str(record["file_content"])
        raw = dict(row)
        raw.update({
            "project_path": str(project_root),
            "codereval_project_tests": True,
            "codereval_project_root": str(project_root),
            "codereval_snapshot_commit": SNAPSHOT_COMMIT,
            "codereval_record_id": record["_id"],
            "codereval_target_file": target_file,
            "codereval_reference_code": reference_code,
            "codereval_test_selectors": TEST_SELECTORS[task_id],
        })
        reference_report = run_codereval_project_tests(reference_code, raw, timeout=args.timeout)
        mutation_report = run_codereval_project_tests(_mutation(reference_code), raw, timeout=args.timeout)
        accepted = exact_snapshot and reference_report.passed is True and mutation_report.passed is False
        audit_rows.append({
            "task_id": task_id,
            "target_file": target_file,
            "function": record["name"],
            "selectors": TEST_SELECTORS[task_id],
            "exact_snapshot": exact_snapshot,
            "reference_passed": reference_report.passed,
            "reference_returncode": reference_report.returncode,
            "mutation_detected": mutation_report.passed is False,
            "mutation_returncode": mutation_report.returncode,
            "accepted": accepted,
            "reference_output": (reference_report.stdout + reference_report.stderr)[-1000:],
            "mutation_output": (mutation_report.stdout + mutation_report.stderr)[-1000:],
        })
        if accepted:
            prepared.append(raw)
        print(
            f"{task_id:<12} snapshot={exact_snapshot} "
            f"reference={reference_report.passed} mutation_detected={mutation_report.passed is False}",
            flush=True,
        )

    audit = {
        "snapshot_commit": SNAPSHOT_COMMIT,
        "project_root": str(project_root),
        "n_input": len(input_rows),
        "n_accepted": len(prepared),
        "all_reference_passed": all(row["reference_passed"] is True for row in audit_rows),
        "all_mutations_detected": all(row["mutation_detected"] for row in audit_rows),
        "rows": audit_rows,
    }
    out_path = Path(args.out)
    audit_path = Path(args.audit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row) + "\n" for row in prepared), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("n_input", "n_accepted", "all_reference_passed", "all_mutations_detected")}, indent=2))
    return 0 if len(prepared) == len(input_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
