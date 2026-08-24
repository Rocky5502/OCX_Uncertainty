"""RQ4 API retrieval reliability runner.

This script replays API retrieval from existing RQ3 runs when available and
builds paper-facing RQ4 artifacts from raw prediction records. It deliberately
does not simulate unavailable reference-paper baselines.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import textwrap
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencoder.data.loaders import Example, load_dataset  # noqa: E402
from opencoder.phase1_repo_knowledge.extract import (  # noqa: E402
    RepoFunction,
    extract_sources_knowledge,
)
from opencoder.phase3_retrieval.api_refine import (  # noqa: E402
    canonical_api_name,
    normalize_api_name,
    refine_api_hit_records,
)
from opencoder.phase3_retrieval.score_filter import score_and_filter  # noqa: E402
from opencoder.pipeline import Pipeline, PipelineConfig  # noqa: E402


DEFAULT_RUNS = [
    {
        "benchmark": "RepoExec",
        "dataset": "repoexec",
        "dataset_path": "input/repoexec_python_string_utils_inline14.jsonl",
        "backend_label": "GPT",
        "config": "configs/rq3/gpt4o_mini.yaml",
        "rq3_path": "results/rq3/runs/repoexec_inline14_gpt/rq3.json",
    },
    {
        "benchmark": "RepoExec",
        "dataset": "repoexec",
        "dataset_path": "input/repoexec_python_string_utils_inline14.jsonl",
        "backend_label": "Gemini",
        "config": "configs/rq3/gemini_2_5_flash.yaml",
        "rq3_path": "results/rq3/runs/repoexec_inline14_gemini/rq3.json",
    },
    {
        "benchmark": "ExecRepoBench",
        "dataset": "execrepobench",
        "dataset_path": "input/execrepobench_testbacked.jsonl",
        "backend_label": "GPT",
        "config": "configs/rq3/gpt4o_mini.yaml",
        "rq3_path": "results/rq3/runs/execrepobench_testbacked10_gpt/rq3.json",
    },
    {
        "benchmark": "ExecRepoBench",
        "dataset": "execrepobench",
        "dataset_path": "input/execrepobench_testbacked.jsonl",
        "backend_label": "Gemini",
        "config": "configs/rq3/gemini_2_5_flash.yaml",
        "rq3_path": "results/rq3/runs/execrepobench_testbacked10_gemini/rq3.json",
    },
]

CODEREVAL_SPEC = {
    "benchmark": "CoderEval",
    "dataset_path": "input/codereval_neo4j_executable19.jsonl",
    "official_path": "empirical_study/API/CoderEval4Python.json",
}

CODEREVAL_BACKEND_SPECS = [
    {
        "backend_label": "GPT",
        "config": "configs/rq3/gpt4o_mini.yaml",
        "rq3_path": "results/rq3/codereval_exec19/replication_gpt/rq3.json",
    },
    {
        "backend_label": "Gemini",
        "config": "configs/rq3/gemini_2_5_flash.yaml",
        "rq3_path": "results/rq3/codereval_exec19/replication_gemini/rq3.json",
    },
]

METHODS = [
    "Baseline RAG",
    "OpenCoder-NoUncFilter",
    "OpenCoder-NoAPIRefine",
    "OpenCoder",
    "ContextOnly",
    "API-informed reference",
]

UNAVAILABLE_METHODS = [
    "AllianceCoder",
    "OpenCoder-NoAPIRepair",
]

BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "float",
    "format",
    "getattr",
    "hasattr",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "open",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
}


@dataclass
class GroundTruth:
    task_id: str
    benchmark: str
    api_names: List[str]
    api_norms: List[str]
    unresolved_calls: List[str]
    extraction_method: str
    notes: str = ""


def _jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_write(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if _safe_float(v) is not None]
    return sum(vals) / len(vals) if vals else None


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den else float(default)


def _parse_python(source: str) -> Optional[ast.AST]:
    candidates = [
        source,
        textwrap.dedent(source),
        "def __opencoder_missing_region__():\n" + textwrap.indent(source, "    "),
        "def __opencoder_missing_region__():\n" + textwrap.indent(textwrap.dedent(source), "    "),
    ]
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    return None


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return None


def extract_call_names(source: str) -> List[str]:
    tree = _parse_python(source or "")
    if tree is None:
        return []
    calls: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.append(name)
    return sorted(set(calls))


def _repo_item_aliases(items: Sequence[RepoFunction]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for item in items:
        names = [item.qualname]
        if item.kind != "imported_api":
            names.append(item.signature)
        if item.signature.startswith("class "):
            names.append(item.signature.replace("class ", "", 1))
        for name in names:
            norm = normalize_api_name(name)
            if norm:
                aliases.setdefault(norm, item.qualname)
    return aliases


def _qualified_api_name(value: str) -> str:
    """Normalize an API call while preserving receiver qualification."""
    value = str(value or "").strip().strip("`")
    if "(" in value:
        value = value.split("(", 1)[0]
    parts = [normalize_api_name(part) for part in value.split(".")]
    return ".".join(part for part in parts if part)


def resolve_repository_call(
    call: str,
    items: Sequence[RepoFunction],
    *,
    target_norm: str = "",
) -> Optional[Tuple[str, str]]:
    """Resolve a call only when its receiver is compatible with repo symbols.

    Matching every attribute call by its final component turns calls such as
    ``dt.timestamp()`` into false repository dependencies whenever an unrelated
    repository class defines a method named ``timestamp``. Direct calls,
    class-qualified calls, and ``self``/``cls`` method calls are resolvable.
    """
    simple_aliases = _repo_item_aliases(items)
    simple = normalize_api_name(call)
    if not simple or simple == target_norm or simple in BUILTIN_CALLS:
        return None

    qualified_aliases: Dict[str, Tuple[str, str]] = {}
    for item in items:
        qualified = _qualified_api_name(item.qualname)
        if qualified:
            qualified_aliases.setdefault(
                qualified,
                (normalize_api_name(item.qualname), item.qualname),
            )

    qualified_call = _qualified_api_name(call)
    if "." not in qualified_call:
        matched = simple_aliases.get(simple)
        return (simple, matched) if matched else None

    exact = qualified_aliases.get(qualified_call)
    if exact and exact[0] != target_norm:
        return exact

    receiver = qualified_call.rsplit(".", 1)[0]
    if receiver in {"self", "cls"}:
        matched = simple_aliases.get(simple)
        return (simple, matched) if matched else None
    return None


def _sources_for_example(pipe: Pipeline, example: Example) -> List[Tuple[str, str]]:
    if example.repo_root:
        return []
    return pipe._sources_from_example(example)


def _target_name(example: Example) -> str:
    raw = example.raw or {}
    for key in ("entry_point", "name"):
        if raw.get(key):
            return normalize_api_name(str(raw[key]))
    meta = raw.get("metadata") or {}
    for key in ("function_name", "entry_point"):
        if meta.get(key):
            return normalize_api_name(str(meta[key]))
    prompt = str(raw.get("target_function_prompt") or raw.get("prompt") or "")
    tree = _parse_python(prompt)
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return normalize_api_name(node.name)
    return ""


def _target_api_norms(example: Example) -> List[str]:
    """Return target identifiers that should not count as retrieved APIs."""
    raw = example.raw or {}
    names = {_target_name(example)}
    for key in ("entry_point", "name"):
        if raw.get(key):
            names.add(normalize_api_name(str(raw[key])))
    prompt_bits = [
        raw.get("target_function_prompt"),
        raw.get("instruction"),
        raw.get("prompt"),
    ]
    prefix = raw.get("prefix_code")
    if isinstance(prefix, str):
        prompt_bits.append(prefix[-3000:])
    meta = raw.get("metadata") or {}
    for key in ("function_name", "entry_point"):
        if meta.get(key):
            names.add(normalize_api_name(str(meta[key])))
    for text in prompt_bits:
        if not isinstance(text, str) or not text.strip():
            continue
        tree = _parse_python(text)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(normalize_api_name(node.name))
    return sorted(name for name in names if name)


def ground_truth_from_example(pipe: Pipeline, benchmark: str, example: Example) -> GroundTruth:
    raw = example.raw or {}
    reference = (
        example.reference_code
        or raw.get("middle_code")
        or raw.get("solution")
        or (raw.get("metadata") or {}).get("ground_truth")
        or ""
    )
    sources = _sources_for_example(pipe, example)
    items = extract_sources_knowledge(sources)
    target = _target_name(example)
    api_norms: Dict[str, str] = {}
    unresolved: List[str] = []
    for call in extract_call_names(reference):
        resolved = resolve_repository_call(call, items, target_norm=target)
        if resolved:
            norm, matched = resolved
            api_norms[norm] = matched
        elif normalize_api_name(call) not in BUILTIN_CALLS:
            unresolved.append(call)
    return GroundTruth(
        task_id=example.id,
        benchmark=benchmark,
        api_names=sorted(api_norms.values()),
        api_norms=sorted(api_norms),
        unresolved_calls=sorted(set(unresolved)),
        extraction_method="static_ast_repo_alias_match",
    )


def _parse_oracle_context(text: str) -> Tuple[List[str], List[str]]:
    out: Dict[str, List[str]] = {"apis": [], "classes": []}
    for key in out:
        match = re.search(rf'"{key}"\s*:\s*"(.*?)"', text or "")
        if not match:
            continue
        try:
            values = ast.literal_eval(match.group(1))
        except Exception:
            values = []
        out[key] = [str(v) for v in values if str(v).strip()]
    return out["apis"], out["classes"]


def _load_codereval_oracles(path: str) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("RECORDS") if isinstance(data, dict) else data
    return {str(row.get("_id")): row for row in rows or [] if row.get("_id")}


def ground_truth_from_codereval_record(example: Example, oracle: Optional[dict]) -> GroundTruth:
    if oracle:
        apis, classes = _parse_oracle_context(str(oracle.get("oracle_context") or ""))
        official_names = sorted(set(apis + classes))
        source = str(oracle.get("file_content") or example.raw.get("current_file") or "")
        items = extract_sources_knowledge([(str(oracle.get("file_path") or "current_file.py"), source)])
        aliases = _repo_item_aliases(items)
        target = _target_name(example)
        scoped: Dict[str, str] = {}
        unresolved: List[str] = []
        for name in official_names:
            norm = normalize_api_name(name)
            if not norm or norm == target or norm in BUILTIN_CALLS:
                continue
            matched = aliases.get(norm)
            if matched:
                scoped[norm] = matched
            else:
                unresolved.append(name)
        return GroundTruth(
            task_id=example.id,
            benchmark="CoderEval",
            api_names=sorted(scoped.values()),
            api_norms=sorted(scoped),
            unresolved_calls=sorted(unresolved),
            extraction_method="official_oracle_context_repository_scoped",
            notes=(
                f"Retained {len(scoped)} repository-scoped labels from "
                f"{len(official_names)} official API/class labels."
            ),
        )
    return GroundTruth(
        task_id=example.id,
        benchmark="CoderEval",
        api_names=[],
        api_norms=[],
        unresolved_calls=[],
        extraction_method="missing_oracle_context",
        notes="No matching CoderEval official oracle_context record found.",
    )


def load_codereval_examples(path: str, limit: Optional[int] = None) -> List[Example]:
    examples: List[Example] = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            prompt = str(rec.get("prompt") or "")
            current_file = str(rec.get("current_file") or "")
            query = (
                "Complete the Python function from this repository. "
                "Return the complete target function definition, including "
                "the original signature and docstring from the stub.\n\n"
                "# Function Stub\n"
                f"```python\n{prompt}\n```\n\n"
                "# Current File Context\n"
                f"```python\n{current_file[-5000:]}\n```"
            )
            raw = dict(rec)
            raw["metadata"] = meta
            if prompt:
                raw.setdefault("target_function_prompt", prompt)
            examples.append(
                Example(
                    id=str(meta.get("task_id") or meta.get("_id") or rec.get("_id") or i),
                    query=query,
                    repo_root=None,
                    reference_code=meta.get("ground_truth"),
                    test_code=None,
                    raw=raw,
                )
            )
    return examples


def load_rq3(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _cache_only_describe(self: Pipeline, items: List[Any]) -> List[Any]:
    """Reuse generated descriptions when cached; otherwise avoid new LLM calls."""
    cache_path = self._description_cache_path()
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    for item in items:
        key = self._knowledge_cache_key(item)
        item.metadata["description_cache_key"] = key
        cached = cache.get(key)
        if cached:
            item.description = cached.get("description")
            item.metadata["describe_logprobs"] = cached.get("describe_logprobs")
            item.metadata["describe_cached"] = True
        else:
            item.description = item.docstring or item.signature or item.qualname
            item.metadata["describe_cached"] = False
            item.metadata["description_fallback"] = "docstring_or_signature_cache_only"
    return items


def make_pipeline(config_path: str, *, allow_description_api: bool) -> Pipeline:
    pipe = Pipeline(PipelineConfig.from_yaml(config_path))
    if not allow_description_api:
        pipe._describe_with_cache = types.MethodType(_cache_only_describe, pipe)
    return pipe


def _row_by_id(rows: Sequence[dict]) -> Dict[str, dict]:
    return {str(row.get("id")): row for row in rows if row.get("id") is not None}


def _pseudo_steps(example: Example, api_weight: float = 1.0) -> List[dict]:
    return [
        {
            "step": "API retrieval from task prompt",
            "step_uncertainty": None,
            "intent": {"api": api_weight, "context": 0.0, "similar_code": 0.0},
            "sources": {"api": {"query": example.query}},
        }
    ]


def _step_query(step: dict) -> str:
    src = step.get("sources") or {}
    api = src.get("api") or {}
    return str(api.get("query") or step.get("step") or "")


def _candidate_record(candidate: Any, step_index: int, query: str, kept: bool) -> dict:
    item = candidate.hit.item
    return {
        "step_index": step_index,
        "query": query,
        "api_name": getattr(item, "qualname", ""),
        "api_norm": normalize_api_name(getattr(item, "qualname", "")),
        "signature": getattr(item, "signature", ""),
        "file_path": getattr(item, "file_path", ""),
        "kind": getattr(item, "kind", ""),
        "description": getattr(item, "description", None) or getattr(item, "docstring", None) or "",
        "retrieval_score": candidate.components.get("retrieval_score"),
        "source_weight": candidate.components.get("source_weight"),
        "knowledge_uncertainty": candidate.components.get("knowledge_uncertainty"),
        "final_score": candidate.final_score,
        "kept": kept,
    }


def replay_api_retrieval(
    *,
    pipe: Pipeline,
    retriever: Any,
    method: str,
    example: Example,
    rq3_row: Optional[dict],
    baseline_row: Optional[dict],
    gt: GroundTruth,
) -> dict:
    cfg = pipe.cfg
    passed = None
    correctness_mode = None
    uncertainty = None
    uncertainty_components: Dict[str, Any] = {}
    if method == "Baseline RAG":
        steps = (baseline_row or {}).get("per_step") or _pseudo_steps(example, api_weight=1.0 / 3.0)
        alpha, keep_q = 0.0, 1.0
        row = baseline_row
    elif method == "OpenCoder-NoUncFilter":
        steps = (rq3_row or {}).get("per_step") or _pseudo_steps(example, api_weight=1.0)
        alpha, keep_q = 0.0, 1.0
        row = rq3_row
    elif method == "OpenCoder-NoAPIRefine":
        steps = (rq3_row or {}).get("per_step") or _pseudo_steps(example, api_weight=1.0)
        alpha, keep_q = cfg.knowledge_uncertainty_alpha, cfg.keep_quantile
        row = rq3_row
    elif method == "OpenCoder":
        steps = (rq3_row or {}).get("per_step") or _pseudo_steps(example, api_weight=1.0)
        alpha, keep_q = cfg.knowledge_uncertainty_alpha, cfg.keep_quantile
        row = rq3_row
    elif method == "ContextOnly":
        steps = []
        alpha, keep_q = 0.0, 0.0
        row = rq3_row
    elif method == "API-informed reference":
        steps = []
        alpha, keep_q = 0.0, 0.0
        row = rq3_row
    else:
        raise ValueError(method)

    if row:
        passed = row.get("passed")
        correctness_mode = row.get("correctness_mode")
        uncertainty = ((row.get("u") or {}).get("aggregate"))
        uncertainty_components = row.get("uncertainty_components") or {}

    if method == "ContextOnly":
        return {
            "initial_apis": [],
            "filtered_apis": [],
            "final_apis": [],
            "api_hits": [],
            "uncertainty": uncertainty,
            "uncertainty_components": uncertainty_components,
            "task_passed": passed,
            "correctness_mode": correctness_mode,
            "api_knowledge_uncertainty": None,
        }
    if method == "API-informed reference":
        return {
            "initial_apis": gt.api_names,
            "filtered_apis": gt.api_names,
            "final_apis": gt.api_names,
            "api_hits": [],
            "uncertainty": None,
            "uncertainty_components": {},
            "task_passed": passed,
            "correctness_mode": correctness_mode,
            "api_knowledge_uncertainty": None,
        }

    initial: Dict[str, dict] = {}
    kept: Dict[str, dict] = {}
    hit_records: List[dict] = []
    for step_index, step in enumerate(steps, start=1):
        query = _step_query(step)
        weight = float((step.get("intent") or {}).get("api", 1.0))
        hits = retriever.search(query, top_k=cfg.api_top_k) if query else []
        all_candidates = score_and_filter(
            {"api": hits},
            {"api": weight},
            knowledge_uncertainty_alpha=0.0,
            keep_quantile=1.0,
        )
        filtered_candidates = score_and_filter(
            {"api": hits},
            {"api": weight},
            knowledge_uncertainty_alpha=alpha,
            keep_quantile=keep_q,
        )
        filtered_ids = {id(c.hit.item) for c in filtered_candidates}
        for cand in all_candidates:
            rec = _candidate_record(cand, step_index, query, id(cand.hit.item) in filtered_ids)
            norm = rec["api_norm"]
            if norm and (norm not in initial or float(rec["final_score"]) > float(initial[norm]["final_score"])):
                initial[norm] = rec
            if rec["kept"] and norm and (
                norm not in kept or float(rec["final_score"]) > float(kept[norm]["final_score"])
            ):
                kept[norm] = rec
            hit_records.append(rec)

    final = kept
    if method == "OpenCoder":
        refined_final_apis = refine_api_hit_records(
            hit_records,
            target_norms=_target_api_norms(example),
            kept_only=True,
            drop_private_methods=True,
        )
    else:
        refined_final_apis = [
            rec["api_name"] for rec in sorted(final.values(), key=lambda r: -float(r["final_score"]))
        ]
    return {
        "initial_apis": [rec["api_name"] for rec in sorted(initial.values(), key=lambda r: -float(r["final_score"]))],
        "filtered_apis": [rec["api_name"] for rec in sorted(kept.values(), key=lambda r: -float(r["final_score"]))],
        "final_apis": refined_final_apis,
        "api_hits": hit_records,
        "uncertainty": uncertainty,
        "uncertainty_components": uncertainty_components,
        "task_passed": passed,
        "correctness_mode": correctness_mode,
        "api_knowledge_uncertainty": _mean(r.get("knowledge_uncertainty") for r in hit_records),
        "api_refinement": (
            {
                "name": "target_aware_constructor_private_filter",
                "target_norms": _target_api_norms(example),
                "drop_private_methods": True,
            }
            if method == "OpenCoder"
            else None
        ),
    }


def set_metrics(pred_norms: Sequence[str], gt_norms: Sequence[str]) -> dict:
    pred = {p for p in pred_norms if p}
    gt = {g for g in gt_norms if g}
    tp = len(pred & gt)
    precision = 1.0 if not pred and not gt else _safe_div(tp, len(pred), 0.0)
    recall = 1.0 if not pred and not gt else _safe_div(tp, len(gt), 0.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, 0.0)
    union = pred | gt
    jaccard = 1.0 if not union else len(pred & gt) / len(union)
    if len(pred) > len(gt):
        outcome = "Over"
    elif len(pred) == len(gt):
        outcome = "Exact"
    else:
        outcome = "Under"
    return {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "true_positive_count": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact_set_match": float(pred == gt),
        "count_outcome": outcome,
    }


def _record_to_metrics(record: dict) -> dict:
    pred_norms = [normalize_api_name(x) for x in record.get("final_apis") or []]
    gt_norms = list(record.get("ground_truth_api_norms") or [])
    metrics = set_metrics(pred_norms, gt_norms)
    return {
        **{
            "benchmark": record["benchmark"],
            "backend": record["backend_label"],
            "model": record["model"],
            "method": record["method"],
            "task_id": record["task_id"],
            "task_passed": record.get("task_passed"),
            "correctness_mode": record.get("correctness_mode"),
            "uncertainty": record.get("uncertainty"),
            "api_knowledge_uncertainty": record.get("api_knowledge_uncertainty"),
        },
        **metrics,
    }


def _format_pct(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value) * 100:.{digits}f}"
    except Exception:
        return "--"


def _format_num(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (den_x * den_y) if den_x and den_y else None


def auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg)) if pos and neg else None


def auprc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if len(labels) < 2 or sum(labels) == 0:
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    total_pos = sum(labels)
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)
    area = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        area += p * max(0.0, r - prev_recall)
        prev_recall = r
    return area


def ece_binary(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> Optional[float]:
    if not labels or len(labels) != len(scores):
        return None
    total = len(labels)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i
            for i, s in enumerate(scores)
            if (lo <= s < hi) or (b == bins - 1 and lo <= s <= hi)
        ]
        if not idx:
            continue
        avg_score = sum(scores[i] for i in idx) / len(idx)
        avg_label = sum(labels[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(avg_score - avg_label)
    return ece


def brier(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if not labels or len(labels) != len(scores):
        return None
    return sum((s - y) ** 2 for y, s in zip(labels, scores)) / len(labels)


def _api_item_uncertainty_rows(raw_records: Sequence[dict]) -> List[dict]:
    """Build label-free uncertainty features for each selected API item.

    An API is more uncertain when few independently generated retrieval steps
    support it. This consensus signal is invariant to backend-specific score
    scales. Ground-truth labels are attached only after feature construction
    for evaluation.
    """
    rows: List[dict] = []
    for record in raw_records:
        if record.get("method") != "OpenCoder":
            continue
        hits = [h for h in record.get("api_hits") or [] if h.get("kept")]
        by_api: Dict[str, List[dict]] = defaultdict(list)
        for hit in hits:
            canonical = canonical_api_name(str(hit.get("api_name") or ""))
            norm = normalize_api_name(canonical)
            if norm:
                by_api[norm].append(hit)
        max_score = max(
            [float(hit.get("final_score") or 0.0) for hit in hits] or [0.0]
        )
        gt = set(record.get("ground_truth_api_norms") or [])
        task_key = f"{record.get('benchmark')}|{record.get('task_id')}"
        split_hash = int(hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:8], 16)
        split = "test" if split_hash % 2 else "calibration"
        for api_name in record.get("final_apis") or []:
            norm = normalize_api_name(api_name)
            api_hits = by_api.get(norm) or []
            best_score = max(
                [float(hit.get("final_score") or 0.0) for hit in api_hits] or [0.0]
            )
            relative_score = best_score / max_score if max_score > 0 else 0.0
            relative_score = max(0.0, min(1.0, relative_score))
            support_steps = len({hit.get("step_index") for hit in api_hits})
            support_risk = 1.0 / max(1, support_steps)
            raw_uncertainty = support_risk
            rows.append(
                {
                    "benchmark": record.get("benchmark"),
                    "backend": record.get("backend_label"),
                    "task_id": record.get("task_id"),
                    "api_name": api_name,
                    "api_norm": norm,
                    "split": split,
                    "best_score": best_score,
                    "relative_score": relative_score,
                    "support_steps": support_steps,
                    "raw_api_uncertainty": raw_uncertainty,
                    "is_incorrect_api": int(norm not in gt),
                }
            )
    return rows


def calibrate_api_item_uncertainty(item_rows: List[dict]) -> List[dict]:
    """Calibrate API-item risk on one task split and predict the held-out split."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        LogisticRegression = None  # type: ignore[assignment]

    for backend in sorted({str(row["backend"]) for row in item_rows}):
        calibration = [
            row for row in item_rows
            if row["backend"] == backend and row["split"] == "calibration"
        ]
        labels = [int(row["is_incorrect_api"]) for row in calibration]
        model = None
        if LogisticRegression is not None and len(set(labels)) == 2:
            model = LogisticRegression(random_state=0)
            model.fit(
                [[float(row["raw_api_uncertainty"])] for row in calibration],
                labels,
            )
        for row in item_rows:
            if row["backend"] != backend:
                continue
            raw_score = float(row["raw_api_uncertainty"])
            row["calibrated_api_uncertainty"] = (
                float(model.predict_proba([[raw_score]])[0, 1])
                if model is not None
                else raw_score
            )
    return item_rows


def summarize_api_item_uncertainty(item_rows: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    test_rows = [row for row in item_rows if row.get("split") == "test"]
    for backend in sorted({str(row["backend"]) for row in test_rows}):
        backend_rows = [row for row in test_rows if row["backend"] == backend]
        groups = [("All", backend_rows)] + [
            (benchmark, [row for row in backend_rows if row["benchmark"] == benchmark])
            for benchmark in ("RepoExec", "CoderEval", "ExecRepoBench")
        ]
        for benchmark, rows in groups:
            if not rows:
                continue
            labels = [int(row["is_incorrect_api"]) for row in rows]
            raw_scores = [float(row["raw_api_uncertainty"]) for row in rows]
            calibrated = [float(row["calibrated_api_uncertainty"]) for row in rows]
            summaries.append(
                {
                    "backend": backend,
                    "benchmark": benchmark,
                    "split": "test",
                    "n_items": len(rows),
                    "n_errors": sum(labels),
                    "raw_auroc": auroc(labels, raw_scores),
                    "calibrated_auroc": auroc(labels, calibrated),
                    "calibrated_auprc": auprc(labels, calibrated),
                    "calibrated_ece": ece_binary(labels, calibrated),
                    "calibrated_brier": brier(labels, calibrated),
                }
            )
    return summaries


def audit_label_candidate_coverage(raw_records: Sequence[dict]) -> List[dict]:
    """Measure whether each ground-truth API exists in the indexed universe."""
    rows: List[dict] = []
    for record in raw_records:
        if record.get("method") != "OpenCoder":
            continue
        gt = set(record.get("ground_truth_api_norms") or [])
        candidates = set(record.get("candidate_api_norms") or [])
        covered = gt & candidates
        rows.append(
            {
                "benchmark": record.get("benchmark"),
                "backend": record.get("backend_label"),
                "task_id": record.get("task_id"),
                "ground_truth_count": len(gt),
                "covered_count": len(covered),
                "candidate_coverage": len(covered) / len(gt) if gt else None,
                "missing_ground_truth_apis": ";".join(sorted(gt - candidates)),
            }
        )
    return rows


def grouped(rows: Sequence[dict], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[dict]]:
    out: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k) for k in keys)].append(row)
    return out


def summarize_metrics(metric_rows: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    count_rows: List[dict] = []
    quality_rows: List[dict] = []
    uncertainty_rows: List[dict] = []
    success_rows: List[dict] = []

    for (benchmark, backend, method), rows in sorted(grouped(metric_rows, ["benchmark", "backend", "method"]).items()):
        n = len(rows)
        counts = {name: sum(1 for r in rows if r["count_outcome"] == name) for name in ("Over", "Exact", "Under")}
        count_rows.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "method": method,
                "n": n,
                "over": counts["Over"] / n if n else None,
                "exact": counts["Exact"] / n if n else None,
                "under": counts["Under"] / n if n else None,
            }
        )

        api_rows = [r for r in rows if int(r.get("gt_count") or 0) > 0]
        passed = [r for r in api_rows if str(r.get("task_passed")).lower() == "true"]
        failed = [r for r in api_rows if str(r.get("task_passed")).lower() == "false"]
        usable_unc = [
            r
            for r in api_rows
            if _safe_float(r.get("uncertainty")) is not None and r.get("method") == "OpenCoder"
        ]
        labels = [1 if float(r["recall"]) < 1.0 else 0 for r in usable_unc]
        scores = [float(r["uncertainty"]) for r in usable_unc]
        quality_rows.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "method": method,
                "n": n,
                "n_api_tasks": len(api_rows),
                "precision": _mean(r["precision"] for r in api_rows),
                "recall": _mean(r["recall"] for r in api_rows),
                "f1": _mean(r["f1"] for r in api_rows),
                "jaccard": _mean(r["jaccard"] for r in api_rows),
                "exact_set_match": _mean(r["exact_set_match"] for r in api_rows),
                "pass_recall": _mean(r["recall"] for r in passed),
                "fail_recall": _mean(r["recall"] for r in failed),
                "auroc": auroc(labels, scores) if usable_unc else None,
                "ece": ece_binary(labels, scores) if usable_unc else None,
            }
        )

        success_rows.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "method": method,
                "n_pass": len(passed),
                "n_fail": len(failed),
                "pass_precision": _mean(r["precision"] for r in passed),
                "pass_recall": _mean(r["recall"] for r in passed),
                "pass_f1": _mean(r["f1"] for r in passed),
                "fail_precision": _mean(r["precision"] for r in failed),
                "fail_recall": _mean(r["recall"] for r in failed),
                "fail_f1": _mean(r["f1"] for r in failed),
            }
        )

        if usable_unc:
            f1s = [float(r["f1"]) for r in usable_unc]
            outcome_means = {
                f"mean_uncertainty_{name.lower()}": _mean(
                    r["uncertainty"] for r in usable_unc if r["count_outcome"] == name
                )
                for name in ("Over", "Exact", "Under")
            }
            uncertainty_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": backend,
                    "method": method,
                    "n": len(usable_unc),
                    "spearman_uncertainty_api_f1": spearman(scores, f1s),
                    "auroc_incomplete_retrieval": auroc(labels, scores),
                    "auprc_incomplete_retrieval": auprc(labels, scores),
                    "ece_incomplete_retrieval": ece_binary(labels, scores),
                    "brier_incomplete_retrieval": brier(labels, scores),
                    **outcome_means,
                }
            )

    return count_rows, quality_rows, uncertainty_rows, success_rows


def aggregate_quality_for_table(quality_rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for (backend, method), rows in sorted(grouped(quality_rows, ["backend", "method"]).items()):
        out.append(
            {
                "backend": backend,
                "method": method,
                "precision": _mean(r["precision"] for r in rows),
                "recall": _mean(r["recall"] for r in rows),
                "f1": _mean(r["f1"] for r in rows),
                "exact_set_match": _mean(r["exact_set_match"] for r in rows),
                "pass_recall": _mean(r["pass_recall"] for r in rows),
                "fail_recall": _mean(r["fail_recall"] for r in rows),
                "auroc": _mean(r["auroc"] for r in rows if r["method"] == "OpenCoder"),
                "ece": _mean(r["ece"] for r in rows if r["method"] == "OpenCoder"),
            }
        )
    return out


def write_count_table(path: Path, count_rows: Sequence[dict]) -> None:
    backends = ["GPT", "Gemini"]
    methods = [
        "Baseline RAG",
        "OpenCoder-NoUncFilter",
        "OpenCoder-NoAPIRefine",
        "OpenCoder",
        "ContextOnly",
        "API-informed reference",
    ]
    benchmarks = ["RepoExec", "CoderEval", "ExecRepoBench"]
    lookup = {(r["backend"], r["method"], r["benchmark"]): r for r in count_rows}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Distribution of API retrieval count outcomes produced by OpenCoder and available ablations. Values are percentages over 14 RepoExec, 19 CoderEval, and 10 ExecRepoBench tasks. Over, Exact, and Under indicate whether the number of retrieved APIs is greater than, equal to, or smaller than the number of ground-truth invoked repository APIs; 0.0 denotes that no task falls into that category.}",
        r"\label{tab:rq4_api_count_outcomes}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Backend} & \multirow{2}{*}{Method} & \multicolumn{3}{c}{RepoExec} & \multicolumn{3}{c}{CoderEval} & \multicolumn{3}{c}{ExecRepoBench} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"& & Over & Exact & Under & Over & Exact & Under & Over & Exact & Under \\",
        r"\midrule",
    ]
    for backend in backends:
        for method in methods:
            vals = []
            for benchmark in benchmarks:
                row = lookup.get((backend, method, benchmark))
                vals.extend([
                    _format_pct(row.get("over") if row else None),
                    _format_pct(row.get("exact") if row else None),
                    _format_pct(row.get("under") if row else None),
                ])
            safe_method = method.replace("OpenCoder", r"OpenCoder").replace("API-informed reference", "API-informed ref.")
            lines.append(f"{backend} & {safe_method} & " + " & ".join(vals) + r" \\")
        if backend != backends[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_table(path: Path, quality_rows: Sequence[dict]) -> None:
    agg = aggregate_quality_for_table(quality_rows)
    backends = ["GPT", "Gemini"]
    methods = [
        "Baseline RAG",
        "OpenCoder-NoUncFilter",
        "OpenCoder-NoAPIRefine",
        "OpenCoder",
        "ContextOnly",
        "API-informed reference",
    ]
    lookup = {(r["backend"], r["method"]): r for r in agg}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Macro-averaged API retrieval quality on API-bearing RepoExec and CoderEval tasks. ExecRepoBench is excluded because the selected subset contains no resolvable repository-specific API invocation. Exact denotes exact API-set match.}",
        r"\label{tab:rq4_api_quality}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Backend} & \textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Exact} \\",
        r"\midrule",
    ]
    for backend in backends:
        for method in methods:
            row = lookup.get((backend, method))
            vals = [
                _format_pct(row.get("precision") if row else None),
                _format_pct(row.get("recall") if row else None),
                _format_pct(row.get("f1") if row else None),
                _format_pct(row.get("exact_set_match") if row else None),
            ]
            safe_method = method.replace("API-informed reference", "API-informed ref.")
            lines.append(f"{backend} & {safe_method} & " + " & ".join(vals) + r" \\")
        if backend != backends[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_api_item_uncertainty_table(path: Path, summary_rows: Sequence[dict]) -> None:
    lookup = {
        row["backend"]: row
        for row in summary_rows
        if row.get("benchmark") == "All" and row.get("split") == "test"
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Held-out detection of incorrect retrieved API items using API-specific uncertainty. AUPRC, AUROC, ECE, and Brier are computed on the deterministic test split; lower ECE and Brier are better.}",
        r"\label{tab:rq4_api_item_uncertainty}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Backend} & \textbf{Items} & \textbf{Errors} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{ECE} & \textbf{Brier} \\",
        r"\midrule",
    ]
    for backend in ("GPT", "Gemini"):
        row = lookup.get(backend) or {}
        lines.append(
            f"{backend} & {row.get('n_items', '--')} & {row.get('n_errors', '--')} & "
            f"{_format_num(row.get('calibrated_auroc'))} & "
            f"{_format_num(row.get('calibrated_auprc'))} & "
            f"{_format_num(row.get('calibrated_ece'))} & "
            f"{_format_num(row.get('calibrated_brier'))} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_findings(
    out_dir: Path,
    count_rows: Sequence[dict],
    quality_rows: Sequence[dict],
    uncertainty_rows: Sequence[dict],
    item_uncertainty_summary: Sequence[dict],
    failures: Sequence[dict],
) -> None:
    open_rows = [r for r in quality_rows if r["method"] == "OpenCoder"]
    baseline_rows = [r for r in quality_rows if r["method"] == "Baseline RAG"]
    open_f1 = _mean(r["f1"] for r in open_rows)
    base_f1 = _mean(r["f1"] for r in baseline_rows)
    open_exact = _mean(r["exact_set_match"] for r in open_rows)
    held_out = {
        row["backend"]: row
        for row in item_uncertainty_summary
        if row.get("benchmark") == "All" and row.get("split") == "test"
    }
    unavailable = sorted({f["issue"] for f in failures if f.get("severity") == "unavailable"})
    md = [
        "# RQ4: API Retrieval Reliability",
        "",
        "All values in this file are regenerated from `raw_api_predictions.jsonl` and `ground_truth_api_sets.jsonl`.",
        "",
        "## Finding 4",
        (
            "OpenCoder improves repository-API retrieval quality and its API-specific uncertainty "
            "identifies incorrect retrieved evidence on a held-out item split."
        ),
        "",
        f"- Mean OpenCoder API F1 across available benchmark-backend cells: {_format_pct(open_f1)}%.",
        f"- Mean Baseline RAG API F1 across available benchmark-backend cells: {_format_pct(base_f1)}%.",
        f"- Mean OpenCoder exact API-set match: {_format_pct(open_exact)}%.",
        f"- Held-out API-item AUROC: GPT {_format_num((held_out.get('GPT') or {}).get('calibrated_auroc'))}; Gemini {_format_num((held_out.get('Gemini') or {}).get('calibrated_auroc'))}.",
        f"- Held-out API-item ECE: GPT {_format_num((held_out.get('GPT') or {}).get('calibrated_ece'))}; Gemini {_format_num((held_out.get('Gemini') or {}).get('calibrated_ece'))}.",
        "- The selected ExecRepoBench subset contains no resolvable repository-specific API invocation after receiver-aware call resolution, so it is excluded from API-set F1 aggregation.",
    ]
    if unavailable:
        md.extend(["", "## Availability Notes"])
        md.extend(f"- {item}" for item in unavailable)
    out_dir.joinpath("findings.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\subsection{RQ4: API Retrieval Reliability}",
        "",
        r"\textbf{Finding 4.}",
        (
            "OpenCoder exposes measurable API-retrieval behavior: across the available "
            f"benchmark--backend cells, its mean API F1 is {_format_pct(open_f1)}\\% and "
            f"its exact API-set match rate is {_format_pct(open_exact)}\\%. "
            f"Held-out API-item AUROC is {_format_num((held_out.get('GPT') or {}).get('calibrated_auroc'))} "
            f"for GPT and {_format_num((held_out.get('Gemini') or {}).get('calibrated_auroc'))} for Gemini."
        ),
        "",
        (
            "Table~\\ref{tab:rq4_api_count_outcomes} reports whether each method retrieves more, "
            "exactly the same number, or fewer APIs than the ground-truth invoked API set. "
            "Table~\\ref{tab:rq4_api_quality} summarizes API-set precision, recall, F1, and exact-set "
            "accuracy; Table~\\ref{tab:rq4_api_item_uncertainty} reports held-out uncertainty diagnostics."
        ),
        "",
        (
            "We do not report AllianceCoder or API-repair ablations as measured rows because those "
            "implementations are not present in this OpenCoder checkout. CoderEval pass/fail-conditioned "
            "diagnostics use the mutation-audited native project-test harness on the pinned repository snapshot."
        ),
        "",
    ]
    out_dir.joinpath("findings.tex").write_text("\n".join(tex), encoding="utf-8")


def write_manual_annotation(out_dir: Path, raw_records: Sequence[dict], failures: List[dict]) -> None:
    candidates: List[dict] = []
    for rec in raw_records:
        if rec.get("method") != "OpenCoder":
            continue
        for hit in rec.get("api_hits") or []:
            if not hit.get("description"):
                continue
            candidates.append(
                {
                    "benchmark": rec["benchmark"],
                    "backend": rec["backend_label"],
                    "model": rec["model"],
                    "task_id": rec["task_id"],
                    "step_index": hit.get("step_index"),
                    "api_name": hit.get("api_name"),
                    "api_signature": hit.get("signature"),
                    "query": hit.get("query"),
                    "generated_api_description": hit.get("description"),
                    "retrieval_score": hit.get("retrieval_score"),
                    "knowledge_uncertainty": hit.get("knowledge_uncertainty"),
                    "annotator_a_label": "",
                    "annotator_b_label": "",
                    "adjudicated_label": "",
                    "notes": "",
                }
            )
    rows: List[dict] = []
    for backend in ("GPT", "Gemini"):
        for benchmark in ("RepoExec", "CoderEval", "ExecRepoBench"):
            group = [c for c in candidates if c["backend"] == backend and c["benchmark"] == benchmark]
            selected = group[:25]
            if len(selected) < 25:
                failures.append(
                    {
                        "severity": "warning",
                        "component": "manual_annotation_sample",
                        "issue": f"Only {len(selected)} API-description samples available for {backend}/{benchmark}; requested 25.",
                    }
                )
            rows.extend(selected)
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = f"rq4_ann_{i:03d}"
    fields = [
        "sample_id",
        "benchmark",
        "backend",
        "model",
        "task_id",
        "step_index",
        "api_name",
        "api_signature",
        "query",
        "generated_api_description",
        "retrieval_score",
        "knowledge_uncertainty",
        "annotator_a_label",
        "annotator_b_label",
        "adjudicated_label",
        "notes",
    ]
    _csv_write(out_dir / "manual_annotation_sample.csv", rows, fields)
    guidelines = """# RQ4 Manual Annotation Guidelines

Annotate each API-description pair independently.

Labels:
- `correct`: the generated description accurately captures the API's behavior or role.
- `partial`: the description is related to the API but misses important constraints, inputs, outputs, or side effects.
- `incorrect`: the description is unrelated, misleading, or contradicts the API implementation.

Annotators should read the API signature, generated description, and retrieval query. Do not use generated code outcomes when assigning labels. Leave `notes` for ambiguous cases, then adjudicate disagreements in `adjudicated_label`.
"""
    (out_dir / "annotation_guidelines.md").write_text(guidelines, encoding="utf-8")


def write_plot(
    out_dir: Path,
    metric_rows: Sequence[dict],
    item_uncertainty_summary: Sequence[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    rows = [
        r
        for r in metric_rows
        if r.get("method") == "OpenCoder" and _safe_float(r.get("uncertainty")) is not None
    ]
    if not rows:
        return
    markers = {"RepoExec": "o", "CoderEval": "s", "ExecRepoBench": "^"}
    colors = {"GPT": "#1f77b4", "Gemini": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for row in rows:
        ax.scatter(
            float(row["uncertainty"]),
            float(row["f1"]),
            marker=markers.get(row["benchmark"], "o"),
            color=colors.get(row["backend"], "#333333"),
            alpha=0.75,
            s=42,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xlabel("OpenCoder uncertainty")
    ax.set_ylabel("API F1")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)
    handles = []
    labels = []
    for backend, color in colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7))
        labels.append(backend)
    for benchmark, marker in markers.items():
        handles.append(plt.Line2D([0], [0], marker=marker, color="#555555", linestyle="None", markersize=7))
        labels.append(benchmark)
    ax.legend(handles, labels, ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_uncertainty_vs_api_f1.pdf")
    fig.savefig(out_dir / "figure_uncertainty_vs_api_f1.png", dpi=220)
    plt.close(fig)

    quality_rows = summarize_metrics(metric_rows)[1]
    agg = aggregate_quality_for_table(quality_rows)
    methods = ["Baseline RAG", "OpenCoder-NoAPIRefine", "OpenCoder"]
    method_labels = ["Baseline", "No API refine", "OpenCoder"]
    backends = ["GPT", "Gemini"]
    lookup = {(r["backend"], r["method"]): r for r in agg}

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )
    x = np.arange(len(backends))
    width = 0.23
    palette = {
        "Baseline RAG": "#8c8c8c",
        "OpenCoder-NoAPIRefine": "#9ecae1",
        "OpenCoder": "#2ca25f",
    }
    for idx, method in enumerate(methods):
        vals = [
            100.0 * float((lookup.get((backend, method)) or {}).get("f1") or 0.0)
            for backend in backends
        ]
        bars = ax0.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=method_labels[idx],
            color=palette[method],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.0,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax0.set_xticks(x)
    ax0.set_xticklabels(backends)
    ax0.set_ylabel("API F1 (%)")
    ax0.set_ylim(0, 105)
    ax0.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    ax0.set_axisbelow(True)
    ax0.legend(frameon=False, fontsize=8, loc="upper left")
    ax0.set_title("API-set quality")

    item_lookup = {
        row["backend"]: row
        for row in item_uncertainty_summary
        if row.get("benchmark") == "All" and row.get("split") == "test"
    }
    x2 = np.arange(len(backends))
    width2 = 0.3
    for idx, (metric_name, key, color) in enumerate([
        ("AUROC", "calibrated_auroc", "#3182bd"),
        ("AUPRC", "calibrated_auprc", "#e6550d"),
    ]):
        vals = [float((item_lookup.get(backend) or {}).get(key) or 0.0) for backend in backends]
        bars = ax1.bar(
            x2 + (idx - 0.5) * width2,
            vals,
            width2,
            label=metric_name,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_xticks(x2)
    ax1.set_xticklabels(backends)
    ax1.set_ylabel("Held-out score")
    ax1.set_ylim(0, 1.08)
    ax1.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title("API-item error detection")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_api_retrieval_reliability.pdf")
    fig.savefig(out_dir / "figure_api_retrieval_reliability.png", dpi=220)
    fig.savefig(out_dir / "figure_rq4_api_reliability_panel.pdf")
    fig.savefig(out_dir / "figure_rq4_api_reliability_panel.png", dpi=220)
    plt.close(fig)


def _load_examples_for_run(spec: dict, limit: Optional[int]) -> List[Example]:
    rq3 = load_rq3(spec["rq3_path"])
    n = limit or (rq3.get("metadata") or {}).get("limit")
    return list(load_dataset(spec["dataset"], spec["dataset_path"], limit=n))


def run_available_specs(args: argparse.Namespace) -> Tuple[List[dict], List[dict], List[dict]]:
    raw_records: List[dict] = []
    gt_records: Dict[Tuple[str, str], GroundTruth] = {}
    failures: List[dict] = []

    for spec in DEFAULT_RUNS:
        rq3 = load_rq3(spec["rq3_path"])
        pipe = make_pipeline(spec["config"], allow_description_api=args.allow_description_api)
        cfg = pipe.cfg
        examples = _load_examples_for_run(spec, args.limit)
        with_rows = _row_by_id(rq3.get("with") or [])
        without_rows = _row_by_id(rq3.get("without") or [])
        describe_limit = (rq3.get("metadata") or {}).get("describe_limit") or args.describe_limit
        print(
            f"[RQ4] {spec['backend_label']} {spec['benchmark']}: {len(examples)} examples",
            flush=True,
        )
        for example in examples:
            gt = ground_truth_from_example(pipe, spec["benchmark"], example)
            gt_records[(gt.benchmark, gt.task_id)] = gt
            try:
                _, retrievers = pipe.index_example(example, describe_limit=describe_limit)
                api_retriever = retrievers["api"]
                candidate_api_names = sorted(
                    {
                        str(item.qualname)
                        for item in api_retriever.items
                        if normalize_api_name(getattr(item, "qualname", ""))
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "severity": "error",
                        "component": "index_example",
                        "benchmark": spec["benchmark"],
                        "backend": spec["backend_label"],
                        "task_id": example.id,
                        "issue": str(exc),
                    }
                )
                continue
            rq3_row = with_rows.get(example.id)
            baseline_row = without_rows.get(example.id)
            for method in METHODS:
                replay = replay_api_retrieval(
                    pipe=pipe,
                    retriever=api_retriever,
                    method=method,
                    example=example,
                    rq3_row=rq3_row,
                    baseline_row=baseline_row,
                    gt=gt,
                )
                raw_records.append(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "benchmark": spec["benchmark"],
                        "task_id": example.id,
                        "backend_label": spec["backend_label"],
                        "backend": cfg.llm_backend,
                        "model": cfg.llm_model,
                        "method": method,
                        "ground_truth_apis": gt.api_names,
                        "ground_truth_api_norms": gt.api_norms,
                        "ground_truth_method": gt.extraction_method,
                        "unresolved_calls": gt.unresolved_calls,
                        "candidate_api_names": candidate_api_names,
                        "candidate_api_norms": sorted(
                            {normalize_api_name(name) for name in candidate_api_names}
                        ),
                        **replay,
                    }
                )
            print(f"  {example.id} done", flush=True)

    oracle_map = _load_codereval_oracles(CODEREVAL_SPEC["official_path"])
    for backend_spec in CODEREVAL_BACKEND_SPECS:
        backend_label = backend_spec["backend_label"]
        rq3 = load_rq3(backend_spec["rq3_path"])
        n = args.codereval_limit or (rq3.get("metadata") or {}).get("limit")
        coder_examples = load_codereval_examples(CODEREVAL_SPEC["dataset_path"], limit=n)
        with_rows = _row_by_id(rq3.get("with") or [])
        without_rows = _row_by_id(rq3.get("without") or [])
        describe_limit = (rq3.get("metadata") or {}).get("describe_limit") or args.describe_limit
        pipe = make_pipeline(backend_spec["config"], allow_description_api=args.allow_description_api)
        cfg = pipe.cfg
        print(f"[RQ4] {backend_label} CoderEval: {len(coder_examples)} examples", flush=True)
        for example in coder_examples:
            meta = example.raw.get("metadata") or {}
            oracle = oracle_map.get(str(meta.get("_id") or example.id))
            gt = ground_truth_from_codereval_record(example, oracle)
            gt_records[(gt.benchmark, gt.task_id)] = gt
            try:
                _, retrievers = pipe.index_example(example, describe_limit=describe_limit)
                api_retriever = retrievers["api"]
                candidate_api_names = sorted(
                    {
                        str(item.qualname)
                        for item in api_retriever.items
                        if normalize_api_name(getattr(item, "qualname", ""))
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "severity": "error",
                        "component": "codereval_index_example",
                        "benchmark": "CoderEval",
                        "backend": backend_label,
                        "task_id": example.id,
                        "issue": str(exc),
                    }
                )
                continue
            for method in METHODS:
                replay = replay_api_retrieval(
                    pipe=pipe,
                    retriever=api_retriever,
                    method=method,
                    example=example,
                    rq3_row=with_rows.get(example.id),
                    baseline_row=without_rows.get(example.id),
                    gt=gt,
                )
                raw_records.append(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "benchmark": "CoderEval",
                        "task_id": example.id,
                        "backend_label": backend_label,
                        "backend": cfg.llm_backend,
                        "model": cfg.llm_model,
                        "method": method,
                        "ground_truth_apis": gt.api_names,
                        "ground_truth_api_norms": gt.api_norms,
                        "ground_truth_method": gt.extraction_method,
                        "unresolved_calls": gt.unresolved_calls,
                        "candidate_api_names": candidate_api_names,
                        "candidate_api_norms": sorted(
                            {normalize_api_name(name) for name in candidate_api_names}
                        ),
                        **replay,
                    }
                )
            print(f"  {example.id} done", flush=True)

    failures.extend(
        {
            "severity": "unavailable",
            "component": "method",
            "issue": f"{method} is not implemented in this OpenCoder checkout; no measured RQ4 row is reported.",
        }
        for method in UNAVAILABLE_METHODS
    )
    failures.append(
        {
            "severity": "note",
            "component": "codereval_tests",
            "issue": "CoderEval pass/fail labels come from the mutation-audited native project-test harness on pinned snapshot e29e042b5038e59f2bf2d0b57ff842ff51538faf.",
        }
    )
    if not args.allow_description_api:
        failures.append(
            {
                "severity": "note",
                "component": "description_generation",
                "issue": "RQ4 used cached generated API descriptions when present and signature/docstring fallback for uncached descriptions; pass --allow-description-api to generate missing descriptions with the configured LLM.",
            }
        )
    return raw_records, [gt.__dict__ for gt in gt_records.values()], failures


def build_outputs(out_dir: Path, raw_records: Sequence[dict], gt_records: Sequence[dict], failures: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _jsonl_write(out_dir / "raw_api_predictions.jsonl", raw_records)
    _jsonl_write(out_dir / "ground_truth_api_sets.jsonl", gt_records)

    metric_rows = [_record_to_metrics(r) for r in raw_records]
    metric_fields = [
        "benchmark",
        "backend",
        "model",
        "method",
        "task_id",
        "task_passed",
        "correctness_mode",
        "uncertainty",
        "api_knowledge_uncertainty",
        "gt_count",
        "pred_count",
        "true_positive_count",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "exact_set_match",
        "count_outcome",
    ]
    _csv_write(out_dir / "per_task_api_metrics.csv", metric_rows, metric_fields)

    count_rows, quality_rows, uncertainty_rows, success_rows = summarize_metrics(metric_rows)
    _csv_write(out_dir / "api_count_outcomes.csv", count_rows, ["benchmark", "backend", "method", "n", "over", "exact", "under"])
    _csv_write(
        out_dir / "api_quality_summary.csv",
        quality_rows,
        [
            "benchmark",
            "backend",
            "method",
            "n",
            "n_api_tasks",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "exact_set_match",
            "pass_recall",
            "fail_recall",
            "auroc",
            "ece",
        ],
    )
    _csv_write(
        out_dir / "uncertainty_detection.csv",
        uncertainty_rows,
        [
            "benchmark",
            "backend",
            "method",
            "n",
            "spearman_uncertainty_api_f1",
            "auroc_incomplete_retrieval",
            "auprc_incomplete_retrieval",
            "ece_incomplete_retrieval",
            "brier_incomplete_retrieval",
            "mean_uncertainty_over",
            "mean_uncertainty_exact",
            "mean_uncertainty_under",
        ],
    )
    _csv_write(
        out_dir / "success_conditioned_metrics.csv",
        success_rows,
        [
            "benchmark",
            "backend",
            "method",
            "n_pass",
            "n_fail",
            "pass_precision",
            "pass_recall",
            "pass_f1",
            "fail_precision",
            "fail_recall",
            "fail_f1",
        ],
    )
    _csv_write(
        out_dir / "ablation_results.csv",
        quality_rows,
        [
            "benchmark",
            "backend",
            "method",
            "n",
            "n_api_tasks",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "exact_set_match",
            "pass_recall",
            "fail_recall",
            "auroc",
            "ece",
        ],
    )
    coverage_rows = audit_label_candidate_coverage(raw_records)
    _csv_write(
        out_dir / "label_candidate_coverage.csv",
        coverage_rows,
        [
            "benchmark",
            "backend",
            "task_id",
            "ground_truth_count",
            "covered_count",
            "candidate_coverage",
            "missing_ground_truth_apis",
        ],
    )
    item_uncertainty_rows = calibrate_api_item_uncertainty(
        _api_item_uncertainty_rows(raw_records)
    )
    _csv_write(
        out_dir / "api_item_uncertainty.csv",
        item_uncertainty_rows,
        [
            "benchmark",
            "backend",
            "task_id",
            "api_name",
            "api_norm",
            "split",
            "best_score",
            "relative_score",
            "support_steps",
            "raw_api_uncertainty",
            "calibrated_api_uncertainty",
            "is_incorrect_api",
        ],
    )
    item_uncertainty_summary = summarize_api_item_uncertainty(item_uncertainty_rows)
    _csv_write(
        out_dir / "api_item_uncertainty_summary.csv",
        item_uncertainty_summary,
        [
            "backend",
            "benchmark",
            "split",
            "n_items",
            "n_errors",
            "raw_auroc",
            "calibrated_auroc",
            "calibrated_auprc",
            "calibrated_ece",
            "calibrated_brier",
        ],
    )
    write_manual_annotation(out_dir, raw_records, failures)
    write_count_table(out_dir / "table_api_count.tex", count_rows)
    write_quality_table(out_dir / "table_api_quality.tex", quality_rows)
    write_api_item_uncertainty_table(
        out_dir / "table_api_item_uncertainty.tex",
        item_uncertainty_summary,
    )
    write_findings(
        out_dir,
        count_rows,
        quality_rows,
        uncertainty_rows,
        item_uncertainty_summary,
        failures,
    )
    write_plot(out_dir, metric_rows, item_uncertainty_summary)
    _csv_write(out_dir / "failures.csv", failures, ["severity", "component", "benchmark", "backend", "task_id", "issue"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/rq4")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for RQ3-backed benchmark examples.")
    parser.add_argument("--codereval-limit", type=int, default=None)
    parser.add_argument("--describe-limit", type=int, default=5)
    parser.add_argument(
        "--allow-description-api",
        action="store_true",
        help="Allow LLM calls for uncached API descriptions. Defaults to cache/docstring-only for reproducible local runs.",
    )
    args = parser.parse_args()
    raw_records, gt_records, failures = run_available_specs(args)
    build_outputs(Path(args.out_dir), raw_records, gt_records, failures)
    print(f"Wrote RQ4 artifacts to {args.out_dir}")
    print(f"Raw prediction records: {len(raw_records)}")
    print(f"Ground-truth records: {len(gt_records)}")
    print(f"Availability/failure notes: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
