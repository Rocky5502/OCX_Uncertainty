"""End-to-end OpenCoder pipeline.

Wires Phases I-V together. The pipeline is structured so RQ1 ablations
(toggle one retrieval source on/off) and RQ2 evaluations (with/without
uncertainty-aware scoring) reuse the exact same execution path.
"""
from __future__ import annotations

import ast
import os
import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .data.loaders import Example
from .embeddings import Encoder
from .llm import LLMClient
from .phase1_repo_knowledge import (
    describe_functions,
    extract_repo_knowledge,
    extract_sources_knowledge,
    profile_knowledge_uncertainty,
)
from .evaluation.metrics import pass_at_ks_from_samples, pass_rate_variance
from .phase2_query import (
    ImplementationStep,
    RetrievalIntent,
    decompose_into_steps,
    estimate_step_uncertainty,
    predict_retrieval_intent,
)
from .phase3_retrieval import (
    APIRetriever,
    ContextRetriever,
    SimilarCodeRetriever,
    fuse_evidence,
)
from .phase3_retrieval.score_filter import (
    merge_step_candidates,
    merge_step_candidates_balanced,
    score_and_filter,
)
from .phase4_generation import generate_target_code
from .phase5_verify import (
    is_opencoderx_execrepobench_record,
    normalize_execrepobench_function,
    repair_code,
    run_codereval_project_tests,
    run_execrepobench_function_tests,
    run_repo_completion_tests,
    run_tests,
    static_check,
)
from .phase5_verify.test_validate import TestReport
from .uncertainty import normalize_code


def re_safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


@dataclass
class PipelineConfig:
    llm_backend: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: Optional[float] = 0.2
    llm_max_tokens: int = 1024
    llm_timeout: int = 120
    llm_seed: Optional[int] = None
    llm_reasoning_effort: Optional[str] = None
    llm_thinking_budget: Optional[int] = None
    embedding_model: str = "microsoft/unixcoder-base"
    embedding_device: str = "cpu"
    api_top_k: int = 10
    context_top_k: int = 10
    similar_code_top_k: int = 10
    fused_top_k: int = 12
    keep_quantile: float = 0.7
    knowledge_uncertainty_alpha: float = 0.5
    n_samples_for_uncertainty: int = 5
    token_entropy_weight: float = 0.4
    self_consistency_weight: float = 0.4
    semantic_variance_weight: float = 0.2
    max_repair_rounds: int = 2
    evaluate_samples: bool = True
    include_test_context: bool = False
    cache_dir: str = "cache"
    enable_sources: tuple = ("api", "context", "similar_code")  # for RQ1 ablation
    uncertainty_aware: bool = True                              # for RQ2 ablation
    uncertainty_decomposition: Optional[bool] = None
    uncertainty_filtering: Optional[bool] = None
    uncertainty_guided_generation: Optional[bool] = None
    uncertainty_verified_selection: Optional[bool] = None
    uncertainty_triggered_repair: Optional[bool] = None
    whole_task_retrieval_anchor: Optional[bool] = None
    source_balanced_fusion: Optional[bool] = None
    max_source_fraction: float = 0.5

    def feature_enabled(self, name: str) -> bool:
        """Resolve an RQ2 component switch with backward-compatible defaults."""
        value = getattr(self, name)
        return self.uncertainty_aware if value is None else bool(value)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        llm = data.get("llm", {})
        embedding = data.get("embedding", {})
        retrieval = data.get("retrieval", {})
        uncertainty = data.get("uncertainty", {})
        verification = data.get("verification", {})
        experiment = data.get("experiment", {})
        flat = {
            "llm_backend": llm.get("backend", cls.llm_backend),
            "llm_model": llm.get("model", cls.llm_model),
            "llm_temperature": llm.get("temperature", cls.llm_temperature),
            "llm_max_tokens": llm.get("max_tokens", cls.llm_max_tokens),
            "llm_timeout": llm.get("timeout", cls.llm_timeout),
            "llm_seed": llm.get("seed", experiment.get("seed")),
            "llm_reasoning_effort": llm.get("reasoning_effort"),
            "llm_thinking_budget": llm.get("thinking_budget"),
            "embedding_model": embedding.get("model", cls.embedding_model),
            "embedding_device": embedding.get("device", cls.embedding_device),
            "api_top_k": retrieval.get("api_top_k", cls.api_top_k),
            "context_top_k": retrieval.get("context_top_k", cls.context_top_k),
            "similar_code_top_k": retrieval.get("similar_code_top_k", cls.similar_code_top_k),
            "fused_top_k": retrieval.get("fused_top_k", cls.fused_top_k),
            "keep_quantile": retrieval.get(
                "uncertainty_filter_quantile", cls.keep_quantile
            ),
            "knowledge_uncertainty_alpha": retrieval.get(
                "knowledge_uncertainty_alpha", cls.knowledge_uncertainty_alpha
            ),
            "n_samples_for_uncertainty": llm.get(
                "n_samples_for_uncertainty", cls.n_samples_for_uncertainty
            ),
            "token_entropy_weight": uncertainty.get(
                "token_entropy_weight", cls.token_entropy_weight
            ),
            "self_consistency_weight": uncertainty.get(
                "self_consistency_weight", cls.self_consistency_weight
            ),
            "semantic_variance_weight": uncertainty.get(
                "semantic_variance_weight", cls.semantic_variance_weight
            ),
            "max_repair_rounds": verification.get(
                "max_repair_rounds", cls.max_repair_rounds
            ),
            "include_test_context": retrieval.get(
                "include_test_context", cls.include_test_context
            ),
            "cache_dir": data.get("cache_dir", cls.cache_dir),
            "uncertainty_decomposition": uncertainty.get("decomposition"),
            "uncertainty_filtering": uncertainty.get("filtering"),
            "uncertainty_guided_generation": uncertainty.get("guided_generation"),
            "uncertainty_verified_selection": verification.get("verified_selection"),
            "uncertainty_triggered_repair": verification.get("uncertainty_triggered_repair"),
            "whole_task_retrieval_anchor": retrieval.get("whole_task_anchor"),
            "source_balanced_fusion": retrieval.get("source_balanced_fusion"),
            "max_source_fraction": retrieval.get(
                "max_source_fraction", cls.max_source_fraction
            ),
        }
        return cls(**flat)


@dataclass
class PipelineRun:
    example_id: str
    code: str
    uncertainty_trace: Dict[str, float]
    uncertainty_components: Dict[str, float] = field(default_factory=dict)
    per_step: List[Dict[str, Any]] = field(default_factory=list)
    fused_evidence: str = ""
    fused_evidence_ids: List[Dict[str, Any]] = field(default_factory=list)
    generated_samples: List[str] = field(default_factory=list)
    generation_raw_responses: List[str] = field(default_factory=list)
    generation_response_metadata: List[Dict[str, Any]] = field(default_factory=list)
    generation_integrity: Dict[str, Any] = field(default_factory=dict)
    static_report: Dict[str, Any] = field(default_factory=dict)
    test_report: Dict[str, Any] = field(default_factory=dict)
    source_diagnostics: Dict[str, Any] = field(default_factory=dict)
    correctness_mode: str = "unknown"
    sample_correctness: List[bool] = field(default_factory=list)
    pass_at_k: Dict[str, float] = field(default_factory=dict)
    pass_rate_variance: float = 0.0
    repair_rounds: int = 0
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    initial_test_report: Dict[str, Any] = field(default_factory=dict)
    post_selection_test_report: Dict[str, Any] = field(default_factory=dict)
    verified_selection_applied: bool = False


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = LLMClient(
            backend=cfg.llm_backend,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
            seed=cfg.llm_seed,
            reasoning_effort=cfg.llm_reasoning_effort,
            thinking_budget=cfg.llm_thinking_budget,
        )
        self.encoder = Encoder(model_name=cfg.embedding_model, device=cfg.embedding_device)

    # ---- Phase I ----
    def index_repo(self, repo_root: str, describe_limit: Optional[int] = None):
        items = extract_repo_knowledge(repo_root)
        sources = []
        for it in {x.file_path for x in items}:
            p = os.path.join(repo_root, it)
            try:
                sources.append((it, open(p, encoding="utf-8", errors="ignore").read()))
            except Exception:
                pass
        return self.index_sources(sources, items=items, describe_limit=describe_limit)

    @staticmethod
    def _target_name(example: Example) -> Optional[str]:
        raw = example.raw or {}
        metadata = raw.get("metadata") or {}
        target = (
            raw.get("entry_point")
            or raw.get("target_function_name")
            or raw.get("target_function")
            or raw.get("function_name")
            or metadata.get("function_name")
        )
        return str(target) if target else None

    @staticmethod
    def _sanitize_target_source(source: str, target_name: str) -> str:
        """Replace target implementations with a stub before retrieval indexing."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            if target_name in source:
                raise ValueError(
                    f"cannot safely remove target {target_name!r} from unparsable source"
                ) from exc
            return source

        changed = False

        class TargetBodyStripper(ast.NodeTransformer):
            def visit_FunctionDef(self, node):  # noqa: N802
                nonlocal changed
                if node.name == target_name:
                    node.body = [ast.copy_location(ast.Pass(), node)]
                    changed = True
                    return node
                return self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):  # noqa: N802
                nonlocal changed
                if node.name == target_name:
                    node.body = [ast.copy_location(ast.Pass(), node)]
                    changed = True
                    return node
                return self.generic_visit(node)

        sanitized = TargetBodyStripper().visit(tree)
        if not changed:
            return source
        ast.fix_missing_locations(sanitized)
        return ast.unparse(sanitized)

    def _index_repo_for_example(
        self,
        repo_root: str,
        example: Example,
        describe_limit: Optional[int],
    ):
        target_name = self._target_name(example)
        if not target_name:
            raise ValueError(
                f"cannot safely index repository for {example.id!r}: target name unavailable"
            )
        extracted = extract_repo_knowledge(repo_root)
        sources: List[Tuple[str, str]] = []
        for relative_path in sorted({item.file_path for item in extracted}):
            path = os.path.join(repo_root, relative_path)
            try:
                source = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            sources.append(
                (
                    relative_path,
                    self._sanitize_target_source(source, target_name),
                )
            )
        items = [
            item
            for item in extract_sources_knowledge(sources)
            if item.qualname.rsplit(".", 1)[-1] != target_name
        ]
        return self.index_sources(
            sources,
            items=items,
            describe_limit=describe_limit,
        )

    def index_sources(
        self,
        sources: Sequence[Tuple[str, str]],
        *,
        items: Optional[List[Any]] = None,
        describe_limit: Optional[int] = None,
    ):
        items = list(items) if items is not None else extract_sources_knowledge(sources)
        if describe_limit is not None:
            limit = max(0, int(describe_limit))
            described = self._describe_with_cache(items[:limit])
            remaining = items[limit:]
            for item in remaining:
                if not item.description:
                    item.description = item.docstring or item.signature or item.qualname
                item.metadata["description_fallback"] = "description_limit"
            items = described + remaining
        else:
            items = self._describe_with_cache(items)
        items = profile_knowledge_uncertainty(items)

        api = APIRetriever(self.encoder)
        api.build(items)
        sim = SimilarCodeRetriever(self.encoder)
        sim.build(items)

        ctx = ContextRetriever(self.encoder)
        ctx.build_from_files(sources)
        return items, {"api": api, "context": ctx, "similar_code": sim}

    def _description_cache_path(self) -> Path:
        if self.cfg.llm_thinking_budget is not None:
            reasoning = f"_thinking_budget_{self.cfg.llm_thinking_budget}"
        elif self.cfg.llm_reasoning_effort:
            reasoning = f"_reasoning_{self.cfg.llm_reasoning_effort}"
        else:
            reasoning = ""
        safe = re_safe_name(
            f"{self.cfg.llm_backend}_{self.cfg.llm_model}{reasoning}"
        )
        return Path(self.cfg.cache_dir) / "descriptions" / f"{safe}.json"

    @staticmethod
    def _knowledge_cache_key(item: Any) -> str:
        payload = "\n".join([
            getattr(item, "file_path", ""),
            getattr(item, "qualname", ""),
            getattr(item, "signature", ""),
            getattr(item, "body", ""),
        ])
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _describe_with_cache(self, items: List[Any]) -> List[Any]:
        cache_path = self._description_cache_path()
        cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        missing = []
        for item in items:
            key = self._knowledge_cache_key(item)
            item.metadata["description_cache_key"] = key
            cached = cache.get(key)
            if cached:
                item.description = cached.get("description")
                item.metadata["describe_logprobs"] = cached.get("describe_logprobs")
                item.metadata["describe_cached"] = True
                item.metadata["description_source"] = cached.get(
                    "description_source", "legacy_unknown"
                )
                item.metadata["description_model"] = cached.get("model")
            else:
                missing.append(item)

        if missing:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(missing, start=1):
                describe_functions([item], self.llm)
                key = item.metadata.get("description_cache_key")
                if key and item.description and not item.metadata.get("describe_error"):
                    item.metadata["description_source"] = "llm"
                    item.metadata["description_model"] = self.llm.model
                    cache[key] = {
                        "description": item.description,
                        "describe_logprobs": item.metadata.get("describe_logprobs"),
                        "description_source": "llm",
                        "model": self.llm.model,
                    }
                else:
                    item.metadata["description_source"] = "fallback_after_error"
                if index % 10 == 0:
                    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return items

    def index_example(
        self,
        example: Example,
        fallback_repo_root: Optional[str] = None,
        describe_limit: Optional[int] = None,
    ):
        repo_root = example.repo_root or fallback_repo_root
        if repo_root and os.path.isdir(repo_root):
            return self._index_repo_for_example(
                repo_root,
                example,
                describe_limit,
            )

        sources = self._sources_from_example(example)
        if sources:
            return self.index_sources(sources, describe_limit=describe_limit)
        raise FileNotFoundError(
            f"No repository root or in-record context found for example {example.id!r}."
        )

    @staticmethod
    def _is_test_path(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or "/tests/" in f"/{normalized.strip('/')}/"
        )

    def _sources_from_example(self, example: Example) -> List[Tuple[str, str]]:
        raw = example.raw or {}
        sources: List[Tuple[str, str]] = []

        current_file = raw.get("current_file")
        if isinstance(current_file, str) and current_file.strip():
            meta = raw.get("metadata") or {}
            fpath = meta.get("fpath_tuple")
            file_path = "/".join(fpath) if isinstance(fpath, list) else "current_file.py"
            sources.append((file_path, current_file))

        for i, entry in enumerate(raw.get("context_code") or []):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                path, text = str(entry[0]).lstrip("/"), str(entry[1])
            elif isinstance(entry, dict):
                path = str(entry.get("file_path") or entry.get("path") or f"context_{i}.py")
                text = str(entry.get("code") or entry.get("text") or "")
            else:
                path, text = f"context_{i}.py", str(entry)
            if text.strip() and (
                self.cfg.include_test_context or not self._is_test_path(path)
            ):
                sources.append((path, text))

        prefix = raw.get("prefix_code")
        suffix = raw.get("suffix_code")
        if isinstance(prefix, str) or isinstance(suffix, str):
            file_path = str(raw.get("file_name") or "target.py").lstrip("/")
            placeholder_indent = self._expected_completion_indent(example)
            sources.append(
                (
                    file_path,
                    f"{prefix or ''}{placeholder_indent}pass\n{suffix or ''}",
                )
            )

        return sources

    @staticmethod
    def _reference_report(code: str, reference: str | None) -> TestReport:
        if reference is None:
            return TestReport(
                passed=None,
                stdout="",
                stderr="(no tests or reference provided)",
                returncode=0,
            )
        passed = normalize_code(code) == normalize_code(reference)
        return TestReport(
            passed=passed,
            stdout="reference exact match" if passed else "",
            stderr="" if passed else "reference exact match failed",
            returncode=0 if passed else 1,
        )

    @staticmethod
    def _is_completion_record(example: Example) -> bool:
        raw = example.raw or {}
        return isinstance(raw.get("prefix_code"), str) or isinstance(raw.get("suffix_code"), str)

    @staticmethod
    def _full_completion_code(code: str, example: Example) -> str:
        raw = example.raw or {}
        return f"{raw.get('prefix_code') or ''}{code}{raw.get('suffix_code') or ''}"

    @staticmethod
    def _expected_completion_indent(example: Example) -> str:
        raw = example.raw or {}
        prefix = raw.get("prefix_code") or ""
        trailing = prefix.rsplit("\n", 1)[-1]
        if trailing and trailing.strip() == "":
            return trailing
        for line in reversed(prefix.splitlines()):
            if line.strip():
                indent = line[: len(line) - len(line.lstrip(" "))]
                if line.rstrip().endswith(":"):
                    return indent + "    "
                return indent
        return ""

    @staticmethod
    def _normalize_completion_code(code: str, example: Example) -> str:
        raw = example.raw or {}
        if is_opencoderx_execrepobench_record(raw):
            try:
                return normalize_execrepobench_function(
                    code,
                    str(raw.get("solution") or example.reference_code or ""),
                )
            except (SyntaxError, ValueError):
                return code
        if not Pipeline._is_completion_record(example):
            return code
        expected = Pipeline._expected_completion_indent(example)
        if not expected:
            return code
        lines = code.splitlines()
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            return code
        min_indent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
        target_indent = len(expected)
        delta = target_indent - min_indent
        normalized = []
        for line in lines:
            if not line.strip():
                normalized.append("")
            elif delta > 0:
                normalized.append(" " * delta + line)
            elif delta < 0:
                normalized.append(line[min(-delta, len(line) - len(line.lstrip(" "))):])
            else:
                normalized.append(line)
        candidate = "\n".join(normalized).rstrip() + "\n"
        full_candidate = Pipeline._full_completion_code(candidate, example)
        if static_check(full_candidate).ok:
            return candidate

        repaired = Pipeline._shift_generated_tail_left(candidate, len(expected))
        if repaired != candidate and static_check(Pipeline._full_completion_code(repaired, example)).ok:
            return repaired
        return candidate

    @staticmethod
    def _shift_generated_tail_left(code: str, expected_indent_len: int) -> str:
        lines = code.splitlines()
        first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first_idx is None:
            return code
        first = lines[first_idx]
        first_indent = len(first) - len(first.lstrip(" "))
        if first_indent != expected_indent_len or not first.rstrip().endswith(":"):
            return code
        shifted = list(lines)
        for i in range(first_idx + 1, len(shifted)):
            line = shifted[i]
            indent = len(line) - len(line.lstrip(" "))
            if line.strip() and indent > expected_indent_len:
                shifted[i] = line[min(4, indent - expected_indent_len):]
        return "\n".join(shifted).rstrip() + "\n"

    @staticmethod
    def _function_completion_code(code: str, example: Example) -> str:
        raw = example.raw or {}
        stub = str(raw.get("target_function_prompt") or raw.get("instruction") or "").rstrip()
        if not stub.lstrip().startswith(("def ", "async def ")):
            return code
        stripped = code.lstrip()
        if stripped.startswith(("def ", "async def ", "class ")):
            return code

        lines = code.strip("\n").splitlines()
        nonblank = [line for line in lines if line.strip()]
        if not nonblank:
            return stub + "\n    pass\n"
        indents = [len(line) - len(line.lstrip(" ")) for line in nonblank]
        fixed: List[str] = []
        if all(indent == 0 for indent in indents):
            fixed = [("    " + line if line.strip() else "") for line in lines]
        elif min(indents) == 0:
            for line in lines:
                if not line.strip():
                    fixed.append("")
                elif line.startswith(" "):
                    fixed.append(line)
                else:
                    fixed.append("    " + line)
        else:
            fixed = lines
        return stub + "\n" + "\n".join(fixed) + ("\n" if code.endswith("\n") else "")

    def _validate_code(self, code: str, example: Example):
        code = self._normalize_completion_code(code, example)
        code = self._function_completion_code(code, example)
        raw = example.raw or {}
        if is_opencoderx_execrepobench_record(raw):
            static_target = (
                str(raw.get("execution_prefix_code") or "")
                + code
                + str(raw.get("execution_suffix_code") or "")
            )
            static_rep = static_check(static_target)
            if not static_rep.ok:
                return static_rep, TestReport(
                    passed=False,
                    stdout="",
                    stderr=static_rep.syntax_error or "static check failed",
                    returncode=1,
                )
            return static_rep, run_execrepobench_function_tests(code, raw)
        execution_prefix = ""
        if not self._is_completion_record(example):
            raw = example.raw or {}
            execution_prefix = str(raw.get("execution_prefix_code") or "")
        test_target = (
            f"{execution_prefix.rstrip()}\n\n{code.lstrip()}"
            if execution_prefix.strip()
            else code
        )
        if (example.raw or {}).get("codereval_project_tests"):
            static_target = textwrap.dedent(code)
        else:
            static_target = self._full_completion_code(code, example) if self._is_completion_record(example) else test_target
        static_rep = static_check(static_target)
        if not static_rep.ok:
            return static_rep, TestReport(
                passed=False,
                stdout="",
                stderr=static_rep.syntax_error or "static check failed",
                returncode=1,
            )
        if self._is_completion_record(example):
            repo_report = run_repo_completion_tests(code, example.raw)
            if repo_report.passed is not None:
                return static_rep, repo_report
        if (example.raw or {}).get("codereval_project_tests"):
            return static_rep, run_codereval_project_tests(code, example.raw)
        if example.test_code:
            extra_pythonpath = None
            if not self._is_completion_record(example):
                extra_pythonpath = (example.raw or {}).get("extra_pythonpath")
            return static_rep, run_tests(
                test_target,
                example.test_code,
                extra_pythonpath=extra_pythonpath,
            )
        return static_rep, self._reference_report(code, example.reference_code)

    @staticmethod
    def _correctness_mode(example: Example) -> str:
        if is_opencoderx_execrepobench_record(example.raw or {}):
            return "repository_tests"
        if (example.raw or {}).get("codereval_project_tests"):
            return "repository_tests"
        if Pipeline._is_completion_record(example):
            raw = example.raw or {}
            for entry in raw.get("context_code") or []:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) >= 2
                    and Pipeline._is_test_path(str(entry[0]))
                ):
                    return "repository_tests"
                if isinstance(entry, dict) and Pipeline._is_test_path(
                    str(entry.get("file_path") or entry.get("path") or "")
                ):
                    return "repository_tests"
        if example.test_code:
            return "execution_tests"
        if example.reference_code is not None:
            return "reference_exact_match"
        return "unknown"

    def _should_repair(self, example: Example, static_rep: Any, test_rep: TestReport) -> bool:
        if not self.cfg.feature_enabled("uncertainty_triggered_repair") or test_rep.passed is not False:
            return False
        if not static_rep.ok:
            return True
        return self._correctness_mode(example) in {"execution_tests", "repository_tests"}

    @staticmethod
    def _summarize_source_diagnostics(per_step_debug: List[Dict[str, Any]]) -> Dict[str, Any]:
        acc: Dict[str, Dict[str, Any]] = {}
        for step in per_step_debug:
            for source, stats in step.get("sources", {}).items():
                bucket = acc.setdefault(
                    source,
                    {
                        "n_steps": 0,
                        "n_retrieved": 0,
                        "n_kept": 0,
                        "top_retrieval_scores": [],
                        "mean_retrieval_scores": [],
                        "mean_knowledge_uncertainties": [],
                    },
                )
                bucket["n_steps"] += 1
                bucket["n_retrieved"] += stats.get("n_retrieved", 0)
                bucket["n_kept"] += stats.get("n_kept", 0)
                series_keys = {
                    "top_retrieval_score": "top_retrieval_scores",
                    "mean_retrieval_score": "mean_retrieval_scores",
                    "mean_knowledge_uncertainty": "mean_knowledge_uncertainties",
                }
                for key, series_key in series_keys.items():
                    if stats.get(key) is not None:
                        bucket[series_key].append(stats[key])

        out: Dict[str, Any] = {}
        for source, stats in acc.items():
            n_steps = max(1, stats["n_steps"])

            def mean(values):
                return float(sum(values) / len(values)) if values else None

            out[source] = {
                "n_steps": stats["n_steps"],
                "n_retrieved": stats["n_retrieved"],
                "n_kept": stats["n_kept"],
                "avg_retrieved_per_step": stats["n_retrieved"] / n_steps,
                "avg_kept_per_step": stats["n_kept"] / n_steps,
                "mean_top_retrieval_score": mean(stats["top_retrieval_scores"]),
                "mean_retrieval_score": mean(stats["mean_retrieval_scores"]),
                "mean_knowledge_uncertainty": mean(stats["mean_knowledge_uncertainties"]),
            }
        return out

    @staticmethod
    def _repair_test_context(example: Example, max_chars: int = 6000) -> str:
        raw = example.raw or {}
        chunks: List[str] = []
        if example.test_code:
            chunks.append(f"# Provided Tests\n```python\n{example.test_code[:max_chars]}\n```")
        for entry in raw.get("context_code") or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                path, text = str(entry[0]), str(entry[1])
            elif isinstance(entry, dict):
                path = str(entry.get("file_path") or entry.get("path") or "")
                text = str(entry.get("code") or entry.get("text") or "")
            else:
                continue
            if Pipeline._is_test_path(path) and text.strip():
                chunks.append(f"# Repository Test File: {path}\n```python\n{text[:max_chars]}\n```")
        joined = "\n\n".join(chunks)
        return joined[:max_chars]

    def _select_verified_sample(
        self,
        samples: Sequence[str],
        example: Example,
    ) -> tuple[Optional[str], Optional[Any], Optional[TestReport]]:
        if self._correctness_mode(example) not in {"execution_tests", "repository_tests"}:
            return None, None, None
        for sample in samples:
            candidate = self._normalize_completion_code(sample, example)
            static_rep, test_rep = self._validate_code(candidate, example)
            if test_rep.passed is True:
                return candidate, static_rep, test_rep
        return None, None, None

    def prepare_query(
        self,
        example: Example,
        *,
        uncertainty_decomposition: Optional[bool] = None,
    ) -> tuple[List[ImplementationStep], List[RetrievalIntent]]:
        """Build a reusable Phase-II plan for paired experimental conditions."""
        enabled = (
            self.cfg.feature_enabled("uncertainty_decomposition")
            if uncertainty_decomposition is None
            else uncertainty_decomposition
        )
        if enabled:
            steps = decompose_into_steps(example.query, self.llm)
            steps = estimate_step_uncertainty(steps, self.llm)
            return steps, predict_retrieval_intent(steps, self.llm)

        steps = [ImplementationStep(index=1, description=example.query, uncertainty=0.0)]
        weights = {s: 1.0 / 3.0 for s in ("api", "context", "similar_code")}
        queries = {s: example.query for s in weights}
        return steps, [RetrievalIntent(step_index=1, source_weights=weights, queries=queries)]

    # ---- End-to-end ----
    def run(
        self,
        example: Example,
        retrievers: Dict[str, Any],
        prepared_query: Optional[tuple[List[ImplementationStep], List[RetrievalIntent]]] = None,
    ) -> PipelineRun:
        cfg = self.cfg
        # Phase II
        if prepared_query is None:
            steps, intents = self.prepare_query(example)
        else:
            steps, intents = prepared_query

        # Phase III
        per_step_candidates = []
        per_step_debug = []
        filtering_enabled = cfg.feature_enabled("uncertainty_filtering")
        alpha = cfg.knowledge_uncertainty_alpha if filtering_enabled else 0.0
        keep_q = cfg.keep_quantile if filtering_enabled else 1.0
        if cfg.feature_enabled("whole_task_retrieval_anchor"):
            anchor_hits: Dict[str, Any] = {}
            anchor_debug: Dict[str, Any] = {}
            anchor_weight = 1.0 / max(1, len(cfg.enable_sources))
            for src in cfg.enable_sources:
                top_k = (
                    getattr(cfg, f"{src}_top_k", 10)
                    if src != "similar_code"
                    else cfg.similar_code_top_k
                )
                hits = retrievers[src].search(example.query, top_k=top_k)
                anchor_hits[src] = hits
                scores = [hit.score for hit in hits]
                ku = [
                    float(hit.metadata.get("knowledge_uncertainty"))
                    for hit in hits
                    if hit.metadata.get("knowledge_uncertainty") is not None
                ]
                anchor_debug[src] = {
                    "query": example.query,
                    "n_retrieved": len(hits),
                    "top_retrieval_score": max(scores) if scores else None,
                    "mean_retrieval_score": (
                        sum(scores) / len(scores) if scores else None
                    ),
                    "mean_knowledge_uncertainty": (
                        sum(ku) / len(ku) if ku else None
                    ),
                    "n_kept": 0,
                }
            anchor_candidates = score_and_filter(
                anchor_hits,
                {source: anchor_weight for source in cfg.enable_sources},
                knowledge_uncertainty_alpha=alpha,
                keep_quantile=keep_q,
            )
            for candidate in anchor_candidates:
                debug = anchor_debug[candidate.hit.source]
                debug["n_kept"] = debug.get("n_kept", 0) + 1
            per_step_candidates.append(anchor_candidates)
            per_step_debug.append({
                "phase": "whole_task_anchor",
                "step": "Whole-task retrieval anchor",
                "step_uncertainty": 0.0,
                "intent": {
                    source: anchor_weight for source in cfg.enable_sources
                },
                "n_candidates": len(anchor_candidates),
                "sources": anchor_debug,
            })
        for step, intent in zip(steps, intents):
            hits_by_source = {}
            source_debug: Dict[str, Any] = {}
            for src in cfg.enable_sources:
                q = intent.queries.get(src, step.description)
                top_k = getattr(cfg, f"{src}_top_k", 10) if src != "similar_code" else cfg.similar_code_top_k
                hits_by_source[src] = retrievers[src].search(q, top_k=top_k)
                hits = hits_by_source[src]
                scores = [h.score for h in hits]
                ku = [
                    float(h.metadata.get("knowledge_uncertainty"))
                    for h in hits
                    if h.metadata.get("knowledge_uncertainty") is not None
                ]
                source_debug[src] = {
                    "query": q,
                    "n_retrieved": len(hits),
                    "top_retrieval_score": max(scores) if scores else None,
                    "mean_retrieval_score": (sum(scores) / len(scores)) if scores else None,
                    "mean_knowledge_uncertainty": (sum(ku) / len(ku)) if ku else None,
                    "n_kept": 0,
                }
            sw = {s: (intent.source_weights.get(s, 0.0) if s in cfg.enable_sources else 0.0)
                  for s in ("api", "context", "similar_code")}
            cands = score_and_filter(
                hits_by_source, sw,
                knowledge_uncertainty_alpha=alpha,
                keep_quantile=keep_q,
            )
            for cand in cands:
                source_debug.setdefault(cand.hit.source, {"n_kept": 0})
                source_debug[cand.hit.source]["n_kept"] = source_debug[cand.hit.source].get("n_kept", 0) + 1
            per_step_candidates.append(cands)
            per_step_debug.append({
                "step": step.description,
                "step_uncertainty": step.uncertainty,
                "intent": intent.source_weights,
                "n_candidates": len(cands),
                "sources": source_debug,
            })

        if cfg.feature_enabled("source_balanced_fusion"):
            merged = merge_step_candidates_balanced(
                per_step_candidates,
                cfg.fused_top_k,
                max_source_fraction=cfg.max_source_fraction,
            )
        else:
            merged = merge_step_candidates(per_step_candidates, cfg.fused_top_k)
        evidence = fuse_evidence(merged)
        evidence_ids = []
        for candidate in merged:
            item = candidate.hit.item
            item_metadata = getattr(item, "metadata", {}) or {}
            identity = "|".join([
                str(candidate.hit.source),
                str(getattr(item, "file_path", "")),
                str(getattr(item, "qualname", "")),
                str(getattr(item, "signature", "")),
                str(getattr(item, "start_line", "")),
            ])
            frozen_document_id = item_metadata.get("document_id")
            evidence_ids.append({
                "id": frozen_document_id or (
                    "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
                ),
                "source": candidate.hit.source,
                "file_path": getattr(item, "file_path", None),
                "qualname": getattr(item, "qualname", None),
                "score": candidate.final_score,
            })
        source_diagnostics = self._summarize_source_diagnostics(per_step_debug)

        # Phase IV
        query_unc = {
            "n_steps": len(steps),
            "mean_step_uncertainty": (sum(s.uncertainty for s in steps) / max(1, len(steps))),
        }
        gen = generate_target_code(
            example.query, evidence, query_unc, self.llm, self.encoder,
            n_samples=cfg.n_samples_for_uncertainty,
            uncertainty_weights={
                "token_entropy_weight": cfg.token_entropy_weight,
                "self_consistency_weight": cfg.self_consistency_weight,
                "semantic_variance_weight": cfg.semantic_variance_weight,
            },
            use_uncertainty_guidance=cfg.feature_enabled("uncertainty_guided_generation"),
            completion_mode=self._is_completion_record(example),
            expected_indent=self._expected_completion_indent(example),
            sampling_temperature=cfg.llm_temperature,
        )

        # Phase V
        code = self._normalize_completion_code(gen.code, example)
        static_rep, test_rep = self._validate_code(code, example)
        initial_test_report = dict(test_rep.__dict__)
        verified_selection_applied = False
        if cfg.feature_enabled("uncertainty_verified_selection"):
            verified_code, verified_static, verified_test = self._select_verified_sample(
                gen.samples,
                example,
            )
            if verified_code is not None and verified_static is not None and verified_test is not None:
                code = verified_code
                static_rep = verified_static
                test_rep = verified_test
                verified_selection_applied = True
        post_selection_test_report = dict(test_rep.__dict__)
        rounds = 0
        repair_history: List[Dict[str, Any]] = []
        repairable = self._should_repair(example, static_rep, test_rep)
        while test_rep.passed is False and repairable and rounds < cfg.max_repair_rounds:
            before_report = dict(test_rep.__dict__)
            diag = f"static: {static_rep.__dict__}\ntests:\n{test_rep.stderr}\n{test_rep.stdout}"
            test_context = self._repair_test_context(example)
            if test_context:
                diag = f"{diag}\n\n# Test Context For Repair\n{test_context}"
            repair_prompt_audit: Dict[str, object] = {}
            code = repair_code(
                code,
                diag,
                self.llm,
                task=example.query,
                completion_mode=self._is_completion_record(example),
                expected_indent=self._expected_completion_indent(example),
                prompt_audit=repair_prompt_audit,
            )
            code = self._normalize_completion_code(code, example)
            static_rep, test_rep = self._validate_code(code, example)
            repairable = self._should_repair(example, static_rep, test_rep)
            rounds += 1
            repair_history.append({
                "round": rounds,
                "input_test_report": before_report,
                "output_static_report": dict(static_rep.__dict__),
                "output_test_report": dict(test_rep.__dict__),
                "prompt_budget": repair_prompt_audit,
            })

        sample_correctness: List[bool] = []
        pass_ks: Dict[str, float] = {}
        prv = 0.0
        if cfg.evaluate_samples:
            known = False
            for sample in gen.samples:
                _, sample_report = self._validate_code(sample, example)
                if sample_report.passed is not None:
                    known = True
                    sample_correctness.append(bool(sample_report.passed))
            if known:
                valid_ks = tuple(k for k in (1, 3, 5) if k <= len(sample_correctness))
                pass_ks = dict(pass_at_ks_from_samples([sample_correctness], ks=valid_ks))
                prv = pass_rate_variance([sample_correctness])

        return PipelineRun(
            example_id=example.id,
            code=code,
            uncertainty_trace=gen.trace.to_dict(),
            uncertainty_components=gen.components,
            per_step=per_step_debug,
            fused_evidence=evidence,
            fused_evidence_ids=evidence_ids,
            generated_samples=gen.samples,
            generation_raw_responses=gen.raw_responses,
            generation_response_metadata=gen.response_metadata,
            generation_integrity=gen.generation_integrity,
            static_report=static_rep.__dict__,
            test_report=test_rep.__dict__,
            source_diagnostics=source_diagnostics,
            correctness_mode=self._correctness_mode(example),
            sample_correctness=sample_correctness,
            pass_at_k=pass_ks,
            pass_rate_variance=prv,
            repair_rounds=rounds,
            repair_history=repair_history,
            initial_test_report=initial_test_report,
            post_selection_test_report=post_selection_test_report,
            verified_selection_applied=verified_selection_applied,
        )
