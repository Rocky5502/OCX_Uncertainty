"""CrossCodeEval's native multilingual completion metrics.

The implementation mirrors the benchmark's Apache-2.0 evaluator while
excluding its unrelated local-model and PyTorch dependencies.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fuzzywuzzy import fuzz
from tree_sitter import Language, Parser


IDENTIFIER_REGEX = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
STRING_PATTERN = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')


@dataclass(frozen=True)
class CrossCodeEvalMetrics:
    exact_match: float
    edit_similarity: float
    identifier_exact_match: float
    identifier_precision: float
    identifier_recall: float
    identifier_f1: float
    postprocessed_prediction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossCodeEvalEvaluator:
    """Evaluate native line-completion output with the official conventions."""

    def __init__(self, benchmark_root: str | Path):
        self.root = Path(benchmark_root)
        self._parsers: dict[str, Parser] = {}

    def _parser(self, language: str) -> Parser:
        if language not in self._parsers:
            library = self.root / "build" / f"{language}-lang-parser.so"
            if not library.is_file():
                raise FileNotFoundError(f"CrossCodeEval parser is missing: {library}")
            parser = Parser()
            parser.set_language(Language(
                str(library),
                "c_sharp" if language == "csharp" else language,
            ))
            self._parsers[language] = parser
        return self._parsers[language]

    @lru_cache(maxsize=8)
    def _keywords(self, language: str) -> frozenset[str]:
        if language == "python":
            return frozenset(word for word in keyword.kwlist if word not in {"True", "False"})
        keyword_path = self.root / "scripts" / "keywords" / f"{language}.txt"
        if not keyword_path.is_file():
            raise FileNotFoundError(f"CrossCodeEval keyword list is missing: {keyword_path}")
        return frozenset(
            line.strip()
            for line in keyword_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    @staticmethod
    def _remove_comments(code: str) -> str:
        code = re.sub(r"#.*", "", code)
        return re.sub(r"//.*", "", code)

    def _parse_valid(self, parser: Parser, code: str) -> bool:
        tree = parser.parse(code.encode("utf-8"))

        def has_error(node: Any) -> bool:
            if node.type == "ERROR":
                return True
            return any(has_error(child) for child in node.children)

        try:
            return not has_error(tree.root_node)
        except RecursionError:
            return False

    def _postprocess(self, prompt: str, completion: str, language: str) -> str:
        parser = self._parser(language)
        if language in {"java", "csharp", "typescript"}:
            for index, char in enumerate(completion):
                if char in {";", "}", "{"}:
                    return completion[: index + 1]
            return completion
        if language == "python":
            for index, _ in enumerate(completion):
                if not self._parse_valid(parser, prompt + completion[: index + 1]):
                    continue
                if index + 1 < len(completion) and completion[index + 1] == "\n":
                    return completion[: index + 1].rstrip()
            return completion
        raise ValueError(f"unsupported CrossCodeEval language: {language}")

    @staticmethod
    def extract_completion(text: str) -> str:
        value = str(text or "").strip()
        match = re.search(r"```(?:[A-Za-z0-9_+#.-]+)?\s*(.*?)```", value, re.DOTALL)
        return (match.group(1) if match else value).strip()

    def _identifiers(self, code: str, language: str) -> list[str]:
        without_strings = STRING_PATTERN.sub("", code)
        keywords = self._keywords(language)
        return [
            token
            for token in re.findall(r"\w+", without_strings)
            if IDENTIFIER_REGEX.match(token) and token not in keywords
        ]

    def evaluate(
        self,
        *,
        prompt: str,
        prediction: str,
        target: str,
        language: str,
    ) -> CrossCodeEvalMetrics:
        extracted = self.extract_completion(prediction)
        processed = self._remove_comments(
            self._postprocess(prompt, extracted, language)
        )
        target = self._remove_comments(target)
        pred_lines = [line.strip() for line in processed.splitlines() if line.strip()]
        target_lines = [line.strip() for line in target.splitlines() if line.strip()]
        pred_ids = set(self._identifiers(processed, language))
        target_ids = set(self._identifiers(target, language))
        true_positive = len(pred_ids & target_ids)
        false_positive = len(pred_ids - target_ids)
        false_negative = len(target_ids - pred_ids)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative else 0.0
        )
        return CrossCodeEvalMetrics(
            exact_match=float(pred_lines == target_lines),
            edit_similarity=float(fuzz.ratio(processed.strip(), target.strip())) / 100.0,
            identifier_exact_match=float(pred_ids == target_ids),
            identifier_precision=precision,
            identifier_recall=recall,
            identifier_f1=f1,
            postprocessed_prediction=processed,
        )
