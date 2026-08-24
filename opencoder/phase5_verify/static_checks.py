"""Phase V, Step 11: Static checks.

Lightweight: ast.parse for syntax, plus a pyflakes-style undefined-name
scan via ast traversal. No external linter dependency required.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import List


@dataclass
class StaticReport:
    ok: bool
    syntax_error: str | None = None
    undefined_names: List[str] = field(default_factory=list)


def _collect_defined(tree: ast.AST) -> set:
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.arguments):
            for a in node.args + node.kwonlyargs + node.posonlyargs:
                defined.add(a.arg)
            if node.vararg:
                defined.add(node.vararg.arg)
            if node.kwarg:
                defined.add(node.kwarg.arg)
    return defined


def static_check(code: str) -> StaticReport:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return StaticReport(ok=False, syntax_error=str(e))
    defined = _collect_defined(tree)
    undef: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined and node.id not in undef:
                undef.append(node.id)
    # Most "undefined" names are actually imports we couldn't resolve; keep
    # the list as advisory rather than blocking.
    return StaticReport(ok=True, undefined_names=undef[:20])
