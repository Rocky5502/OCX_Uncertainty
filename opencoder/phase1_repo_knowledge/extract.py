"""Phase I, Step 1: Extract Knowledge.

Walks a repository, parses each .py file with Python's ast module, and
emits one record per function/method/class. Pure stdlib — no external
parser. For other languages, extend with a tree-sitter backend.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass
class RepoFunction:
    file_path: str
    qualname: str
    signature: str
    docstring: Optional[str]
    body: str
    start_line: int
    end_line: int
    kind: str = "function"          # function | method | class | imported_api
    description: Optional[str] = None
    uncertainty: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def _iter_py_files(root: str) -> Iterator[str]:
    skip = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def extract_source_knowledge(file_path: str, source: str) -> List[RepoFunction]:
    """Extract knowledge items from one in-memory Python source string."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []

    items: List[RepoFunction] = []
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    local_definitions = {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parents.get(node)
            is_method = isinstance(parent, ast.ClassDef)
            qualname = f"{parent.name}.{node.name}" if is_method else node.name
            sig = f"{node.name}({', '.join(a.arg for a in node.args.args)})"
            items.append(
                RepoFunction(
                    file_path=file_path,
                    qualname=qualname,
                    signature=sig,
                    docstring=ast.get_docstring(node),
                    body=_segment(source, node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    kind="method" if is_method else "function",
                )
            )
        elif isinstance(node, ast.ClassDef):
            items.append(
                RepoFunction(
                    file_path=file_path,
                    qualname=node.name,
                    signature=f"class {node.name}",
                    docstring=ast.get_docstring(node),
                    body=_segment(source, node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    kind="class",
                )
            )
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                if local_name in local_definitions:
                    continue
                items.append(
                    RepoFunction(
                        file_path=file_path,
                        qualname=local_name,
                        signature=f"imported {local_name} from {module}",
                        docstring=None,
                        body=_segment(source, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        kind="imported_api",
                    )
                )
    return items


def extract_sources_knowledge(sources: Sequence[Tuple[str, str]]) -> List[RepoFunction]:
    items: List[RepoFunction] = []
    for file_path, source in sources:
        items.extend(extract_source_knowledge(file_path, source))
    return items


def extract_repo_knowledge(root: str, max_files: Optional[int] = None) -> List[RepoFunction]:
    items: List[RepoFunction] = []
    for i, path in enumerate(_iter_py_files(root)):
        if max_files is not None and i >= max_files:
            break
        try:
            src = open(path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        items.extend(extract_source_knowledge(os.path.relpath(path, root), src))
    return items
