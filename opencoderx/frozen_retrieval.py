"""Build task retrievers from the frozen leakage-audited knowledge index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opencoder.phase1_repo_knowledge import RepoFunction, profile_knowledge_uncertainty
from opencoder.phase3_retrieval import APIRetriever, ContextRetriever, SimilarCodeRetriever
from opencoder.phase3_retrieval.context_retriever import ContextChunk


def _read_repository_documents(path: str | Path, repository: str) -> list[dict[str, Any]]:
    documents = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("repository") or "") == repository:
                documents.append(row)
    return documents


def _split_api_content(content: str) -> tuple[str, str | None]:
    signature, separator, description = content.partition("\n")
    return signature, description if separator and description.strip() else None


def build_frozen_retrievers(
    index_path: str | Path,
    repository: str,
    encoder: Any,
) -> tuple[list[RepoFunction], dict[str, Any]]:
    """Load one repository's provider-independent frozen candidate pools."""
    documents = _read_repository_documents(index_path, repository)
    if not documents:
        raise ValueError(f"no frozen retrieval documents for repository {repository!r}")

    api_items: list[RepoFunction] = []
    similar_items: list[RepoFunction] = []
    context_items: list[ContextChunk] = []
    for row in documents:
        source = str(row.get("source_type") or "")
        content = str(row.get("content") or "")
        common = {
            "file_path": str(row.get("path") or ""),
            "qualname": str(row.get("qualified_name") or ""),
            "start_line": int(row.get("line") or 0),
            "end_line": int(row.get("line") or 0),
            "metadata": {
                "document_id": str(row.get("document_id") or ""),
                "frozen_index": Path(index_path).name,
                "repository": repository,
            },
        }
        if source == "api":
            signature, description = _split_api_content(content)
            api_items.append(RepoFunction(
                signature=signature,
                description=description,
                docstring=description,
                body=content,
                kind="function",
                **common,
            ))
        elif source == "similar_code":
            similar_items.append(RepoFunction(
                signature=content.splitlines()[0] if content else "",
                description=None,
                docstring=None,
                body=content,
                kind="function",
                **common,
            ))
        elif source == "context":
            context_items.append(ContextChunk(
                file_path=str(row.get("path") or ""),
                text=content,
                metadata=common["metadata"],
            ))

    profile_knowledge_uncertainty(api_items)
    profile_knowledge_uncertainty(similar_items)
    api = APIRetriever(encoder)
    api.build(api_items)
    similar = SimilarCodeRetriever(encoder)
    similar.build(similar_items)
    context = ContextRetriever(encoder)
    context.index.build(context_items, [item.text for item in context_items])
    return [*api_items, *similar_items], {
        "api": api,
        "context": context,
        "similar_code": similar,
    }
