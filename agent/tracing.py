"""Optional MLflow tracing.

If mlflow is installed (eval jobs, notebooks), nodes produce spans, and retrieval is logged as
a RETRIEVER span whose outputs follow MLflow's document schema, which the RetrievalGroundedness
judge reads. If mlflow is absent (the deployed app), these helpers are no-ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:  # pragma: no cover - depends on the environment
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False


def trace(name: str, span_type: str = "CHAIN") -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        if not _HAS_MLFLOW:
            return fn
        return mlflow.trace(name=name, span_type=span_type)(fn)  # type: ignore[return-value]

    return decorator


def to_documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunks -> MLflow retriever documents ({page_content, metadata, id})."""
    return [
        {
            "id": c.get("chunk_id"),
            "page_content": c.get("text", ""),
            "metadata": {k: v for k, v in c.items() if k not in ("text", "chunk_id")},
        }
        for c in chunks
    ]


def from_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"chunk_id": d.get("id"), "text": d.get("page_content", ""), **d.get("metadata", {})} for d in docs]
