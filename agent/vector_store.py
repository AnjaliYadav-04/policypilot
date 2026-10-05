"""VectorStore: one interface, two backends.

* databricks: Databricks Vector Search, Delta-Sync index over ``<catalog>.<schema>.chunks``.
* local: an in-memory index loaded from the JSONL file written by scripts/build_local_index.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from agent.config import Settings
from agent.embeddings import Embedder, get_embedder

RETURN_COLUMNS = [
    "chunk_id",
    "ticker",
    "company",
    "form",
    "filing_date",
    "accession_number",
    "section",
    "text",
    "source_url",
]


class VectorStore(Protocol):
    def search(self, query: str, k: int, ticker: str | None = None) -> list[dict[str, Any]]: ...


class DatabricksVectorStore:
    def __init__(self, index_name: str, embedder: Embedder, workspace_client: Any = None):
        self.index_name = index_name
        self._embedder = embedder
        self._w = workspace_client

    def _client(self) -> Any:
        if self._w is None:
            from databricks.sdk import WorkspaceClient

            self._w = WorkspaceClient()
        return self._w

    def search(self, query: str, k: int, ticker: str | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "index_name": self.index_name,
            "columns": RETURN_COLUMNS,
            "query_vector": self._embedder.embed_query(query),
            "num_results": k,
        }
        if ticker:
            kwargs["filters_json"] = json.dumps({"ticker": ticker})
        response = self._client().vector_search_indexes.query_index(**kwargs)
        return parse_query_response(response)


def parse_query_response(response: Any) -> list[dict[str, Any]]:
    """Turn a QueryVectorIndexResponse into dicts. The last column is the similarity score."""
    if response is None or response.result is None or not response.result.data_array:
        return []
    columns = [c.name for c in response.manifest.columns]
    rows = []
    for values in response.result.data_array:
        row = dict(zip(columns, values))
        row["score"] = float(row.pop("score", 0.0) or 0.0)
        rows.append(row)
    return rows


class LocalVectorStore:
    def __init__(self, path: str, embedder: Embedder):
        self.path = Path(path)
        self._embedder = embedder
        self._rows: list[dict[str, Any]] | None = None
        self._matrix: Any = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(
                f"No local index at {self.path}. Run: python scripts/build_local_index.py --tickers AAPL MSFT"
            )
        import numpy as np

        rows, vectors = [], []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                vectors.append(record.pop("embedding"))
                rows.append(record)
        self._rows = rows
        self._matrix = np.asarray(vectors, dtype="float32")

    def search(self, query: str, k: int, ticker: str | None = None) -> list[dict[str, Any]]:
        import numpy as np

        self._load()
        assert self._rows is not None
        candidates = [i for i, r in enumerate(self._rows) if not ticker or r.get("ticker") == ticker]
        if not candidates:
            return []
        q = np.asarray(self._embedder.embed_query(query), dtype="float32")
        scores = self._matrix[candidates] @ q
        top = np.argsort(-scores)[:k]
        return [{**self._rows[candidates[i]], "score": float(scores[i])} for i in top]


def get_vector_store(settings: Settings) -> VectorStore:
    embedder = get_embedder(settings.embedding_model)
    if settings.is_databricks:
        return DatabricksVectorStore(settings.index_name, embedder)
    return LocalVectorStore(settings.local_index_path, embedder)
