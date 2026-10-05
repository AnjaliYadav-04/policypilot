"""Self-managed embeddings (all-MiniLM-L6-v2, 384 dims).

The same model embeds chunks at ingestion time and questions at query time, which is what
"self-managed embeddings" on the Delta-Sync index requires. Vectors are L2-normalised so
similarity ranking matches cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self._load().encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
