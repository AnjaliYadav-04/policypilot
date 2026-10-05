"""Runtime settings, read from environment variables.

PP_ENV selects the backends:
  * ``databricks``: Unity AI Gateway endpoint + Databricks Vector Search (production)
  * ``local``:      Groq API directly + an in-memory index built from a JSONL file (dev)

The LangGraph agent itself (agent/graph.py) is identical in both modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_ENVS = ("local", "databricks")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    env: str = "local"
    catalog: str = "policypilot_dev"
    schema: str = "filings"
    vs_endpoint: str = "policypilot-vs-endpoint"
    llm_endpoint: str = "policypilot-groq-gateway"
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    local_index_path: str = "data/local_chunks.jsonl"
    top_k: int = 6
    max_answer_tokens: int = 1500
    strict_numbers: bool = True

    @property
    def chunks_table(self) -> str:
        return f"{self.catalog}.{self.schema}.chunks"

    @property
    def index_name(self) -> str:
        return f"{self.catalog}.{self.schema}.chunks_index"

    @property
    def is_databricks(self) -> bool:
        return self.env == "databricks"


def load_settings() -> Settings:
    env = _env("PP_ENV", "local").strip().lower()
    if env not in VALID_ENVS:
        raise ValueError(f"PP_ENV must be one of {VALID_ENVS}, got {env!r}")
    return Settings(
        env=env,
        catalog=_env("PP_CATALOG", Settings.catalog),
        schema=_env("PP_SCHEMA", Settings.schema),
        vs_endpoint=_env("PP_VS_ENDPOINT", Settings.vs_endpoint),
        llm_endpoint=_env("PP_LLM_ENDPOINT", Settings.llm_endpoint),
        groq_model=_env("PP_GROQ_MODEL", Settings.groq_model),
        groq_base_url=_env("PP_GROQ_BASE_URL", Settings.groq_base_url),
        embedding_model=_env("PP_EMBEDDING_MODEL", Settings.embedding_model),
        local_index_path=_env("PP_LOCAL_INDEX", Settings.local_index_path),
        top_k=int(_env("PP_TOP_K", str(Settings.top_k))),
        max_answer_tokens=int(_env("PP_MAX_TOKENS", str(Settings.max_answer_tokens))),
        strict_numbers=_env_bool("PP_STRICT_NUMBERS", Settings.strict_numbers),
    )
