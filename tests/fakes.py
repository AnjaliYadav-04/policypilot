"""Test doubles for the LLM and the vector store."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.config import Settings
from agent.llm import GuardrailBlocked
from agent.nodes import AgentDeps

APPLE_CHUNKS = [
    {
        "chunk_id": "AAPL_1_item_1a_0001",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "form": "10-K",
        "filing_date": "2025-10-31",
        "accession_number": "0000320193-25-000079",
        "section": "Item 1A. Risk Factors",
        "text": "The Company depends on component and product manufacturing and logistical services provided "
        "by outsourcing partners, many of which are located outside of the U.S. Substantially all of the "
        "Company's manufacturing is performed in whole or in part by outsourcing partners located primarily "
        "in China mainland, India, Japan, South Korea, Taiwan and Vietnam.",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "score": 0.82,
    },
    {
        "chunk_id": "AAPL_1_item_1a_0002",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "form": "10-K",
        "filing_date": "2025-10-31",
        "accession_number": "0000320193-25-000079",
        "section": "Item 1A. Risk Factors",
        "text": "Certain components are currently obtained from single or limited sources. The Company had "
        "$1,234 million of purchase commitments at year end.",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "score": 0.79,
    },
]


class FakeLLM:
    """Returns scripted replies in order; a reply may be an exception to raise."""

    def __init__(self, replies: list[str | Exception]):
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, max_tokens=None, temperature=0.0) -> str:  # noqa: ANN001
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeStore:
    def __init__(self, chunks: list[dict[str, Any]] | Callable[..., list[dict[str, Any]]]):
        self._chunks = chunks
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, k: int, ticker: str | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, k, ticker))
        chunks = self._chunks(query, k, ticker) if callable(self._chunks) else self._chunks
        return [c for c in chunks if not ticker or c["ticker"] == ticker][:k]


def make_deps(replies: list[str | Exception], chunks=APPLE_CHUNKS, **settings: Any) -> AgentDeps:  # noqa: ANN001
    return AgentDeps(llm=FakeLLM(replies), store=FakeStore(chunks), settings=Settings(**settings))


PLAN_APPLE = '{"search_query": "supply chain outsourcing partners single source components", "ticker": "AAPL"}'
GOOD_ANSWER = (
    "Apple's 10-K describes several supply chain risks.\n"
    "- It relies on outsourcing partners located primarily in China mainland, India and Vietnam [1].\n"
    "- Certain components come from single or limited sources [2].\n"
    "- It had $1,234 million of purchase commitments at year end [2]."
)

__all__ = ["APPLE_CHUNKS", "FakeLLM", "FakeStore", "GOOD_ANSWER", "GuardrailBlocked", "PLAN_APPLE", "make_deps"]
