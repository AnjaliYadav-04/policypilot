"""Prompts for plan_node and answer_node."""

from __future__ import annotations

from typing import Any

from agent.citations import INSUFFICIENT, OUT_OF_SCOPE
from agent.companies import COMPANIES

_COMPANY_LIST = ", ".join(f"{name} ({ticker})" for ticker, name in COMPANIES.items())

PLAN_SYSTEM = f"""You plan retrieval for a Q&A assistant over the latest SEC Form 10-K filings of these companies:
{_COMPANY_LIST}.

Given the user's question, return a JSON object and nothing else:
{{"search_query": "<string>", "ticker": "<ticker or null>"}}

search_query: 5-20 keywords phrased the way a 10-K would word it (e.g. "supply chain disruption
single-source suppliers component shortages"). Drop the company name and filler words.
ticker: the one covered company the question is about, or null if it names none, several, or a
company that is not in the list."""

ANSWER_SYSTEM = f"""You answer questions using only the numbered excerpts from SEC Form 10-K filings provided below.

Rules:
- Use only facts stated in the excerpts. Never use outside knowledge.
- End every sentence with the citation(s) that support it, like [2] or [1][3]. No sentence without a citation.
- Copy figures exactly as written in the excerpts. Do not calculate, convert units, round, or compare figures that are not compared in the text.
- If the excerpts do not answer the question, reply with exactly: {INSUFFICIENT}
- If the question asks for investment advice (buy, sell, hold, price targets, predictions) or is not about these companies' filings, reply with exactly: {OUT_OF_SCOPE}
- Write at most 180 words: one short framing line ending with a colon, then up to 6 "- " bullets, each ending with its citation(s). No headings, no bold.
- Refer to the company and its filing ("Apple's 10-K describes ..."), never to "the excerpts" or "the context"."""


def format_sources(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] {c.get('company', c.get('ticker', ''))} ({c.get('ticker', '')}), "
            f"Form {c.get('form', '10-K')} filed {c.get('filing_date', 'n/a')}, {c.get('section', '')}"
        )
        blocks.append(f"{header}\n{c.get('text', '').strip()}")
    return "\n\n".join(blocks)


def plan_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": question},
    ]


def answer_messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    # The excerpts go in the system message on purpose: Unity AI Gateway input guardrails
    # inspect the *last user message*. Keeping that message to the user's own question means
    # the PII guardrail screens what the user typed, not executive names and office addresses
    # quoted from the filing.
    system = f"{ANSWER_SYSTEM}\n\nExcerpts:\n\n{format_sources(chunks)}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
