"""The four agent nodes. Each takes the state plus injected dependencies and returns a
partial state update, so they can be unit-tested with fakes and no LangGraph installed."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.citations import renumber_citations, verify_answer
from agent.companies import detect_ticker, normalize_ticker
from agent.config import Settings
from agent.llm import GuardrailBlocked, LLMClient
from agent.prompts import answer_messages, plan_messages
from agent.state import AgentState
from agent.tracing import from_documents, to_documents, trace
from agent.vector_store import VectorStore

REFUSALS: dict[str, str] = {
    "insufficient_evidence": (
        "The 10-K passages I found don't answer this, so I won't guess. Try naming the company, "
        "or ask about something a 10-K covers, such as risk factors, business lines or results of operations."
    ),
    "no_evidence": (
        "I found no passages for this in the indexed 10-K filings. Check that the company is one "
        "of the 20 covered, or rephrase using terms a filing would use."
    ),
    "citation_check_failed": (
        "I drafted an answer, but parts of it weren't backed by a cited 10-K passage, so I'm not "
        "showing it. Try a narrower question."
    ),
    "out_of_scope": (
        "I only answer questions about what these companies disclose in their 10-K filings. "
        "I can't give investment advice or answer unrelated questions."
    ),
    "blocked": (
        "This request was blocked by the AI Gateway's safety and privacy guardrails. Remove any "
        "personal details and ask about a company's filing."
    ),
}


def is_refusal_text(text: str) -> bool:
    return text.strip() in REFUSALS.values()


@dataclass
class AgentDeps:
    llm: LLMClient
    store: VectorStore
    settings: Settings


def _step(node: str, started: float, **info: Any) -> dict[str, Any]:
    return {"node": node, "ms": round((time.perf_counter() - started) * 1000), **info}


def _refuse(reason: str, node: str, started: float, details: list[str] | None = None) -> AgentState:
    return {
        "refused": True,
        "refusal_reason": reason,
        "refusal_details": details or [],
        "answer": REFUSALS[reason],
        "citations": [],
        "trace": [_step(node, started, refused=reason)],
    }


def parse_plan(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of the planner's reply (tolerates code fences and chatter)."""
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        raise ValueError("planner returned no JSON object")
    plan = json.loads(match.group(0))
    if not isinstance(plan, dict):
        raise ValueError("planner JSON is not an object")
    return plan


@trace("plan_node", span_type="LLM")
def plan_node(state: AgentState, deps: AgentDeps) -> AgentState:
    started = time.perf_counter()
    question = state["question"]
    try:
        plan = parse_plan(deps.llm.complete(plan_messages(question), max_tokens=400))
        planned = True
    except GuardrailBlocked as exc:
        return _refuse("blocked", "plan_node", started, [str(exc)[:300]])
    except ValueError:  # includes json.JSONDecodeError
        plan, planned = {}, False

    search_query = str(plan.get("search_query") or "").strip() or question
    ticker = (
        normalize_ticker(state.get("ticker_hint"))
        or normalize_ticker(plan.get("ticker"))
        or detect_ticker(question)
    )
    return {
        "search_query": search_query,
        "ticker": ticker,
        "trace": [_step("plan_node", started, search_query=search_query, ticker=ticker, planned=planned)],
    }


@trace("vector_search", span_type="RETRIEVER")
def _retrieve_documents(store: VectorStore, query: str, k: int, ticker: str | None) -> list[dict[str, Any]]:
    return to_documents(store.search(query, k=k, ticker=ticker))


def retrieve_node(state: AgentState, deps: AgentDeps) -> AgentState:
    started = time.perf_counter()
    query = state.get("search_query") or state["question"]
    chunks = from_documents(_retrieve_documents(deps.store, query, deps.settings.top_k, state.get("ticker")))
    if not chunks:
        return _refuse("no_evidence", "retrieve_node", started)
    return {
        "chunks": chunks,
        "trace": [_step("retrieve_node", started, hits=len(chunks), top_score=round(chunks[0].get("score", 0.0), 4))],
    }


@trace("answer_node", span_type="LLM")
def answer_node(state: AgentState, deps: AgentDeps) -> AgentState:
    started = time.perf_counter()
    try:
        draft = deps.llm.complete(answer_messages(state["question"], state["chunks"]))
    except GuardrailBlocked as exc:
        return _refuse("blocked", "answer_node", started, [str(exc)[:300]])
    return {"draft": draft, "trace": [_step("answer_node", started, chars=len(draft))]}


@trace("verify_node", span_type="PARSER")
def verify_node(state: AgentState, deps: AgentDeps) -> AgentState:
    started = time.perf_counter()
    chunks = state.get("chunks", [])
    result = verify_answer(
        state.get("draft", ""),
        num_sources=len(chunks),
        source_texts=[c.get("text", "") for c in chunks],
        strict_numbers=deps.settings.strict_numbers,
    )
    if not result.ok:
        reason = result.reason if result.reason in REFUSALS else "citation_check_failed"
        return _refuse(reason, "verify_node", started, result.failures)

    answer, kept = renumber_citations(state["draft"])
    citations = [
        {"n": n, **{k: v for k, v in chunks[old - 1].items() if k != "score"}}
        for n, old in enumerate(kept, start=1)
    ]
    return {
        "answer": answer,
        "citations": citations,
        "refused": False,
        "trace": [_step("verify_node", started, claims=len(result.claims), sources_cited=len(citations))],
    }
