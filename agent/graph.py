"""LangGraph wiring: plan -> retrieve -> answer -> verify, with early exit on refusal.

This module is the same in every environment; PP_ENV only changes which LLMClient and
VectorStore get injected (see agent/config.py).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.config import Settings, load_settings
from agent.llm import get_llm_client
from agent.nodes import AgentDeps, answer_node, plan_node, retrieve_node, verify_node
from agent.state import AgentState
from agent.tracing import trace
from agent.vector_store import get_vector_store


def _continue_or_stop(state: AgentState) -> str:
    return "stop" if state.get("refused") else "continue"


def build_graph(deps: AgentDeps) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("plan_node", lambda s: plan_node(s, deps))
    graph.add_node("retrieve_node", lambda s: retrieve_node(s, deps))
    graph.add_node("answer_node", lambda s: answer_node(s, deps))
    graph.add_node("verify_node", lambda s: verify_node(s, deps))

    graph.add_edge(START, "plan_node")
    graph.add_conditional_edges("plan_node", _continue_or_stop, {"continue": "retrieve_node", "stop": END})
    graph.add_conditional_edges("retrieve_node", _continue_or_stop, {"continue": "answer_node", "stop": END})
    graph.add_conditional_edges("answer_node", _continue_or_stop, {"continue": "verify_node", "stop": END})
    graph.add_edge("verify_node", END)
    return graph.compile()


def default_deps(settings: Settings | None = None) -> AgentDeps:
    settings = settings or load_settings()
    return AgentDeps(llm=get_llm_client(settings), store=get_vector_store(settings), settings=settings)


@lru_cache(maxsize=1)
def get_graph() -> Any:
    return build_graph(default_deps())


def to_response(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "refused": bool(state.get("refused")),
        "refusal_reason": state.get("refusal_reason", ""),
        "refusal_details": state.get("refusal_details", []),
        "ticker": state.get("ticker"),
        "search_query": state.get("search_query", ""),
        "trace": state.get("trace", []),
    }


@trace("policypilot", span_type="AGENT")
def run_agent(question: str, ticker: str | None = None, graph: Any = None) -> dict[str, Any]:
    graph = graph or get_graph()
    final = graph.invoke({"question": question.strip(), "ticker_hint": ticker, "trace": []})
    return to_response(final)
