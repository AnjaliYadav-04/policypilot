"""Shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    ticker_hint: str | None  # company picked in the UI; overrides detection

    # plan_node
    search_query: str
    ticker: str | None

    # retrieve_node
    chunks: list[dict[str, Any]]

    # answer_node
    draft: str

    # verify_node
    answer: str
    citations: list[dict[str, Any]]

    # any node may refuse
    refused: bool
    refusal_reason: str
    refusal_details: list[str]

    # per-node diagnostics, appended to by every node
    trace: Annotated[list[dict[str, Any]], operator.add]
