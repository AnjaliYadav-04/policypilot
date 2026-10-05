import pytest

pytest.importorskip("langgraph")

from agent.graph import build_graph, to_response  # noqa: E402
from tests.fakes import GOOD_ANSWER, PLAN_APPLE, make_deps  # noqa: E402


def test_graph_end_to_end_with_fakes():
    graph = build_graph(make_deps([PLAN_APPLE, GOOD_ANSWER]))
    result = to_response(graph.invoke({"question": "What supply chain risks does Apple disclose?", "trace": []}))
    assert not result["refused"]
    assert len(result["citations"]) == 2
    assert [t["node"] for t in result["trace"]] == ["plan_node", "retrieve_node", "answer_node", "verify_node"]


def test_graph_stops_at_refusal():
    graph = build_graph(make_deps([PLAN_APPLE], chunks=[]))
    result = to_response(graph.invoke({"question": "Anything?", "trace": []}))
    assert result["refused"] and result["refusal_reason"] == "no_evidence"
    assert [t["node"] for t in result["trace"]] == ["plan_node", "retrieve_node"]
