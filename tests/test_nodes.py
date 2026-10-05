"""Exercise the four nodes in graph order without LangGraph."""

from agent.citations import INSUFFICIENT, OUT_OF_SCOPE
from agent.nodes import REFUSALS, answer_node, parse_plan, plan_node, retrieve_node, verify_node
from tests.fakes import GOOD_ANSWER, PLAN_APPLE, GuardrailBlocked, make_deps


def run(question, deps, ticker_hint=None):  # noqa: ANN001
    state = {"question": question, "ticker_hint": ticker_hint, "trace": []}
    for node in (plan_node, retrieve_node, answer_node, verify_node):
        update = node(state, deps)
        trace = update.pop("trace", [])
        state.update(update)
        state["trace"] = state["trace"] + trace
        if state.get("refused"):
            break
    return state


def test_happy_path_returns_cited_answer():
    deps = make_deps([PLAN_APPLE, GOOD_ANSWER])
    state = run("What supply chain risks does Apple disclose?", deps)
    assert not state.get("refused"), state.get("refusal_details")
    assert state["ticker"] == "AAPL"
    assert [c["n"] for c in state["citations"]] == [1, 2]
    assert state["citations"][0]["chunk_id"] == "AAPL_1_item_1a_0001"
    assert [t["node"] for t in state["trace"]] == ["plan_node", "retrieve_node", "answer_node", "verify_node"]
    assert deps.store.calls[0][2] == "AAPL"


def test_excerpts_are_not_in_the_last_user_message():
    deps = make_deps([PLAN_APPLE, GOOD_ANSWER])
    run("What supply chain risks does Apple disclose?", deps)
    answer_call = deps.llm.calls[1]
    assert answer_call[-1] == {"role": "user", "content": "What supply chain risks does Apple disclose?"}
    assert "outsourcing partners" in answer_call[0]["content"]


def test_ui_company_overrides_planner():
    deps = make_deps(['{"search_query": "risks", "ticker": "MSFT"}', GOOD_ANSWER])
    state = run("What risks are disclosed?", deps, ticker_hint="AAPL")
    assert state["ticker"] == "AAPL"


def test_bad_planner_json_falls_back_to_question_and_detection():
    deps = make_deps(["sure! here is my plan", GOOD_ANSWER])
    state = run("What supply chain risks does Apple disclose?", deps)
    assert state["search_query"] == "What supply chain risks does Apple disclose?"
    assert state["ticker"] == "AAPL"
    assert state["trace"][0]["planned"] is False


def test_no_hits_refuses_before_answering():
    deps = make_deps([PLAN_APPLE], chunks=[])
    state = run("What supply chain risks does Apple disclose?", deps)
    assert state["refused"] and state["refusal_reason"] == "no_evidence"
    assert len(deps.llm.calls) == 1


def test_insufficient_evidence_refuses():
    state = run("Does Apple plan restaurants on Mars?", make_deps([PLAN_APPLE, INSUFFICIENT]))
    assert state["refusal_reason"] == "insufficient_evidence"
    assert state["answer"] == REFUSALS["insufficient_evidence"]


def test_investment_advice_refuses():
    state = run("Should I buy Apple stock?", make_deps([PLAN_APPLE, OUT_OF_SCOPE]))
    assert state["refusal_reason"] == "out_of_scope"


def test_uncited_or_invented_figures_fail_the_gate():
    draft = "Apple had $9,999 million of purchase commitments [2]."
    state = run("What commitments does Apple have?", make_deps([PLAN_APPLE, draft]))
    assert state["refusal_reason"] == "citation_check_failed"
    assert "9999" in state["refusal_details"][0]


def test_guardrail_block_becomes_refusal():
    state = run("my SSN is 123-45-6789", make_deps([GuardrailBlocked("400 PII detected; blocked by guardrail")]))
    assert state["refusal_reason"] == "blocked"


def test_parse_plan_handles_fences():
    assert parse_plan('```json\n{"search_query": "x", "ticker": null}\n```') == {"search_query": "x", "ticker": None}
