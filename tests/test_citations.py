from agent.citations import INSUFFICIENT, OUT_OF_SCOPE, parse_ids, renumber_citations, split_claims, verify_answer

SOURCES = ["Revenue was $383,285 million in fiscal 2023.", "Gross margin was 44.1 percent."]


def test_parse_ids_handles_groups_and_lists():
    assert parse_ids("a [1] b [2, 3] c [1][4]") == [1, 2, 3, 1, 4]


def test_split_claims_skips_lead_ins_and_handles_abbreviations():
    text = "Key points:\n- Apple sells in the U.S. and abroad [1].\n- Margins rose sharply [2]. Costs fell a lot [2]."
    assert split_claims(text) == [
        "Apple sells in the U.S. and abroad [1].",
        "Margins rose sharply [2].",
        "Costs fell a lot [2].",
    ]


def test_short_framing_line_before_bullets_is_not_a_claim():
    text = "Apple's 10-K describes several supply chain risks.\n- It relies on outsourcing partners in Asia [1]."
    assert split_claims(text) == ["It relies on outsourcing partners in Asia [1]."]


def test_framing_line_with_a_figure_must_be_cited():
    text = "Apple reported $383,285 million of revenue.\n- It relies on outsourcing partners in Asia [1]."
    assert verify_answer(text, 2, SOURCES).failures[0].startswith("Uncited claim")


def test_citation_after_period_stays_with_its_sentence():
    claims = split_claims("Apple relies on partners in Asia. [1] It also uses single sources. [2]")
    assert all(parse_ids(c) for c in claims), claims


def test_verify_passes_fully_cited_answer():
    draft = "Revenue was $383,285 million in fiscal 2023 [1]. Gross margin was 44.1 percent [2]."
    result = verify_answer(draft, num_sources=2, source_texts=SOURCES)
    assert result.ok, result.failures
    assert result.cited_ids == [1, 2]


def test_verify_rejects_uncited_claim():
    draft = "Revenue was $383,285 million in fiscal 2023 [1]. The company expects strong growth next year."
    result = verify_answer(draft, num_sources=2, source_texts=SOURCES)
    assert not result.ok
    assert result.reason == "citation_check_failed"
    assert "Uncited claim" in result.failures[0]


def test_verify_rejects_citation_to_unretrieved_source():
    result = verify_answer("Gross margin was 44.1 percent for the year [3].", num_sources=2, source_texts=SOURCES)
    assert not result.ok and "wasn't retrieved" in result.failures[0]


def test_verify_rejects_hallucinated_figure():
    result = verify_answer("Gross margin was 47.5 percent for the year [2].", num_sources=2, source_texts=SOURCES)
    assert not result.ok and "47.5" in result.failures[0]


def test_strict_numbers_can_be_disabled():
    result = verify_answer(
        "Gross margin was 47.5 percent for the year [2].", num_sources=2, source_texts=SOURCES, strict_numbers=False
    )
    assert result.ok


def test_form_names_and_years_are_not_treated_as_figures():
    draft = "Apple's 10-K for fiscal 2023 reports revenue of $383,285 million [1]."
    assert verify_answer(draft, num_sources=2, source_texts=SOURCES).ok


def test_sentinels_map_to_reasons():
    assert verify_answer(INSUFFICIENT, 2).reason == "insufficient_evidence"
    assert verify_answer(OUT_OF_SCOPE, 2).reason == "out_of_scope"


def test_renumber_orders_by_first_use_and_splits_groups():
    text, order = renumber_citations("A [4]. B [2, 4]. C [2].")
    assert order == [4, 2]
    assert text == "A [1]. B [2][1]. C [2]."
