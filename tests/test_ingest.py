from ingest.chunking import build_chunks, chunk_text, embedding_input
from ingest.sec_edgar import Filing, html_to_text
from ingest.sections import extract_sections

FILLER = "The Company faces risks that could materially affect its business and results. " * 40


def _ten_k() -> str:
    toc = "\n".join(
        [
            "Item 1. Business 1",
            "Item 1A. Risk Factors 5",
            "Item 1B. Unresolved Staff Comments 17",
            "Item 7. Management's Discussion and Analysis of Financial Condition 22",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 30",
            "Item 8. Financial Statements and Supplementary Data 31",
        ]
    )
    body = "\n".join(
        [
            "Item 1. Business",
            "BUSINESS BODY " + FILLER,
            "Item 1A. Risk Factors",
            "RISK BODY " + FILLER,
            "Item 1B. Unresolved Staff Comments",
            "None.",
            "Item 7. Management\u2019s Discussion and Analysis of Financial Condition and Results of Operations",
            "MDA BODY " + FILLER,
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            "Item 8. Financial Statements and Supplementary Data",
        ]
    )
    return toc + "\n" + body


def test_extract_sections_skips_table_of_contents():
    sections = extract_sections(_ten_k())
    assert set(sections) == {"item_1", "item_1a", "item_7"}
    assert "BUSINESS BODY" in sections["item_1"] and "RISK BODY" not in sections["item_1"]
    assert "RISK BODY" in sections["item_1a"] and "MDA BODY" not in sections["item_1a"]
    assert "MDA BODY" in sections["item_7"]


def test_chunk_text_sizes_and_overlap():
    text = "\n".join(f"Paragraph {i} " + "word " * 90 for i in range(12))
    chunks = chunk_text(text, target_words=300, overlap_words=50)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 300 + 90 + 50 for c in chunks)
    first_tail = chunks[0].split()[-50:]
    assert chunks[1].split()[:50] == first_tail


def test_long_paragraph_is_windowed():
    chunks = chunk_text("word " * 2000, target_words=300, overlap_words=50)
    assert len(chunks) >= 6


def test_build_chunks_ids_and_metadata():
    filing = Filing("AAPL", "Apple Inc.", 320193, "10-K", "2025-10-31", "2025-09-27", "0000320193-25-000079", "aapl.htm")
    chunks = build_chunks(filing, {"item_1a": "RISK " + FILLER})
    assert chunks[0].chunk_id == "AAPL_000032019325000079_item_1a_0000"
    assert chunks[0].section == "Item 1A. Risk Factors"
    assert chunks[0].source_url.endswith("/320193/000032019325000079/aapl.htm")
    assert embedding_input(chunks[0]).startswith("Apple Inc. (AAPL) 10-K, Item 1A. Risk Factors.")


def test_html_to_text_drops_scripts_xbrl_header_and_page_numbers():
    html = (
        "<html><head><title>x</title></head><body><ix:header>hidden</ix:header>"
        "<script>bad()</script><p>Item&nbsp;1A. Risk Factors</p><p>12</p><p>Real text.</p></body></html>"
    )
    text = html_to_text(html)
    assert "hidden" not in text and "bad()" not in text
    assert text.splitlines() == ["Item 1A. Risk Factors", "Real text."]
