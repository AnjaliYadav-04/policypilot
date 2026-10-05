from agent.companies import detect_ticker, normalize_ticker


def test_detects_names_and_tickers():
    assert detect_ticker("What supply chain risks does Apple disclose?") == "AAPL"
    assert detect_ticker("Compare NVDA export risks") == "NVDA"
    assert detect_ticker("What does J&J say about talc litigation?") == "JNJ"
    assert detect_ticker("How does Visa describe fraud risk?") == "V"


def test_avoids_false_positives():
    assert detect_ticker("What is artificial intelligence?") is None  # 'intel' inside a word
    assert detect_ticker("V for vendetta") is None  # single-letter tickers need a $ prefix
    assert detect_ticker("What did $V report?") == "V"
    assert detect_ticker("banana bread recipe") is None


def test_first_mention_wins():
    assert detect_ticker("Does Microsoft or Apple disclose more AI risk?") == "MSFT"


def test_normalize_ticker():
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker("$tsla") == "TSLA"
    assert normalize_ticker("Google") == "GOOGL"
    assert normalize_ticker("IBM") is None
    assert normalize_ticker(None) is None
