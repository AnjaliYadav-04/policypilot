"""The 20 companies PolicyPilot covers, plus helpers to recognise them in free text."""

from __future__ import annotations

import re

COMPANIES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "The Procter & Gamble Company",
    "XOM": "Exxon Mobil Corporation",
    "UNH": "UnitedHealth Group Incorporated",
    "HD": "The Home Depot, Inc.",
    "KO": "The Coca-Cola Company",
    "PFE": "Pfizer Inc.",
    "DIS": "The Walt Disney Company",
    "INTC": "Intel Corporation",
    "NFLX": "Netflix, Inc.",
}

# Lower-case names people actually type, mapped to tickers.
ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",
    "visa": "V",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "j&j": "JNJ",
    "walmart": "WMT",
    "procter & gamble": "PG",
    "procter and gamble": "PG",
    "p&g": "PG",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "home depot": "HD",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pfizer": "PFE",
    "disney": "DIS",
    "walt disney": "DIS",
    "intel": "INTC",
    "netflix": "NFLX",
}

_ALIAS_RE = re.compile(
    r"(?<![\w&])("
    + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True))
    + r")(?![\w&])",
    re.IGNORECASE,
)
# Tickers are matched case-sensitively, and single-letter tickers only with a "$" prefix,
# so ordinary words ("V", "PG-13", "HD video") don't trigger false matches.
_MULTI = "|".join(t for t in COMPANIES if len(t) > 1)
_SINGLE = "|".join(t for t in COMPANIES if len(t) == 1)
_TICKER_RE = re.compile(rf"(?<![\w$-])\$?({_MULTI})(?![\w-])|\$({_SINGLE})(?!\w)")


def normalize_ticker(value: object) -> str | None:
    """Return the canonical ticker if ``value`` names a covered company, else None."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper().lstrip("$")
    if cleaned in COMPANIES:
        return cleaned
    return ALIASES.get(value.strip().lower())


def detect_ticker(text: str) -> str | None:
    """Find the first covered company mentioned in ``text`` (by name or ticker)."""
    hits: list[tuple[int, str]] = []
    for m in _ALIAS_RE.finditer(text):
        hits.append((m.start(), ALIASES[m.group(1).lower()]))
    for m in _TICKER_RE.finditer(text):
        hits.append((m.start(), m.group(1) or m.group(2)))
    return min(hits)[1] if hits else None


def company_label(ticker: str) -> str:
    return f"{COMPANIES.get(ticker, ticker)} ({ticker})"
