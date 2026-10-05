"""Locate 10-K items (Business, Risk Factors, MD&A) in flattened filing text.

Every 10-K repeats item headings in its table of contents, so for each item we take every
candidate start, find the next end heading, and keep the longest span. TOC entries produce
tiny spans and lose to the real section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEP = r"\s*[.:\-\u2013\u2014]?\s*"


@dataclass(frozen=True)
class SectionSpec:
    code: str
    label: str
    start: str
    ends: tuple[str, ...]


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        code="item_1",
        label="Item 1. Business",
        start=rf"item\s*1(?![0-9a-z]){_SEP}business\b",
        ends=(rf"item\s*1a{_SEP}risk\s+factors",),
    ),
    SectionSpec(
        code="item_1a",
        label="Item 1A. Risk Factors",
        start=rf"item\s*1a{_SEP}risk\s+factors",
        ends=(rf"item\s*1b{_SEP}unresolved", rf"item\s*1c{_SEP}cybersecurity", rf"item\s*2{_SEP}properties"),
    ),
    SectionSpec(
        code="item_7",
        label="Item 7. Management's Discussion and Analysis",
        start=rf"item\s*7(?![0-9a-z]){_SEP}management[\u2019']?s\s+discussion",
        ends=(rf"item\s*7a{_SEP}quantitative", rf"item\s*8{_SEP}financial\s+statements"),
    ),
)

SECTION_LABELS = {s.code: s.label for s in SECTIONS}
MIN_SECTION_CHARS = 1500


def _longest_span(text: str, spec: SectionSpec) -> str | None:
    starts = [m for m in re.finditer(spec.start, text, flags=re.IGNORECASE)]
    end_re = re.compile("|".join(f"(?:{e})" for e in spec.ends), flags=re.IGNORECASE)
    best: tuple[int, int] | None = None
    for m in starts:
        end = end_re.search(text, m.end())
        stop = end.start() if end else len(text)
        if best is None or stop - m.start() > best[1] - best[0]:
            best = (m.start(), stop)
    if best is None:
        return None
    body = text[best[0] : best[1]].strip()
    return body if len(body) >= MIN_SECTION_CHARS else None


def extract_sections(text: str) -> dict[str, str]:
    """Return {section_code: section_text} for every section that could be located."""
    found = {}
    for spec in SECTIONS:
        body = _longest_span(text, spec)
        if body:
            found[spec.code] = body
    return found
