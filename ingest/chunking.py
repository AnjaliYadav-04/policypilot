"""Paragraph-aware chunking with word overlap, plus the Delta row schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ingest.sec_edgar import Filing
from ingest.sections import SECTION_LABELS


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    company: str
    form: str
    filing_date: str
    accession_number: str
    section: str
    chunk_index: int
    text: str
    source_url: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    step = max(1, size - overlap)
    return [words[i : i + size] for i in range(0, max(1, len(words) - overlap), step)]


def chunk_text(text: str, target_words: int = 320, overlap_words: int = 60, min_words: int = 60) -> list[str]:
    """Pack paragraphs into ~target_words chunks; overlap the tail of each chunk into the next.

    Paragraphs longer than 1.5x the target are split into overlapping word windows first.
    A short final chunk is merged into the previous one.
    """
    paragraphs: list[list[str]] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            continue
        if len(words) > target_words * 1.5:
            paragraphs.extend(_windows(words, target_words, overlap_words))
        else:
            paragraphs.append(words)

    chunks: list[list[str]] = []
    current: list[str] = []
    for words in paragraphs:
        if current and len(current) + len(words) > target_words:
            chunks.append(current)
            current = current[-overlap_words:] if overlap_words else []
        current = current + words
    if current:
        if chunks and len(current) < min_words:
            chunks[-1] = chunks[-1] + current[overlap_words:]
        else:
            chunks.append(current)
    return [" ".join(c) for c in chunks]


def build_chunks(filing: Filing, sections: dict[str, str], **chunk_kwargs: Any) -> list[Chunk]:
    accession = filing.accession_number.replace("-", "")
    out: list[Chunk] = []
    for code, body in sections.items():
        for i, text in enumerate(chunk_text(body, **chunk_kwargs)):
            out.append(
                Chunk(
                    chunk_id=f"{filing.ticker}_{accession}_{code}_{i:04d}",
                    ticker=filing.ticker,
                    company=filing.company,
                    form=filing.form,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    section=SECTION_LABELS.get(code, code),
                    chunk_index=i,
                    text=text,
                    source_url=filing.url,
                )
            )
    return out


def embedding_input(chunk: Chunk | dict[str, Any]) -> str:
    """Text that gets embedded: a short header gives the vector company and section context."""
    c = chunk if isinstance(chunk, dict) else chunk.to_row()
    return f"{c['company']} ({c['ticker']}) {c['form']}, {c['section']}. {c['text']}"
