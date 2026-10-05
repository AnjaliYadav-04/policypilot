"""Citation parsing and the hard citation gate.

Pure Python with no third-party imports, so it can be unit-tested anywhere and reused by the
evaluation scorers.

Rules enforced by :func:`verify_answer`:
  1. The draft must not be one of the model's refusal sentinels.
  2. Every claim (sentence or bullet of 4+ words) carries at least one ``[n]`` marker. A short,
     figure-free line that introduces a bulleted list is treated as framing, not a claim.
  3. Every marker points at a source that was actually retrieved (1..num_sources).
  4. With ``strict_numbers``, every figure in a claim appears verbatim in a cited source,
     so a hallucinated dollar amount or percentage fails the gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITE_GROUP = r"(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+"
_BULLET_RE = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u201c])")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_FORM_RE = re.compile(r"\b(?:10-K|10-Q|8-K|20-F)(?:/A)?\b", re.IGNORECASE)

# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = (
    "U.S.", "U.K.", "E.U.", "e.g.", "i.e.", "Inc.", "Corp.", "Co.", "Ltd.", "No.",
    "Nos.", "vs.", "approx.", "Mr.", "Ms.", "Dr.", "St.", "Jan.", "Feb.", "Mar.",
    "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.",
)
_DOT = "\u2024"  # one-dot leader: a placeholder that is not a sentence terminator

MIN_CLAIM_WORDS = 4


@dataclass
class VerificationResult:
    ok: bool
    reason: str = ""
    claims: list[str] = field(default_factory=list)
    cited_ids: list[int] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def parse_ids(text: str) -> list[int]:
    """All source ids referenced in ``text``, in order of appearance (with repeats)."""
    ids: list[int] = []
    for group in CITATION_RE.findall(text):
        ids.extend(int(x) for x in re.split(r"\s*,\s*", group))
    return ids


def _protect_abbreviations(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = re.sub(rf"(?<!\w){re.escape(abbr)}", abbr.replace(".", _DOT), text)
    return text


def _move_citations_before_punctuation(text: str) -> str:
    """'... suppliers. [1][2] Next' -> '... suppliers [1][2]. Next' so splitting keeps them."""
    return re.sub(rf"([.!?])\s*({_CITE_GROUP})", lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)


def split_claims(text: str) -> list[str]:
    """Split an answer into checkable claims: sentences and bullet items, minus lead-ins."""
    text = _move_citations_before_punctuation(_protect_abbreviations(text))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    claims: list[str] = []
    for i, raw_line in enumerate(lines):
        is_bullet = bool(_BULLET_RE.match(raw_line))
        line = _BULLET_RE.sub("", raw_line).strip()
        if not line or line.endswith(":"):
            continue
        if not is_bullet and _is_lead_in(line, lines[i + 1] if i + 1 < len(lines) else ""):
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = re.sub(r"\s+", " ", sentence).strip().replace(_DOT, ".")
            if len(CITATION_RE.sub("", sentence).split()) >= MIN_CLAIM_WORDS:
                claims.append(sentence)
    return claims


def _is_lead_in(line: str, next_line: str) -> bool:
    """A short, figure-free line introducing a bulleted list is framing, not a claim."""
    return (
        bool(_BULLET_RE.match(next_line))
        and not parse_ids(line)
        and len(line.split()) <= 20
        and not _figures(line)
    )


def _normalise_number(token: str) -> str:
    return token.replace(",", "").rstrip(".")


def _figures(claim: str) -> list[str]:
    """Figures worth checking: drop citation markers, years and single digits."""
    body = _FORM_RE.sub(" ", CITATION_RE.sub(" ", claim))
    out = []
    for token in _NUMBER_RE.findall(body):
        norm = _normalise_number(token)
        if len(norm.replace(".", "")) < 2 or _YEAR_RE.match(norm):
            continue
        out.append(norm)
    return out


def verify_answer(
    draft: str,
    num_sources: int,
    source_texts: list[str] | None = None,
    strict_numbers: bool = True,
) -> VerificationResult:
    text = (draft or "").strip()
    if OUT_OF_SCOPE in text:
        return VerificationResult(ok=False, reason="out_of_scope")
    if INSUFFICIENT in text:
        return VerificationResult(ok=False, reason="insufficient_evidence")

    claims = split_claims(text)
    if not claims:
        return VerificationResult(ok=False, reason="empty_answer", failures=["no checkable claims"])

    normalised_sources = [_normalise_number(s) for s in (source_texts or [])]
    failures: list[str] = []
    cited: list[int] = []

    for claim in claims:
        ids = parse_ids(claim)
        if not ids:
            failures.append(f"Uncited claim: {claim[:120]}")
            continue
        bad = [i for i in ids if i < 1 or i > num_sources]
        if bad:
            failures.append(f"Citation to a source that wasn't retrieved {bad}: {claim[:120]}")
            continue
        cited.extend(ids)
        if strict_numbers and normalised_sources:
            evidence = " ".join(normalised_sources[i - 1] for i in set(ids))
            missing = [f for f in _figures(claim) if f not in evidence]
            if missing:
                failures.append(f"Figure(s) {missing} not found in cited source(s) {sorted(set(ids))}")

    return VerificationResult(
        ok=not failures,
        reason="" if not failures else "citation_check_failed",
        claims=claims,
        cited_ids=list(dict.fromkeys(cited)),
        failures=failures,
    )


def renumber_citations(text: str) -> tuple[str, list[int]]:
    """Renumber markers 1..m in order of first use and normalise '[1, 3]' to '[1][3]'.

    Returns the rewritten text and the original ids, in their new order, so callers can
    build the matching source list.
    """
    order = list(dict.fromkeys(parse_ids(text)))
    mapping = {old: new for new, old in enumerate(order, start=1)}

    def _sub(match: re.Match[str]) -> str:
        ids = [int(x) for x in re.split(r"\s*,\s*", match.group(1))]
        return "".join(f"[{mapping[i]}]" for i in dict.fromkeys(ids))

    return CITATION_RE.sub(_sub, text), order
