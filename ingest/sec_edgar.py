"""Minimal, polite SEC EDGAR client for fetching each company's latest Form 10-K.

SEC fair-access rules: identify yourself with a descriptive User-Agent ("Name email@domain")
and stay under 10 requests per second.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: int
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession_nodash=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "url": self.url}


class EdgarClient:
    def __init__(self, user_agent: str, min_interval: float = 0.15, session: requests.Session | None = None):
        if not user_agent or "@" not in user_agent:
            raise ValueError('SEC requires a User-Agent like "Your Name your.email@example.com"')
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._min_interval = min_interval
        self._last_call = 0.0
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str, retries: int = 4) -> requests.Response:
        for attempt in range(retries):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            response = self._session.get(url, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError(f"unreachable: {url}")

    def cik_for(self, ticker: str) -> int:
        if self._ciks is None:
            data = self._get(TICKER_MAP_URL).json()
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        try:
            return self._ciks[ticker.upper()]
        except KeyError as exc:
            raise KeyError(f"Ticker {ticker} not found in SEC ticker map") from exc

    def latest_10k(self, ticker: str) -> Filing:
        cik = self.cik_for(ticker)
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form == "10-K":  # original filings only, not 10-K/A amendments
                return Filing(
                    ticker=ticker.upper(),
                    company=data.get("name", ticker),
                    cik=cik,
                    form=form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent.get("reportDate", [""] * len(recent["form"]))[i],
                    accession_number=recent["accessionNumber"][i],
                    primary_document=recent["primaryDocument"][i],
                )
        raise LookupError(f"No 10-K in recent filings for {ticker}")

    def fetch_html(self, filing: Filing) -> str:
        return self._get(filing.url).text


def html_to_text(html: str) -> str:
    """Flatten 10-K HTML (often inline XBRL) into plain text with one block per line."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):  # hidden XBRL metadata
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or re.fullmatch(r"(?:page )?\d{1,3}", line, flags=re.I):  # page numbers
            continue
        lines.append(line)
    return "\n".join(lines)
