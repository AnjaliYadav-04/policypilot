"""Build data/local_chunks.jsonl for PP_ENV=local development.

Runs the same ingestion code as notebooks/seed_chunks_table.py, but writes a JSONL file
instead of a Delta table, so you can iterate on the agent without a Databricks workspace.

    python scripts/build_local_index.py --tickers AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.companies import COMPANIES  # noqa: E402
from agent.config import load_settings  # noqa: E402
from agent.embeddings import get_embedder  # noqa: E402
from ingest.chunking import build_chunks, embedding_input  # noqa: E402
from ingest.sec_edgar import EdgarClient, html_to_text  # noqa: E402
from ingest.sections import extract_sections  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=list(COMPANIES), help="default: all 20")
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    parser.add_argument("--out", default=load_settings().local_index_path)
    args = parser.parse_args()

    settings = load_settings()
    edgar = EdgarClient(args.user_agent)
    embedder = get_embedder(settings.embedding_model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for ticker in args.tickers:
            filing = edgar.latest_10k(ticker)
            sections = extract_sections(html_to_text(edgar.fetch_html(filing)))
            chunks = build_chunks(filing, sections)
            vectors = embedder.embed([embedding_input(c) for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                fh.write(json.dumps({**chunk.to_row(), "embedding": vector}) + "\n")
            total += len(chunks)
            print(f"{ticker}: 10-K filed {filing.filing_date}, sections={sorted(sections)}, chunks={len(chunks)}")
    print(f"Wrote {total} chunks to {out}")


if __name__ == "__main__":
    main()
