# Databricks notebook source
# MAGIC %md
# MAGIC # Seed `filings.chunks`
# MAGIC
# MAGIC Fetches each covered company's latest Form 10-K from SEC EDGAR, extracts Item 1, Item 1A and
# MAGIC Item 7, chunks them, embeds every chunk with `all-MiniLM-L6-v2` (self-managed embeddings), and
# MAGIC MERGEs the rows into a Delta table with Change Data Feed enabled, which the Vector Search
# MAGIC Delta-Sync index reads from. Safe to re-run: rows are keyed by `chunk_id`, and chunks from
# MAGIC superseded filings are deleted.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-ingest.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

dbutils.widgets.text("catalog", "policypilot_dev")  # noqa: F821
dbutils.widgets.text("schema", "filings")  # noqa: F821
dbutils.widgets.text("tickers", "")  # noqa: F821  (blank = all 20)
dbutils.widgets.text("sec_user_agent", "")  # noqa: F821  e.g. "Anjali Yadav anjali@example.com"

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821
user_agent = dbutils.widgets.get("sec_user_agent")  # noqa: F821
table = f"{catalog}.{schema}.chunks"

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(".."))  # repo root, synced by the bundle

from agent.companies import COMPANIES
from agent.config import Settings
from agent.embeddings import get_embedder
from ingest.chunking import build_chunks, embedding_input
from ingest.sec_edgar import EdgarClient, html_to_text
from ingest.sections import extract_sections

tickers = [t.strip().upper() for t in dbutils.widgets.get("tickers").split(",") if t.strip()] or list(COMPANIES)  # noqa: F821

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")  # noqa: F821
spark.sql(  # noqa: F821
    f"""
    CREATE TABLE IF NOT EXISTS {table} (
      chunk_id STRING NOT NULL,
      ticker STRING,
      company STRING,
      form STRING,
      filing_date STRING,
      accession_number STRING,
      section STRING,
      chunk_index INT,
      text STRING,
      source_url STRING,
      embedding ARRAY<FLOAT>,
      ingested_at TIMESTAMP
    )
    COMMENT 'PolicyPilot: 10-K chunks with all-MiniLM-L6-v2 embeddings'
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

# COMMAND ----------

edgar = EdgarClient(user_agent)
embedder = get_embedder(Settings().embedding_model)

rows, latest_accessions = [], {}
for ticker in tickers:
    filing = edgar.latest_10k(ticker)
    sections = extract_sections(html_to_text(edgar.fetch_html(filing)))
    chunks = build_chunks(filing, sections)
    vectors = embedder.embed([embedding_input(c) for c in chunks])
    rows.extend({**c.to_row(), "embedding": v} for c, v in zip(chunks, vectors))
    latest_accessions[ticker] = filing.accession_number
    print(f"{ticker}: filed {filing.filing_date}, sections={sorted(sections)}, chunks={len(chunks)}")

print(f"Prepared {len(rows)} chunks for {len(tickers)} companies")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

row_schema = StructType(
    [StructField(n, StringType()) for n in ("chunk_id", "ticker", "company", "form", "filing_date", "accession_number", "section")]
    + [StructField("chunk_index", IntegerType()), StructField("text", StringType()), StructField("source_url", StringType())]
    + [StructField("embedding", ArrayType(FloatType()))]
)
staged = spark.createDataFrame(rows, schema=row_schema).withColumn("ingested_at", F.current_timestamp())  # noqa: F821
staged.createOrReplaceTempView("staged_chunks")

spark.sql(  # noqa: F821
    f"""
    MERGE INTO {table} AS t
    USING staged_chunks AS s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED AND (t.text <> s.text) THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)

# One 10-K per company: drop chunks from older filings of the tickers we just refreshed.
for ticker, accession in latest_accessions.items():
    spark.sql(f"DELETE FROM {table} WHERE ticker = '{ticker}' AND accession_number <> '{accession}'")  # noqa: F821

display(spark.sql(f"SELECT ticker, count(*) AS chunks, max(filing_date) AS filed FROM {table} GROUP BY ticker ORDER BY ticker"))  # noqa: F821
