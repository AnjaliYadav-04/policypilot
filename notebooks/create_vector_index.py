# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search endpoint and Delta-Sync index
# MAGIC
# MAGIC Creates `policypilot-vs-endpoint` (if missing) and a Delta-Sync index over the chunks table using
# MAGIC the precomputed `embedding` column (self-managed, 384 dims). On later runs it triggers a sync,
# MAGIC which reads only the rows that changed, via Change Data Feed.

# COMMAND ----------

dbutils.widgets.text("catalog", "policypilot_dev")  # noqa: F821
dbutils.widgets.text("schema", "filings")  # noqa: F821
dbutils.widgets.text("vs_endpoint", "policypilot-vs-endpoint")  # noqa: F821

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821
endpoint_name = dbutils.widgets.get("vs_endpoint")  # noqa: F821
source_table = f"{catalog}.{schema}.chunks"
index_name = f"{catalog}.{schema}.chunks_index"

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

if endpoint_name not in {e.name for e in w.vector_search_endpoints.list_endpoints()}:
    print(f"Creating endpoint {endpoint_name} (takes several minutes)...")
    w.vector_search_endpoints.create_endpoint_and_wait(name=endpoint_name, endpoint_type=EndpointType.STANDARD)
print(f"Endpoint {endpoint_name} is ready")

# COMMAND ----------

try:
    w.vector_search_indexes.get_index(index_name)
    print(f"Index {index_name} exists; triggering a sync")
    w.vector_search_indexes.sync_index(index_name)
except NotFound:
    print(f"Creating index {index_name}")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=384)],
            columns_to_sync=[
                "chunk_id", "ticker", "company", "form", "filing_date",
                "accession_number", "section", "text", "source_url", "embedding",
            ],
        ),
    )

# COMMAND ----------

deadline = time.time() + 45 * 60
while True:
    status = w.vector_search_indexes.get_index(index_name).status
    print(f"ready={status.ready} rows={status.indexed_row_count} {status.message or ''}")
    if status.ready:
        break
    if time.time() > deadline:
        raise TimeoutError(f"{index_name} not ready after 45 minutes")
    time.sleep(30)
