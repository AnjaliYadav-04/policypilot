# Databricks notebook source
# MAGIC %md
# MAGIC # Unity AI Gateway endpoint for Groq
# MAGIC
# MAGIC Creates (or updates) the external-model endpoint `policypilot-groq-gateway`, which proxies
# MAGIC OpenAI-compatible chat calls to Groq's `openai/gpt-oss-120b`. The Groq key is never in code:
# MAGIC the endpoint reads it from the Key Vault-backed secret scope `policypilot-kv-scope`.
# MAGIC
# MAGIC Gateway settings applied here: PII blocking on input, safety filtering on input and output,
# MAGIC and usage tracking. The off-topic and investment-advice guardrails are custom LLM-evaluator
# MAGIC guardrails, added once in the endpoint's **Guardrails** tab (see README).

# COMMAND ----------

dbutils.widgets.text("endpoint_name", "policypilot-groq-gateway")  # noqa: F821
dbutils.widgets.text("secret_scope", "policypilot-kv-scope")  # noqa: F821
dbutils.widgets.text("secret_key", "groq-api-key")  # noqa: F821
dbutils.widgets.text("model", "openai/gpt-oss-120b")  # noqa: F821

endpoint_name = dbutils.widgets.get("endpoint_name")  # noqa: F821
scope = dbutils.widgets.get("secret_scope")  # noqa: F821
key = dbutils.widgets.get("secret_key")  # noqa: F821
model = dbutils.widgets.get("model")  # noqa: F821

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()

served_config = {
    "served_entities": [
        {
            "name": "groq-gpt-oss-120b",
            "external_model": {
                "name": model,
                "provider": "custom",  # any OpenAI-compatible API
                "task": "llm/v1/chat",
                "custom_provider_config": {
                    "custom_provider_url": "https://api.groq.com/openai/v1/chat/completions",
                    "bearer_token_auth": {"token": f"{{{{secrets/{scope}/{key}}}}}"},
                },
            },
        }
    ]
}

ai_gateway = {
    "usage_tracking_config": {"enabled": True},
    "guardrails": {
        "input": {"pii": {"behavior": "BLOCK"}, "safety": True},
        "output": {"safety": True},
    },
}

try:
    w.api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint_name}")
    print(f"Updating {endpoint_name}")
    w.api_client.do("PUT", f"/api/2.0/serving-endpoints/{endpoint_name}/config", body=served_config)
    w.api_client.do("PUT", f"/api/2.0/serving-endpoints/{endpoint_name}/ai-gateway", body=ai_gateway)
except NotFound:
    print(f"Creating {endpoint_name}")
    w.api_client.do(
        "POST",
        "/api/2.0/serving-endpoints",
        body={"name": endpoint_name, "config": served_config, "ai_gateway": ai_gateway},
    )

# COMMAND ----------

# Smoke test through the same OpenAI-compatible path the app uses.
client = w.serving_endpoints.get_open_ai_client()
reply = client.chat.completions.create(
    model=endpoint_name,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    max_tokens=200,
)
print(reply.choices[0].message.content)
