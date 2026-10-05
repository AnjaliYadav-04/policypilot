"""LLMClient: one interface, two backends.

* databricks: the Unity AI Gateway external-model endpoint (``policypilot-groq-gateway``),
  called through the workspace's OpenAI-compatible API with the app's own credentials.
* local: Groq's OpenAI-compatible API directly, using GROQ_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from agent.config import Settings

_GUARDRAIL_HINTS = ("guardrail", "pii", "safety", "unsafe", "blocked", "jailbreak", "policy")


class GuardrailBlocked(RuntimeError):
    """The gateway refused the request or response (PII, safety, or a custom guardrail)."""


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str: ...


def _is_guardrail_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    message = str(exc).lower()
    return status in (400, 403) and any(h in message for h in _GUARDRAIL_HINTS)


class OpenAICompatibleClient:
    def __init__(self, client: Any, model: str, default_max_tokens: int = 1500):
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens or self._default_max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # the openai SDK raises BadRequestError etc.
            if _is_guardrail_error(exc):
                raise GuardrailBlocked(str(exc)) from exc
            raise
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise GuardrailBlocked("Response was filtered by the gateway.")
        return (choice.message.content or "").strip()


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.is_databricks:
        from databricks.sdk import WorkspaceClient

        # Inside Databricks Apps, DATABRICKS_HOST / CLIENT_ID / CLIENT_SECRET are injected for
        # the app's auto-created service principal, so no token is configured anywhere.
        openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        return OpenAICompatibleClient(openai_client, settings.llm_endpoint, settings.max_answer_tokens)

    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("PP_ENV=local needs GROQ_API_KEY. Copy .env.example to .env and set it.")
    openai_client = OpenAI(api_key=api_key, base_url=settings.groq_base_url)
    return OpenAICompatibleClient(openai_client, settings.groq_model, settings.max_answer_tokens)
