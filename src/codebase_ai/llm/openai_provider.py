"""OpenAI adapter (lazy SDK import), using the Chat Completions streaming API."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from codebase_ai.llm.base import (
    ErrorKind,
    FinishReason,
    Message,
    ProviderError,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)

_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length", "content_filter": "refusal"}


class OpenAIProvider:
    name = "openai"
    sends_code_off_machine = True
    context_window = None
    max_output_tokens = None

    def __init__(self, model: str, api_key: str | None = None, *, timeout: float = 120.0, client: Any = None):
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ProviderError(
                    "OPENAI_API_KEY is not set (needed for LLM_PROVIDER=openai).", "misconfigured", provider=self.name
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "The OpenAI SDK is not installed. Run: pip install -e '.[openai]'",
                    "misconfigured",
                    provider=self.name,
                ) from exc
            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def stream(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        client = self._get_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *({"role": m.role, "content": m.content} for m in messages)],
            "stream": True,
            "stream_options": {"include_usage": True},
            # `max_completion_tokens` works on every chat model; newer ones reject the older `max_tokens`.
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:  # some newer models reject it, so it is only sent when asked for
            params["temperature"] = temperature

        finish: FinishReason = "other"
        usage: Usage | None = None
        model_name: str | None = None
        refused = False
        try:
            for chunk in client.chat.completions.create(**params):
                model_name = getattr(chunk, "model", None) or model_name
                if getattr(chunk, "usage", None) is not None:
                    usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                for choice in getattr(chunk, "choices", None) or []:
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        yield TextDelta(delta.content)
                    if getattr(delta, "refusal", None):  # a structured refusal is streamed in its own field
                        refused = True
                        yield TextDelta(delta.refusal)
                    if choice.finish_reason:
                        finish = _FINISH.get(choice.finish_reason, "other")
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamDone(finish="refusal" if refused else finish, usage=usage, model=model_name)

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        import openai

        def err(message: str, kind: ErrorKind, retryable: bool = False) -> ProviderError:
            return ProviderError(message, kind, provider=self.name, retryable=retryable)

        if isinstance(exc, openai.AuthenticationError):
            return err("OpenAI rejected the API key. Check OPENAI_API_KEY.", "auth")
        if isinstance(exc, openai.PermissionDeniedError):
            return err("This OpenAI API key is not allowed to use that model.", "permission")
        if isinstance(exc, openai.NotFoundError):
            return err(f"OpenAI does not know the model '{self.model}'. Check OPENAI_MODEL.", "not_found")
        if isinstance(exc, openai.RateLimitError):
            return err(
                "OpenAI rate limit or quota reached. Wait a moment, or check the plan and billing.",
                "rate_limit",
                retryable=True,
            )
        if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
            return err(f"OpenAI rejected the request: {getattr(exc, 'message', exc)}", "bad_request")
        if isinstance(exc, openai.APITimeoutError):
            return err("The request to OpenAI timed out. Try again, or raise LLM_TIMEOUT_SECONDS.", "timeout", True)
        if isinstance(exc, openai.APIConnectionError):
            return err("Could not reach OpenAI. Check the network connection.", "connection", retryable=True)
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0)
            if status >= 500:
                return err(f"OpenAI had a server problem (HTTP {status}). Try again shortly.", "server", True)
            return err(f"OpenAI returned an error (HTTP {status}): {getattr(exc, 'message', exc)}", "unknown")
        return err(f"OpenAI request failed: {exc}", "unknown")
