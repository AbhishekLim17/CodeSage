"""Anthropic adapter (lazy SDK import).

Notes that follow from the current Claude API rather than from habit:

* Current Claude models reject sampling parameters, so ``temperature`` is never sent.
* Thinking is always on for the default model and counts against ``max_tokens``; the caller passes a generous limit.
* A safety classifier can decline a request (``stop_reason == "refusal"``). That is reported as ``finish="refusal"``,
  and, unless disabled, the request also asks the API to retry a decline on another model server-side.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"  # the `fallbacks: "default"` form
_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "refusal",
}


class AnthropicProvider:
    name = "anthropic"
    sends_code_off_machine = True
    context_window = None
    max_output_tokens = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        effort: str | None = None,
        refusal_fallback: bool = True,
        client: Any = None,
    ):
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._effort = effort
        self._fallback = refusal_fallback
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "The Anthropic SDK is not installed. Run: pip install -e '.[anthropic]'",
                    "misconfigured",
                    provider=self.name,
                ) from exc
            # With no key here the SDK falls back to its own credential lookup (environment, `ant auth login`).
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            try:
                self._client = anthropic.Anthropic(**kwargs)
            except Exception as exc:
                raise self._translate(exc) from exc
        return self._client

    def stream(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,  # accepted for the common interface; Claude models reject it
    ) -> Iterator[StreamEvent]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self._effort:
            params["output_config"] = {"effort": self._effort}

        produced = False
        try:
            for event in self._run(params, fallback=self._fallback):
                produced = True
                yield event
        except ProviderError as error:
            # A 400 on the fallback-enabled request most likely means this model or account does not accept the
            # fallbacks parameter. Retry once without it, and stop asking, rather than failing every request.
            if not (self._fallback and error.kind == "bad_request" and not produced):
                raise
            log.warning("A request with refusal fallbacks was rejected (%s); retrying without them.", error)
            self._fallback = False
            yield from self._run(params, fallback=False)

    def _run(self, params: dict[str, Any], *, fallback: bool) -> Iterator[StreamEvent]:
        client = self._get_client()
        try:
            if fallback:
                manager = client.beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **params)
            else:
                manager = client.messages.stream(**params)
            with manager as stream:
                for text in stream.text_stream:
                    if text:
                        yield TextDelta(text)
                final = stream.get_final_message()
        except Exception as exc:
            raise self._translate(exc) from exc
        usage = getattr(final, "usage", None)
        yield StreamDone(
            finish=_FINISH.get(getattr(final, "stop_reason", None) or "", "other"),
            usage=Usage(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)) if usage else None,
            model=getattr(final, "model", None),
        )

    def _translate(self, exc: Exception) -> ProviderError:
        """Map an SDK exception to a ProviderError. The order matters: most specific class first."""
        if isinstance(exc, ProviderError):
            return exc
        try:
            import anthropic
        except ImportError:  # pragma: no cover - the SDK was importable a moment ago
            return ProviderError(f"Anthropic request failed: {exc}", "unknown", provider=self.name)

        def err(message: str, kind: ErrorKind, retryable: bool = False) -> ProviderError:
            return ProviderError(message, kind, provider=self.name, retryable=retryable)

        # With no key anywhere the SDK raises a plain TypeError from its header check, before any request is made.
        no_credentials = isinstance(exc, TypeError) and "authentication method" in str(exc)
        if no_credentials or isinstance(exc, anthropic.AuthenticationError | anthropic.CredentialsError):
            return err(
                "Anthropic rejected the credentials, or none were found. Set ANTHROPIC_API_KEY in .env "
                "(or log in with `ant auth login`).",
                "auth",
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return err("This Anthropic API key is not allowed to use that model or feature.", "permission")
        if isinstance(exc, anthropic.NotFoundError):
            return err(f"Anthropic does not know the model '{self.model}'. Check ANTHROPIC_MODEL.", "not_found")
        if isinstance(exc, anthropic.RateLimitError):
            return err("Anthropic rate limit reached. Wait a moment and try again.", "rate_limit", retryable=True)
        if isinstance(
            exc, anthropic.BadRequestError | anthropic.UnprocessableEntityError | anthropic.RequestTooLargeError
        ):
            return err(f"Anthropic rejected the request: {getattr(exc, 'message', exc)}", "bad_request")
        if isinstance(exc, anthropic.APITimeoutError):
            return err("The request to Anthropic timed out. Try again, or raise LLM_TIMEOUT_SECONDS.", "timeout", True)
        if isinstance(exc, anthropic.APIConnectionError):
            return err("Could not reach Anthropic. Check the network connection.", "connection", retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", 0)
            if status >= 500:
                return err(f"Anthropic had a server problem (HTTP {status}). Try again shortly.", "server", True)
            return err(f"Anthropic returned an error (HTTP {status}): {getattr(exc, 'message', exc)}", "unknown")
        return err(f"Anthropic request failed: {exc}", "unknown")
