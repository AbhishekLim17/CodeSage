"""LLMProvider protocol, Message type, ProviderError, and provider factory from settings.

Everything above this layer (the answerer, the CLI, the UI) depends only on these types, never on a vendor SDK.
Each adapter turns its SDK's streaming API into the same small event stream, and turns every SDK failure into a
``ProviderError`` with a message that is safe to show to a user (it never contains a key or a request header).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from codebase_ai.config import Settings

Role = Literal["user", "assistant"]
# What a provider's own stop reason means to us: finished normally, hit the output limit, declined, or something else.
FinishReason = Literal["stop", "length", "refusal", "other"]
ErrorKind = Literal[
    "auth",
    "permission",
    "rate_limit",
    "timeout",
    "connection",
    "bad_request",
    "not_found",
    "server",
    "misconfigured",
    "unknown",
]
PROVIDER_NAMES = ("anthropic", "openai", "ollama")


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class TextDelta:
    """A piece of the answer text, in order."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """Always the last event of a stream that completed."""

    finish: FinishReason = "stop"
    usage: Usage | None = None
    model: str | None = None  # the model that actually answered, when the provider reports it


StreamEvent = TextDelta | StreamDone


@dataclass(frozen=True)
class LLMResponse:
    text: str
    finish: FinishReason
    usage: Usage | None
    model: str | None


class ProviderError(RuntimeError):
    """A provider call failed or is misconfigured. ``str(error)`` is safe to show to a user."""

    def __init__(self, message: str, kind: ErrorKind = "unknown", *, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.provider = provider
        self.retryable = retryable


class LLMProvider(Protocol):
    """Streams a model's answer as ``TextDelta`` events followed by one ``StreamDone``."""

    name: str
    model: str
    sends_code_off_machine: bool  # True when the prompt (which contains retrieved code) leaves this computer
    context_window: int | None  # tokens the provider will actually accept, when it limits that itself (e.g. Ollama)
    max_output_tokens: int | None  # a cap the provider applies to ``max_tokens``, if any

    def stream(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]: ...


def complete(
    provider: LLMProvider,
    system: str,
    messages: Sequence[Message],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> LLMResponse:
    """Run a request to the end and return the whole answer (a convenience over ``provider.stream``)."""
    parts: list[str] = []
    done = StreamDone()
    for event in provider.stream(system, messages, max_tokens=max_tokens, temperature=temperature):
        if isinstance(event, TextDelta):
            parts.append(event.text)
        else:
            done = event
    return LLMResponse(text="".join(parts), finish=done.finish, usage=done.usage, model=done.model)


def create_provider(
    settings: Settings, *, provider: str | None = None, model: str | None = None
) -> LLMProvider:
    """Build the provider selected by ``LLM_PROVIDER`` (or ``provider``), with ``model`` overriding the default.

    SDKs are imported only for the provider actually used, so a user who only wants Ollama never installs the others.
    """
    name = provider or settings.llm_provider
    if name == "anthropic":
        from codebase_ai.llm.anthropic_provider import AnthropicProvider

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicProvider(
            model or settings.anthropic_model,
            key,
            timeout=settings.llm_timeout_seconds,
            effort=settings.anthropic_effort,
            refusal_fallback=settings.anthropic_refusal_fallback,
        )
    if name == "openai":
        from codebase_ai.llm.openai_provider import OpenAIProvider

        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIProvider(model or settings.openai_model, key, timeout=settings.llm_timeout_seconds)
    if name == "ollama":
        from codebase_ai.llm.ollama_provider import OllamaProvider

        return OllamaProvider(
            model or settings.ollama_model,
            settings.ollama_host,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.llm_timeout_seconds,
        )
    raise ProviderError(f"Unknown LLM provider '{name}'; choose from {', '.join(PROVIDER_NAMES)}.", "misconfigured")
