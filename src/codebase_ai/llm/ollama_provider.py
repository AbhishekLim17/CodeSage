"""Ollama adapter for fully local operation (lazy SDK import).

Two Ollama behaviours shape this adapter:

* Its default context window is far smaller than a retrieved prompt, and it truncates silently. ``num_ctx`` is always
  set, and ``context_window`` tells the answerer how much prompt it may build.
* Local models are run with a low temperature unless the caller asks otherwise, which suits grounded code answers.
* Even at temperature 0, a request answered straight after another can come out differently from the same request
  sent to a freshly loaded model (research/DETERMINISM_CHECK.md). ``keep_alive=0`` unloads the model after every
  request, so each answer starts fresh and repeats exactly; the study's runs use it.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.parse import urlparse

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

DEFAULT_TEMPERATURE = 0.0
OUTPUT_CAP = 4096  # local models are slow and their context window is shared with the prompt
_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an Ollama response, which is a dict-like model in some versions and a plain dict in others."""
    if obj is None:
        return default
    try:
        value = obj[name]
    except (KeyError, TypeError, IndexError):
        value = getattr(obj, name, default)
    return default if value is None else value


def is_local_host(host: str) -> bool:
    """Whether an Ollama address points at this computer (``localhost``, a loopback address, or a bare port)."""
    parsed = urlparse(host if "://" in host else f"http://{host}")
    name = (parsed.hostname or "").lower()
    if not name:
        return True  # ":11434" or "" means the Ollama client's default, which is local
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        num_ctx: int = 16_384,
        timeout: float = 120.0,
        client: Any = None,
        keep_alive: float | str | None = None,
    ):
        self.model = model
        self.keep_alive = keep_alive  # None: Ollama's default (5 minutes); 0: unload after every request
        self._host = host
        # Ollama on another machine is still "off machine": the prompt, with the retrieved code, crosses the network.
        self.sends_code_off_machine = not is_local_host(host)
        self._timeout = timeout
        self._client = client
        self.context_window: int | None = num_ctx
        self.max_output_tokens: int | None = OUTPUT_CAP

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise ProviderError(
                    "The Ollama SDK is not installed. Run: pip install -e '.[ollama]'",
                    "misconfigured",
                    provider=self.name,
                ) from exc
            self._client = Client(host=self._host, timeout=self._timeout)
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
        options = {
            "num_ctx": self.context_window,
            "num_predict": min(max_tokens, OUTPUT_CAP),
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        }
        chat_messages = [{"role": "system", "content": system}, *({"role": m.role, "content": m.content} for m in messages)]
        finish: FinishReason = "other"
        usage: Usage | None = None
        try:
            chunks = client.chat(
                model=self.model, messages=chat_messages, stream=True, options=options, keep_alive=self.keep_alive
            )
            for chunk in chunks:
                text = _field(_field(chunk, "message"), "content", "")
                if text:
                    yield TextDelta(text)
                if _field(chunk, "done", False):
                    finish = _FINISH.get(_field(chunk, "done_reason", "stop"), "other")
                    usage = Usage(_field(chunk, "prompt_eval_count"), _field(chunk, "eval_count"))
        except Exception as exc:
            raise self._translate(exc) from exc
        yield StreamDone(finish=finish, usage=usage, model=self.model)

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc

        def err(message: str, kind: ErrorKind, retryable: bool = False) -> ProviderError:
            return ProviderError(message, kind, provider=self.name, retryable=retryable)

        try:
            import httpx
        except ImportError:  # pragma: no cover - a dependency of the ollama package
            httpx = None
        try:
            from ollama import ResponseError
        except ImportError:  # pragma: no cover
            ResponseError = ()  # type: ignore[assignment,misc]

        if isinstance(exc, ResponseError):
            status = getattr(exc, "status_code", None)
            if status == 404:
                return err(
                    f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}", "not_found"
                )
            return err(f"Ollama returned an error (HTTP {status}): {getattr(exc, 'error', exc)}", "unknown")
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return err("The request to Ollama timed out. Try again, or raise LLM_TIMEOUT_SECONDS.", "timeout", True)
        if isinstance(exc, ConnectionError) or (httpx is not None and isinstance(exc, httpx.TransportError)):
            return err(
                f"Could not reach Ollama at {self._host}. Is it running? Start it with: ollama serve",
                "connection",
                retryable=True,
            )
        return err(f"Ollama request failed: {exc}", "unknown")
