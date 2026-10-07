"""Provider adapters, exercised through the *real* vendor SDKs against a scripted fake HTTP server.

Nothing here touches the network or needs a key: the SDK builds the request and parses the response exactly as it
would in production, and the fake transport records what was sent and replies with what the real API would.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import httpx2
import pytest

from codebase_ai.config import Settings
from codebase_ai.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from codebase_ai.llm.base import (
    LLMResponse,
    Message,
    ProviderError,
    StreamDone,
    TextDelta,
    Usage,
    complete,
    create_provider,
)
from codebase_ai.llm.ollama_provider import OUTPUT_CAP, OllamaProvider
from codebase_ai.llm.openai_provider import OpenAIProvider

MESSAGES = [Message("user", "How does login work?")]
SYSTEM = "You explain code."


class Recorder:
    """Collects the requests a fake server saw."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def record(self, request) -> dict:
        body = json.loads(request.content) if request.content else {}
        entry = {"path": request.url.path, "headers": dict(request.headers), "body": body}
        self.requests.append(entry)
        return entry

    @property
    def last(self) -> dict:
        return self.requests[-1]


def run(provider, **kwargs):
    return list(provider.stream(SYSTEM, MESSAGES, max_tokens=kwargs.pop("max_tokens", 1000), **kwargs))


def texts(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


# =============================================================================================================
# Anthropic (SDK 1.x, built on httpx2)
# =============================================================================================================


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def anthropic_stream(chunks: list[str], *, stop_reason: str = "end_turn", model: str = "claude-opus-5-5") -> bytes:
    events: list[tuple[str, dict]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 321, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ]
    events += [
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}})
        for c in chunks
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return sse(*events)


def anthropic_error(status: int, kind: str, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})


def anthropic_provider(handler: Callable, recorder: Recorder | None = None, **kwargs) -> AnthropicProvider:
    import anthropic

    def transport(request):
        if recorder is not None:
            recorder.record(request)
        return handler(request)

    client = anthropic.Anthropic(
        api_key=kwargs.pop("api_key", "sk-ant-test-key"),
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(transport)),
        max_retries=0,  # the SDK would otherwise sleep and retry 429/5xx, which makes error tests slow
    )
    return AnthropicProvider("claude-opus-5-5", client=client, **kwargs)


def ok_stream(chunks=("Login checks ", "the suspended flag [1].")) -> Callable:
    return lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=anthropic_stream(list(chunks))
    )


def test_anthropic_streams_text_then_reports_usage_model_and_finish():
    events = run(anthropic_provider(ok_stream()))
    assert texts(events) == "Login checks the suspended flag [1]."
    done = events[-1]
    assert isinstance(done, StreamDone) and done.finish == "stop"
    assert done.model == "claude-opus-5-5"
    assert done.usage == Usage(input_tokens=321, output_tokens=42)
    assert all(isinstance(e, TextDelta) for e in events[:-1])


def test_anthropic_request_carries_model_system_messages_and_never_a_temperature():
    seen = Recorder()
    run(anthropic_provider(ok_stream(), seen, refusal_fallback=False), temperature=0.7)
    request = seen.last
    assert request["path"] == "/v1/messages"
    body = request["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 1000 and body["stream"] is True
    assert body["system"] == SYSTEM
    assert body["messages"] == [{"role": "user", "content": "How does login work?"}]
    assert "temperature" not in body  # current Claude models reject sampling parameters
    assert "fallbacks" not in body and "output_config" not in body
    assert "server-side-fallback" not in request["headers"].get("anthropic-beta", "")
    assert request["headers"]["x-api-key"] == "sk-ant-test-key"


def test_anthropic_asks_for_refusal_fallbacks_by_default():
    seen = Recorder()
    run(anthropic_provider(ok_stream(), seen))
    request = seen.last
    assert request["body"]["fallbacks"] == "default"
    assert FALLBACK_BETA in request["headers"]["anthropic-beta"]


def test_anthropic_effort_is_sent_only_when_configured():
    seen = Recorder()
    run(anthropic_provider(ok_stream(), seen, effort="low", refusal_fallback=False))
    assert seen.last["body"]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"), ("refusal", "refusal"), ("pause_turn", "other")],
)
def test_anthropic_stop_reasons_are_normalised(stop_reason, finish):
    handler = lambda r: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=anthropic_stream(["x"], stop_reason=stop_reason)
    )
    assert run(anthropic_provider(handler, refusal_fallback=False))[-1].finish == finish


def test_a_rejected_fallbacks_request_is_retried_once_without_them_and_remembered():
    seen = Recorder()

    def handler(request):
        if b'"fallbacks"' in request.content:
            return anthropic_error(400, "invalid_request_error", "fallbacks: unsupported for this model")
        return ok_stream()(request)

    provider = anthropic_provider(handler, seen)
    assert texts(run(provider)) == "Login checks the suspended flag [1]."
    assert ["fallbacks" in r["body"] for r in seen.requests] == [True, False]
    run(provider)  # later calls do not ask again
    assert ["fallbacks" in r["body"] for r in seen.requests] == [True, False, False]


def test_a_400_without_fallbacks_is_reported_not_retried():
    seen = Recorder()
    handler = lambda r: anthropic_error(400, "invalid_request_error", "max_tokens too large")
    with pytest.raises(ProviderError, match="max_tokens too large") as info:
        run(anthropic_provider(handler, seen, refusal_fallback=False))
    assert info.value.kind == "bad_request" and len(seen.requests) == 1


def test_a_400_after_fallbacks_are_dropped_is_still_reported():
    seen = Recorder()
    handler = lambda r: anthropic_error(400, "invalid_request_error", "system prompt too long")
    with pytest.raises(ProviderError, match="system prompt too long"):
        run(anthropic_provider(handler, seen))
    assert len(seen.requests) == 2  # one attempt with fallbacks, one without, then the error


@pytest.mark.parametrize(
    ("status", "kind", "retryable", "needle"),
    [
        (401, "auth", False, "ANTHROPIC_API_KEY"),
        (403, "permission", False, "not allowed"),
        (404, "not_found", False, "claude-opus-5-5"),
        (429, "rate_limit", True, "rate limit"),
        (500, "server", True, "HTTP 500"),
        (529, "server", True, "HTTP 529"),
    ],
)
def test_anthropic_http_errors_become_readable_provider_errors(status, kind, retryable, needle):
    handler = lambda r: anthropic_error(status, "api_error", "boom")
    with pytest.raises(ProviderError, match=needle) as info:
        run(anthropic_provider(handler, refusal_fallback=False))
    assert (info.value.kind, info.value.retryable, info.value.provider) == (kind, retryable, "anthropic")


def test_anthropic_timeouts_and_connection_failures_are_distinguished():
    def timeout(request):
        raise httpx2.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as slow:
        run(anthropic_provider(timeout, refusal_fallback=False))
    with pytest.raises(ProviderError) as down:
        run(anthropic_provider(refused, refusal_fallback=False))
    assert (slow.value.kind, down.value.kind) == ("timeout", "connection")
    assert slow.value.retryable and down.value.retryable


def test_the_api_key_never_appears_in_an_error_message():
    handler = lambda r: anthropic_error(401, "authentication_error", "invalid x-api-key")
    with pytest.raises(ProviderError) as info:
        run(anthropic_provider(handler, api_key="sk-ant-super-secret-value", refusal_fallback=False))
    assert "sk-ant-super-secret-value" not in str(info.value)


def test_anthropic_flags_that_code_leaves_the_machine():
    provider = anthropic_provider(ok_stream())
    assert provider.sends_code_off_machine is True and provider.context_window is None


# =============================================================================================================
# OpenAI (SDK 3.x, built on httpx2)
# =============================================================================================================


def openai_chunk(**fields) -> str:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return f"data: {json.dumps({**base, **fields})}\n\n"


def openai_stream(chunks: list[str], *, finish: str = "stop", usage: bool = True, refusal: str | None = None) -> bytes:
    parts = [openai_chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])]
    parts += [openai_chunk(choices=[{"index": 0, "delta": {"content": c}, "finish_reason": None}]) for c in chunks]
    if refusal:
        parts.append(openai_chunk(choices=[{"index": 0, "delta": {"refusal": refusal}, "finish_reason": None}]))
    parts.append(openai_chunk(choices=[{"index": 0, "delta": {}, "finish_reason": finish}]))
    if usage:
        parts.append(
            openai_chunk(choices=[], usage={"prompt_tokens": 210, "completion_tokens": 33, "total_tokens": 243})
        )
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def openai_provider(handler: Callable, recorder: Recorder | None = None, **kwargs) -> OpenAIProvider:
    import openai

    def transport(request):
        if recorder is not None:
            recorder.record(request)
        return handler(request)

    client = openai.OpenAI(
        api_key="sk-test-openai-key",
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(transport)),
        max_retries=0,
    )
    return OpenAIProvider("gpt-4o-mini", "sk-test-openai-key", client=client, **kwargs)


def openai_ok(chunks=("Login checks ", "suspension [1].")) -> Callable:
    return lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=openai_stream(list(chunks)))


def test_openai_streams_text_and_reports_usage_and_finish():
    events = run(openai_provider(openai_ok()))
    assert texts(events) == "Login checks suspension [1]."
    done = events[-1]
    assert done == StreamDone(finish="stop", usage=Usage(210, 33), model="gpt-4o-mini")


def test_openai_request_shape():
    seen = Recorder()
    run(openai_provider(openai_ok(), seen), max_tokens=777)
    request = seen.last
    assert request["path"].endswith("/chat/completions")
    body = request["body"]
    assert body["model"] == "gpt-4o-mini" and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 777 and "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": SYSTEM}
    assert body["messages"][1] == {"role": "user", "content": "How does login work?"}
    assert "temperature" not in body
    assert request["headers"]["authorization"] == "Bearer sk-test-openai-key"


def test_openai_temperature_is_sent_only_when_asked_for():
    seen = Recorder()
    run(openai_provider(openai_ok(), seen), temperature=0.2)
    assert seen.last["body"]["temperature"] == 0.2


@pytest.mark.parametrize(
    ("finish", "expected"), [("stop", "stop"), ("length", "length"), ("content_filter", "refusal"), ("tool_calls", "other")]
)
def test_openai_finish_reasons_are_normalised(finish, expected):
    handler = lambda r: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=openai_stream(["x"], finish=finish)
    )
    assert run(openai_provider(handler))[-1].finish == expected


def test_an_openai_structured_refusal_is_shown_and_flagged():
    handler = lambda r: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=openai_stream([], refusal="I can't help with that.")
    )
    events = run(openai_provider(handler))
    assert texts(events) == "I can't help with that." and events[-1].finish == "refusal"


def openai_error(status: int, code: str, message: str = "boom") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "type": code, "code": code}})


@pytest.mark.parametrize(
    ("status", "kind", "needle"),
    [
        (401, "auth", "OPENAI_API_KEY"),
        (403, "permission", "not allowed"),
        (404, "not_found", "gpt-4o-mini"),
        (429, "rate_limit", "quota"),
        (400, "bad_request", "rejected the request"),
        (500, "server", "HTTP 500"),
    ],
)
def test_openai_http_errors_become_readable_provider_errors(status, kind, needle):
    with pytest.raises(ProviderError, match=needle) as info:
        run(openai_provider(lambda r: openai_error(status, "err")))
    assert info.value.kind == kind and info.value.provider == "openai"


def test_openai_timeouts_and_connection_failures_are_distinguished():
    def timeout(request):
        raise httpx2.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as slow:
        run(openai_provider(timeout))
    with pytest.raises(ProviderError) as down:
        run(openai_provider(refused))
    assert (slow.value.kind, down.value.kind) == ("timeout", "connection")


def test_openai_without_a_key_explains_what_to_set():
    with pytest.raises(ProviderError, match="OPENAI_API_KEY") as info:
        run(OpenAIProvider("gpt-4o-mini", None))
    assert info.value.kind == "misconfigured"


def test_openai_key_never_appears_in_errors():
    with pytest.raises(ProviderError) as info:
        run(openai_provider(lambda r: openai_error(401, "invalid_api_key", "Incorrect API key: sk-test-openai-key")))
    assert "sk-test-openai-key" not in str(info.value)


# =============================================================================================================
# Ollama (local; SDK on standard httpx)
# =============================================================================================================


def ollama_lines(chunks: list[str], *, done_reason: str = "stop") -> bytes:
    lines = [{"model": "m", "created_at": "2026-01-01T00:00:00Z", "message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
    lines.append(
        {
            "model": "m", "created_at": "2026-01-01T00:00:00Z", "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": done_reason, "prompt_eval_count": 180, "eval_count": 25,
        }
    )
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode()


def ollama_provider(handler: Callable, recorder: Recorder | None = None, **kwargs) -> OllamaProvider:
    import ollama

    def transport(request):
        if recorder is not None:
            recorder.record(request)
        return handler(request)

    client = ollama.Client(host="http://localhost:11434", transport=httpx.MockTransport(transport))
    return OllamaProvider("qwen2.5-coder:7b", "http://localhost:11434", client=client, **kwargs)


def ollama_ok(chunks=("Login checks ", "suspension [1].")) -> Callable:
    return lambda r: httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=ollama_lines(list(chunks)))


def test_ollama_streams_text_and_reports_usage_and_finish():
    events = run(ollama_provider(ollama_ok()))
    assert texts(events) == "Login checks suspension [1]."
    assert events[-1] == StreamDone(finish="stop", usage=Usage(180, 25), model="qwen2.5-coder:7b")


def test_ollama_always_sets_the_context_window_and_caps_the_output():
    seen = Recorder()
    provider = ollama_provider(ollama_ok(), seen, num_ctx=20_000)
    run(provider, max_tokens=16_000)
    body = seen.last["body"]
    assert seen.last["path"] == "/api/chat" and body["stream"] is True
    assert body["options"]["num_ctx"] == 20_000
    assert body["options"]["num_predict"] == OUTPUT_CAP  # 16,000 requested, capped for a local model
    assert body["options"]["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": SYSTEM}
    assert (provider.context_window, provider.max_output_tokens, provider.sends_code_off_machine) == (20_000, OUTPUT_CAP, False)


def test_ollama_keeps_its_default_model_lifetime_unless_told_to_unload_after_each_request():
    seen = Recorder()
    run(ollama_provider(ollama_ok(), seen))
    assert "keep_alive" not in seen.last["body"]  # Ollama's own default: the model stays loaded for a while
    run(ollama_provider(ollama_ok(), seen, keep_alive=0))
    assert seen.last["body"]["keep_alive"] == 0  # unloaded after the answer, so the next one starts fresh (D11)


def test_ollama_small_requests_are_not_inflated_and_temperature_can_be_set():
    seen = Recorder()
    run(ollama_provider(ollama_ok(), seen), max_tokens=300, temperature=0.4)
    assert seen.last["body"]["options"]["num_predict"] == 300 and seen.last["body"]["options"]["temperature"] == 0.4


def test_ollama_length_finish():
    handler = lambda r: httpx.Response(200, content=ollama_lines(["x"], done_reason="length"))
    assert run(ollama_provider(handler))[-1].finish == "length"


def test_a_missing_ollama_model_tells_the_user_to_pull_it():
    handler = lambda r: httpx.Response(404, json={"error": "model 'qwen2.5-coder:7b' not found"})
    with pytest.raises(ProviderError, match="ollama pull qwen2.5-coder:7b") as info:
        run(ollama_provider(handler))
    assert info.value.kind == "not_found"


def test_ollama_not_running_and_timeouts_are_distinguished():
    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="ollama serve") as down:
        run(ollama_provider(refused))
    with pytest.raises(ProviderError) as timed_out:
        run(ollama_provider(slow))
    assert (down.value.kind, timed_out.value.kind) == ("connection", "timeout")


# =============================================================================================================
# Shared behaviour
# =============================================================================================================


def test_complete_collects_a_whole_answer():
    response = complete(anthropic_provider(ok_stream(), refusal_fallback=False), SYSTEM, MESSAGES, max_tokens=500)
    assert response == LLMResponse("Login checks the suspended flag [1].", "stop", Usage(321, 42), "claude-opus-5-5")


def test_all_adapters_follow_the_same_event_protocol():
    for provider in (
        anthropic_provider(ok_stream(), refusal_fallback=False),
        openai_provider(openai_ok()),
        ollama_provider(ollama_ok()),
    ):
        events = run(provider)
        assert isinstance(events[-1], StreamDone) and all(isinstance(e, TextDelta) for e in events[:-1])
        assert texts(events).endswith("[1].")


def settings_with(monkeypatch, **env) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_the_factory_builds_the_selected_provider(monkeypatch):
    s = settings_with(monkeypatch, ANTHROPIC_API_KEY="sk-ant-x", OPENAI_API_KEY="sk-o-x")
    assert isinstance(create_provider(s), AnthropicProvider)
    assert isinstance(create_provider(s, provider="openai"), OpenAIProvider)
    ollama = create_provider(s, provider="ollama")
    assert isinstance(ollama, OllamaProvider) and ollama.context_window == 16_384


def test_the_factory_applies_settings_and_model_override(monkeypatch):
    s = settings_with(monkeypatch, ANTHROPIC_EFFORT="low", ANTHROPIC_REFUSAL_FALLBACK="false", OLLAMA_NUM_CTX="8192")
    claude = create_provider(s, model="claude-sonnet-5-5")
    assert claude.model == "claude-sonnet-5-5" and claude._effort == "low" and claude._fallback is False
    assert create_provider(s, provider="ollama").context_window == 8192
    assert create_provider(s).model == "claude-opus-5-5"  # the default model


def test_the_factory_rejects_unknown_providers(monkeypatch):
    with pytest.raises(ProviderError, match="Unknown LLM provider 'skynet'"):
        create_provider(settings_with(monkeypatch), provider="skynet")


def test_provider_errors_carry_kind_and_retryability():
    error = ProviderError("x", "rate_limit", provider="anthropic", retryable=True)
    assert (error.kind, error.provider, error.retryable, str(error)) == ("rate_limit", "anthropic", True, "x")


def test_anthropic_with_no_credentials_anywhere_says_to_set_the_key(monkeypatch, tmp_path):
    """The SDK raises a bare TypeError for this; the user must see an actionable message, not that."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.setenv(name, str(tmp_path))  # no stored login to find
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")  # if a login were found, nothing is reachable
    with pytest.raises(ProviderError) as caught:
        run(AnthropicProvider("claude-opus-5-5", None))
    assert caught.value.kind == "auth"
    assert "ANTHROPIC_API_KEY" in str(caught.value)
    assert "TypeError" not in str(caught.value)


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("localhost:11434", True),
        ("http://LOCALHOST:11434", True),
        ("http://ollama.localhost:11434", True),
        ("", True),
        ("http://192.168.1.20:11434", False),
        ("http://gpu-box.internal:11434", False),
        ("https://ollama.example.com", False),
        ("gpu-box:11434", False),
        ("http://localhost.evil.example:11434", False),
    ],
)
def test_ollama_only_counts_as_on_this_machine_when_its_host_is_local(host, local):
    provider = OllamaProvider("m", host, client=object())
    assert provider.sends_code_off_machine is (not local)
