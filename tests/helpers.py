"""Shared test helpers (importable as ``helpers``; ``tests`` is on pytest's pythonpath)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from codebase_ai.chunking import Chunk, chunk_file
from codebase_ai.index.keyword_index import split_identifiers
from codebase_ai.ingest.languages import classify
from codebase_ai.ingest.walker import SourceFile, load_source_file
from codebase_ai.llm.base import FinishReason, Message, StreamDone, StreamEvent, TextDelta, Usage

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing identifier words get similar vectors.

    Lets tests exercise the full pipeline without downloading a model.
    """

    DIM = 64

    def __init__(self, model_id: str = "fake:hash-64"):
        self.model_id = model_id

    @property
    def dim(self) -> int:
        return self.DIM

    def _vector(self, text: str) -> list[float]:
        counts: Counter[int] = Counter()
        for token in split_identifiers(text):
            counts[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1
        if not counts:
            return [1.0] + [0.0] * (self.DIM - 1)
        norm = math.sqrt(sum(v * v for v in counts.values()))
        return [counts.get(i, 0) / norm for i in range(self.DIM)]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_source(text: str, path: str = "mod.py", file_hash: str = "hash") -> SourceFile:
    """Build a SourceFile in memory (no disk access)."""
    kind, language = classify(path) or ("code", "python")
    return SourceFile(
        path=path,
        abs_path=Path(path),
        kind=kind,
        language=language,
        size=len(text),
        file_hash=file_hash,
        text=text,
    )


def chunk_fixture(relpath: str, **kwargs) -> list[Chunk]:
    """Chunk a file from the fixture repo."""
    source, reason = load_source_file(FIXTURE_REPO / relpath, relpath)
    assert source is not None, reason
    return chunk_file(source, "repo", **kwargs)


def line_of(relpath: str, marker: str) -> int:
    """1-based number of the first line of a fixture file containing ``marker``."""
    for number, line in enumerate((FIXTURE_REPO / relpath).read_text(encoding="utf-8").split("\n"), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not found in {relpath}")


def line_count(relpath: str) -> int:
    return len((FIXTURE_REPO / relpath).read_text(encoding="utf-8").rstrip("\n").split("\n"))


class ScriptedProvider:
    """A fake LLM: replays a fixed reply (or computes one from the prompt) and records every call.

    ``reply`` is the whole answer text, or a function of the prompt. ``fail_after_chars`` raises ``error`` once that
    many characters have been streamed, to test a connection that drops mid-answer.
    """

    def __init__(
        self,
        reply: str | Callable[[str], str] = "It is handled in [1].",
        *,
        finish: FinishReason = "stop",
        chunk_size: int = 8,
        fail_after_chars: int | None = None,
        error: Exception | None = None,
        name: str = "scripted",
        model: str = "scripted-1",
        reported_model: str | None = "",
        sends_code_off_machine: bool = False,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ):
        self.reply = reply
        self.finish: FinishReason = finish
        self.chunk_size = chunk_size
        self.fail_after_chars = fail_after_chars
        self.error = error
        self.name = name
        self.model = model
        self.reported_model = model if reported_model == "" else reported_model  # None = provider reports none
        self.sends_code_off_machine = sends_code_off_machine
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.calls: list[dict] = []

    def stream(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        prompt = "\n".join(m.content for m in messages)
        self.calls.append(
            {"system": system, "messages": list(messages), "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.error is not None and self.fail_after_chars is None:
            raise self.error
        text = self.reply(prompt) if callable(self.reply) else self.reply
        sent = 0
        for start in range(0, len(text), self.chunk_size):
            piece = text[start : start + self.chunk_size]
            yield TextDelta(piece)
            sent += len(piece)
            if self.fail_after_chars is not None and sent >= self.fail_after_chars and self.error is not None:
                raise self.error
        yield StreamDone(self.finish, Usage(len(prompt) // 4, len(text) // 4), self.reported_model)
