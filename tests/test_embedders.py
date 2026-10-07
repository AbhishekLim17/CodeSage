from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from codebase_ai.config import Settings
from codebase_ai.index.embedder import (
    EmbeddingError,
    LocalEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class FakeSentenceModel:
    def __init__(self):
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        assert kwargs.get("normalize_embeddings") is True
        return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype="float32")


def test_local_embedder_applies_the_models_prefixes():
    model = FakeSentenceModel()
    e = LocalEmbedder("BAAI/bge-small-en-v1.5", model=model)
    assert e.model_id == "local:BAAI/bge-small-en-v1.5" and e.dim == 3
    e.embed_documents(["def f(): pass"])
    e.embed_query("where is f")
    assert model.calls[0] == ["def f(): pass"]
    assert model.calls[1] == ["Represent this sentence for searching relevant passages: where is f"]


def test_local_embedder_e5_prefixes_and_unknown_model_has_none():
    model = FakeSentenceModel()
    LocalEmbedder("intfloat/e5-small-v2", model=model).embed_documents(["x"])
    LocalEmbedder("some/unknown-model", model=model).embed_query("y")
    assert model.calls == [["passage: x"], ["y"]]


def test_local_embedder_returns_plain_lists_and_handles_empty_input():
    e = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2", model=FakeSentenceModel())
    vectors = e.embed_documents(["a", "b"])
    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] and isinstance(vectors[0], list)
    assert e.embed_documents([]) == []


class FakeOpenAI:
    def __init__(self, fail=False):
        self.calls: list[int] = []
        self.fail = fail
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        if self.fail:
            raise RuntimeError("rate limited")
        self.calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 0.0]) for t in input])


def test_openai_embedder_batches_and_reports_dimension():
    client = FakeOpenAI()
    e = OpenAIEmbedder("text-embedding-3-small", "key", client=client)
    vectors = e.embed_documents(["x"] * 600)
    assert len(vectors) == 600 and client.calls == [256, 256, 88]
    assert e.dim == 2
    assert e.embed_query("abc") == [3.0, 0.0]
    assert e.model_id == "openai:text-embedding-3-small"


def test_openai_embedder_wraps_provider_errors():
    e = OpenAIEmbedder("m", "key", client=FakeOpenAI(fail=True))
    with pytest.raises(EmbeddingError, match="rate limited"):
        e.embed_query("x")


def test_openai_embedder_without_key_explains_what_to_set():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbedder("m", None).embed_query("x")


class FakeOllama:
    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, model, input):
        if self.fail:
            raise ConnectionError("connection refused")
        return {"embeddings": [[0.5, 0.5, 0.0] for _ in input]}


def test_ollama_embedder_round_trip():
    e = OllamaEmbedder("nomic-embed-text", "http://localhost:11434", client=FakeOllama())
    assert e.embed_documents(["a", "b"]) == [[0.5, 0.5, 0.0]] * 2
    assert e.dim == 3 and e.model_id == "ollama:nomic-embed-text"


def test_ollama_embedder_error_names_host_and_model():
    e = OllamaEmbedder("nomic-embed-text", "http://localhost:11434", client=FakeOllama(fail=True))
    with pytest.raises(EmbeddingError, match="localhost:11434.*nomic-embed-text"):
        e.embed_query("x")


@pytest.mark.parametrize(
    ("provider", "cls"),
    [("local", LocalEmbedder), ("openai", OpenAIEmbedder), ("ollama", OllamaEmbedder)],
)
def test_create_embedder_follows_settings(provider, cls, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    embedder = create_embedder(Settings(_env_file=None))
    assert isinstance(embedder, cls)
    assert embedder.model_id == Settings(_env_file=None).embedding_model_id()


def test_a_model_that_fails_to_load_gives_a_readable_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no such model")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", boom)
    with pytest.raises(EmbeddingError, match="Could not load embedding model 'some/model': no such model"):
        LocalEmbedder("some/model")._load()


def test_loading_a_model_turns_off_its_progress_bars(monkeypatch):
    # Seen for real: "Loading weights: 100%|####|" landed in the middle of `ask`, `search` and `eval` output.
    from transformers.utils import logging as transformers_logging

    transformers_logging.enable_progress_bar()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *args, **kwargs: FakeSentenceModel())
    LocalEmbedder("some/model")._load()
    assert not transformers_logging.is_progress_bar_enabled()


def test_remote_code_is_never_trusted(monkeypatch):
    seen = {}

    def fake_ctor(name, **kwargs):
        seen.update(kwargs)
        return FakeSentenceModel()

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_ctor)
    LocalEmbedder("some/model")._load()
    assert seen == {"trust_remote_code": False}
