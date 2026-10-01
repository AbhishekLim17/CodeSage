"""Embedder protocol plus local (sentence-transformers), OpenAI and Ollama implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from codebase_ai.config import Settings


class EmbeddingError(RuntimeError):
    """An embedding provider failed or is misconfigured; the message is safe to show to the user."""


class Embedder(Protocol):
    """Turns text into vectors. ``model_id`` is stored in the index; vectors from different ids don't mix."""

    model_id: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# (document prefix, query prefix) that each model was trained with.
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "BAAI/bge-small-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "BAAI/bge-base-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "Snowflake/snowflake-arctic-embed-s": ("", "Represent this sentence for searching relevant passages: "),
    "intfloat/e5-small-v2": ("passage: ", "query: "),
    "sentence-transformers/all-MiniLM-L6-v2": ("", ""),
}


class LocalEmbedder:
    """Runs a sentence-transformers model on this machine; nothing leaves the computer."""

    def __init__(self, model_name: str, *, batch_size: int = 32, model: Any = None):
        self.model_name = model_name
        self.model_id = f"local:{model_name}"
        self.batch_size = batch_size
        self.doc_prefix, self.query_prefix = MODEL_PREFIXES.get(model_name, ("", ""))
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # trust_remote_code stays False: never execute code shipped with a downloaded model.
                self._model = SentenceTransformer(self.model_name, trust_remote_code=False)
            except Exception as exc:
                raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {exc}") from exc
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # renamed in newer sentence-transformers; keep working with both
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.doc_prefix + t for t in texts]
        vectors = self._load().encode(
            prefixed, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._load().encode(
            [self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )
        return vector[0].tolist()


class OpenAIEmbedder:
    """OpenAI embeddings API. Sends chunk text to OpenAI."""

    _MAX_BATCH = 256

    def __init__(self, model: str, api_key: str | None, *, client: Any = None):
        self.model_id = f"openai:{model}"
        self._model = model
        self._client = client
        self._api_key = api_key
        self._dim: int | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise EmbeddingError("OPENAI_API_KEY is not set (needed for EMBEDDING_PROVIDER=openai).")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingError("Install the OpenAI extra: pip install -e '.[openai]'") from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = list(texts[i : i + self._MAX_BATCH])
            try:
                response = self._get_client().embeddings.create(model=self._model, input=batch)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc
            out.extend(item.embedding for item in response.data)
        return out

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class OllamaEmbedder:
    """Embeddings from a local Ollama server; nothing leaves the computer."""

    _MAX_BATCH = 64

    def __init__(self, model: str, host: str, *, client: Any = None):
        self.model_id = f"ollama:{model}"
        self._model = model
        self._host = host
        self._client = client
        self._dim: int | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise EmbeddingError("Install the Ollama extra: pip install -e '.[ollama]'") from exc
            self._client = Client(host=self._host)
        return self._client

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = list(texts[i : i + self._MAX_BATCH])
            try:
                response = self._get_client().embed(model=self._model, input=batch)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"Ollama embedding request failed ({self._host}, model '{self._model}'): {exc}"
                ) from exc
            out.extend(list(v) for v in response["embeddings"])
        return out

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``EMBEDDING_PROVIDER``."""
    if settings.embedding_provider == "local":
        return LocalEmbedder(settings.local_embedding_model)
    if settings.embedding_provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbedder(settings.openai_embedding_model, key)
    return OllamaEmbedder(settings.ollama_embedding_model, settings.ollama_host)
