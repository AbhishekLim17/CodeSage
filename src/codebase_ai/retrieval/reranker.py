"""Optional cross-encoder reranker (off by default).

Vector search compares a question and a chunk through two separate embeddings. A cross-encoder reads the question and
the chunk *together* and scores how well the chunk answers it, which is slower but usually sharper. The retriever can
re-order its top candidates with one (``Retriever(reranker=...)``, ``RERANKER_MODEL`` in settings).

It is off by default because it has to earn its place: it adds latency and a second model to download, and it is only
worth enabling if the evaluation shows a gain (``eval/run_rerank_eval.py``; results in ``docs/QUALITY_EVAL.md``).
Like the local embedder it runs on this machine and sends nothing anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from codebase_ai.config import Settings


class RerankError(RuntimeError):
    """The reranker could not be loaded or run; the message is safe to show to the user."""


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, higher meaning a better answer to ``query``."""
        ...


class CrossEncoderReranker:
    """A sentence-transformers ``CrossEncoder`` running on this machine."""

    def __init__(self, model_name: str, *, batch_size: int = 16, max_length: int = 512, model: Any = None) -> None:
        self.model_name = model_name
        self.model_id = f"cross-encoder:{model_name}"
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                from codebase_ai.index.embedder import quiet_model_loading

                quiet_model_loading()
                # trust_remote_code stays False: never execute code shipped with a downloaded model.
                self._model = CrossEncoder(self.model_name, max_length=self.max_length, trust_remote_code=False)
            except Exception as exc:
                raise RerankError(f"Could not load reranker model '{self.model_name}': {exc}") from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        try:
            scores = self._load().predict(
                [(query, text) for text in texts], batch_size=self.batch_size, show_progress_bar=False
            )
        except RerankError:
            raise
        except Exception as exc:
            raise RerankError(f"The reranker '{self.model_name}' failed: {exc}") from exc
        return [float(s) for s in scores]


def create_reranker(settings: Settings) -> Reranker | None:
    """The reranker selected by ``RERANKER_MODEL``, or ``None`` when reranking is off."""
    if not settings.reranker_model:
        return None
    return CrossEncoderReranker(settings.reranker_model)
