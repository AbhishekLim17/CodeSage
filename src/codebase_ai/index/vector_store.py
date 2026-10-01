"""VectorStore interface and ChromaDB implementation."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from codebase_ai.chunking.models import Chunk, SearchHit

_COLLECTION = "chunks"
_WRITE_BATCH = 1000  # stay well under Chroma's per-call maximum


class VectorStore(Protocol):
    """Nearest-neighbour search over chunk embeddings. Chunk text lives in the keyword index / chunk table."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def query(self, embedding: Sequence[float], k: int) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def close(self) -> None:
        """Release files and handles held by the store. Safe to call more than once."""
        ...


class ChromaVectorStore:
    """Persistent, embedded ChromaDB collection using cosine distance. Telemetry is disabled."""

    def __init__(self, path: Path):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open()
        self._closed = False

    def close(self) -> None:
        """Let go of Chroma's files. Without this every opened index keeps handles until the process exits, which on
        Windows (512 open files per process) ends a long session or a big test run with "Too many open files"."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._client, "close", None)  # Chroma 1.x; older versions have nothing to release
        if closer is not None:
            with contextlib.suppress(Exception):  # best effort: a failure to close must not hide the real result
                closer()

    def _open(self):
        return self._client.get_or_create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), _WRITE_BATCH):
            batch = chunks[i : i + _WRITE_BATCH]
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=[list(e) for e in embeddings[i : i + _WRITE_BATCH]],
                metadatas=[
                    {"path": c.path, "start_line": c.start_line, "end_line": c.end_line} for c in batch
                ],
            )

    def delete(self, ids: Sequence[str]) -> None:
        for i in range(0, len(ids), _WRITE_BATCH):
            self._collection.delete(ids=list(ids[i : i + _WRITE_BATCH]))

    def query(self, embedding: Sequence[float], k: int) -> list[SearchHit]:
        total = self._collection.count()
        if total == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)], n_results=min(k, total), include=["distances"]
        )
        return [
            SearchHit(chunk_id=chunk_id, score=1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(_COLLECTION)
        self._collection = self._open()
