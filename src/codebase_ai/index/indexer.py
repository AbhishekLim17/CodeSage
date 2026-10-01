"""Orchestrates walk, chunk, embed and store; re-indexes only changed files."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from codebase_ai.chunking import CHUNKER_VERSION, Chunk, chunk_file
from codebase_ai.index.embedder import Embedder
from codebase_ai.index.keyword_index import KeywordIndex
from codebase_ai.index.manifest import Manifest
from codebase_ai.index.vector_store import ChromaVectorStore, VectorStore
from codebase_ai.ingest.walker import DEFAULT_LIMITS, SourceFile, WalkLimits, WalkReport, walk_repo

_META_MODEL = "embedding_model_id"
_META_DIM = "embedding_dim"
_META_CHUNKER = "chunker_version"
_META_ROOT = "repo_root"


class IndexMismatchError(RuntimeError):
    """The stored index was built with different settings and can't be updated in place."""


def repo_id_for(root: Path) -> str:
    """Stable, human-readable id for a repo root, e.g. ``MagnaFlow-1a2b3c4d``."""
    resolved = str(Path(root).resolve())
    digest = hashlib.sha1(os.path.normcase(resolved).encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(resolved).name).strip("-") or "repo"
    return f"{slug}-{digest}"


class RepoIndex:
    """Open handles to the three stores (vectors, keyword/chunks, manifest) of one indexed repo."""

    def __init__(self, repo_root: str | Path, index_root: str | Path, vectors: VectorStore | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.repo_id = repo_id_for(self.repo_root)
        self.dir = Path(index_root) / self.repo_id
        self.manifest = Manifest(self.dir / "manifest.db")
        self.keyword = KeywordIndex(self.dir / "chunks.db")
        self.vectors: VectorStore = vectors if vectors is not None else ChromaVectorStore(self.dir / "chroma")

    def is_built(self) -> bool:
        return self.manifest.get_meta(_META_MODEL) is not None

    def info(self) -> dict[str, str | None]:
        return {
            "repo_root": self.manifest.get_meta(_META_ROOT),
            "embedding_model_id": self.manifest.get_meta(_META_MODEL),
            "embedding_dim": self.manifest.get_meta(_META_DIM),
            "chunker_version": self.manifest.get_meta(_META_CHUNKER),
        }

    def close(self) -> None:
        self.manifest.close()
        self.keyword.close()
        closer = getattr(self.vectors, "close", None)  # a custom VectorStore need not have one
        if closer is not None:
            closer()


@dataclass(frozen=True)
class IndexStatus:
    """What is stored for one repository, as read by ``index_status``."""

    embedding_model_id: str
    files: int
    chunks: int
    location: Path


def index_status(repo_root: str | Path, index_root: str | Path) -> IndexStatus | None:
    """Summarize a repository's index, or ``None`` if it has none.

    Cheap and read-only: it opens neither the vector store nor anything for a repository that was never indexed, so a
    UI can call it on every refresh without creating empty index folders.
    """
    directory = Path(index_root) / repo_id_for(Path(repo_root))
    if not (directory / "manifest.db").exists() or not (directory / "chunks.db").exists():
        return None
    manifest = Manifest(directory / "manifest.db")
    keyword = KeywordIndex(directory / "chunks.db")
    try:
        model = manifest.get_meta(_META_MODEL)
        if model is None:
            return None
        return IndexStatus(model, len(manifest.files()), keyword.count(), directory)
    finally:
        manifest.close()
        keyword.close()


@dataclass
class IndexReport:
    """Outcome of one indexing run. Also passed (live) to the progress callback."""

    embedding_model_id: str
    full: bool = False
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    seconds: float = 0.0
    skipped: Counter[str] = field(default_factory=Counter)
    skipped_examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def files_indexed(self) -> int:
        return self.files_new + self.files_changed


@dataclass
class _Pending:
    source: SourceFile
    chunks: list[Chunk]
    old_ids: tuple[str, ...]


class Indexer:
    """Builds or incrementally updates a ``RepoIndex``. Unchanged files (same sha256) cost nothing."""

    def __init__(
        self,
        index: RepoIndex,
        embedder: Embedder,
        *,
        limits: WalkLimits = DEFAULT_LIMITS,
        chunk_max_lines: int = 120,
        window_lines: int = 60,
        window_overlap: int = 10,
        batch_size: int = 64,
    ):
        self.index = index
        self.embedder = embedder
        self.limits = limits
        self.chunk_max_lines = chunk_max_lines
        self.window_lines = window_lines
        self.window_overlap = window_overlap
        self.batch_size = batch_size

    def _check_compatible(self) -> None:
        manifest = self.index.manifest
        stored_model = manifest.get_meta(_META_MODEL)
        if stored_model is None:
            return
        problems = []
        if stored_model != self.embedder.model_id:
            problems.append(f"embedding model changed ({stored_model} -> {self.embedder.model_id})")
        if manifest.get_meta(_META_CHUNKER) != str(CHUNKER_VERSION):
            problems.append("chunking logic changed")
        if problems:
            raise IndexMismatchError(
                "The existing index was built with different settings: "
                + "; ".join(problems)
                + ". Re-run with --full to rebuild it."
            )

    def _wipe(self) -> None:
        self.index.vectors.reset()
        self.index.keyword.reset()
        self.index.manifest.reset()

    def _flush(self, pending: list[_Pending], report: IndexReport) -> None:
        if not pending:
            return
        chunks = [c for p in pending for c in p.chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            vectors.extend(self.embedder.embed_documents([c.embed_text for c in batch]))
        new_ids = {c.id for c in chunks}
        # Write-ahead: note every id that may exist for these files (hash "" never matches a real file) so that a
        # crash between the store writes below and the final manifest update cannot leave orphaned chunks behind;
        # the next run treats the files as changed and deletes all ids listed here that it no longer produces.
        for p in pending:
            self.index.manifest.upsert_file(p.source.path, "", sorted({*p.old_ids, *(c.id for c in p.chunks)}))
        self.index.manifest.commit()
        stale = [i for p in pending for i in p.old_ids if i not in new_ids]
        if stale:
            self.index.vectors.delete(stale)
            self.index.keyword.delete(stale)
            report.chunks_removed += len(stale)
        if chunks:
            self.index.vectors.upsert(chunks, vectors)
            self.index.keyword.upsert(chunks)
        for p in pending:
            self.index.manifest.upsert_file(p.source.path, p.source.file_hash, [c.id for c in p.chunks])
        self.index.manifest.commit()
        report.chunks_added += len(chunks)
        pending.clear()

    def run(
        self, *, full: bool = False, progress: Callable[[IndexReport], None] | None = None
    ) -> IndexReport:
        """Index the repo. Raises ``IndexMismatchError`` if the stored index needs ``full=True``."""
        started = time.perf_counter()
        if full:
            self._wipe()
        else:
            self._check_compatible()
        manifest = self.index.manifest
        manifest.set_meta(_META_MODEL, self.embedder.model_id)
        manifest.set_meta(_META_DIM, str(self.embedder.dim))
        manifest.set_meta(_META_CHUNKER, str(CHUNKER_VERSION))
        manifest.set_meta(_META_ROOT, str(self.index.repo_root))
        manifest.commit()

        report = IndexReport(embedding_model_id=self.embedder.model_id, full=full)
        walk_report = WalkReport()
        known = manifest.files()
        seen: set[str] = set()
        pending: list[_Pending] = []
        pending_chunks = 0

        for source in walk_repo(self.index.repo_root, self.limits, walk_report):
            seen.add(source.path)
            record = known.get(source.path)
            report.files_seen = walk_report.files_seen
            if record is not None and record.file_hash == source.file_hash:
                report.files_unchanged += 1
                continue
            chunks = list(
                {
                    c.id: c
                    for c in chunk_file(
                        source,
                        self.index.repo_id,
                        max_lines=self.chunk_max_lines,
                        window_lines=self.window_lines,
                        window_overlap=self.window_overlap,
                    )
                }.values()
            )
            if record is None:
                report.files_new += 1
            else:
                report.files_changed += 1
            pending.append(_Pending(source, chunks, record.chunk_ids if record else ()))
            pending_chunks += len(chunks)
            if pending_chunks >= self.batch_size:
                self._flush(pending, report)
                pending_chunks = 0
                if progress:
                    progress(report)
        self._flush(pending, report)

        # Files that were indexed before but are no longer indexable (deleted, now ignored, now flagged secret).
        gone = [path for path in known if path not in seen]
        stale_ids = [i for path in gone for i in known[path].chunk_ids]
        if stale_ids:
            self.index.vectors.delete(stale_ids)
            self.index.keyword.delete(stale_ids)
        for path in gone:
            manifest.delete_file(path)
        manifest.commit()
        report.files_removed = len(gone)
        report.chunks_removed += len(stale_ids)

        report.files_seen = walk_report.files_seen
        report.skipped = walk_report.skipped
        report.skipped_examples = walk_report.examples
        report.seconds = time.perf_counter() - started
        if progress:
            progress(report)
        return report
