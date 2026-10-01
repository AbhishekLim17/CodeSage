"""SQLite FTS5 keyword index with identifier-splitting (camelCase / snake_case) tokenisation.

This module also holds the chunk table (the source of truth for chunk text and metadata), so a search
result from any retriever can be turned back into a full ``Chunk`` with ``get_chunks``.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from codebase_ai.chunking.models import Chunk, SearchHit

_T = TypeVar("_T")
_WORDS = re.compile(r"\w+")
_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SQL_BATCH = 500

# Dropped from *queries* only; the index keeps everything.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "with"]
)


def split_identifiers(text: str) -> list[str]:
    """Lower-case tokens where ``getUserById`` and ``get_user_by_id`` both yield ``get user by id``.

    Compound identifiers are also kept whole (``get_user_by_id``) so exact-name searches rank first.
    """
    tokens: list[str] = []
    for word in _WORDS.findall(text):
        parts = [p.lower() for p in _PARTS.findall(word)]
        if len(parts) != 1:
            tokens.append(word.lower())
        tokens.extend(parts)
    return [t for t in tokens if len(t) > 1]


def query_tokens(query: str) -> list[str]:
    """Tokens for a natural-language query: identifier-split, stopwords removed, de-duplicated."""
    seen: dict[str, None] = {}
    for token in split_identifiers(query):
        if token not in _STOPWORDS:
            seen.setdefault(token)
    return list(seen)


def _batches(items: Sequence[_T]) -> Iterable[Sequence[_T]]:
    for i in range(0, len(items), _SQL_BATCH):
        yield items[i : i + _SQL_BATCH]


class KeywordIndex:
    """BM25 keyword search (SQLite FTS5) plus the chunk table."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                id TEXT UNIQUE NOT NULL,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                file_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path, symbol);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                meta_tokens, body_tokens, tokenize = "unicode61 tokenchars '_'"
            );
            """
        )

    def close(self) -> None:
        self._db.close()

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.delete([c.id for c in chunks])
        for chunk in chunks:
            row = chunk.to_row()
            cursor = self._db.execute(
                "INSERT INTO chunks (id, repo_id, path, language, kind, symbol, start_line, end_line, text, file_hash)"
                " VALUES (:id, :repo_id, :path, :language, :kind, :symbol, :start_line, :end_line, :text, :file_hash)",
                row,
            )
            meta = " ".join(split_identifiers(f"{chunk.path} {chunk.symbol or ''}"))
            body = " ".join(split_identifiers(chunk.text))
            self._db.execute(
                "INSERT INTO chunks_fts (rowid, meta_tokens, body_tokens) VALUES (?, ?, ?)",
                (cursor.lastrowid, meta, body),
            )
        self._db.commit()

    def delete(self, ids: Sequence[str]) -> None:
        for batch in _batches(list(ids)):
            marks = ",".join("?" * len(batch))
            rowids = [
                r[0] for r in self._db.execute(f"SELECT rowid FROM chunks WHERE id IN ({marks})", batch)
            ]
            for rowid_batch in _batches(rowids):
                rmarks = ",".join("?" * len(rowid_batch))
                self._db.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({rmarks})", rowid_batch)
            self._db.execute(f"DELETE FROM chunks WHERE id IN ({marks})", batch)
        self._db.commit()

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Top ``k`` chunks by BM25; path and symbol matches count 4x a body match."""
        tokens = query_tokens(query)
        if not tokens or k <= 0:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        rows = self._db.execute(
            "SELECT c.id AS id, bm25(chunks_fts, 4.0, 1.0) AS score FROM chunks_fts"
            " JOIN chunks c ON c.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (match, k),
        ).fetchall()
        return [SearchHit(chunk_id=r["id"], score=-float(r["score"])) for r in rows]

    def get_chunks(self, ids: Sequence[str]) -> list[Chunk]:
        """Full chunks for ``ids``, in the same order; unknown ids are skipped."""
        found: dict[str, Chunk] = {}
        for batch in _batches(list(ids)):
            marks = ",".join("?" * len(batch))
            for row in self._db.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", batch):
                found[row["id"]] = Chunk.from_row(dict(row))
        return [found[i] for i in ids if i in found]

    def find_chunks(self, path: str, *, symbol: str | None = None, kind: str | None = None) -> list[Chunk]:
        """Chunks of one file, optionally narrowed by exact symbol and/or kind, in line order."""
        sql, params = "SELECT * FROM chunks WHERE path = ?", [path]
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._db.execute(sql + " ORDER BY start_line, end_line", params)
        return [Chunk.from_row(dict(row)) for row in rows]

    def outline(self) -> list[tuple[str, str, str, str, int, int]]:
        """``(path, language, kind, symbol, start_line, end_line)`` for every chunk, without the text.

        Enough to describe the shape of a repository (which files exist, what they define) at the cost of one query.
        """
        rows = self._db.execute(
            "SELECT path, language, kind, symbol, start_line, end_line FROM chunks ORDER BY path, start_line"
        )
        return [tuple(row) for row in rows]  # type: ignore[misc]

    def file_heads(self) -> dict[str, str]:
        """The text of the chunk that starts at line 1 of each file (where a file's leading comment lives)."""
        heads: dict[str, str] = {}
        for row in self._db.execute("SELECT path, text FROM chunks WHERE start_line = 1 ORDER BY end_line"):
            heads.setdefault(row["path"], row["text"])  # if several chunks start there, the shortest is enough
        return heads

    def count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def reset(self) -> None:
        self._db.execute("DELETE FROM chunks_fts")
        self._db.execute("DELETE FROM chunks")
        self._db.commit()
