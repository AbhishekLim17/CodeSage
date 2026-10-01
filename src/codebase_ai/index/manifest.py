"""File-hash manifest and index metadata (embedding model id, dimension, chunker version)."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileRecord:
    file_hash: str
    chunk_ids: tuple[str, ...]


class Manifest:
    """Which files are indexed (by content hash) and which chunk ids each produced, plus index metadata."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                chunk_ids TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self._db.close()

    def commit(self) -> None:
        self._db.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def files(self) -> dict[str, FileRecord]:
        rows = self._db.execute("SELECT path, file_hash, chunk_ids FROM files").fetchall()
        return {p: FileRecord(h, tuple(json.loads(ids))) for p, h, ids in rows}

    def upsert_file(self, path: str, file_hash: str, chunk_ids: list[str]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO files (path, file_hash, chunk_ids, indexed_at) VALUES (?, ?, ?, ?)",
            (path, file_hash, json.dumps(chunk_ids), time.time()),
        )

    def delete_file(self, path: str) -> None:
        self._db.execute("DELETE FROM files WHERE path = ?", (path,))

    def reset(self) -> None:
        self._db.execute("DELETE FROM files")
        self._db.execute("DELETE FROM meta")
        self._db.commit()
