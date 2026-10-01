"""Chunk dataclass: id, path, language, kind, symbol, line range, text, embed_text, file_hash."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

# Bump when chunking output changes; indexes built with another version must be rebuilt.
CHUNKER_VERSION = 1

# function | method | class | type | module | doc_section | window
KIND_FUNCTION = "function"
KIND_METHOD = "method"
KIND_CLASS = "class"
KIND_TYPE = "type"
KIND_MODULE = "module"
KIND_DOC_SECTION = "doc_section"
KIND_WINDOW = "window"


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a file. Line numbers are 1-based and inclusive."""

    repo_id: str
    path: str
    language: str
    kind: str
    symbol: str | None
    start_line: int
    end_line: int
    text: str
    file_hash: str
    id: str = field(init=False)

    def __post_init__(self) -> None:
        content_hash = hashlib.sha1(self.text.encode("utf-8")).hexdigest()
        raw = f"{self.repo_id}|{self.path}|{self.start_line}|{self.end_line}|{content_hash}"
        object.__setattr__(self, "id", hashlib.sha1(raw.encode("utf-8")).hexdigest())

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def embed_text(self) -> str:
        """What gets embedded: a location/symbol header plus the raw text (the raw text is what is cited)."""
        header = " • ".join(part for part in (self.path, self.symbol, self.kind) if part)
        return f"{header}\n{self.text}"

    def to_row(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "path": self.path,
            "language": self.language,
            "kind": self.kind,
            "symbol": self.symbol or "",
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "file_hash": self.file_hash,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Chunk:
        return cls(
            repo_id=str(row["repo_id"]),
            path=str(row["path"]),
            language=str(row["language"]),
            kind=str(row["kind"]),
            symbol=str(row["symbol"]) or None,
            start_line=int(row["start_line"]),  # type: ignore[call-overload]
            end_line=int(row["end_line"]),  # type: ignore[call-overload]
            text=str(row["text"]),
            file_hash=str(row["file_hash"]),
        )


@dataclass(frozen=True)
class SearchHit:
    """A ranked retrieval result. Higher ``score`` is better; scales differ between searchers."""

    chunk_id: str
    score: float
