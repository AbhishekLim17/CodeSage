"""Fallback line-window chunker (with overlap) for unsupported languages, config and data files."""

from __future__ import annotations

import re

from codebase_ai.chunking.models import KIND_WINDOW, Chunk
from codebase_ai.ingest.walker import SourceFile

_WORD = re.compile(r"\w")


def has_content(lines: list[str], start: int, end: int) -> bool:
    """True if any line in ``lines[start..end]`` (0-based, inclusive) contains a word character."""
    return any(_WORD.search(line) for line in lines[start : end + 1])


def window_ranges(start: int, end: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Split the inclusive range ``start..end`` into windows of ``window`` lines overlapping by ``overlap``.

    Always returns at least one range; the final window ends exactly at ``end``.
    """
    if window <= 0 or not 0 <= overlap < window:
        raise ValueError("need window > 0 and 0 <= overlap < window")
    ranges: list[tuple[int, int]] = []
    s = start
    while True:
        e = min(s + window - 1, end)
        ranges.append((s, e))
        if e >= end:
            return ranges
        s = e - overlap + 1


def chunk_windows(
    source: SourceFile, repo_id: str, *, window_lines: int = 60, window_overlap: int = 10
) -> list[Chunk]:
    """Chunk a whole file into overlapping line windows, skipping windows with no real content."""
    lines = source.text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    chunks = []
    for s, e in window_ranges(0, len(lines) - 1, window_lines, window_overlap):
        if not has_content(lines, s, e):
            continue
        chunks.append(
            Chunk(
                repo_id=repo_id,
                path=source.path,
                language=source.language,
                kind=KIND_WINDOW,
                symbol=None,
                start_line=s + 1,
                end_line=e + 1,
                text="\n".join(lines[s : e + 1]),
                file_hash=source.file_hash,
            )
        )
    return chunks
