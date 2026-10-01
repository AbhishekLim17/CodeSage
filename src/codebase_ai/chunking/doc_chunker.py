"""Split markdown/rst/text by heading, keeping the heading path."""

from __future__ import annotations

import re
from dataclasses import dataclass

from codebase_ai.chunking.models import KIND_DOC_SECTION, Chunk
from codebase_ai.chunking.window_chunker import has_content, window_ranges
from codebase_ai.ingest.walker import SourceFile

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")


@dataclass
class _Section:
    start: int  # 0-based row of first line
    path: str | None  # "Setup > Environment"
    end: int = -1


def _sections(lines: list[str]) -> list[_Section]:
    sections: list[_Section] = [_Section(start=0, path=None)]
    stack: list[tuple[int, str]] = []  # (level, title)
    fence: str | None = None
    for row, line in enumerate(lines):
        marker = _FENCE.match(line)
        if marker:
            fence = None if fence == marker.group(1) else (fence or marker.group(1))
            continue
        if fence:
            continue
        heading = _HEADING.match(line)
        if not heading:
            continue
        level, title = len(heading.group(1)), heading.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(_Section(start=row, path=" > ".join(t for _, t in stack)))
    for i, section in enumerate(sections):
        section.end = (sections[i + 1].start if i + 1 < len(sections) else len(lines)) - 1
    return sections


def _non_blank(lines: list[str], start: int, end: int) -> int:
    return sum(1 for line in lines[start : end + 1] if line.strip())


def chunk_markdown(source: SourceFile, repo_id: str, *, max_lines: int = 120) -> list[Chunk]:
    """One chunk per heading section; oversized sections are split into windows.

    A heading with no body of its own (e.g. ``# Title`` directly followed by ``## Sub``) is merged into the
    next section so the parent heading stays with its first child.
    """
    lines = source.text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []

    chunks: list[Chunk] = []
    carry_start: int | None = None  # start row of heading-only sections waiting to be merged forward
    sections = _sections(lines)
    for i, section in enumerate(sections):
        start = section.start if carry_start is None else carry_start
        end = section.end
        if not has_content(lines, section.start, end):
            continue
        is_last = i == len(sections) - 1
        if section.path is not None and _non_blank(lines, section.start, end) == 1 and not is_last:
            carry_start = start
            continue
        carry_start = None
        for s, e in window_ranges(start, end, max_lines, min(10, max_lines - 1)):
            if not has_content(lines, s, e):
                continue
            while e > s and not lines[e].strip():
                e -= 1
            chunks.append(
                Chunk(
                    repo_id=repo_id,
                    path=source.path,
                    language=source.language,
                    kind=KIND_DOC_SECTION,
                    symbol=section.path,
                    start_line=s + 1,
                    end_line=e + 1,
                    text="\n".join(lines[s : e + 1]),
                    file_hash=source.file_hash,
                )
            )
    return chunks
