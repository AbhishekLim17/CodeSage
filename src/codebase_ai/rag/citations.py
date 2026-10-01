"""Parse [n] markers and check they refer to supplied sources; flag ungrounded answers.

Two checks, both done on the finished answer text:

* **Markers.** ``[1]``, ``[1, 2]``, ``[2-4]`` and ``[Source 3]`` must refer to a source that was actually supplied.
  Markers to sources that do not exist are removed from the text and reported. Brackets inside code (``items[1]``,
  fenced blocks, inline code) are array indexing, not citations, and are left alone.
* **Locations.** A ``path/to/file.py:12-30`` written in the prose must match a supplied source that contains those
  lines. A file the model never saw, or lines outside what it saw, is reported as unverified.

What this cannot do is check that a cited source actually *supports* the sentence it is attached to; that needs a
judge (`answer_eval.py`; built, not yet run live). A valid marker means "this source was supplied", not "this claim is true".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from codebase_ai.ingest.languages import CODE_EXTENSIONS, CONFIG_EXTENSIONS, DOC_EXTENSIONS
from codebase_ai.retrieval.retriever import Source

_CODE_SPAN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
# A marker follows whitespace, or a character that is neither a word character nor ")" (which would make it indexing).
_MARKER = re.compile(
    r"(?:(?P<lead>\s)|(?<![\w)]))"
    r"\[(?:sources?\s*)?(?P<body>\d+(?:\s*(?:,|-|–|and)\s*\d+)*)\]",
    re.IGNORECASE,
)
_EXTENSIONS = sorted(
    {ext.lstrip(".") for table in (CODE_EXTENSIONS, DOC_EXTENSIONS, CONFIG_EXTENSIONS) for ext in table},
    key=len,
    reverse=True,
)
_LOCATION = re.compile(
    r"(?<![\w./-])(?P<path>(?:[\w.@+-]+/)*[\w.@+-]+\.(?:" + "|".join(map(re.escape, _EXTENSIONS)) + r"))"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?(?!\w)"
)
_MAX_RANGE = 50


@dataclass(frozen=True)
class CitationReport:
    """What the answer's citations amount to, against the sources that were supplied."""

    clean_text: str  # the answer with markers to non-existent sources removed
    cited: tuple[int, ...]  # valid source numbers, in order of first appearance
    invalid: tuple[int, ...]  # numbers that matched no supplied source (removed from clean_text)
    uncited: tuple[int, ...]  # supplied sources the answer never cited
    unverified_locations: tuple[str, ...]  # path:line mentions that match no supplied source
    marker_count: int  # number of citation markers found (each ``[1][2]`` counts twice)

    @property
    def grounded(self) -> bool:
        """The answer cites at least one supplied source."""
        return bool(self.cited)

    @property
    def validity(self) -> float | None:
        """Share of cited numbers that were valid, or ``None`` if there were no citations at all."""
        total = len(self.cited) + len(self.invalid)
        return None if total == 0 else len(self.cited) / total


def _numbers(body: str) -> list[int]:
    """``"1, 3-5"`` -> ``[1, 3, 4, 5]``. An absurdly long range is treated as its two endpoints."""
    numbers: list[int] = []
    for part in re.split(r"\s*(?:,|and)\s*", body, flags=re.IGNORECASE):
        ends = re.split(r"\s*[-–]\s*", part)
        if len(ends) == 2:
            low, high = int(ends[0]), int(ends[1])
            numbers += list(range(low, high + 1)) if low <= high and high - low <= _MAX_RANGE else [low, high]
        elif part.strip():
            numbers.append(int(part))
    return numbers


def _prose_and_code(text: str) -> list[tuple[bool, str]]:
    """Split into ``(is_code, chunk)`` pieces so markers are only read from prose."""
    pieces: list[tuple[bool, str]] = []
    position = 0
    for match in _CODE_SPAN.finditer(text):
        pieces.append((False, text[position : match.start()]))
        pieces.append((True, match.group()))
        position = match.end()
    pieces.append((False, text[position:]))
    return pieces


def _location_matches(path: str, start: int, end: int, sources: Sequence[Source]) -> bool:
    return any(
        (source.path == path or source.path.endswith("/" + path))
        and source.start_line <= start
        and end <= source.end_line
        for source in sources
    )


def validate_citations(text: str, sources: Sequence[Source]) -> CitationReport:
    """Check the answer's citations against ``sources`` (numbered from 1 in the order given)."""
    count = len(sources)
    cited: dict[int, None] = {}
    invalid: dict[int, None] = {}
    markers = 0
    previous_end = -1  # end of the last marker read; a bracket right after ``]`` is a citation only if it chains from it

    def rewrite(match: re.Match[str]) -> str:
        nonlocal markers, previous_end
        lead = match.group("lead") or ""
        follows_bracket = match.start() > 0 and match.string[match.start() - 1] == "]"
        if not lead and follows_bracket and match.start() != previous_end:
            return match.group()  # ``matrix[1][2]``: indexing, not a citation
        previous_end = match.end()
        numbers = _numbers(match.group("body"))
        markers += len(numbers)
        valid = [n for n in numbers if 1 <= n <= count]
        for n in numbers:
            (cited if 1 <= n <= count else invalid).setdefault(n)
        if len(valid) == len(numbers):
            return match.group()
        if not valid:
            # Drop the marker with the space in front of it, unless another marker follows and needs that space.
            return lead if match.string.startswith("[", match.end()) else ""
        return f"{lead}[{', '.join(map(str, valid))}]"

    def process(chunk: str) -> str:
        nonlocal previous_end
        previous_end = -1
        return _MARKER.sub(rewrite, chunk)

    clean = "".join(chunk if is_code else process(chunk) for is_code, chunk in _prose_and_code(text))

    unverified: dict[str, None] = {}
    for match in _LOCATION.finditer(_FENCED_BLOCK.sub(" ", text)):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start or not _location_matches(match.group("path"), start, end, sources):
            unverified.setdefault(match.group())

    return CitationReport(
        clean_text=clean,
        cited=tuple(cited),
        invalid=tuple(invalid),
        uncited=tuple(n for n in range(1, count + 1) if n not in cited),
        unverified_locations=tuple(unverified),
        marker_count=markers,
    )
