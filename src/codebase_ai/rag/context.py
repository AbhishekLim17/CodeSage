"""Format retrieved chunks as numbered sources with path:start-end labels."""

from __future__ import annotations

import re
from collections.abc import Sequence

from codebase_ai.retrieval.retriever import Source

_BACKTICK_RUN = re.compile(r"`+")
_CLOSING_TAG = "</sources>"
_MAX_SYMBOLS_SHOWN = 3


def code_fence(text: str) -> str:
    """A fence of backticks longer than any run inside ``text``, so the code can never close its own block."""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    return "`" * max(3, longest + 1)


def describe_source(source: Source) -> str:
    """A short human label: ``method UserService.normalize``, ``class header UserService``, ``doc_section Setup``."""
    symbols = list(source.symbols[:_MAX_SYMBOLS_SHOWN])
    if len(source.symbols) > _MAX_SYMBOLS_SHOWN:
        symbols.append("...")
    named = ", ".join(symbols)
    if source.role == "map":
        return "generated map of the repository's directories and files"
    if source.role == "context":
        return f"class header {named}".strip()
    kind = "/".join(source.kinds)
    return f"{kind} {named}".strip() if named else kind


def source_header(number: int, source: Source) -> str:
    return f"[{number}] {source.location} - {describe_source(source)}"


def format_sources(sources: Sequence[Source]) -> str:
    """Numbered source blocks, best first, numbered from 1 in the order given.

    The numbers are what the model cites, so ``sources[n - 1]`` is always the source cited as ``[n]``.
    """
    blocks = []
    for number, source in enumerate(sources, start=1):
        # The literal closing tag would end the <sources> section early, so it is written differently in the prompt.
        text = source.text.replace(_CLOSING_TAG, "<\\/sources>")
        fence = code_fence(text)
        blocks.append(f"{source_header(number, source)}\n{fence}{source.language}\n{text}\n{fence}")
    return "\n\n".join(blocks)
