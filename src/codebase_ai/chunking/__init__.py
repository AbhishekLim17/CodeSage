"""Turn files into retrievable chunks."""

from __future__ import annotations

from codebase_ai.chunking.code_chunker import chunk_code
from codebase_ai.chunking.doc_chunker import chunk_markdown
from codebase_ai.chunking.models import CHUNKER_VERSION, Chunk, SearchHit
from codebase_ai.chunking.window_chunker import chunk_windows
from codebase_ai.ingest.languages import KIND_CODE, KIND_DOC
from codebase_ai.ingest.walker import SourceFile

__all__ = ["CHUNKER_VERSION", "Chunk", "SearchHit", "chunk_file"]


def chunk_file(
    source: SourceFile,
    repo_id: str,
    *,
    max_lines: int = 120,
    window_lines: int = 60,
    window_overlap: int = 10,
) -> list[Chunk]:
    """Chunk one file with the best strategy for its type; never raises for bad input.

    Code -> syntax-tree chunks (fallback: line windows); markdown -> heading sections; anything else -> windows.
    """
    if source.kind == KIND_CODE:
        chunks = chunk_code(
            source,
            repo_id,
            max_lines=max_lines,
            window_lines=window_lines,
            window_overlap=window_overlap,
        )
        if chunks is not None:
            return chunks
    elif source.kind == KIND_DOC and source.language == "markdown":
        return chunk_markdown(source, repo_id, max_lines=max_lines)
    return chunk_windows(source, repo_id, window_lines=window_lines, window_overlap=window_overlap)
