from __future__ import annotations

import pytest

from codebase_ai.chunking.code_chunker import supported_languages
from codebase_ai.ingest.languages import CODE_EXTENSIONS, classify


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app.py", ("code", "python")),
        ("web/App.jsx", ("code", "javascript")),
        ("web/index.mjs", ("code", "javascript")),
        ("web/types.ts", ("code", "typescript")),
        ("web/Widget.TSX", ("code", "tsx")),
        ("Main.java", ("code", "java")),
        ("cmd/server.go", ("code", "go")),
        ("README.md", ("doc", "markdown")),
        ("notes.txt", ("doc", "text")),
        ("package.json", ("config", "json")),
        ("firestore.rules", ("config", "rules")),
        ("Dockerfile", ("config", "dockerfile")),
        (".env.example", ("config", "dotenv")),
    ],
)
def test_classify(path, expected):
    assert classify(path) == expected


@pytest.mark.parametrize("path", ["photo.png", "archive.zip", "Makefile.bak", "noextension", "data.parquet"])
def test_unsupported_types_return_none(path):
    assert classify(path) is None


def test_every_grammar_language_is_reachable_from_an_extension():
    languages = set(CODE_EXTENSIONS.values())
    assert supported_languages() <= languages
