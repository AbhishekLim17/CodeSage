"""Map file extensions to languages and tree-sitter grammars."""

from __future__ import annotations

from pathlib import PurePosixPath

# extension -> language name, for source code
CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
}

DOC_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
}

CONFIG_EXTENSIONS: dict[str, str] = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".rules": "rules",
    ".gradle": "gradle",
    ".xml": "xml",
}

# whole-filename matches (case-sensitive) for files without a useful extension
CONFIG_FILENAMES: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    ".gitignore": "gitignore",
    ".env.example": "dotenv",
    ".env.sample": "dotenv",
    ".env.template": "dotenv",
}

KIND_CODE = "code"
KIND_DOC = "doc"
KIND_CONFIG = "config"


def classify(path: str) -> tuple[str, str] | None:
    """Return ``(kind, language)`` for an indexable file, or ``None`` if the type is unsupported.

    ``path`` is repo-relative with forward slashes; only the file name matters.
    """
    name = PurePosixPath(path).name
    if name in CONFIG_FILENAMES:
        return KIND_CONFIG, CONFIG_FILENAMES[name]
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return KIND_CODE, CODE_EXTENSIONS[suffix]
    if suffix in DOC_EXTENSIONS:
        return KIND_DOC, DOC_EXTENSIONS[suffix]
    if suffix in CONFIG_EXTENSIONS:
        return KIND_CONFIG, CONFIG_EXTENSIONS[suffix]
    return None
