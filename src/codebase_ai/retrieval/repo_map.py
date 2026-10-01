"""A compact map of a repository, built from the index, for questions about the project as a whole.

"What does this project do?" is not answered by any one chunk. The map gives the answering model the shape of the
repository: the directories (with file counts, languages and what they define), the notable files with the first line
of their leading comment or docstring, and the README as a real, citable source.

Everything is derived from what is already in the index, with no model call and no extra index-time work, so it costs
nothing to keep and can never be out of date. What a directory *contains* is stated; what it is *for* is left to
the model to infer from names, symbols and docstrings, and the map says it is generated so it is never mistaken
for source code. (Model-written summaries per directory are a possible later addition; they would add cost and a
second thing that can be wrong, so they were not built without evidence that this version falls short.)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from codebase_ai.chunking.models import KIND_CLASS, KIND_FUNCTION, KIND_TYPE
from codebase_ai.index.indexer import RepoIndex
from codebase_ai.ingest.languages import CODE_EXTENSIONS, DOC_EXTENSIONS
from codebase_ai.retrieval.retriever import Source, estimate_tokens, is_test_path

MAP_PATH = "(repository map)"
README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
_SYMBOL_KINDS = {KIND_CLASS: 0, KIND_TYPE: 1, KIND_FUNCTION: 2}  # what to list first as "what this directory defines"
_MAX_SYMBOLS = 4
_MAX_ROOT_FILES = 14
_MAX_NOTABLE_FILES = 10
_MAX_DOCS = 8
_CODE_LANGUAGES = frozenset(CODE_EXTENSIONS.values()) - {"html", "css", "scss", "sql", "shell"}
_DOC_LANGUAGES = frozenset(DOC_EXTENSIONS.values()) - {"text"}  # prose with a title; not requirements.txt and the like
_DIR_SHARE = 0.7  # of the budget, for directory lines; the rest is for notable files
_README_LINES = 60


@dataclass(frozen=True)
class RepoMap:
    text: str
    directories: tuple[str, ...]  # every directory named in the text, without a trailing slash
    files: int
    readme_path: str | None = None
    notable_files: tuple[str, ...] = field(default_factory=tuple)

    def lists(self, directory: str) -> bool:
        """Whether the map names ``directory`` (``"src/services"``), directly or inside a collapsed chain."""
        wanted = directory.strip("/")
        return any(d == wanted or d.startswith(wanted + "/") for d in self.directories)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


# --- what a file says about itself --------------------------------------------------------------------------------

_NOISE = re.compile(r"^@|copyright|license|licen[cs]e|all rights reserved|eslint|prettier|noqa|type:\s*ignore|^#!", re.IGNORECASE)


def _first_sentence(text: str, limit: int = 110, minimum: int = 8) -> str | None:
    line = " ".join(text.split())
    if not line or _NOISE.search(line) or len(line) < minimum:
        return None
    line = re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0]
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}$")  # reStructuredText heading underlines: ====, ----, ~~~~
_DOTTED_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.\w+)+$")  # "requests.api": a module restating its own name


def _rst_title(lines: list[str]) -> str | None:
    """The first reStructuredText heading: a line with an underline (and possibly an overline) of punctuation."""
    for index in range(1, len(lines)):
        if _UNDERLINE.match(lines[index].strip()) and lines[index - 1].strip() and not _UNDERLINE.match(lines[index - 1].strip()):
            return _first_sentence(lines[index - 1], minimum=3)
    return None


def describe_file(head: str, language: str) -> str | None:
    """First meaningful line of a file's leading docstring or comment, or ``None`` if it has none."""
    lines = head.splitlines()[:40]
    if language == "rst":
        return _rst_title(lines)
    if language == "python":
        text = "\n".join(lines).lstrip()
        text = re.sub(r"^(?:#[^\n]*\n\s*)+", "", text)  # comments above the docstring
        found = re.match(r'(?:[rRuUbB]{0,2})("""|\'\'\')(.*?)(?:\1|$)', text, re.DOTALL)
        if found:
            for line in found.group(2).splitlines():
                stripped = line.strip()
                if _UNDERLINE.match(stripped) or _DOTTED_NAME.match(stripped):
                    continue  # the docstring's own title line; the sentence after it says what the module does
                described = _first_sentence(line)
                if described:
                    return described
        return None
    collected: list[str] = []
    in_block = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("#!"):
            continue  # an interpreter line is not part of the description
        if not stripped or stripped in {"'use strict';", '"use strict";'}:
            if collected:
                break
            continue
        if in_block:
            body = re.sub(r"\*/.*$", "", stripped).lstrip("*").strip()
            if body:
                collected.append(body)
            if "*/" in stripped:
                break
            continue
        if stripped.startswith("/*"):
            body = re.sub(r"^/\*+", "", stripped)
            in_block = "*/" not in body
            body = re.sub(r"\*/.*$", "", body).lstrip("*").strip()
            if body:
                collected.append(body)
            if not in_block:
                break
            continue
        if stripped.startswith(("//", "#")) and not stripped.startswith("#!"):
            collected.append(stripped.lstrip("/#").strip())
            continue
        break
    for line in collected:
        described = _first_sentence(line, minimum=3 if language in _DOC_LANGUAGES else 8)
        if described:
            return described
    return None


# --- directory tree ------------------------------------------------------------------------------------------------


@dataclass
class _Dir:
    path: str  # "" for the root
    files: set[str] = field(default_factory=set)  # files anywhere below
    direct_files: set[str] = field(default_factory=set)
    children: dict[str, _Dir] = field(default_factory=dict)
    languages: Counter[str] = field(default_factory=Counter)


def _build_tree(file_languages: dict[str, str]) -> _Dir:
    root = _Dir("")
    for path, language in file_languages.items():
        parts = PurePosixPath(path).parts
        node = root
        node.files.add(path)
        node.languages[language] += 1
        for depth, part in enumerate(parts[:-1], start=1):
            node = node.children.setdefault(part, _Dir("/".join(parts[:depth])))
            node.files.add(path)
            node.languages[language] += 1
        node.direct_files.add(path)
    return root


def _collapse(node: _Dir) -> _Dir:
    """Merge chains of directories that only contain one directory (``src/pkg/`` shows as one entry)."""
    node.children = {name: _collapse(child) for name, child in node.children.items()}
    merged: dict[str, _Dir] = {}
    for name, child in node.children.items():
        while len(child.children) == 1 and not child.direct_files:
            (only,) = child.children.values()
            child = only
        merged[name] = child
    node.children = merged
    return node


def _walk(node: _Dir, depth: int = 0):
    for name in sorted(node.children):
        child = node.children[name]
        yield depth + 1, child
        yield from _walk(child, depth + 1)


def _files(count: int) -> str:
    return f"{count} file" + ("" if count == 1 else "s")


def _dir_line(node: _Dir, depth: int, symbols: list[str], purpose: str | None) -> str:
    languages = ", ".join(lang for lang, _ in node.languages.most_common(2))
    line = f"{'  ' * (depth - 1)}{node.path}/  ({_files(len(node.files))}, {languages})"
    if purpose:
        line += f" - {purpose}"
    if symbols:
        line += "  defines " + ", ".join(symbols)
    return line


_PURPOSE_FILES = ("__init__.py", "readme.md")


def _purpose_for(node: _Dir, heads: dict[str, str], languages: dict[str, str]) -> str | None:
    """What the directory says it is for: its package docstring or its own README title."""
    for path in sorted(node.direct_files):
        if PurePosixPath(path).name.lower() in _PURPOSE_FILES and path in heads:
            described = describe_file(heads[path], languages[path])
            if described:
                return described
    return None


def _directory_symbols(outline: list[tuple[str, str, str, str, int, int]]) -> dict[str, list[tuple[int, int, str]]]:
    """Per file: the top-level classes, types and functions worth naming, ranked (kind first, then size)."""
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    for path, _language, kind, symbol, start, end in outline:
        rank = _SYMBOL_KINDS.get(kind)
        if rank is None or not symbol or "." in symbol or symbol.startswith("_") or is_test_path(path):
            continue
        by_file.setdefault(path, []).append((rank, -(end - start), symbol))
    return by_file


def _symbols_for(node: _Dir, by_file: dict[str, list[tuple[int, int, str]]]) -> list[str]:
    """What the files directly in this directory define (subdirectories get their own lines)."""
    candidates = [item for path in node.direct_files for item in by_file.get(path, ())]
    seen: list[str] = []
    for _rank, _size, symbol in sorted(candidates):
        if symbol not in seen:
            seen.append(symbol)
        if len(seen) >= _MAX_SYMBOLS:
            break
    return seen


def build_repo_map(index: RepoIndex, *, budget_tokens: int = 1500) -> RepoMap:
    """Render the repository's shape into at most about ``budget_tokens`` tokens."""
    outline = index.keyword.outline()
    heads = index.keyword.file_heads()
    file_languages: dict[str, str] = {}
    for path, language, *_ in outline:
        file_languages.setdefault(path, language)
    if not file_languages:
        return RepoMap(text="(the index is empty)", directories=(), files=0)

    root = _collapse(_build_tree(file_languages))
    by_file = _directory_symbols(outline)
    budget_chars = budget_tokens * 4

    languages = ", ".join(f"{lang} {n}" for lang, n in root.languages.most_common(6))
    header = [
        "Repository map (generated from the index; it is not a source file).",
        f"{len(file_languages)} files: {languages}.",
    ]
    root_files = sorted(root.direct_files, key=lambda p: (p not in README_NAMES, p.startswith("."), p))
    if root_files:
        shown = root_files[:_MAX_ROOT_FILES]
        more = f", and {len(root_files) - len(shown)} more" if len(root_files) > len(shown) else ""
        header.append("Files at the top level: " + ", ".join(shown) + more + ".")
    header.append("")

    # Directories, biggest and shallowest first, until their share of the budget is used.
    ranked = sorted(_walk(root), key=lambda item: (item[0], -len(item[1].files), item[1].path))
    purposes = {node.path: _purpose_for(node, heads, file_languages) for _, node in ranked}
    used = sum(len(line) + 1 for line in header)
    chosen: list[tuple[int, _Dir]] = []
    for depth, node in ranked:
        if is_test_path(node.path + "/x"):
            continue
        line = _dir_line(node, depth, _symbols_for(node, by_file), purposes[node.path])
        if used + len(line) + 1 > budget_chars * _DIR_SHARE and chosen:
            continue
        chosen.append((depth, node))
        used += len(line) + 1
    chosen.sort(key=lambda item: item[1].path)  # tree order: a parent precedes its children

    body = [_dir_line(node, depth, _symbols_for(node, by_file), purposes[node.path]) for depth, node in chosen]
    tests = sorted({n.path for _, n in _walk(root) if is_test_path(n.path + "/x")})
    top_tests = [t for t in tests if not any(t.startswith(other + "/") for other in tests if other != t)]
    if top_tests:
        counts = {t: len({p for p in file_languages if p.startswith(t + "/")}) for t in top_tests}
        body.append("Tests: " + ", ".join(f"{t}/ ({_files(n)})" for t, n in counts.items()) + ".")

    # Documentation a newcomer would open, then code files that describe themselves; shallowest first.
    docs: list[tuple[int, str, str]] = []
    code: list[tuple[int, str, str]] = []
    for path, language in file_languages.items():
        if is_test_path(path) or path in README_NAMES or path not in heads:  # the README is its own source
            continue
        depth = len(PurePosixPath(path).parts)
        if language in _DOC_LANGUAGES and depth <= 3:
            docs.append((depth, path, describe_file(heads[path], language) or ""))
        elif language in _CODE_LANGUAGES and PurePosixPath(path).name != "__init__.py":  # those label their directory
            description = describe_file(heads[path], language)
            if description:
                code.append((depth, path, description))
    used = sum(len(line) + 1 for line in header + body)
    notable: list[str] = []

    def section(title: str, entries: list[tuple[int, str, str]], limit: int) -> list[str]:
        nonlocal used
        lines: list[str] = []
        for _depth, path, description in sorted(entries, key=lambda e: (e[0], e[1].startswith("."), e[1]))[:limit]:
            line = f"  {path}" + (f" - {description}" if description else "")
            if used + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            notable.append(path)
            used += len(line) + 1
        return ["", title, *lines] if lines else []

    body += section("Documentation files (title or first heading):", docs, _MAX_DOCS)
    body += section("Code files that describe themselves (first line of their header comment or docstring):", code, _MAX_NOTABLE_FILES)

    text = "\n".join([*header, *body])
    directories = tuple(sorted({*(node.path for _, node in chosen), *top_tests}))
    readme = next((p for p in file_languages if p in README_NAMES), None)
    return RepoMap(text=text, directories=directories, files=len(file_languages), readme_path=readme, notable_files=tuple(notable))


_LISTED = re.compile(r"([\w.@+/-]+)/ {1,2}\(\d+ files?")


def directories_named_in(text: str) -> frozenset[str]:
    """The directories a rendered map names (its directory lines and its ``Tests:`` line)."""
    return frozenset(_LISTED.findall(text))


def map_lists_directory(text: str, directory: str) -> bool:
    """Whether a rendered map names ``directory``, directly or as part of a longer listed path."""
    wanted = directory.strip("/")
    return any(d == wanted or d.startswith(wanted + "/") for d in directories_named_in(text))


# --- as sources for the prompt ------------------------------------------------------------------------------------


def map_source(repo_map: RepoMap) -> Source:
    """The map as a numbered, citable source. Its location is not a file, and the header says so."""
    return Source(
        path=MAP_PATH,
        language="text",
        start_line=1,
        end_line=repo_map.text.count("\n") + 1,
        text=repo_map.text,
        symbols=(),
        kinds=("map",),
        score=0.0,
        rank=0,
        chunk_ids=(),
        role="map",
    )


def readme_source(index: RepoIndex, repo_map: RepoMap) -> Source | None:
    """The top of the README as a real source (exact lines), or ``None`` if the repository has none indexed."""
    if repo_map.readme_path is None:
        return None
    chunks = index.keyword.find_chunks(repo_map.readme_path)
    if not chunks:
        return None
    lines: dict[int, str] = {}
    for chunk in chunks:
        for offset, line in enumerate(chunk.text.split("\n")):
            lines.setdefault(chunk.start_line + offset, line)
    last = min(max(lines), min(lines) + _README_LINES - 1)
    numbers = [n for n in sorted(lines) if n <= last]
    return Source(
        path=repo_map.readme_path,
        language=chunks[0].language,
        start_line=numbers[0],
        end_line=numbers[-1],
        text="\n".join(lines[n] for n in numbers),
        symbols=(),
        kinds=("doc_section",),
        score=0.0,
        rank=0,
        chunk_ids=tuple(c.id for c in chunks[:1]),
    )
