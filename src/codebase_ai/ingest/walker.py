"""Walk a repo honouring .gitignore and deny lists; skip binaries and oversized files; yield files with sha256."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pathspec

from codebase_ai.ingest.languages import KIND_CONFIG, classify
from codebase_ai.ingest.secrets import find_secret, is_secret_filename

# Directories that never contain first-party source worth indexing.
DENIED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "bower_components",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".next",
        ".nuxt",
        ".turbo",
        ".vite",
        ".cache",
        ".idea",
        ".vscode",
        ".gradle",
        ".terraform",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "coverage",
        "vendor",
        ".codebase_ai",
    }
)

_LOCKFILE_NAMES = frozenset({"package-lock.json", "pnpm-lock.yaml", "go.sum", "bun.lockb"})
_GENERATED_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")
_EXAMPLES_PER_REASON = 5


@dataclass(frozen=True)
class WalkLimits:
    max_file_bytes: int = 1_000_000
    max_config_bytes: int = 200_000
    max_line_chars: int = 2_000


DEFAULT_LIMITS = WalkLimits()


@dataclass(frozen=True)
class SourceFile:
    path: str  # repo-relative, forward slashes
    abs_path: Path
    kind: str  # code | doc | config
    language: str
    size: int
    file_hash: str  # sha256 hex of the raw bytes
    text: str  # decoded, "\n" line endings


@dataclass
class WalkReport:
    """What the walk saw and why files were skipped (reason -> count, plus a few example paths)."""

    files_seen: int = 0
    files_yielded: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def skip(self, reason: str, path: str) -> None:
        self.skipped[reason] += 1
        samples = self.examples.setdefault(reason, [])
        if len(samples) < _EXAMPLES_PER_REASON:
            samples.append(path)


def load_source_file(
    abs_path: Path, rel_path: str, limits: WalkLimits = DEFAULT_LIMITS
) -> tuple[SourceFile | None, str | None]:
    """Read one file. Returns ``(file, None)`` or ``(None, skip_reason)``."""
    if is_secret_filename(rel_path):
        return None, "secret_file"
    name = PurePosixPath(rel_path).name
    if name in _LOCKFILE_NAMES or name.endswith(_GENERATED_SUFFIXES):
        return None, "lockfile_or_generated"
    classified = classify(rel_path)
    if classified is None:
        return None, "unsupported_type"
    kind, language = classified
    try:
        size = abs_path.stat().st_size
        cap = limits.max_config_bytes if kind == KIND_CONFIG else limits.max_file_bytes
        if size > cap:
            return None, "too_large"
        data = abs_path.read_bytes()
    except OSError:
        return None, "unreadable"
    if b"\x00" in data[:8192]:
        return None, "binary"
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    if not text.strip():
        return None, "empty"
    if max(len(line) for line in text.split("\n")) > limits.max_line_chars:
        return None, "minified_or_generated"
    if find_secret(text) is not None:
        return None, "secret_pattern"
    return (
        SourceFile(
            path=rel_path,
            abs_path=abs_path,
            kind=kind,
            language=language,
            size=size,
            file_hash=hashlib.sha256(data).hexdigest(),
            text=text,
        ),
        None,
    )


def _load_gitignore(path: str) -> pathspec.GitIgnoreSpec:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return pathspec.GitIgnoreSpec.from_lines(fh.read().splitlines())


def _is_ignored(rel: str, is_dir: bool, specs: tuple[tuple[str, pathspec.GitIgnoreSpec], ...]) -> bool:
    for base, spec in specs:
        sub = rel if not base else rel[len(base) + 1 :]
        if spec.match_file(sub + "/" if is_dir else sub):
            return True
    return False


def walk_repo(
    root: str | Path,
    limits: WalkLimits = DEFAULT_LIMITS,
    report: WalkReport | None = None,
) -> Iterator[SourceFile]:
    """Yield every indexable file under ``root`` in a deterministic order (names sorted within each directory).

    Honours the root ``.gitignore`` and any nested ``.gitignore`` files, never follows symlinks,
    and records every skip in ``report`` (pass your own to read it after iteration).
    """
    root = Path(root).resolve()
    if report is None:
        report = WalkReport()
    chains: dict[str, tuple[tuple[str, pathspec.GitIgnoreSpec], ...]] = {"": ()}

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = "" if Path(dirpath) == root else Path(dirpath).relative_to(root).as_posix()
        specs = chains.pop(rel_dir, ())
        gitignore = os.path.join(dirpath, ".gitignore")
        if os.path.isfile(gitignore) and not os.path.islink(gitignore):
            specs = specs + ((rel_dir, _load_gitignore(gitignore)),)

        kept: list[str] = []
        for name in sorted(dirnames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if os.path.islink(os.path.join(dirpath, name)):
                report.skip("symlink", rel)
            elif name in DENIED_DIRS:
                report.skip("denied_dir", rel)
            elif _is_ignored(rel, True, specs):
                report.skip("gitignored", rel)
            else:
                kept.append(name)
                chains[rel] = specs
        dirnames[:] = kept

        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            full = os.path.join(dirpath, name)
            report.files_seen += 1
            if os.path.islink(full):
                report.skip("symlink", rel)
                continue
            if _is_ignored(rel, False, specs):
                report.skip("gitignored", rel)
                continue
            source, reason = load_source_file(Path(full), rel, limits)
            if source is None:
                report.skip(reason or "unknown", rel)
                continue
            report.files_yielded += 1
            yield source
