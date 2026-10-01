"""AST-based chunking with tree-sitter: one chunk per function/class/method plus module-level chunks.

Every line that contains a word character ends up in at least one chunk: definitions become
function/method/class/type chunks, and whatever is left between them (imports, constants, script code)
becomes ``module`` chunks. Definitions keep the comment block directly above them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from functools import cache

from codebase_ai.chunking.models import (
    KIND_CLASS,
    KIND_FUNCTION,
    KIND_METHOD,
    KIND_MODULE,
    KIND_TYPE,
    Chunk,
)
from codebase_ai.chunking.window_chunker import has_content, window_ranges
from codebase_ai.ingest.walker import SourceFile

log = logging.getLogger(__name__)

_COMMENTS = frozenset({"comment", "line_comment", "block_comment"})
# Expression node types that are "a function" when they are the value of a declaration.
_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})


@dataclass(frozen=True)
class LangSpec:
    """Which tree-sitter node types mean what, for one language."""

    grammar: str
    functions: frozenset[str] = frozenset()  # named callable definitions
    methods: frozenset[str] = frozenset()  # functions that are methods even at top level (Go)
    classes: frozenset[str] = frozenset()  # containers whose body holds member definitions
    types: frozenset[str] = frozenset()  # atomic named types (interface, enum, type alias)
    declarations: frozenset[str] = frozenset()  # `const f = () => {}` style declarations
    fields: frozenset[str] = frozenset()  # class fields that may hold a function (`handler = () => {}`)
    wrappers: dict[str, tuple[str, ...]] = field(default_factory=dict)  # node type -> fields to unwrap through


_JS = LangSpec(
    grammar="javascript",
    functions=frozenset({"function_declaration", "generator_function_declaration", "method_definition"}),
    classes=frozenset({"class_declaration"}),
    declarations=frozenset({"lexical_declaration", "variable_declaration"}),
    fields=frozenset({"field_definition"}),
    wrappers={"export_statement": ("declaration", "value")},
)
_TS = LangSpec(
    grammar="typescript",
    functions=_JS.functions,
    classes=frozenset({"class_declaration", "abstract_class_declaration"}),
    types=frozenset({"interface_declaration", "enum_declaration", "type_alias_declaration"}),
    declarations=_JS.declarations,
    fields=frozenset({"field_definition", "public_field_definition"}),
    wrappers=_JS.wrappers,
)

SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        grammar="python",
        functions=frozenset({"function_definition"}),
        classes=frozenset({"class_definition"}),
        wrappers={"decorated_definition": ("definition",)},
    ),
    "javascript": _JS,
    "typescript": _TS,
    "tsx": replace(_TS, grammar="tsx"),
    "java": LangSpec(
        grammar="java",
        functions=frozenset({"method_declaration", "constructor_declaration"}),
        classes=frozenset({"class_declaration", "record_declaration"}),
        types=frozenset({"interface_declaration", "enum_declaration", "annotation_type_declaration"}),
    ),
    "go": LangSpec(
        grammar="go",
        functions=frozenset({"function_declaration", "method_declaration"}),
        methods=frozenset({"method_declaration"}),
        types=frozenset({"type_declaration"}),
    ),
}


def supported_languages() -> frozenset[str]:
    return frozenset(SPECS)


@cache
def _parser(grammar: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(grammar)


@dataclass
class _Def:
    start: int  # 0-based first row (includes attached comments/decorators)
    end: int  # 0-based last row, inclusive
    kind: str
    symbol: str
    children: list[_Def] = field(default_factory=list)


def _rows(node) -> tuple[int, int]:
    start, end = node.start_point.row, node.end_point.row
    if node.end_point.column == 0 and end > start:  # node swallowed the trailing newline
        end -= 1
    return start, end


def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _name(node) -> str | None:
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named)
    if node.type == "type_declaration":  # Go: type_declaration -> type_spec(name)
        for child in node.named_children:
            spec_name = child.child_by_field_name("name")
            if spec_name is not None:
                return _text(spec_name)
    return None


def _go_receiver(node) -> str | None:
    receiver = node.child_by_field_name("receiver")
    stack = [receiver] if receiver is not None else []
    while stack:
        current = stack.pop(0)
        if current.type == "type_identifier":
            return _text(current)
        stack.extend(current.named_children)
    return None


def _is_function_value(value) -> bool:
    """`() => {}`, `function () {}`, or a call wrapping one, e.g. `memo(() => ...)`."""
    if value is None:
        return False
    if value.type in _FUNCTION_VALUES:
        return True
    if value.type == "call_expression":
        args = value.child_by_field_name("arguments")
        return args is not None and any(a.type in _FUNCTION_VALUES for a in args.named_children)
    return False


def _qualify(parent: str, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _as_def(node, spec: LangSpec, parent: str, in_class: bool) -> _Def | None:
    inner = node
    while inner.type in spec.wrappers:
        nxt = None
        for field_name in spec.wrappers[inner.type]:
            nxt = inner.child_by_field_name(field_name)
            if nxt is not None:
                break
        if nxt is None:
            return None
        inner = nxt
    start, end = _rows(node)
    kind_type = inner.type

    if kind_type in spec.classes or kind_type == "class":
        name = _name(inner) or "default"
        d = _Def(start, end, KIND_CLASS, _qualify(parent, name))
        body = inner.child_by_field_name("body")
        if body is not None:
            d.children = _collect(body, spec, d.symbol, in_class=True)
        return d
    if kind_type in spec.types:
        return _Def(start, end, KIND_TYPE, _qualify(parent, _name(inner) or "default"))
    if kind_type in spec.functions:
        name = _name(inner) or "default"
        if kind_type in spec.methods:
            receiver = _go_receiver(inner)
            symbol = f"{receiver}.{name}" if receiver else name
            return _Def(start, end, KIND_METHOD, symbol)
        kind = KIND_METHOD if in_class else KIND_FUNCTION
        return _Def(start, end, kind, _qualify(parent, name))
    if kind_type in spec.declarations:
        for declarator in inner.named_children:
            name_node = declarator.child_by_field_name("name")
            if (
                declarator.type == "variable_declarator"
                and name_node is not None
                and name_node.type == "identifier"
                and _is_function_value(declarator.child_by_field_name("value"))
            ):
                return _Def(start, end, KIND_FUNCTION, _qualify(parent, _text(name_node)))
        return None
    if kind_type in spec.fields:
        name_node = inner.child_by_field_name("property") or inner.child_by_field_name("name")
        if name_node is not None and _is_function_value(inner.child_by_field_name("value")):
            return _Def(start, end, KIND_METHOD, _qualify(parent, _text(name_node)))
        return None
    if kind_type in _FUNCTION_VALUES:  # `export default () => {}` / `export default function () {}`
        return _Def(start, end, KIND_FUNCTION, _qualify(parent, "default"))
    return None


def _collect(container, spec: LangSpec, parent: str, in_class: bool) -> list[_Def]:
    """Definitions directly inside ``container``, each extended upward over its leading comment block."""
    defs: list[_Def] = []
    comment_start = comment_end = -1  # contiguous comment block waiting for a definition
    prev_end = -1
    for node in container.named_children:
        start, end = _rows(node)
        if node.type in _COMMENTS:
            if start <= prev_end:  # trailing comment on the previous statement's line
                comment_start = -1
            elif comment_start >= 0 and start <= comment_end + 1:
                comment_end = end
            else:
                comment_start, comment_end = start, end
            prev_end = max(prev_end, end)
            continue
        d = _as_def(node, spec, parent, in_class)
        if d is not None:
            if comment_start >= 0 and comment_end + 1 >= d.start:
                d.start = min(d.start, comment_start)
            defs.append(d)
        comment_start = -1
        prev_end = end
    return defs


def _well_formed(defs: list[_Def], lo: int, hi: int) -> bool:
    """Definitions are inside ``lo..hi``, in order, non-overlapping, and children sit inside their parent."""
    previous_end = lo - 1
    for d in defs:
        if not previous_end < d.start <= d.end <= hi or not _well_formed(d.children, d.start, d.end):
            return False
        previous_end = d.end
    return True


class _Emitter:
    def __init__(self, source: SourceFile, repo_id: str, max_lines: int, window: int, overlap: int):
        self.source = source
        self.repo_id = repo_id
        self.max_lines = max_lines
        self.window = window
        self.overlap = overlap
        self.lines = source.text.split("\n")
        self.chunks: list[Chunk] = []

    def _make(self, kind: str, symbol: str | None, start: int, end: int) -> None:
        self.chunks.append(
            Chunk(
                repo_id=self.repo_id,
                path=self.source.path,
                language=self.source.language,
                kind=kind,
                symbol=symbol,
                start_line=start + 1,
                end_line=end + 1,
                text="\n".join(self.lines[start : end + 1]),
                file_hash=self.source.file_hash,
            )
        )

    def emit_range(self, kind: str, symbol: str | None, start: int, end: int) -> None:
        """One chunk for the range, or overlapping windows (same kind/symbol) if it is too long."""
        if end - start + 1 <= self.max_lines:
            self._make(kind, symbol, start, end)
            return
        for s, e in window_ranges(start, end, self.window, self.overlap):
            if has_content(self.lines, s, e):
                self._make(kind, symbol, s, e)

    def emit_leftovers(self, kind: str, symbol: str | None, lo: int, hi: int, covered: list[_Def]) -> None:
        """Chunks for the parts of ``lo..hi`` that no definition covers (blank/punctuation-only parts dropped)."""
        cursor = lo
        for d in sorted(covered, key=lambda x: x.start):
            self._leftover(kind, symbol, cursor, d.start - 1)
            cursor = d.end + 1
        self._leftover(kind, symbol, cursor, hi)

    def _leftover(self, kind: str, symbol: str | None, start: int, end: int) -> None:
        while start <= end and not self.lines[start].strip():
            start += 1
        while end >= start and not self.lines[end].strip():
            end -= 1
        if start <= end and has_content(self.lines, start, end):
            self.emit_range(kind, symbol, start, end)

    def emit_def(self, d: _Def) -> None:
        if d.kind == KIND_CLASS and d.children and d.end - d.start + 1 > self.max_lines:
            self.emit_leftovers(KIND_CLASS, d.symbol, d.start, d.end, d.children)
            for child in d.children:
                self.emit_def(child)
        else:
            self.emit_range(d.kind, d.symbol, d.start, d.end)


def chunk_code(
    source: SourceFile,
    repo_id: str,
    *,
    max_lines: int = 120,
    window_lines: int = 60,
    window_overlap: int = 10,
) -> list[Chunk] | None:
    """Chunk a source file by its syntax tree.

    Returns ``None`` (caller falls back to line windows) if the language is unsupported, the parser fails,
    or the syntax tree reports positions that cannot be right.
    """
    spec = SPECS.get(source.language)
    if spec is None:
        return None
    emitter = _Emitter(source, repo_id, max_lines, window_lines, window_overlap)
    try:
        tree = _parser(spec.grammar).parse(source.text.encode("utf-8"))
        defs = _collect(tree.root_node, spec, parent="", in_class=False)
        if not _well_formed(defs, 0, len(emitter.lines) - 1):
            raise ValueError("syntax tree positions fall outside the file")
        emitter.emit_leftovers(KIND_MODULE, None, 0, len(emitter.lines) - 1, defs)
        for d in defs:
            emitter.emit_def(d)
    except Exception:
        log.warning("tree-sitter chunking failed for %s; falling back to windows", source.path, exc_info=True)
        return None
    emitter.chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return emitter.chunks
