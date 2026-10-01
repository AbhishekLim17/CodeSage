from __future__ import annotations

import re

import pytest

from codebase_ai.chunking import chunk_file
from helpers import FIXTURE_REPO, chunk_fixture, line_count, line_of, make_source

PY = "app/user_service.py"
JS = "web/cart.js"
TS = "web/catalog.ts"
TSX = "web/Widget.tsx"
JAVA = "java/OrderService.java"
GO = "go/server.go"
CODE_FIXTURES = [PY, JS, TS, TSX, JAVA, GO]


def by_symbol(chunks):
    return {c.symbol: c for c in chunks if c.symbol}


def span(chunk):
    return chunk.start_line, chunk.end_line


# --- invariants that must hold for every language and every size limit ---------------------------------


@pytest.mark.parametrize("relpath", CODE_FIXTURES)
@pytest.mark.parametrize("max_lines", [120, 12, 6, 3])
def test_chunks_are_exact_and_cover_every_meaningful_line(relpath, max_lines):
    lines = (FIXTURE_REPO / relpath).read_text(encoding="utf-8").rstrip("\n").split("\n")
    chunks = chunk_fixture(relpath, max_lines=max_lines, window_lines=max_lines, window_overlap=1)
    assert chunks
    covered: set[int] = set()
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    missing = [n for n, line in enumerate(lines, start=1) if re.search(r"\w", line) and n not in covered]
    assert missing == [], f"lines not in any chunk: {missing}"
    assert len({c.id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("relpath", CODE_FIXTURES)
def test_chunks_are_sorted_and_carry_file_metadata(relpath):
    chunks = chunk_fixture(relpath)
    assert [span(c) for c in chunks] == sorted(span(c) for c in chunks)
    assert {c.path for c in chunks} == {relpath}
    assert len({c.file_hash for c in chunks}) == 1


# --- Python ---------------------------------------------------------------------------------------------


def test_python_top_level_definitions():
    chunks = chunk_fixture(PY)
    symbols = by_symbol(chunks)
    assert set(symbols) == {"User", "ValidationError", "UserService", "make_service"}

    # decorator is part of the class chunk
    assert symbols["User"].kind == "class"
    assert span(symbols["User"]) == (line_of(PY, "@dataclass"), line_of(PY, "email: str"))
    # comment directly above a definition is attached to it
    assert span(symbols["ValidationError"]) == (line_of(PY, "# Raised when"), line_of(PY, "    pass"))
    assert span(symbols["UserService"]) == (line_of(PY, "class UserService"), line_of(PY, "self.normalize(email))"))
    assert symbols["make_service"].kind == "function"
    assert span(symbols["make_service"]) == (line_of(PY, "def make_service"), line_of(PY, "return UserService(repository)"))


def test_python_module_level_code_becomes_module_chunks():
    modules = [c for c in chunk_fixture(PY) if c.kind == "module"]
    assert [span(c) for c in modules] == [
        (1, line_of(PY, "EMAIL_RE = ")),  # docstring, imports, constants
        (line_of(PY, "if __name__"), line_of(PY, "print(make_service")),
    ]
    assert all(c.symbol is None for c in modules)


def test_python_large_class_is_split_into_header_and_methods():
    chunks = chunk_fixture(PY, max_lines=12)
    symbols = by_symbol(chunks)
    assert {s for s in symbols if s.startswith("UserService")} == {
        "UserService",
        "UserService.__init__",
        "UserService.create_user",
        "UserService.normalize",
        "UserService.find_by_email",
    }
    header = symbols["UserService"]
    assert header.kind == "class"
    assert span(header) == (line_of(PY, "class UserService"), line_of(PY, '"""Creates and looks up users."""'))
    create = symbols["UserService.create_user"]
    assert create.kind == "method"
    assert span(create) == (line_of(PY, "# Validate then persist"), line_of(PY, "        return user"))
    # decorators belong to the method
    assert span(symbols["UserService.normalize"])[0] == line_of(PY, "@staticmethod")
    # async methods are found too
    assert symbols["UserService.find_by_email"].kind == "method"
    # small definitions are not split
    assert span(symbols["User"]) == (line_of(PY, "@dataclass"), line_of(PY, "email: str"))


def test_python_long_function_is_split_into_overlapping_windows_with_same_symbol():
    body = "\n".join(f"    x{i} = {i}" for i in range(199))
    chunks = chunk_file(make_source(f"def big():\n{body}\n", "big.py"), "repo")
    assert [span(c) for c in chunks] == [(1, 60), (51, 110), (101, 160), (151, 200)]
    assert {(c.symbol, c.kind) for c in chunks} == {("big", "function")}


def test_python_syntax_errors_still_produce_full_coverage():
    text = "def ok():\n    return 1\n\ndef broken(:\n    pass\n\nclass Fine:\n    x = 1\n"
    chunks = chunk_file(make_source(text, "bad.py"), "repo")
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    assert {1, 2, 4, 5, 7, 8} <= covered
    assert {"ok", "Fine"} <= {c.symbol for c in chunks}


def test_python_file_with_only_statements_has_no_symbols():
    chunks = chunk_file(make_source("import os\nprint(os.name)\n", "script.py"), "repo")
    assert [(c.kind, c.symbol, span(c)) for c in chunks] == [("module", None, (1, 2))]


def test_python_nested_functions_stay_inside_their_parent():
    text = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    chunks = chunk_file(make_source(text, "n.py"), "repo")
    assert [(c.symbol, span(c)) for c in chunks] == [("outer", (1, 4))]


def test_trailing_comment_is_not_stolen_by_the_next_definition():
    text = "x = 1  # about x\ndef f():\n    return x\n"
    chunks = chunk_file(make_source(text, "t.py"), "repo")
    assert span(by_symbol(chunks)["f"]) == (2, 3)


def test_comment_separated_by_blank_line_is_not_attached():
    text = "# licence header\n\ndef f():\n    return 1\n"
    chunks = chunk_file(make_source(text, "t.py"), "repo")
    assert span(by_symbol(chunks)["f"]) == (3, 4)
    assert (1, 1) in {span(c) for c in chunks if c.kind == "module"}


# --- JavaScript -----------------------------------------------------------------------------------------


def test_javascript_functions_arrow_functions_and_class():
    chunks = chunk_fixture(JS)
    symbols = by_symbol(chunks)
    assert set(symbols) == {"cartTotal", "applyDiscount", "formatPrice", "Cart"}

    # JSDoc block above the function is included; `export` is part of the chunk
    assert span(symbols["cartTotal"]) == (line_of(JS, "/**"), line_of(JS, "items.reduce") + 1)
    assert symbols["cartTotal"].kind == "function"
    # `export const f = () => {}` and `const f = async () => {}` are functions
    assert span(symbols["applyDiscount"]) == (line_of(JS, "export const applyDiscount"), line_of(JS, "return total;") + 1)
    assert span(symbols["formatPrice"]) == (line_of(JS, "const formatPrice"), line_of(JS, "toFixed(2)") + 1)
    # `export default class` is unwrapped
    assert symbols["Cart"].kind == "class"
    assert span(symbols["Cart"]) == (line_of(JS, "export default class"), line_count(JS))


def test_javascript_imports_and_constants_are_a_module_chunk():
    (module,) = [c for c in chunk_fixture(JS) if c.kind == "module"]
    assert span(module) == (1, line_of(JS, "TAX_RATE"))


def test_javascript_class_is_split_into_methods_when_large():
    chunks = chunk_fixture(JS, max_lines=8)
    symbols = by_symbol(chunks)
    assert {"Cart", "Cart.constructor", "Cart.add", "Cart.total"} <= set(symbols)
    assert symbols["Cart"].text == "export default class Cart {"
    assert symbols["Cart.add"].kind == "method"
    assert span(symbols["Cart.add"]) == (line_of(JS, "  add(item)"), line_of(JS, "this.items.push") + 1)


def test_javascript_wrapped_function_values_and_class_fields():
    text = (
        "const Memo = React.memo(() => {\n  return 1;\n});\n"
        "const plain = 42;\n"
        "class Btn {\n  onClick = () => {\n    go();\n  };\n  label = 'x';\n}\n"
    )
    chunks = chunk_file(make_source(text, "c.js"), "repo", max_lines=4)
    symbols = by_symbol(chunks)
    assert symbols["Memo"].kind == "function"
    assert "plain" not in symbols  # a plain constant is module code, not a function
    assert symbols["Btn.onClick"].kind == "method"


def test_javascript_export_default_anonymous_function():
    chunks = chunk_file(make_source("export default function () {\n  return 1;\n}\n", "d.js"), "repo")
    assert [(c.symbol, c.kind) for c in chunks] == [("default", "function")]


# --- TypeScript / TSX -----------------------------------------------------------------------------------


def test_typescript_interfaces_enums_aliases_and_classes():
    symbols = by_symbol(chunk_fixture(TS))
    assert set(symbols) == {"Product", "ProductMap", "Status", "Catalog"}
    assert {symbols[s].kind for s in ("Product", "ProductMap", "Status")} == {"type"}
    assert symbols["Catalog"].kind == "class"
    assert span(symbols["Product"]) == (line_of(TS, "export interface"), line_of(TS, "price: number") + 1)
    assert span(symbols["Status"]) == (line_of(TS, "export enum"), line_of(TS, "Retired") + 1)


def test_typescript_class_split_keeps_field_in_header_and_finds_methods():
    symbols = by_symbol(chunk_fixture(TS, max_lines=6))
    assert span(symbols["Catalog"]) == (line_of(TS, "export class Catalog"), line_of(TS, "private products"))
    assert symbols["Catalog.add"].kind == "method"
    assert span(symbols["Catalog.find"]) == (line_of(TS, "  find(id"), line_of(TS, "return this.products[id]") + 1)


def test_tsx_components_and_props_type():
    symbols = by_symbol(chunk_fixture(TSX))
    assert set(symbols) == {"Props", "Widget", "Badge"}
    assert symbols["Props"].kind == "type"
    assert symbols["Widget"].kind == "function"
    assert span(symbols["Badge"]) == (line_of(TSX, "export const Badge"), line_of(TSX, "export const Badge"))


# --- Java -----------------------------------------------------------------------------------------------


def test_java_small_class_is_one_chunk():
    chunks = chunk_fixture(JAVA)
    symbols = by_symbol(chunks)
    assert set(symbols) == {"OrderService"}
    assert symbols["OrderService"].kind == "class"
    assert span(symbols["OrderService"]) == (line_of(JAVA, "public class"), line_count(JAVA))
    modules = [c for c in chunks if c.kind == "module"]
    assert [span(c) for c in modules] == [(1, line_of(JAVA, "import java.util.List"))]


def test_java_large_class_splits_into_constructor_methods_and_nested_types():
    symbols = by_symbol(chunk_fixture(JAVA, max_lines=8))
    assert {"OrderService", "OrderService.OrderService", "OrderService.place", "OrderService.Listener"} <= set(symbols)
    assert symbols["OrderService.OrderService"].kind == "method"
    assert symbols["OrderService.Listener"].kind == "type"
    # javadoc attaches to the method
    assert span(symbols["OrderService.place"]) == (line_of(JAVA, "/** Place"), line_of(JAVA, "orders.add(item)") + 1)
    assert span(symbols["OrderService"]) == (line_of(JAVA, "public class"), line_of(JAVA, "private final"))


# --- Go -------------------------------------------------------------------------------------------------


def test_go_types_methods_and_functions():
    chunks = chunk_fixture(GO)
    symbols = by_symbol(chunks)
    assert set(symbols) == {"Server", "Server.Start", "NewServer"}
    assert symbols["Server"].kind == "type"
    assert span(symbols["Server"]) == (line_of(GO, "// Server handles"), line_of(GO, "Name string") + 1)
    # method symbol is qualified by its receiver type, and keeps its doc comment
    assert symbols["Server.Start"].kind == "method"
    assert span(symbols["Server.Start"]) == (line_of(GO, "// Start begins"), line_of(GO, "return nil") + 1)
    assert symbols["NewServer"].kind == "function"
    (module,) = [c for c in chunks if c.kind == "module"]
    assert span(module) == (1, line_of(GO, 'import "fmt"'))


# --- robustness -----------------------------------------------------------------------------------------


def realistic_js(handlers: int = 14) -> str:
    """A ~400-line Cloud-Functions-style file: doc comments, non-ASCII banners, handlers, template literals."""
    parts = [
        "/**\n * Cloud functions — reminders and account cleanup.\n * Written for the free plan.\n */\n",
        "const functions = require('firebase-functions');\nconst admin = require('firebase-admin');\n",
        "admin.initializeApp();\n\n",
        "// ─── helpers ──────────────────\n",
    ]
    for i in range(handlers):
        parts.append(
            f"// ─── {i}.5 · handler number {i} ⚠️ ───\n"
            f"exports.handler{i} = functions.https.onCall(async (data, ctx) => {{\n"
            f"  const caller = await getCaller{i % 3}(ctx);\n"
            f"  if (!['admin', 'master'].includes(caller?.role))\n"
            f"    throw new functions.https.HttpsError('permission-denied', 'Not authorized');\n"
            f"  try {{\n"
            f"    const snap = await admin.firestore().collection('items').doc(`item-${{data.id}}`).get();\n"
            f"    return {{ ok: true, exists: snap.exists, note: `handled {i}` }};\n"
            f"  }} catch (error) {{\n"
            f"    console.error('handler {i} failed:', error);\n"
            f"    throw new functions.https.HttpsError('internal', error.message);\n"
            f"  }}\n"
            f"}});\n\n"
        )
        if i % 4 == 0:
            parts.append(
                f"// Role checks read the caller's profile {i}.\n"
                f"async function getCaller{i % 3}(ctx) {{\n"
                f"  if (!ctx.auth) return null;\n"
                f"  const doc = await admin.firestore().doc(`users/${{ctx.auth.uid}}`).get();\n"
                f"  return doc.exists ? doc.data() : null;\n"
                f"}}\n\n"
            )
    parts.append("function emailKey(email) {\n  return String(email).toLowerCase();\n}\n")
    return "".join(parts)


def assert_exact_and_covering(text: str, chunks) -> None:
    lines = text.rstrip("\n").split("\n")
    covered: set[int] = set()
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    assert [n for n, line in enumerate(lines, start=1) if re.search(r"\w", line) and n not in covered] == []


def test_soak_repeated_chunking_of_realistic_files_stays_exact():
    """Repeats parse+chunk many times; catches syntax-tree memory-safety problems (see pyproject tree-sitter pin)."""
    js = realistic_js()
    assert js.count("\n") > 200
    for round_number in range(40):
        chunks = chunk_file(make_source(js, "functions/index.js", file_hash=str(round_number)), "repo")
        assert_exact_and_covering(js, chunks)
        assert {"getCaller0", "getCaller1", "getCaller2", "emailKey"} <= {c.symbol for c in chunks}
        for relpath in CODE_FIXTURES:
            fixture_text = (FIXTURE_REPO / relpath).read_text(encoding="utf-8")
            assert_exact_and_covering(fixture_text, chunk_fixture(relpath, max_lines=12, window_lines=12))


def test_impossible_tree_positions_fall_back_to_windows(monkeypatch):
    from codebase_ai.chunking import code_chunker

    garbage = [code_chunker._Def(start=10820, end=10820, kind="function", symbol="emailKey")]
    monkeypatch.setattr(code_chunker, "_collect", lambda *args, **kwargs: garbage)
    text = "function a() {\n  return 1;\n}\n"
    chunks = chunk_file(make_source(text, "a.js"), "repo")
    assert [(c.kind, c.start_line, c.end_line) for c in chunks] == [("window", 1, 3)]


def test_overlapping_definitions_are_rejected_as_malformed(monkeypatch):
    from codebase_ai.chunking import code_chunker

    overlapping = [
        code_chunker._Def(start=0, end=2, kind="function", symbol="a"),
        code_chunker._Def(start=1, end=3, kind="function", symbol="b"),
    ]
    monkeypatch.setattr(code_chunker, "_collect", lambda *args, **kwargs: overlapping)
    chunks = chunk_file(make_source("a\nb\nc\nd\n", "x.js"), "repo")
    assert {c.kind for c in chunks} == {"window"}


def test_errors_while_analysing_the_tree_fall_back_to_windows(monkeypatch, caplog):
    from codebase_ai.chunking import code_chunker

    def boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(code_chunker, "_collect", boom)
    chunks = chunk_file(make_source("def f():\n    return 1\n", "f.py"), "repo")
    assert [c.kind for c in chunks] == ["window"]
    assert "falling back to windows" in caplog.text


# --- embed text -----------------------------------------------------------------------------------------


def test_embed_text_prefixes_path_symbol_and_kind_but_text_stays_raw():
    chunk = by_symbol(chunk_fixture(PY))["make_service"]
    header, _, body = chunk.embed_text.partition("\n")
    assert header == f"{PY} • make_service • function"
    assert body == chunk.text
    assert chunk.location == f"{PY}:{chunk.start_line}-{chunk.end_line}"


def test_chunk_ids_depend_on_repo_path_range_and_content():
    a = chunk_file(make_source("def f():\n    return 1\n", "a.py"), "repo1")[0]
    same = chunk_file(make_source("def f():\n    return 1\n", "a.py"), "repo1")[0]
    other_repo = chunk_file(make_source("def f():\n    return 1\n", "a.py"), "repo2")[0]
    other_path = chunk_file(make_source("def f():\n    return 1\n", "b.py"), "repo1")[0]
    other_text = chunk_file(make_source("def f():\n    return 2\n", "a.py"), "repo1")[0]
    assert a.id == same.id
    assert len({a.id, other_repo.id, other_path.id, other_text.id}) == 4
