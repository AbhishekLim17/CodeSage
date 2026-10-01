from __future__ import annotations

import pytest

from codebase_ai.index.keyword_index import KeywordIndex, query_tokens, split_identifiers
from codebase_ai.ingest.walker import WalkReport, walk_repo
from helpers import FIXTURE_REPO, chunk_fixture


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("getUserById", ["getuserbyid", "get", "user", "by", "id"]),
        ("get_user_by_id", ["get_user_by_id", "get", "user", "by", "id"]),
        ("HTTPServer", ["httpserver", "http", "server"]),
        ("user", ["user"]),
        ("x = 1", []),
        ("cart.js", ["cart", "js"]),
    ],
)
def test_split_identifiers(text, expected):
    assert split_identifiers(text) == expected


def test_camel_and_snake_case_produce_the_same_parts():
    camel = set(split_identifiers("getUserById")) - {"getuserbyid"}
    snake = set(split_identifiers("get_user_by_id")) - {"get_user_by_id"}
    assert camel == snake


def test_query_tokens_drop_stopwords_and_duplicates():
    assert query_tokens("How does the user login work? user!") == ["user", "login", "work"]


@pytest.fixture
def all_chunks(sample_repo):
    from codebase_ai.chunking import chunk_file

    chunks = []
    for source in walk_repo(sample_repo, report=WalkReport()):
        chunks.extend(chunk_file(source, "repo"))
    return chunks


@pytest.fixture
def index(tmp_path, all_chunks):
    idx = KeywordIndex(tmp_path / "kw" / "chunks.db")
    idx.upsert(all_chunks)
    yield idx
    idx.close()


def top(index, query, k=3):
    return [(c.path, c.symbol) for c in index.get_chunks([h.chunk_id for h in index.search(query, k)])]


def test_search_finds_code_by_words_in_its_symbol_and_path(index):
    assert top(index, "cart total")[0] == ("web/cart.js", "cartTotal")


def test_search_matches_snake_case_functions_from_plain_words(index):
    assert top(index, "make service")[0][1] == "make_service"


def test_search_finds_body_terms(index):
    assert top(index, "normalize email")[0][0] == "app/user_service.py"


def test_search_ranks_docs_for_docs_questions(index):
    assert top(index, "run the installer")[0][0] == "README.md"


def test_scores_are_positive_and_sorted_best_first(index):
    hits = index.search("email user", 5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_no_match_and_empty_queries_return_nothing(index):
    assert index.search("zzzqqqxxx", 5) == []
    assert index.search("the of and", 5) == []
    assert index.search("cart", 0) == []


@pytest.mark.parametrize("query", ['foo" OR "bar', "a.b(c)*", "NEAR(x y)", "col:name", "'; DROP TABLE chunks; --", "cart AND"])
def test_query_syntax_characters_cannot_break_the_search(index, query):
    index.search(query, 5)  # must not raise


def test_upsert_is_idempotent(index, all_chunks):
    before = index.count()
    index.upsert(all_chunks)
    assert index.count() == before == len(all_chunks)
    hits = index.search("cart total", 50)
    assert len(hits) == len({h.chunk_id for h in hits})  # no duplicate rows left behind in the FTS table


def test_delete_removes_from_search_and_chunk_table(index):
    hit = index.search("cart total", 1)[0]
    index.delete([hit.chunk_id])
    assert hit.chunk_id not in {h.chunk_id for h in index.search("cart total", 50)}
    assert index.get_chunks([hit.chunk_id]) == []


def test_get_chunks_preserves_order_and_skips_unknown_ids(index):
    ids = [h.chunk_id for h in index.search("email", 3)]
    chunks = index.get_chunks([*reversed(ids), "unknown-id"])
    assert [c.id for c in chunks] == list(reversed(ids))


def test_chunks_round_trip_through_the_index(index):
    original = chunk_fixture("app/user_service.py")
    stored = index.get_chunks([c.id for c in original])
    assert [(c.path, c.symbol, c.kind, c.start_line, c.end_line, c.text) for c in stored] == [
        (c.path, c.symbol, c.kind, c.start_line, c.end_line, c.text) for c in original
    ]


def test_find_chunks_by_path_symbol_and_kind_in_line_order(index):
    in_file = index.find_chunks("app/user_service.py")
    assert len(in_file) > 3
    assert [c.start_line for c in in_file] == sorted(c.start_line for c in in_file)
    assert {c.path for c in in_file} == {"app/user_service.py"}

    (service,) = index.find_chunks("app/user_service.py", symbol="UserService")
    assert service.kind == "class"
    assert index.find_chunks("app/user_service.py", symbol="UserService", kind="method") == []
    assert [c.symbol for c in index.find_chunks("app/user_service.py", kind="function")] == ["make_service"]
    assert index.find_chunks("no/such/file.py") == []


def test_reset_empties_everything(index):
    index.reset()
    assert index.count() == 0
    assert index.search("cart", 5) == []


def test_index_persists_across_reopen(tmp_path, all_chunks):
    path = tmp_path / "persist.db"
    first = KeywordIndex(path)
    first.upsert(all_chunks)
    first.close()
    second = KeywordIndex(path)
    try:
        assert second.count() == len(all_chunks)
        assert second.search("cart total", 1)
    finally:
        second.close()


def test_fixture_repo_exists():
    assert (FIXTURE_REPO / "README.md").is_file()


def test_outline_lists_every_chunk_without_its_text(index, all_chunks):
    outline = index.outline()
    assert len(outline) == len(all_chunks) == index.count()
    assert {row[0] for row in outline} == {c.path for c in all_chunks}
    path, language, kind, _symbol, start, end = next(row for row in outline if row[3] == "cartTotal")
    assert (path, language, kind) == ("web/cart.js", "javascript", "function")
    assert 1 <= start <= end
    assert outline == sorted(outline, key=lambda row: (row[0], row[4]))


def test_file_heads_holds_the_chunk_that_starts_at_line_one_of_each_file(index, all_chunks):
    heads = index.file_heads()
    starts_at_one = {c.path for c in all_chunks if c.start_line == 1}
    assert set(heads) == starts_at_one
    assert heads["app/user_service.py"].startswith('"""User management service."""')
    assert heads["README.md"].startswith("# Sample Repo")


def test_file_heads_prefers_the_shortest_chunk_when_several_start_at_line_one(tmp_path):
    from codebase_ai.chunking import Chunk

    small = Chunk("r", "a.py", "python", "module", None, 1, 2, "one\ntwo", "h")
    large = Chunk("r", "a.py", "python", "function", "f", 1, 9, "\n".join(["x"] * 9), "h")
    idx = KeywordIndex(tmp_path / "kw" / "chunks.db")
    try:
        idx.upsert([large, small])
        assert idx.file_heads() == {"a.py": "one\ntwo"}
    finally:
        idx.close()
