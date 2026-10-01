from __future__ import annotations

from itertools import pairwise

import pytest

from codebase_ai.chunking import Chunk
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.retrieval.retriever import (
    MODES,
    RetrievalError,
    Retriever,
    ScoredChunk,
    Source,
    apply_test_penalty,
    estimate_tokens,
    is_test_path,
    mentions_tests,
)
from helpers import FakeEmbedder


@pytest.fixture
def built(sample_repo, tmp_path, fake_embedder):
    idx = RepoIndex(sample_repo, tmp_path / "idx")
    Indexer(idx, fake_embedder).run()
    yield idx
    idx.close()


@pytest.fixture
def built_small_chunks(sample_repo, tmp_path, fake_embedder):
    """Same repo, but classes over 12 lines are split into header + member chunks."""
    idx = RepoIndex(sample_repo, tmp_path / "idx-small")
    Indexer(idx, fake_embedder, chunk_max_lines=12, window_lines=12, window_overlap=3).run()
    yield idx
    idx.close()


def chunk(path="a.py", start=1, end=10, text=None, symbol=None, kind="function") -> Chunk:
    lines = text if text is not None else "\n".join(f"line {n}" for n in range(start, end + 1))
    return Chunk("repo", path, "python", kind, symbol, start, end, lines, "hash")


def scored(rank: int, **kwargs) -> ScoredChunk:
    return ScoredChunk(chunk(**kwargs), score=1.0 / rank, rank=rank)


# --- ranking modes ---------------------------------------------------------------------------------------


def test_every_mode_finds_the_obvious_file(built, fake_embedder):
    for mode in MODES:
        result = Retriever(built, fake_embedder, mode=mode).retrieve("shopping cart total price")
        assert result.ranked_files[0] == "web/cart.js", mode
        assert result.mode == mode


def test_hybrid_candidates_are_the_union_of_both_searches(built, fake_embedder):
    query = "validate email and create a user"
    vector = {s.chunk.id for s in Retriever(built, fake_embedder, mode="vector", top_k=5).rank(query)}
    keyword = {s.chunk.id for s in Retriever(built, fake_embedder, mode="keyword", top_k=5).rank(query)}
    hybrid = {s.chunk.id for s in Retriever(built, fake_embedder, mode="hybrid", top_k=5).rank(query)}
    assert hybrid == vector | keyword


def test_ranks_are_consecutive_from_one_and_scores_do_not_increase(built, fake_embedder):
    ranked = Retriever(built, fake_embedder).rank("cart total")
    assert [s.rank for s in ranked] == list(range(1, len(ranked) + 1))
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_keyword_mode_needs_no_embedder_and_ignores_model_mismatch(sample_repo, tmp_path):
    idx = RepoIndex(sample_repo, tmp_path / "idx-kw")
    try:
        Indexer(idx, FakeEmbedder("fake:model-a")).run()
        result = Retriever(idx, None, mode="keyword").retrieve("cart total")
        assert result.ranked_files[0] == "web/cart.js"
    finally:
        idx.close()


def test_vector_and_hybrid_modes_require_an_embedder(built):
    for mode in ("vector", "hybrid"):
        with pytest.raises(RetrievalError, match="needs an embedder"):
            Retriever(built, None, mode=mode)


def test_embedding_model_mismatch_asks_for_a_rebuild(built):
    with pytest.raises(RetrievalError, match=r"fake:hash-64.*fake:other.*--full"):
        Retriever(built, FakeEmbedder("fake:other"), mode="hybrid")


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "fuzzy"}, {"top_k": 0}, {"budget_tokens": 0}, {"max_chunks": 0}, {"max_chunks_per_file": 0}],
)
def test_invalid_configuration_is_rejected(built, fake_embedder, kwargs):
    with pytest.raises(ValueError):
        Retriever(built, fake_embedder, **kwargs)


def test_a_query_with_no_keyword_matches_still_works_in_hybrid_mode(built, fake_embedder):
    result = Retriever(built, fake_embedder, mode="hybrid").retrieve("zzzqqq")
    assert result.sources  # vector search always has neighbours
    assert Retriever(built, None, mode="keyword").retrieve("zzzqqq").sources == []


# --- sources are exact ------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["cart total", "how are users created", "start the server", "installer setup steps"])
def test_every_source_is_exactly_the_files_lines(built, fake_embedder, sample_repo, query):
    result = Retriever(built, fake_embedder).retrieve(query)
    assert result.sources
    for source in result.sources:
        file_lines = (sample_repo / source.path).read_text(encoding="utf-8").split("\n")
        assert source.text == "\n".join(file_lines[source.start_line - 1 : source.end_line]), source.location
        assert source.location == f"{source.path}:{source.start_line}-{source.end_line}"


def test_sources_are_best_first_and_never_overlap_within_a_file(built, fake_embedder):
    result = Retriever(built, fake_embedder).retrieve("user email cart server")
    assert [s.rank for s in result.sources] == sorted(s.rank for s in result.sources)
    by_path: dict[str, list[Source]] = {}
    for s in result.sources:
        by_path.setdefault(s.path, []).append(s)
    for spans in by_path.values():
        spans.sort(key=lambda s: s.start_line)
        for earlier, later in pairwise(spans):
            assert earlier.end_line + 1 < later.start_line  # merged if they touch or overlap


def test_result_helpers(built, fake_embedder):
    result = Retriever(built, fake_embedder).retrieve("cart total")
    assert len(result.ranked_files) == len(set(result.ranked_files))
    assert result.tokens == sum(estimate_tokens(s.text) for s in result.sources)
    assert set(result.context_files) <= set(result.ranked_files)


# --- merging ----------------------------------------------------------------------------------------------


def test_overlapping_chunks_merge_into_one_exact_span():
    a, b = scored(1, start=1, end=10), scored(2, start=6, end=15)
    (source,) = Retriever._merge([b, a], set())
    assert (source.start_line, source.end_line) == (1, 15)
    assert source.text == "\n".join(f"line {n}" for n in range(1, 16))
    assert source.rank == 1 and source.score == 1.0
    assert len(source.chunk_ids) == 2


def test_directly_adjacent_chunks_merge_but_a_gap_keeps_them_apart():
    adjacent = Retriever._merge([scored(1, start=1, end=5), scored(2, start=6, end=9)], set())
    assert [(s.start_line, s.end_line) for s in adjacent] == [(1, 9)]
    gap = Retriever._merge([scored(1, start=1, end=5), scored(2, start=7, end=9)], set())
    assert [(s.start_line, s.end_line) for s in gap] == [(1, 5), (7, 9)]


def test_chunks_of_different_files_never_merge_and_sources_sort_by_rank():
    sources = Retriever._merge([scored(2, path="b.py"), scored(1, path="a.py", start=50, end=60)], set())
    assert [(s.path, s.rank) for s in sources] == [("a.py", 1), ("b.py", 2)]


def test_merged_source_lists_symbols_and_kinds_once_each():
    chunks = [
        scored(1, start=1, end=5, symbol="f", kind="function"),
        scored(2, start=4, end=9, symbol="f", kind="function"),
        scored(3, start=10, end=12, symbol="g", kind="method"),
    ]
    (source,) = Retriever._merge(chunks, set())
    assert source.symbols == ("f", "g") and source.kinds == ("function", "method")


def test_a_span_made_only_of_context_chunks_is_marked_context():
    header = scored(1, start=1, end=3, symbol="C", kind="class")
    method = scored(1, start=20, end=25, symbol="C.m", kind="method")
    sources = Retriever._merge([header, method], {header.chunk.id})
    assert {(s.symbols, s.role) for s in sources} == {(("C",), "context"), (("C.m",), "match")}
    merged = Retriever._merge([header, scored(2, start=4, end=8)], {header.chunk.id})
    assert [s.role for s in merged] == ["match"]  # touches a real match, so it is one ordinary span


# --- selection under the budget -----------------------------------------------------------------------------


def make_retriever(built, **kwargs):
    return Retriever(built, None, mode="keyword", **kwargs)


def texts(n_tokens: int) -> str:
    return "x" * (n_tokens * 4)


def test_first_pass_takes_the_best_chunk_of_each_file_before_second_helpings(built):
    ranked = [
        scored(1, path="a.py", start=1, end=2, text=texts(100)),
        scored(2, path="a.py", start=10, end=11, text=texts(100)),
        scored(3, path="a.py", start=20, end=21, text=texts(100)),
        scored(4, path="b.py", start=1, end=2, text=texts(100)),
    ]
    chosen = make_retriever(built, budget_tokens=300)._select(ranked)
    assert {(s.chunk.path, s.rank) for s in chosen} == {("a.py", 1), ("b.py", 4), ("a.py", 2)}


def test_the_top_chunk_is_kept_even_if_it_alone_exceeds_the_budget(built):
    ranked = [scored(1, text=texts(5000)), scored(2, path="b.py", text=texts(10))]
    chosen = make_retriever(built, budget_tokens=100)._select(ranked)
    assert [s.rank for s in chosen] == [1]  # kept despite the budget; nothing else fits after it


def test_budget_is_respected_after_the_first_chunk(built):
    ranked = [scored(r, path=f"f{r}.py", text=texts(400)) for r in range(1, 9)]
    chosen = make_retriever(built, budget_tokens=1000)._select(ranked)
    assert len(chosen) == 2  # 400 + 400 fits, a third would reach 1200


def test_a_too_big_chunk_is_skipped_in_favour_of_a_smaller_lower_ranked_one(built):
    ranked = [scored(1, path="a.py", text=texts(100)), scored(2, path="b.py", text=texts(900)), scored(3, path="c.py", text=texts(50))]
    chosen = make_retriever(built, budget_tokens=200)._select(ranked)
    assert [s.rank for s in chosen] == [1, 3]


def test_max_chunks_and_max_chunks_per_file_are_enforced(built):
    same_file = [scored(r, start=r * 20, end=r * 20 + 2, text=texts(10)) for r in range(1, 8)]
    assert len(make_retriever(built, max_chunks_per_file=2)._select(same_file)) == 2
    many_files = [scored(r, path=f"f{r}.py", text=texts(10)) for r in range(1, 30)]
    assert len(make_retriever(built, max_chunks=5)._select(many_files)) == 5


def test_selection_never_picks_the_same_chunk_twice(built):
    one = scored(1, text=texts(10))
    assert len(make_retriever(built)._select([one, one, one])) == 1


def test_no_candidates_means_no_sources(built):
    assert make_retriever(built)._select([]) == []


# --- class header context -----------------------------------------------------------------------------------


def test_a_retrieved_method_brings_its_class_header(built_small_chunks, fake_embedder):
    result = Retriever(built_small_chunks, fake_embedder, mode="keyword", max_chunks=1).retrieve("normalize")
    match = next(s for s in result.sources if s.role == "match")
    assert "UserService.normalize" in match.symbols
    (header,) = [s for s in result.sources if s.role == "context"]
    assert header.path == match.path and header.symbols == ("UserService",)
    assert header.text.startswith("class UserService")
    assert header.end_line < match.start_line
    assert result.sources.index(header) == result.sources.index(match) + 1


def test_context_can_be_switched_off_and_is_absent_for_plain_functions(built_small_chunks, fake_embedder):
    off = Retriever(built_small_chunks, fake_embedder, mode="keyword", max_chunks=1, add_context=False)
    assert all(s.role == "match" for s in off.retrieve("normalize").sources)
    plain = Retriever(built_small_chunks, fake_embedder, mode="keyword", max_chunks=1).retrieve("make service factory")
    assert all(s.role == "match" for s in plain.sources)


def test_context_is_skipped_when_the_budget_has_no_room(built_small_chunks, fake_embedder):
    tight = Retriever(built_small_chunks, fake_embedder, mode="keyword", max_chunks=1, budget_tokens=1)
    result = tight.retrieve("normalize")
    assert [s.role for s in result.sources] == ["match"]


def test_a_class_header_already_retrieved_is_not_added_twice(built_small_chunks, fake_embedder):
    result = Retriever(built_small_chunks, fake_embedder, mode="keyword").retrieve("UserService creates and looks up users normalize")
    headers = [s for s in result.sources if s.path == "app/user_service.py" and s.text.startswith("class UserService")]
    assert len(headers) <= 1


# --- demoting test files ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_a.py",
        "test/helpers.js",
        "src/lib/foo.test.js",
        "web/App.spec.tsx",
        "scripts/integration/mail-jobs.itest.js",
        "pkg/thing_test.go",
        "conftest.py",
        "tests/fixtures/data.py",
        "src/__tests__/a.js",
        "e2e/login.js",
    ],
)
def test_test_paths_are_recognised(path):
    assert is_test_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/lib/latest.js",
        "src/contest.py",
        "src/attestation.py",
        "src/testing_utils.py",
        "docs/ARCHITECTURE.md",
        "src/codebase_ai/index/embedder.py",
        "src/protest.py",
    ],
)
def test_ordinary_paths_that_merely_contain_test_are_not_test_paths(path):
    assert not is_test_path(path)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how are the tests run", True),
        ("pytest fixtures for the index", True),
        ("unit test helper", True),
        ("Mock the embedder", True),
        ("latest contest attestation", False),
        ("computeCriticalPath", False),
        ("how does login work", False),
    ],
)
def test_mentions_tests(query, expected):
    assert mentions_tests(query) is expected


def pair(path, score):
    return (chunk(path=path), score)


def test_the_penalty_demotes_test_chunks_but_keeps_everything():
    scored = [pair("tests/test_a.py", 1.0), pair("src/a.py", 0.8), pair("src/b.py", 0.3)]
    result = apply_test_penalty(scored, 0.5, "how does a work")
    assert [c.path for c, _ in result] == ["src/a.py", "tests/test_a.py", "src/b.py"]
    assert {c.path: s for c, s in result}["tests/test_a.py"] == pytest.approx(0.5)
    assert len(result) == 3


def test_the_penalty_is_off_when_disabled_or_when_the_query_is_about_tests():
    scored = [pair("tests/test_a.py", 1.0), pair("src/a.py", 0.8)]
    assert apply_test_penalty(scored, 1.0, "how does a work") == scored
    assert apply_test_penalty(scored, 0.1, "how is a tested") == scored


def test_the_penalty_always_lowers_a_score_even_a_negative_one():
    (result,) = apply_test_penalty([pair("tests/test_a.py", -0.2)], 0.5, "x")
    assert result[1] < -0.2


def test_equal_scores_keep_their_original_order_after_the_penalty():
    scored = [pair("src/a.py", 0.5), pair("src/b.py", 0.5), pair("src/c.py", 0.5)]
    assert [c.path for c, _ in apply_test_penalty(scored, 0.5, "x")] == ["src/a.py", "src/b.py", "src/c.py"]


@pytest.mark.parametrize("bad", [0, -0.5, 1.5])
def test_invalid_penalties_are_rejected(built, fake_embedder, bad):
    with pytest.raises(ValueError, match="test_penalty"):
        Retriever(built, fake_embedder, test_penalty=bad)


def test_retriever_ranks_a_test_file_below_the_code_it_tests(sample_repo, tmp_path, fake_embedder):
    (sample_repo / "web" / "cart.test.js").write_text(
        "import { cartTotal } from './cart.js';\n"
        "test('cart total price of the shopping cart', () => {\n  expect(cartTotal([])).toBe(0);\n});\n",
        encoding="utf-8",
    )
    idx = RepoIndex(sample_repo, tmp_path / "idx-tests")
    try:
        Indexer(idx, fake_embedder).run()
        for mode in ("vector", "keyword", "hybrid"):
            demoted = Retriever(idx, fake_embedder, mode=mode, test_penalty=0.05).retrieve("shopping cart total price")
            files = demoted.ranked_files
            assert files.index("web/cart.js") < files.index("web/cart.test.js"), mode
            # a question about tests is ranked exactly as if the penalty were off
            about_tests = "shopping cart total price tests"
            plain = Retriever(idx, fake_embedder, mode=mode).retrieve(about_tests)
            same = Retriever(idx, fake_embedder, mode=mode, test_penalty=0.05).retrieve(about_tests)
            assert same.ranked_files == plain.ranked_files, mode
    finally:
        idx.close()


# --- release notes (changelog) demotion ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["HISTORY.md", "CHANGELOG.md", "docs/CHANGES.rst", "NEWS", "RELEASE_NOTES.md", "release-notes.txt", "CHANGELOG-1.2.md", "whats-new.md"],
)
def test_release_notes_are_recognised(path):
    from codebase_ai.retrieval.retriever import is_changelog_path

    assert is_changelog_path(path)


@pytest.mark.parametrize(
    "path",
    ["src/history.py", "src/requests/history_utils.py", "app/releases.js", "docs/releases/1.2.md", "src/changes/apply.py", "README.md", "docs/guide.rst", "src/news.ts"],
)
def test_code_and_other_documents_are_never_taken_for_release_notes(path):
    from codebase_ai.retrieval.retriever import is_changelog_path

    assert not is_changelog_path(path)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Which release dropped support for an old Python version?", True),
        ("What changed in 2.0?", True),
        ("Show me the changelog", True),
        ("How do I upgrade from 1.x?", True),
        ("How does a session handle redirects?", False),
        ("What happens when a value is None?", False),
    ],
)
def test_questions_about_changes_are_recognised(query, expected):
    from codebase_ai.retrieval.retriever import mentions_changes

    assert mentions_changes(query) is expected


class TestPathPenalties:
    def scored(self):
        return [
            (chunk(path="HISTORY.md", text="x"), 1.0),
            (chunk(path="tests/test_a.py", text="x"), 0.9),
            (chunk(path="src/a.py", text="x"), 0.8),
        ]

    def paths(self, result):
        return [c.path for c, _ in result]

    def test_both_penalties_reorder_and_nothing_is_dropped(self):
        from codebase_ai.retrieval.retriever import apply_path_penalties

        out = apply_path_penalties(self.scored(), "how does it work", test_penalty=0.5, changelog_penalty=0.5)
        assert self.paths(out) == ["src/a.py", "HISTORY.md", "tests/test_a.py"]  # 0.8, 0.5, 0.45

    def test_each_penalty_is_independent(self):
        from codebase_ai.retrieval.retriever import apply_path_penalties

        assert self.paths(apply_path_penalties(self.scored(), "q", changelog_penalty=0.5)) == ["tests/test_a.py", "src/a.py", "HISTORY.md"]
        assert self.paths(apply_path_penalties(self.scored(), "q", test_penalty=0.5)) == ["HISTORY.md", "src/a.py", "tests/test_a.py"]

    def test_a_question_about_changes_keeps_release_notes_in_place_but_tests_are_still_demoted(self):
        from codebase_ai.retrieval.retriever import apply_path_penalties

        out = apply_path_penalties(self.scored(), "which release changed this", test_penalty=0.5, changelog_penalty=0.1)
        assert self.paths(out) == ["HISTORY.md", "src/a.py", "tests/test_a.py"]

    def test_a_question_about_tests_keeps_tests_but_release_notes_are_still_demoted(self):
        from codebase_ai.retrieval.retriever import apply_path_penalties

        out = apply_path_penalties(self.scored(), "how is this tested", test_penalty=0.1, changelog_penalty=0.5)
        assert self.paths(out) == ["tests/test_a.py", "src/a.py", "HISTORY.md"]

    def test_with_both_off_the_input_is_returned_untouched(self):
        from codebase_ai.retrieval.retriever import apply_path_penalties

        scored = self.scored()
        assert apply_path_penalties(scored, "q") is scored

    def test_the_old_test_penalty_function_still_behaves_the_same(self):
        assert self.paths(apply_test_penalty(self.scored(), 0.5, "how does it work")) == ["HISTORY.md", "src/a.py", "tests/test_a.py"]


def test_changelog_penalty_must_be_in_range(built, fake_embedder):
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="changelog_penalty"):
            Retriever(built, fake_embedder, changelog_penalty=bad)


def test_a_retriever_demotes_release_notes_unless_the_question_is_about_releases(tmp_path, fake_embedder):
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "redirects.py").write_text("def follow_redirects(url):\n    return url\n", encoding="utf-8")
    (repo / "HISTORY.md").write_text("# History\n\n- How are redirects followed: redirects are followed, redirects followed.\n", encoding="utf-8")
    index = RepoIndex(repo, tmp_path / "idx-cl")
    try:
        Indexer(index, fake_embedder).run()
        plain = Retriever(index, fake_embedder, test_penalty=1.0).rank("how are redirects followed")
        demoted = Retriever(index, fake_embedder, test_penalty=1.0, changelog_penalty=0.1).rank("how are redirects followed")
        assert plain[0].chunk.path == "HISTORY.md"  # the situation being fixed: prose outranks the code
        assert demoted[0].chunk.path == "src/redirects.py" and {s.chunk.path for s in demoted} == {s.chunk.path for s in plain}
        about = Retriever(index, fake_embedder, test_penalty=1.0, changelog_penalty=0.1).rank("which release changed how redirects are followed")
        assert about[0].chunk.path == "HISTORY.md"
    finally:
        index.close()
