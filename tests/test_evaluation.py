from __future__ import annotations

import json

import pytest

from codebase_ai.chunking import Chunk
from codebase_ai.evaluation import (
    EvalQuestion,
    EvalReport,
    QuestionResult,
    evaluate,
    load_questions,
    symbol_in_sources,
    to_markdown,
)
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.retrieval.retriever import RetrievalResult, Retriever, ScoredChunk, Source
from helpers import FakeEmbedder


def write_questions(tmp_path, *rows):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


GOOD = {"id": "q1", "question": "How does X work?", "gold_files": ["a.py"]}


# --- loading ------------------------------------------------------------------------------------------------


def test_load_questions_reads_all_fields_with_defaults(tmp_path):
    path = write_questions(
        tmp_path,
        {**GOOD, "type": "explain", "acceptable_files": ["README.md"], "gold_symbols": ["X"]},
        "",
        {"id": "q2", "question": "Where?", "gold_files": ["b.py", "c.py"]},
    )
    first, second = load_questions(path)
    assert first == EvalQuestion("q1", "How does X work?", "explain", ("a.py",), ("README.md",), ("X",))
    assert (second.type, second.acceptable_files, second.gold_symbols) == ("other", (), ())
    assert first.lenient_files == {"a.py", "README.md"}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("{not json", r"q\.jsonl:1: not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ({"id": "q", "question": "x"}, "missing or empty 'gold_files'"),
        ({"id": "q", "question": "", "gold_files": ["a"]}, "missing or empty 'question'"),
        ({"question": "x", "gold_files": ["a"]}, "missing or empty 'id'"),
        ({**GOOD, "gold_files": "a.py"}, "'gold_files' must be a list of non-empty strings"),
        ({**GOOD, "acceptable_files": [1]}, "'acceptable_files' must be a list"),
        ({**GOOD, "gold_symbols": [""]}, "'gold_symbols' must be a list"),
    ],
)
def test_bad_question_files_are_rejected_with_the_line_number(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        load_questions(write_questions(tmp_path, row))


def test_duplicate_ids_and_empty_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate id 'q1'"):
        load_questions(write_questions(tmp_path, GOOD, GOOD))
    with pytest.raises(ValueError, match="no questions found"):
        load_questions(write_questions(tmp_path, ""))


# --- symbol matching ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbols", "texts", "expected"),
    [
        (["computeCriticalPath"], ["const x = computeCriticalPath(tasks);"], True),
        (["UserService.create_user"], ["def create_user(self):"], True),  # qualified names match on the last part
        (["cart"], ["const cartTotal = 1;"], False),  # whole words only
        (["cart"], ["my_cart = 1"], False),
        (["$store"], ["use($store)"], True),
        (["a", "missing"], ["x", "a + 1"], True),
        (["missing"], ["nothing here"], False),
        (["x"], [], False),
    ],
)
def test_symbol_in_sources(symbols, texts, expected):
    assert symbol_in_sources(symbols, texts) is expected


# --- scoring with a stub retriever --------------------------------------------------------------------------


def make_chunk(path):
    return Chunk("r", path, "python", "function", None, 1, 2, "text", "h")


def make_source(path, text="body"):
    return Source(path, "python", 1, 2, text, (), (), 1.0, 1, ("id",))


class StubRetriever:
    """Returns canned rankings, so scoring can be tested independently of search quality."""

    mode = "stub"

    def __init__(self, ranked_by_question: dict[str, tuple[list[str], list[Source]]]):
        self.canned = ranked_by_question

    def retrieve(self, query):
        ranked, sources = self.canned[query]
        candidates = [ScoredChunk(make_chunk(p), 1.0 / r, r) for r, p in enumerate(ranked, start=1)]
        return RetrievalResult(query=query, mode="stub", candidates=candidates, sources=sources)


QUESTIONS = [
    EvalQuestion("q1", "one", "explain", ("a.py",), (), ("alpha",)),
    EvalQuestion("q2", "two", "locate", ("b.py",), ("docs.md",)),
    EvalQuestion("q3", "three", "locate", ("c.py",), ()),
]


def stub_report():
    retriever = StubRetriever(
        {
            "one": (["a.py", "x.py"], [make_source("a.py", "def alpha(): ...")]),  # strict hit at rank 1
            "two": (["x.py", "docs.md", "b.py"], [make_source("docs.md"), make_source("x.py")]),  # lenient 2, strict 3
            "three": (["x.py", "y.py"], [make_source("x.py")]),  # miss
        }
    )
    return evaluate(retriever, QUESTIONS)


def test_per_question_ranks_and_context_flags():
    by_id = {r.id: r for r in stub_report().results}
    assert (by_id["q1"].rank, by_id["q1"].lenient_rank, by_id["q1"].in_context) == (1, 1, True)
    assert (by_id["q2"].rank, by_id["q2"].lenient_rank) == (3, 2)
    assert (by_id["q2"].in_context, by_id["q2"].lenient_in_context) == (False, True)
    assert (by_id["q3"].rank, by_id["q3"].lenient_rank, by_id["q3"].in_context) == (None, None, False)
    assert by_id["q1"].top_files == ("a.py", "x.py")


def test_symbol_recall_is_only_computed_for_labelled_questions():
    by_id = {r.id: r for r in stub_report().results}
    assert by_id["q1"].symbol_in_context is True
    assert by_id["q2"].symbol_in_context is None
    assert stub_report().summary["symbol_in_context"] == 1.0


def test_summary_metrics():
    s = stub_report().summary
    assert s["strict_hit@1"] == pytest.approx(1 / 3)
    assert s["strict_hit@3"] == pytest.approx(2 / 3)
    assert s["strict_mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert s["lenient_hit@3"] == pytest.approx(2 / 3)
    assert s["lenient_mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert s["context_recall"] == pytest.approx(1 / 3)
    assert s["lenient_context_recall"] == pytest.approx(2 / 3)
    assert s["avg_sources"] == pytest.approx(4 / 3)
    assert s["avg_tokens"] > 0 and s["avg_ms"] >= 0


def test_summary_omits_symbol_recall_when_no_question_has_symbols():
    retriever = StubRetriever({"two": (["b.py"], [make_source("b.py")])})
    report = evaluate(retriever, [QUESTIONS[1]])
    assert "symbol_in_context" not in report.summary


def test_by_type_and_misses():
    report = stub_report()
    by_type = report.by_type()
    assert set(by_type) == {"explain", "locate"}
    assert by_type["explain"]["strict_hit@1"] == 1.0
    assert by_type["locate"]["strict_hit@1"] == 0.0
    assert [r.id for r in report.misses(5)] == ["q3"]
    assert [r.id for r in report.misses(2)] == ["q2", "q3"]


def test_an_empty_report_has_an_empty_summary():
    assert evaluate(StubRetriever({}), []).summary == {}


def test_an_unanswerable_question_has_no_gold_files_and_is_left_out_of_retrieval_scores(tmp_path):
    unanswerable = {"id": "u1", "type": "unanswerable", "question": "How are failed payments retried?", "gold_files": []}
    loaded = load_questions(write_questions(tmp_path, GOOD, unanswerable, {"id": "u2", "type": "unanswerable", "question": "Refunds?"}))
    assert [q.answerable for q in loaded] == [True, False, False] and loaded[2].gold_files == ()
    with pytest.raises(ValueError, match="an unanswerable question has no gold files"):
        load_questions(write_questions(tmp_path, {**unanswerable, "gold_files": ["a.py"]}))
    # Nothing can be found for it, so it is not scored as a miss; the report says it was left out.
    retriever = StubRetriever({"one": (["a.py"], [make_source("a.py")])})
    report = evaluate(retriever, [QUESTIONS[0], EvalQuestion("u1", "never asked", "unanswerable", ())])
    assert [r.id for r in report.results] == ["q1"] and report.unanswerable == 1
    assert report.summary["strict_hit@1"] == 1.0
    assert "1 unanswerable question(s) left out" in to_markdown([report])


def test_markdown_table_has_one_row_per_report():
    text = to_markdown([stub_report(), stub_report()], title="Suite")
    assert text.startswith("### Suite")
    assert text.count("| stub |") == 2
    assert "33%" in text and "0.444" in text


# --- against a real index -----------------------------------------------------------------------------------


REAL_QUESTIONS = [
    EvalQuestion("r1", "How is the total price of the shopping cart calculated?", "explain", ("web/cart.js",)),
    EvalQuestion("r2", "Where does the server start listening?", "locate", ("go/server.go",), gold_symbols=("Start",)),
    EvalQuestion("r3", "How is a new user validated and created?", "explain", ("app/user_service.py",)),
]


def test_evaluating_a_real_retriever_end_to_end(sample_repo, tmp_path, fake_embedder):
    index = RepoIndex(sample_repo, tmp_path / "idx")
    try:
        Indexer(index, fake_embedder).run()
        for mode in ("vector", "keyword", "hybrid"):
            report = evaluate(Retriever(index, fake_embedder, mode=mode), REAL_QUESTIONS)
            assert report.name == mode
            assert report.summary["strict_hit@3"] == 1.0, mode
            assert report.summary["context_recall"] == 1.0, mode
        hybrid = evaluate(Retriever(index, fake_embedder), REAL_QUESTIONS, name="my-config")
        assert hybrid.name == "my-config"
        assert hybrid.summary["symbol_in_context"] == 1.0
    finally:
        index.close()


def test_fake_embedder_is_deterministic_across_runs(fake_embedder):
    assert fake_embedder.embed_query("cart total") == FakeEmbedder().embed_query("cart total")


# --- symbol lookup queries ----------------------------------------------------------------------------------


def test_symbol_lookup_questions_use_the_bare_symbol_and_keep_the_labels():
    from codebase_ai.evaluation import symbol_lookup_questions

    derived = symbol_lookup_questions(
        [
            EvalQuestion("q1", "How does X work?", "explain", ("a.py",), ("README.md",), ("alpha", "beta")),
            EvalQuestion("q2", "No symbols here", "locate", ("b.py",)),
            EvalQuestion("q3", "One symbol", "locate", ("c.py", "d.py"), (), ("gamma",)),
        ]
    )
    assert [(q.id, q.question, q.type) for q in derived] == [("q1-sym", "alpha", "symbol"), ("q3-sym", "gamma", "symbol")]
    assert derived[0].gold_files == ("a.py",) and derived[0].acceptable_files == ("README.md",)
    assert derived[1].gold_files == ("c.py", "d.py")
    assert all(q.gold_symbols == () for q in derived)


def test_symbol_lookups_of_the_shipped_question_sets_are_usable():
    from pathlib import Path

    from codebase_ai.evaluation import symbol_lookup_questions

    questions_dir = Path(__file__).parent.parent / "eval" / "questions"
    for path in questions_dir.glob("*.jsonl"):
        if path.stem.endswith("_overview"):
            continue  # questions about the whole repository name no symbols
        derived = symbol_lookup_questions(load_questions(path))
        assert derived, path.name
        assert all(q.question.strip() and " " not in q.question for q in derived), path.name


# --- overview questions: gold_dirs and area coverage ------------------------------------------------------------


def _source(path: str, role: str = "match", text: str = "code") -> Source:
    return Source(path, "python", 1, 2, text, (), ("function",), 1.0, 1, ("c",), role)  # type: ignore[arg-type]


MAP_TEXT = "Repository map\n\nsrc/pkg/  (3 files, python)\n  src/pkg/api/  (2 files, python)\nTests: tests/ (4 files)."


def test_gold_dirs_are_loaded_and_validated(tmp_path):
    good = tmp_path / "ok.jsonl"
    good.write_text(json.dumps({"id": "o1", "question": "q?", "gold_files": ["README.md"], "gold_dirs": ["src", "scripts"]}))
    assert load_questions(good)[0].gold_dirs == ("src", "scripts")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "o1", "question": "q?", "gold_files": ["a"], "gold_dirs": "src"}))
    with pytest.raises(ValueError, match="gold_dirs"):
        load_questions(bad)


def test_area_coverage_counts_code_from_inside_a_directory():
    from codebase_ai.evaluation import area_coverage

    sources = [_source("src/pkg/api/routes.py"), _source("docs/guide.md")]
    assert area_coverage(sources, ["src/pkg/api", "docs", "scripts"]) == pytest.approx(2 / 3)


def test_a_directory_only_named_in_the_map_counts_but_a_similar_prefix_does_not():
    from codebase_ai.evaluation import area_coverage

    sources = [_source("(repository map)", role="map", text=MAP_TEXT)]
    assert area_coverage(sources, ["src/pkg", "src/pkg/api", "src", "tests"]) == 1.0
    assert area_coverage(sources, ["src/pk"]) == 0.0  # a prefix of a name is not a directory that is listed
    assert area_coverage(sources, ["scripts"]) == 0.0


def test_a_similarly_named_directory_is_not_covered_by_code_elsewhere():
    from codebase_ai.evaluation import area_coverage

    assert area_coverage([_source("src2/a.py")], ["src"]) == 0.0


def test_no_gold_dirs_means_no_coverage_score():
    from codebase_ai.evaluation import area_coverage

    assert area_coverage([_source("a.py")], []) is None


def test_the_summary_reports_coverage_and_overview_rate_only_when_there_is_something_to_report():
    with_dirs = QuestionResult("a", "overview", 1, 1, True, True, None, 10, 1, 1.0, (), area_coverage=0.5, overview=True)
    without = QuestionResult("b", "overview", 1, 1, True, True, None, 10, 1, 1.0, (), area_coverage=None, overview=False)
    summary = EvalReport("x", [with_dirs, without]).summary
    assert summary["area_coverage"] == 0.5 and summary["overview_rate"] == 0.5
    assert "area_coverage" not in EvalReport("y", [without]).summary
