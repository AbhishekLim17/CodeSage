from __future__ import annotations

from pathlib import Path

import pytest

from codebase_ai.evaluation import load_questions

PROJECT = Path(__file__).parent.parent
ALL_FILES = sorted((PROJECT / "eval" / "questions").glob("*.jsonl"))
OVERVIEW_FILES = [p for p in ALL_FILES if p.stem.endswith("_overview")]  # questions about the whole repository
QUESTION_FILES = [p for p in ALL_FILES if p not in OVERVIEW_FILES]  # questions about specific code
KNOWN_TYPES = {"locate", "explain", "doc"}


def test_the_expected_question_sets_exist():
    assert {p.stem for p in QUESTION_FILES} >= {"magnaflow", "codebase_ai", "rich"}
    assert {p.stem for p in OVERVIEW_FILES} >= {"magnaflow_overview", "codebase_ai_overview"}


@pytest.mark.parametrize("path", QUESTION_FILES, ids=lambda p: p.stem)
def test_question_files_load_and_are_well_formed(path):
    questions = load_questions(path)
    # The blind sets (httpx, jinja2) have 18: questions added after seeing results would no longer be blind.
    assert len(questions) >= 18
    assert {q.type for q in questions} <= KNOWN_TYPES
    for q in questions:
        for label in (*q.gold_files, *q.acceptable_files):
            assert "\\" not in label and not label.startswith("/") and ":" not in label, (q.id, label)
        assert not set(q.gold_files) & set(q.acceptable_files), f"{q.id}: a file is both gold and acceptable"
        assert q.question.rstrip().endswith(("?", ".")), q.id


@pytest.mark.parametrize("path", OVERVIEW_FILES, ids=lambda p: p.stem)
def test_overview_question_files_are_well_formed(path):
    questions = load_questions(path)
    assert len(questions) >= 5
    assert {q.type for q in questions} == {"overview"}
    assert any(q.gold_dirs for q in questions)
    for q in questions:
        for label in (*q.gold_files, *q.acceptable_files, *q.gold_dirs):
            assert "\\" not in label and not label.startswith("/") and ":" not in label, (q.id, label)
        assert not q.gold_symbols, f"{q.id}: overview questions are scored on files and directories"
        assert q.question.rstrip().endswith(("?", ".")), q.id


def test_the_self_overview_questions_point_at_real_files_and_directories_in_this_repo():
    for q in load_questions(PROJECT / "eval" / "questions" / "codebase_ai_overview.jsonl"):
        for label in (*q.gold_files, *q.acceptable_files):
            assert (PROJECT / label).is_file(), f"{q.id}: {label} does not exist"
        for directory in q.gold_dirs:
            assert (PROJECT / directory).is_dir(), f"{q.id}: {directory} is not a directory"


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.stem)
def test_questions_do_not_leak_their_answers(path):
    """A question that names a gold file makes retrieval trivial and the score meaningless."""
    for q in load_questions(path):
        for gold in q.gold_files:
            name = Path(gold).name
            assert name.lower() not in q.question.lower(), f"{q.id} mentions {name}"


def test_self_evaluation_questions_point_at_real_files_in_this_repo():
    for q in load_questions(PROJECT / "eval" / "questions" / "codebase_ai.jsonl"):
        for label in (*q.gold_files, *q.acceptable_files):
            assert (PROJECT / label).is_file(), f"{q.id}: {label} does not exist"
        for symbol in q.gold_symbols:
            sources = "\n".join((PROJECT / f).read_text(encoding="utf-8") for f in q.gold_files if f.endswith(".py"))
            assert symbol in sources, f"{q.id}: symbol {symbol} not found in {q.gold_files}"


@pytest.mark.parametrize("name", ["rich", "httpx", "jinja2"])
def test_installed_libraries_still_have_the_labelled_files(name):
    package = Path(pytest.importorskip(name).__file__).parent
    for q in load_questions(PROJECT / "eval" / "questions" / f"{name}.jsonl"):
        for label in q.gold_files:
            assert (package / label).is_file(), f"{q.id}: {name}/{label} not found ({name} version changed?)"
