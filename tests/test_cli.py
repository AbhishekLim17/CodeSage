from __future__ import annotations

import pytest
from typer.testing import CliRunner

from codebase_ai.cli import app
from helpers import FIXTURE_REPO, FakeEmbedder

runner = CliRunner()


@pytest.fixture
def fake_provider(monkeypatch):
    """Make the CLI use the fake embedder instead of downloading a model."""
    state = {"embedder": FakeEmbedder()}
    monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: state["embedder"])
    return state


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "stats", "chunks", "search", "ask", "serve", "eval"):
        assert command in result.output


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_chunks_command_lists_symbols_with_line_ranges():
    result = runner.invoke(app, ["chunks", str(FIXTURE_REPO / "app" / "user_service.py")])
    assert result.exit_code == 0
    assert "UserService" in result.output and "make_service" in result.output
    assert "python" in result.output


def test_chunks_command_can_print_text():
    result = runner.invoke(app, ["chunks", str(FIXTURE_REPO / "go" / "server.go"), "--text"])
    assert result.exit_code == 0 and "func NewServer" in result.output


def test_chunks_command_refuses_secret_files(tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("KEY=1\n")
    result = runner.invoke(app, ["chunks", str(secret)])
    assert result.exit_code == 1 and "secret_file" in result.output


def test_invalid_configuration_is_reported_not_a_traceback(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "skynet")
    result = runner.invoke(app, ["chunks", str(FIXTURE_REPO / "README.md")])
    assert result.exit_code == 2 and "Invalid configuration" in result.output


def test_index_then_stats_then_search(fake_provider, sample_repo):
    result = runner.invoke(app, ["index", str(sample_repo)])
    assert result.exit_code == 0, result.output
    assert "Files indexed" in result.output and "fake:hash-64" in result.output

    stats = runner.invoke(app, ["stats", str(sample_repo)])
    assert stats.exit_code == 0 and "fake:hash-64" in stats.output and "Chunks" in stats.output

    keyword = runner.invoke(app, ["search", str(sample_repo), "cart total", "--mode", "keyword", "-k", "3"])
    assert keyword.exit_code == 0 and "web/cart.js" in keyword.output

    vector = runner.invoke(app, ["search", str(sample_repo), "validate email and create a user", "-k", "3"])
    assert vector.exit_code == 0 and "app/user_service.py" in vector.output


def test_index_reports_skipped_files(fake_provider, sample_repo):
    (sample_repo / ".env.local").write_text("KEY=1\n")
    result = runner.invoke(app, ["index", str(sample_repo)])
    assert result.exit_code == 0 and "secret_file" in result.output


def test_second_index_run_reports_unchanged_files(fake_provider, sample_repo):
    runner.invoke(app, ["index", str(sample_repo)])
    result = runner.invoke(app, ["index", str(sample_repo)])
    assert result.exit_code == 0 and "Files unchanged" in result.output


def test_switching_model_asks_for_full_reindex(fake_provider, sample_repo):
    runner.invoke(app, ["index", str(sample_repo)])
    fake_provider["embedder"] = FakeEmbedder("fake:other-model")

    refused = runner.invoke(app, ["index", str(sample_repo)])
    assert refused.exit_code == 1 and "--full" in refused.output

    search = runner.invoke(app, ["search", str(sample_repo), "cart"])
    assert search.exit_code == 1 and "--full" in search.output

    rebuilt = runner.invoke(app, ["index", str(sample_repo), "--full"])
    assert rebuilt.exit_code == 0


def test_stats_and_search_without_an_index_explain_what_to_do(fake_provider, sample_repo):
    for args in (["stats", str(sample_repo)], ["search", str(sample_repo), "x"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1 and "codebase-ai index" in result.output


def test_missing_repo_path_is_rejected_by_the_parser(tmp_path):
    result = runner.invoke(app, ["index", str(tmp_path / "does-not-exist")])
    assert result.exit_code == 2


# --- retrieval: search and eval ------------------------------------------------------------------------------


@pytest.fixture
def indexed_repo(fake_provider, sample_repo):
    result = runner.invoke(app, ["index", str(sample_repo)])
    assert result.exit_code == 0, result.output
    return sample_repo


def test_search_defaults_to_vector_and_shows_kind_symbol_and_token_estimate(indexed_repo):
    result = runner.invoke(app, ["search", str(indexed_repo), "shopping cart total price"])
    assert result.exit_code == 0, result.output
    assert "web/cart.js:" in result.output
    assert "[function] cartTotal" in result.output  # the kind label must not be swallowed as rich markup
    assert "(vector)" in result.output and "tokens" in result.output


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_supports_every_mode(indexed_repo, mode):
    result = runner.invoke(app, ["search", str(indexed_repo), "start the server", "--mode", mode])
    assert result.exit_code == 0, result.output
    assert "go/server.go" in result.output and f"({mode})" in result.output


def test_search_limits_sources_and_can_print_text(indexed_repo):
    limited = runner.invoke(app, ["search", str(indexed_repo), "user email cart server", "-k", "2"])
    assert limited.exit_code == 0 and limited.output.count("\n 1.") + limited.output.count("\n 2.") >= 1
    assert " 3. " not in limited.output
    full = runner.invoke(app, ["search", str(indexed_repo), "make service factory", "--text", "-k", "1"])
    assert "def make_service" in full.output


def test_search_shows_the_repository_map_that_ask_adds_for_whole_project_questions(indexed_repo):
    # `search` is how you check what `ask` hands the model, so it must not leave the map out.
    result = runner.invoke(app, ["search", str(indexed_repo), "What does this project do?", "-k", "3"])
    assert result.exit_code == 0, result.output
    assert "repository map is added" in result.output
    assert " 1. (repository map)" in result.output and "[map]" in result.output


def test_search_adds_no_map_to_an_ordinary_question(indexed_repo):
    result = runner.invoke(app, ["search", str(indexed_repo), "shopping cart total price"])
    assert "repository map" not in result.output


def test_indexing_a_folder_with_nothing_to_search_says_so_and_fails(fake_provider, tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    result = runner.invoke(app, ["index", str(tmp_path / "empty")])
    assert result.exit_code == 1 and "Nothing to search" in result.output and "right folder" in result.output


def test_search_reports_no_results_for_a_keyword_query_with_no_matches(indexed_repo):
    result = runner.invoke(app, ["search", str(indexed_repo), "zzzqqq", "--mode", "keyword"])
    assert result.exit_code == 0 and "No results" in result.output


def write_eval_questions(tmp_path, rows=None):
    rows = rows or [
        {"id": "e1", "type": "explain", "question": "How is the total price of the cart calculated?", "gold_files": ["web/cart.js"]},
        {"id": "e2", "type": "locate", "question": "Where does the server start?", "gold_files": ["go/server.go"], "gold_symbols": ["Start"]},
    ]
    import json

    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_eval_compares_modes_and_writes_json(indexed_repo, tmp_path):
    out = tmp_path / "out" / "eval.json"
    result = runner.invoke(
        app, ["eval", str(indexed_repo), "--questions", str(write_eval_questions(tmp_path)), "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    for mode in ("vector", "keyword", "hybrid"):
        assert mode in result.output
    assert "Hit@1" in result.output and "MRR" in result.output and "In ctx" in result.output
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"vector", "keyword", "hybrid"}
    assert data["hybrid"]["summary"]["strict_hit@3"] == 1.0
    assert [q["id"] for q in data["hybrid"]["questions"]] == ["e1", "e2"]


def test_eval_can_run_a_single_keyword_mode_without_an_embedder(indexed_repo, tmp_path, monkeypatch):
    def no_embedder(settings):
        raise AssertionError("keyword-only evaluation must not load an embedding model")

    monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", no_embedder)
    result = runner.invoke(
        app, ["eval", str(indexed_repo), "--questions", str(write_eval_questions(tmp_path)), "--modes", "keyword"]
    )
    assert result.exit_code == 0, result.output


def test_eval_lists_misses_with_gold_and_actual_files(indexed_repo, tmp_path):
    rows = [{"id": "m1", "question": "cart total price", "gold_files": ["config/settings.yaml"]}]
    result = runner.invoke(app, ["eval", str(indexed_repo), "--questions", str(write_eval_questions(tmp_path, rows))])
    assert result.exit_code == 0
    assert "m1" in result.output and "gold: config/settings.yaml" in result.output and "got:" in result.output


def test_eval_warns_about_labelled_files_that_are_not_indexed(indexed_repo, tmp_path):
    rows = [{"id": "w1", "question": "cart", "gold_files": ["web/cart.js", "web/typo.js"]}]
    result = runner.invoke(app, ["eval", str(indexed_repo), "--questions", str(write_eval_questions(tmp_path, rows))])
    assert result.exit_code == 0 and "not in the index" in result.output and "web/typo.js" in result.output


def test_eval_rejects_unknown_modes_and_bad_question_files(indexed_repo, tmp_path):
    questions = write_eval_questions(tmp_path)
    bad_mode = runner.invoke(app, ["eval", str(indexed_repo), "-q", str(questions), "--modes", "vector,magic"])
    assert bad_mode.exit_code == 2 and "magic" in bad_mode.output
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{nope", encoding="utf-8")
    bad_file = runner.invoke(app, ["eval", str(indexed_repo), "-q", str(broken)])
    assert bad_file.exit_code == 2 and "not valid JSON" in bad_file.output


def test_eval_without_an_index_explains_what_to_do(fake_provider, sample_repo, tmp_path):
    result = runner.invoke(app, ["eval", str(sample_repo), "-q", str(write_eval_questions(tmp_path))])
    assert result.exit_code == 1 and "codebase-ai index" in result.output


def test_eval_and_search_refuse_a_model_mismatch(indexed_repo, fake_provider, tmp_path):
    fake_provider["embedder"] = FakeEmbedder("fake:other-model")
    search = runner.invoke(app, ["search", str(indexed_repo), "cart"])
    assert search.exit_code == 1 and "--full" in search.output
    evaluated = runner.invoke(app, ["eval", str(indexed_repo), "-q", str(write_eval_questions(tmp_path))])
    assert evaluated.exit_code == 1 and "--full" in evaluated.output


def test_search_mode_can_be_set_by_the_environment_and_overridden_by_the_flag(indexed_repo, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    from codebase_ai.config import get_settings

    get_settings.cache_clear()
    from_env = runner.invoke(app, ["search", str(indexed_repo), "shopping cart total price"])
    assert from_env.exit_code == 0 and "(keyword)" in from_env.output
    overridden = runner.invoke(app, ["search", str(indexed_repo), "shopping cart total price", "--mode", "hybrid"])
    assert overridden.exit_code == 0 and "(hybrid)" in overridden.output


def test_eval_can_add_identifier_lookups_and_takes_a_test_penalty(indexed_repo, tmp_path):
    out = tmp_path / "with-lookups.json"
    result = runner.invoke(
        app,
        [
            "eval", str(indexed_repo), "-q", str(write_eval_questions(tmp_path)),
            "--symbol-lookups", "--test-penalty", "0.25", "--modes", "vector,hybrid", "--json", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "identifier lookups" in result.output and "test demotion x0.25" in result.output
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["symbol_lookups"]) == {"vector", "hybrid"}
    assert [q["id"] for q in data["symbol_lookups"]["vector"]["questions"]] == ["e2-sym"]


# --- characters the terminal cannot show ----------------------------------------------------------------------


def test_emoji_in_source_does_not_crash_output_on_a_legacy_code_page(fake_provider, sample_repo):
    """Regression: printing a source line with an emoji to a cp1252 stream raised UnicodeEncodeError."""
    (sample_repo / "web" / "flags.js").write_text(
        "// \U0001f9ea experimental flags for the cart — ✓\nexport const cartFlags = { total: true };\n",
        encoding="utf-8",
    )
    legacy = CliRunner(charset="cp1252")
    assert legacy.invoke(app, ["index", str(sample_repo)]).exit_code == 0

    for args in (
        ["search", str(sample_repo), "experimental flags for the cart", "--mode", "keyword", "-k", "3"],
        ["search", str(sample_repo), "experimental flags for the cart", "--text"],
        ["chunks", str(sample_repo / "web" / "flags.js"), "--text"],
    ):
        result = legacy.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)
        assert "Traceback" not in result.output and "UnicodeEncodeError" not in result.output, args
    text = legacy.invoke(app, ["chunks", str(sample_repo / "web" / "flags.js"), "--text"]).output
    assert "experimental flags" in text and "cartFlags" in text  # the rest of the line is still there


def test_command_help_texts_are_shown_whole_and_not_eaten_as_markup():
    """Regression: 'cites ... as [n]' lost its ending because rich read [n] as a markup tag."""
    import re

    # CI forces colour codes into the output, which land in the middle of the sentence; strip them before comparing.
    overview = re.sub(r"\x1b\[[0-9;]*m", "", runner.invoke(app, ["--help"]).output)
    assert "cites the code it is based on by number." in " ".join(overview.split()).replace("│ ", "").replace(" │", "")
    for command in ("index", "stats", "search", "ask", "serve", "eval"):
        help_text = runner.invoke(app, [command, "--help"])
        assert help_text.exit_code == 0, command


def test_version_is_shown_and_matches_the_package_metadata():
    from codebase_ai import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == f"codebase-ai {__version__}"
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == __version__  # one number, in two places: they must not drift apart


def test_python_dash_m_runs_the_same_cli():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "codebase_ai", "--version"], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0 and done.stdout.startswith("codebase-ai ")
