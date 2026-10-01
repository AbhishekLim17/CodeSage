from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codebase_ai.cli import app
from codebase_ai.llm.base import ProviderError
from codebase_ai.llm.base import create_provider as REAL_CREATE_PROVIDER
from codebase_ai.rag.prompts import NO_CONTEXT_ANSWER
from helpers import FakeEmbedder, ScriptedProvider

runner = CliRunner()
QUESTION = "shopping cart total price"


@pytest.fixture
def llm(monkeypatch):
    """The CLI's LLM is a scripted fake; ``llm.provider`` can be swapped and ``llm.requests`` records the calls."""

    class Holder:
        provider = ScriptedProvider("cartTotal adds up the prices [1].")
        requests: list[dict]

    holder = Holder()
    holder.requests = []

    def create(settings, *, provider=None, model=None):
        holder.requests.append({"provider": provider, "model": model})
        return holder.provider

    monkeypatch.setattr("codebase_ai.llm.base.create_provider", create)
    monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: FakeEmbedder())
    return holder


@pytest.fixture
def indexed(llm, sample_repo) -> Path:
    result = runner.invoke(app, ["index", str(sample_repo)])
    assert result.exit_code == 0, result.output
    return sample_repo


def ask(repo: Path, question: str = QUESTION, *options: str):
    return runner.invoke(app, ["ask", str(repo), question, *options])


class TestAsk:
    def test_prints_the_answer_and_the_sources_it_cited(self, indexed):
        result = ask(indexed)
        assert result.exit_code == 0, result.output
        assert "cartTotal adds up the prices [1]." in result.output
        assert "Sources cited" in result.output
        assert "[1] web/cart.js:" in result.output
        assert "scripted scripted-1" in result.output
        assert "Traceback" not in result.output

    def test_retrieved_sources_the_answer_did_not_cite_are_listed_separately(self, indexed):
        result = ask(indexed, "start the server and validate the email of a new user")
        assert result.exit_code == 0, result.output
        assert result.output.index("Sources cited") < result.output.index("Also retrieved (not cited)")

    def test_no_sources_hides_the_list(self, indexed):
        result = ask(indexed, QUESTION, "--no-sources")
        assert result.exit_code == 0
        assert "cartTotal adds up" in result.output
        assert "Sources cited" not in result.output and "Also retrieved" not in result.output

    def test_provider_and_model_options_reach_the_factory(self, indexed, llm):
        assert ask(indexed, QUESTION, "--provider", "ollama", "--model", "llama3").exit_code == 0
        assert llm.requests[-1] == {"provider": "ollama", "model": "llama3"}
        assert ask(indexed).exit_code == 0
        assert llm.requests[-1] == {"provider": None, "model": None}

    def test_an_unknown_provider_is_a_usage_error(self, indexed):
        result = ask(indexed, QUESTION, "--provider", "skynet")
        assert result.exit_code == 2

    def test_the_retrieval_mode_option_is_used(self, indexed, llm):
        result = ask(indexed, "cart total", "--mode", "keyword")
        assert result.exit_code == 0, result.output
        assert "cartTotal adds up" in result.output

    def test_the_question_reaches_the_model_inside_a_sources_block(self, indexed, llm):
        ask(indexed)
        (call,) = llm.provider.calls
        assert "<sources>" in call["prompt"] and call["prompt"].endswith(f"Question: {QUESTION}")

    def test_warnings_are_shown(self, indexed, llm):
        llm.provider = ScriptedProvider("It works [1] and is tested [42].")
        result = ask(indexed)
        assert result.exit_code == 0
        assert "Warning:" in result.output and "[42]" in result.output

    def test_an_uncited_answer_is_flagged(self, indexed, llm):
        llm.provider = ScriptedProvider("Trust me.")
        result = ask(indexed)
        assert "cites no retrieved code" in result.output

    def test_a_truncated_answer_says_so(self, indexed, llm):
        llm.provider = ScriptedProvider("It begins [1] and", finish="length")
        assert "ANSWER_MAX_TOKENS" in ask(indexed).output

    def test_nothing_relevant_means_no_model_call_and_no_source_list(self, indexed, llm):
        result = ask(indexed, "zzzqqq", "--mode", "keyword")
        assert result.exit_code == 0
        assert NO_CONTEXT_ANSWER in result.output.replace("\n", " ")
        assert llm.provider.calls == []
        assert "Sources cited" not in result.output

    def test_code_containing_brackets_is_not_swallowed_as_markup(self, indexed, llm):
        llm.provider = ScriptedProvider("Use `items[bold]` and [red]colors[/red] literally [1].")
        result = ask(indexed)
        assert "[red]colors[/red]" in result.output and "items[bold]" in result.output


class TestPrivacyNotice:
    def test_shown_when_code_leaves_the_machine(self, indexed, llm):
        llm.provider = ScriptedProvider("ok [1].", name="anthropic", model="claude-x", sends_code_off_machine=True)
        result = ask(indexed)
        assert "sent to anthropic (claude-x)" in " ".join(result.output.split())  # the terminal wraps long lines
        assert result.output.index("sent to anthropic") < result.output.index("ok [1].")

    def test_not_shown_for_a_local_model(self, indexed):
        assert "sent to" not in ask(indexed).output

    def test_not_shown_when_nothing_is_sent(self, indexed, llm):
        llm.provider = ScriptedProvider("unused", sends_code_off_machine=True)
        assert "sent to" not in ask(indexed, "zzzqqq", "--mode", "keyword").output


class TestFailures:
    def test_a_missing_index_says_how_to_build_one(self, llm, sample_repo):
        result = ask(sample_repo)
        assert result.exit_code == 1
        assert "codebase-ai index" in result.output

    def test_an_empty_question_is_rejected(self, indexed):
        result = ask(indexed, "   ")
        assert result.exit_code == 2 and "empty" in result.output

    def test_a_provider_that_cannot_be_created(self, indexed, monkeypatch):
        def refuse(settings, *, provider=None, model=None):
            raise ProviderError("Unknown LLM provider 'x'.", "misconfigured")

        monkeypatch.setattr("codebase_ai.llm.base.create_provider", refuse)
        result = ask(indexed)
        assert result.exit_code == 1 and "Unknown LLM provider" in result.output
        assert "Traceback" not in result.output

    def test_an_error_before_any_text_is_reported_plainly(self, indexed, llm):
        llm.provider = ScriptedProvider(error=ProviderError("The API key was rejected.", "auth", provider="scripted"))
        result = ask(indexed)
        assert result.exit_code == 1
        assert "The API key was rejected." in result.output
        assert "incomplete" not in result.output and "Traceback" not in result.output

    def test_an_error_mid_answer_keeps_the_partial_text_and_says_it_is_incomplete(self, indexed, llm):
        error = ProviderError("The connection dropped.", "connection", provider="scripted", retryable=True)
        llm.provider = ScriptedProvider(
            "cartTotal adds up the prices [1].", chunk_size=4, fail_after_chars=12, error=error
        )
        result = ask(indexed)
        assert result.exit_code == 1
        assert "cartTotal ad" in result.output
        assert "The connection dropped." in result.output and "incomplete" in result.output
        assert "Sources cited" not in result.output

    def test_the_real_anthropic_provider_without_a_key_fails_cleanly(self, indexed, monkeypatch):
        monkeypatch.setattr("codebase_ai.llm.base.create_provider", REAL_CREATE_PROVIDER)
        # Should the SDK find stored credentials, the request still goes nowhere: a closed local port.
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
        result = ask(indexed, QUESTION, "--provider", "anthropic")
        assert result.exit_code == 1, result.output
        assert "Traceback" not in result.output and "sk-" not in result.output
        assert "Sources cited" not in result.output

    def test_a_terminal_that_cannot_show_a_character_does_not_crash_the_answer(self, indexed, llm):
        llm.provider = ScriptedProvider("cartTotal ✓ adds up the prices \U0001f9ea [1].")
        result = CliRunner(charset="cp1252").invoke(app, ["ask", str(indexed), QUESTION])
        assert result.exit_code == 0, result.output
        assert "UnicodeEncodeError" not in result.output and "adds up the prices" in result.output


class TestServe:
    @pytest.fixture
    def launched(self, monkeypatch):
        calls: list[dict] = []

        def fake_call(command, env=None):
            calls.append({"command": command, "env": env})
            return calls[-1].get("code", 0)

        monkeypatch.setattr("subprocess.call", fake_call)
        return calls

    def test_launches_streamlit_on_localhost_with_the_ui_script(self, launched):
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0, result.output
        (call,) = launched
        command = call["command"]
        assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
        assert Path(command[4]).name == "streamlit_app.py" and Path(command[4]).exists()
        assert command[command.index("--server.address") + 1] == "localhost"
        assert command[command.index("--server.port") + 1] == "8501"
        assert command[command.index("--browser.gatherUsageStats") + 1] == "false"
        assert command[command.index("--server.fileWatcherType") + 1] == "none"
        assert "CODEBASE_AI_REPO" not in call["env"]

    def test_the_repo_is_handed_to_the_ui_through_the_environment(self, launched, sample_repo):
        assert runner.invoke(app, ["serve", str(sample_repo), "--port", "9000"]).exit_code == 0
        (call,) = launched
        assert call["env"]["CODEBASE_AI_REPO"] == str(sample_repo.resolve())
        assert call["command"][call["command"].index("--server.port") + 1] == "9000"

    def test_the_exit_code_of_the_ui_process_is_passed_on(self, monkeypatch):
        monkeypatch.setattr("subprocess.call", lambda command, env=None: 3)
        assert runner.invoke(app, ["serve"]).exit_code == 3

    def test_ctrl_c_is_a_clean_exit(self, monkeypatch):
        def interrupted(command, env=None):
            raise KeyboardInterrupt

        monkeypatch.setattr("subprocess.call", interrupted)
        assert runner.invoke(app, ["serve"]).exit_code == 0

    def test_a_missing_repo_is_rejected_by_the_parser(self, launched, tmp_path):
        assert runner.invoke(app, ["serve", str(tmp_path / "nope")]).exit_code == 2
        assert launched == []

    def test_a_bad_configuration_stops_before_launching(self, launched, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "skynet")
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 2 and "Invalid configuration" in result.output
        assert launched == []
