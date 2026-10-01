"""Headless tests of the Streamlit app, driven with ``streamlit.testing.v1.AppTest`` and a scripted LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from codebase_ai.config import Settings
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.llm.base import ProviderError
from codebase_ai.rag.prompts import NO_CONTEXT_ANSWER
from codebase_ai.ui.streamlit_app import REPO_ENV, escape_math
from helpers import FakeEmbedder, ScriptedProvider

APP = Path(__file__).parent.parent / "src" / "codebase_ai" / "ui" / "streamlit_app.py"
QUESTION = "shopping cart total price"
TEXT_KINDS = ("markdown", "caption", "text", "error", "warning", "info", "success", "code")


class Llm:
    """Holds the scripted provider the app will get, and what it was asked for."""

    def __init__(self) -> None:
        self.provider = ScriptedProvider("cartTotal adds up the prices [1].")
        self.requests: list[dict] = []


@pytest.fixture
def llm(monkeypatch) -> Llm:
    holder = Llm()

    def create(settings, *, provider=None, model=None):
        holder.requests.append({"provider": provider, "model": model})
        return holder.provider

    st.cache_resource.clear()  # the embedder is cached per process; each test brings its own
    monkeypatch.setattr("codebase_ai.llm.base.create_provider", create)
    monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: FakeEmbedder())
    yield holder
    st.cache_resource.clear()


def build_index(repo: Path, embedder: FakeEmbedder | None = None) -> None:
    index = RepoIndex(repo, Settings(_env_file=None).index_dir)
    try:
        Indexer(index, embedder or FakeEmbedder()).run()
    finally:
        index.close()


@pytest.fixture
def repo(sample_repo, monkeypatch, llm) -> Path:
    monkeypatch.setenv(REPO_ENV, str(sample_repo))
    return sample_repo


@pytest.fixture
def indexed(repo) -> Path:
    build_index(repo)
    return repo


def start() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def page_text(at: AppTest) -> str:
    return "\n".join(str(element.value) for kind in TEXT_KINDS for element in at.get(kind))


def ask(at: AppTest, question: str = QUESTION) -> AppTest:
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def chat_disabled(at: AppTest) -> bool:
    return bool(at.chat_input[0].proto.disabled)


class TestStartup:
    def test_without_a_repository_it_says_where_to_start(self, monkeypatch, llm):
        monkeypatch.delenv(REPO_ENV, raising=False)
        at = start()
        assert "Enter the path or git URL of a repository" in page_text(at)
        assert chat_disabled(at)

    def test_the_repo_from_serve_is_prefilled(self, repo):
        at = start()
        assert at.sidebar.text_input(key="repo_path").value == str(repo)

    def test_an_unindexed_repo_offers_to_index_it_and_blocks_questions(self, repo):
        at = start()
        assert "not indexed yet" in page_text(at)
        assert at.sidebar.button(key="index_button").label == "Index repository"
        assert chat_disabled(at)

    def test_a_missing_folder_is_reported(self, monkeypatch, llm, tmp_path):
        monkeypatch.setenv(REPO_ENV, str(tmp_path / "nope"))
        at = start()
        assert "That folder does not exist." in page_text(at)
        assert chat_disabled(at)

    def test_an_invalid_configuration_is_shown_instead_of_a_crash(self, monkeypatch, llm):
        monkeypatch.setenv("LLM_PROVIDER", "skynet")
        at = start()
        assert "configuration is invalid" in page_text(at)

    def test_an_indexed_repo_shows_its_size_and_enables_questions(self, indexed):
        at = start()
        text = page_text(at)
        assert "files," in text and "chunks" in text and "fake:hash-64" in text
        assert at.sidebar.button(key="index_button").label == "Update index"
        assert not chat_disabled(at)


class TestIndexing:
    def test_indexing_from_the_sidebar_makes_the_repo_ready(self, repo):
        at = start()
        at.sidebar.button(key="index_button").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Indexed" in s.value for s in at.sidebar.success)
        assert at.sidebar.button(key="index_button").label == "Update index"
        assert not chat_disabled(at)

    def test_updating_an_unchanged_repo_says_nothing_was_new(self, indexed):
        at = start()
        at.sidebar.button(key="index_button").click().run()
        assert any("Indexed 0 new or changed" in s.value for s in at.sidebar.success)

    def test_a_changed_embedding_model_blocks_updates_and_explains_the_way_out(self, indexed, monkeypatch):
        monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: FakeEmbedder("fake:other"))
        at = start()
        at.sidebar.button(key="index_button").click().run()
        messages = " ".join(e.value for e in at.sidebar.error)
        assert "different settings" in messages
        assert "Rebuild from scratch" in messages and "--full" not in messages

    def test_rebuilding_from_scratch_uses_the_current_embedding_model(self, indexed, monkeypatch, llm):
        monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: FakeEmbedder("fake:other"))
        st.cache_resource.clear()
        at = start()
        at.sidebar.expander[0].button(key="rebuild_button").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Indexed" in s.value for s in at.sidebar.success)
        assert "fake:other" in page_text(at)


class TestAsking:
    def test_a_question_gets_a_streamed_answer_with_its_sources(self, indexed, llm):
        at = ask(start())
        assert [m.name for m in at.chat_message] == ["user", "assistant"]
        assert at.chat_message[0].markdown[0].value == QUESTION
        assert "cartTotal adds up the prices [1]." in at.chat_message[1].markdown[0].value
        labels = [e.label for e in at.expander if e.label.startswith(("Cited", "Also"))]
        assert labels[0] == "Cited sources (1)"
        assert labels[1].startswith("Also retrieved, not cited (")
        assert "Cited: [1] web/cart.js:" in page_text(at)
        assert not at.warning or all("sent to" not in w.value for w in at.warning)

    def test_the_source_code_itself_can_be_read(self, indexed, llm):
        at = ask(start())
        cited = next(e for e in at.expander if e.label == "Cited sources (1)")
        assert "cartTotal" in cited.code[0].value
        assert any("[1] web/cart.js:" in m.value for m in cited.markdown)

    def test_uncited_sources_are_kept_apart_from_cited_ones(self, indexed, llm):
        at = ask(start(), "start the server and validate the email of a new user")
        cited = next(e for e in at.expander if e.label.startswith("Cited sources"))
        also = next(e for e in at.expander if e.label.startswith("Also retrieved"))
        cited_numbers = {m.value.split("]")[0] for m in cited.markdown}
        also_numbers = {m.value.split("]")[0] for m in also.markdown}
        assert cited_numbers == {"`[1"} and "`[1" not in also_numbers and also_numbers

    def test_the_answering_model_never_sees_the_chat_only_the_question_and_its_sources(self, indexed, llm):
        def reply(prompt: str) -> str:
            return "How is the shopping cart total computed on the server?" if "<conversation>" in prompt else "Done [1]."

        llm.provider = ScriptedProvider(reply)
        ask(ask(start(), "cart total"), "and the server?")
        first, rewrite, second = llm.provider.calls
        assert first["prompt"].endswith("Question: cart total")
        assert "<conversation>" in rewrite["prompt"] and "cart total" in rewrite["prompt"]
        assert second["prompt"].endswith("Question: How is the shopping cart total computed on the server?")
        assert "<conversation>" not in second["prompt"] and "Done [1]." not in second["prompt"]
        assert len(second["messages"]) == 1

    def test_the_sidebar_choices_reach_the_provider(self, indexed, llm):
        at = start()
        at.sidebar.selectbox(key="provider").select("ollama")
        at = at.run()
        at.sidebar.text_input(key="model_ollama").set_value("llama3").run()
        ask(at)
        assert llm.requests[-1] == {"provider": "ollama", "model": "llama3"}

    def test_the_retrieval_mode_can_be_changed(self, indexed, llm):
        at = start()
        at.sidebar.selectbox(key="mode").select("keyword").run()
        ask(at, "cart total")
        assert "cartTotal adds up" in at.chat_message[1].markdown[0].value

    def test_history_survives_reruns_and_can_be_cleared(self, indexed, llm):
        at = ask(start())
        at.run()
        assert len(at.chat_message) == 2 and len(llm.provider.calls) == 1
        at.sidebar.button(key="clear_button").click().run()
        assert len(at.chat_message) == 0

    def test_a_different_repository_starts_a_fresh_conversation(self, indexed, llm, tmp_path):
        at = ask(start())
        other = tmp_path / "other"
        other.mkdir()
        at.sidebar.text_input(key="repo_path").set_value(str(other)).run()
        assert len(at.chat_message) == 0

    def test_dollar_signs_are_not_turned_into_maths(self, indexed, llm):
        llm.provider = ScriptedProvider("Costs $5 and $10, see `$HOME` [1].")
        at = ask(start())
        shown = at.chat_message[1].markdown[0].value
        assert "\\$5" in shown and "\\$10" in shown and "`$HOME`" in shown


class TestWarningsAndFailures:
    def test_an_answer_without_citations_is_flagged(self, indexed, llm):
        llm.provider = ScriptedProvider("It just works.")
        at = ask(start())
        assert any("cites no retrieved code" in w.value for w in at.warning)

    def test_a_citation_to_a_missing_source_is_removed_and_flagged(self, indexed, llm):
        llm.provider = ScriptedProvider("Adds [1] and taxes [99].")
        at = ask(start())
        shown = at.chat_message[1].markdown[0].value
        assert "[99]" not in shown and "[1]" in shown
        assert any("[99]" in w.value and "do not exist" in w.value for w in at.warning)

    def test_a_truncated_answer_is_flagged(self, indexed, llm):
        llm.provider = ScriptedProvider("It begins [1] and", finish="length")
        assert any("ANSWER_MAX_TOKENS" in w.value for w in ask(start()).warning)

    def test_a_refusal_is_explained(self, indexed, llm):
        llm.provider = ScriptedProvider("", finish="refusal")
        at = ask(start())
        assert any("declined" in w.value for w in at.warning)

    def test_nothing_relevant_means_no_model_call(self, indexed, llm):
        at = start()
        at.sidebar.selectbox(key="mode").select("keyword").run()
        ask(at, "zzzqqq")
        assert llm.provider.calls == []
        assert NO_CONTEXT_ANSWER in at.chat_message[1].markdown[0].value
        assert not [e for e in at.expander if e.label.startswith(("Cited", "Also"))]

    def test_a_provider_error_is_shown_in_the_chat_and_the_app_keeps_working(self, indexed, llm):
        llm.provider = ScriptedProvider(error=ProviderError("The API key was rejected.", "auth", provider="scripted"))
        at = ask(start())
        assert [e.value for e in at.chat_message[1].error] == ["The API key was rejected."]
        assert not chat_disabled(at)
        llm.provider = ScriptedProvider("Fine now [1].")
        ask(at)
        assert "Fine now [1]." in at.chat_message[3].markdown[0].value

    def test_an_error_mid_answer_keeps_what_arrived(self, indexed, llm):
        error = ProviderError("The connection dropped.", "connection", provider="scripted", retryable=True)
        llm.provider = ScriptedProvider("cartTotal adds up the prices [1].", chunk_size=4, fail_after_chars=12, error=error)
        at = ask(start())
        assistant = at.chat_message[1]
        assert assistant.markdown[0].value.startswith("cartTotal ad")
        assert "▌" not in assistant.markdown[0].value
        assert "The connection dropped." in assistant.error[0].value and "incomplete" in assistant.error[0].value

    def test_an_unexpected_failure_is_a_message_not_a_stack_trace(self, indexed, llm):
        def explode(prompt: str) -> str:
            raise RuntimeError("boom")

        llm.provider = ScriptedProvider(explode)
        at = ask(start())
        assert "RuntimeError: boom" in at.chat_message[1].error[0].value

    def test_an_index_from_another_embedding_model_blocks_semantic_search_only(self, repo, llm):
        build_index(repo, FakeEmbedder("fake:other"))
        at = start()
        assert chat_disabled(at)
        text = page_text(at)
        assert "fake:other" in text and "fake:hash-64" in text and "keyword retrieval" in text
        at.sidebar.selectbox(key="mode").select("keyword").run()
        assert not chat_disabled(at)
        ask(at, "cart total")
        assert "cartTotal adds up" in at.chat_message[1].markdown[0].value


class TestPrivacy:
    def test_a_hosted_provider_is_announced_before_anything_is_sent(self, indexed, llm):
        llm.provider = ScriptedProvider("ok [1].", name="anthropic", model="claude-x", sends_code_off_machine=True)
        at = start()
        assert any("sent to anthropic" in w.value for w in at.sidebar.warning)
        ask(at)
        assert "Sending your question and" in page_text(at) and "anthropic (claude-x)" in page_text(at)

    def test_a_local_provider_says_the_code_stays_here(self, indexed, llm):
        at = start()
        assert "stay on this computer" in page_text(at)
        assert not at.sidebar.warning
        ask(at)
        assert "Sending your question" not in page_text(at)

    def test_the_api_key_is_reported_as_set_but_never_shown(self, indexed, llm, monkeypatch):
        secret = "sk-ant-this-must-never-be-displayed"
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
        at = ask(start())
        text = page_text(at)
        assert "API key: set" in text and secret not in text
        assert secret not in "".join(str(t.value) for t in at.sidebar.text_input)

    def test_a_missing_key_says_where_to_put_it(self, indexed, llm):
        assert "API key: not set. Add ANTHROPIC_API_KEY to .env" in page_text(start())


class TestEscapeMath:
    def test_prose_dollars_are_escaped(self):
        assert escape_math("costs $5, then $10") == "costs \\$5, then \\$10"

    def test_inline_code_is_left_alone(self):
        assert escape_math("run `echo $HOME` now, $5") == "run `echo $HOME` now, \\$5"

    def test_fenced_code_is_left_alone(self):
        text = "before $1\n```bash\necho $PATH\n```\nafter $2"
        assert escape_math(text) == "before \\$1\n```bash\necho $PATH\n```\nafter \\$2"

    def test_text_without_dollars_is_unchanged(self):
        assert escape_math("nothing here [1]") == "nothing here [1]"


class TestFollowUpsAndOverviews:
    @staticmethod
    def rewriting(standalone: str) -> ScriptedProvider:
        return ScriptedProvider(lambda prompt: standalone if "<conversation>" in prompt else "It applies the tax rate [1].")

    def test_a_rewritten_follow_up_shows_what_was_searched_for(self, indexed, llm):
        llm.provider = self.rewriting("How is tax applied to the shopping cart total?")
        at = ask(ask(start(), "cart total"), "and the tax?")
        assert "Searched for: How is tax applied to the shopping cart total?" in page_text(at)

    def test_a_first_question_has_no_searched_for_line(self, indexed, llm):
        assert "Searched for:" not in page_text(ask(start()))

    def test_a_question_that_failed_leaves_nothing_to_refer_back_to(self, indexed, llm):
        llm.provider = ScriptedProvider(error=ProviderError("Key rejected.", "auth", provider="scripted"))
        at = ask(start())
        llm.provider = ScriptedProvider("Fine [1].")
        ask(at, "cart total")
        assert len(llm.provider.calls) == 1  # no rewrite: there is no earlier answer to resolve "it" against

    def test_a_failed_rewrite_is_reported_and_the_question_still_runs(self, indexed, llm):
        llm.provider = ScriptedProvider(lambda prompt: "  " if "<conversation>" in prompt else "It works [1].")
        at = ask(ask(start(), "cart total"), "and the tax?")
        assert any("could not be turned into a standalone question" in w.value for w in at.warning)
        assert "It works [1]." in at.chat_message[3].markdown[0].value

    def test_an_overview_question_says_it_got_the_map_and_shows_it_as_a_source(self, indexed, llm):
        llm.provider = ScriptedProvider("A small sample project [1] described in its README [2].")
        at = ask(start(), "What does this project do?")
        assert "generated map of the repository was added" in page_text(at)
        cited = next(e for e in at.expander if e.label == "Cited sources (2)")
        assert any("(repository map)" in m.value for m in cited.markdown)
        assert any(c.value.startswith("Repository map (generated from the index") for c in cited.code)
        assert not at.warning

    def test_a_narrow_question_is_not_described_as_an_overview(self, indexed, llm):
        assert "generated map" not in page_text(ask(start()))


class TestGitUrls:
    URL = "https://github.com/acme/shop"

    @pytest.fixture
    def remote(self, monkeypatch, llm):
        """A git URL as the repository, with the network step replaced by a copy of the sample repo."""
        import shutil

        from codebase_ai.ingest.git_source import clone_dir_for
        from helpers import FIXTURE_REPO

        class Fake:
            calls: list[str]
            error: Exception | None = None

        fake = Fake()
        fake.calls = []

        def sync(remote, root, *, branch=None, **kwargs):
            fake.calls.append(remote.display)
            if fake.error is not None:
                raise fake.error
            target = clone_dir_for(remote, root)
            if not target.exists():
                shutil.copytree(FIXTURE_REPO, target)
            return target

        monkeypatch.setattr("codebase_ai.ingest.git_source.sync_repo", sync)
        monkeypatch.setenv(REPO_ENV, self.URL)
        return fake

    def test_an_undownloaded_url_offers_to_clone_and_downloads_nothing_yet(self, remote):
        at = start()
        assert at.sidebar.button(key="index_button").label == "Clone and index"
        assert "has not been downloaded yet" in page_text(at) and chat_disabled(at)
        assert remote.calls == []

    def test_clone_and_index_makes_the_repository_ready_to_ask_about(self, remote, llm):
        at = start()
        at.sidebar.button(key="index_button").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert remote.calls == [self.URL]
        assert any("Indexed" in s.value for s in at.sidebar.success)
        assert at.sidebar.button(key="index_button").label == "Update index" and not chat_disabled(at)
        assert f"Asking about `{self.URL}`" in page_text(at)
        ask(at)
        assert "cartTotal adds up the prices [1]." in at.chat_message[1].markdown[0].value

    def test_a_failed_download_is_reported_and_nothing_else_breaks(self, remote):
        from codebase_ai.ingest.git_source import GitSourceError

        remote.error = GitSourceError("The repository was not found, or you do not have access to it.")
        at = start()
        at.sidebar.button(key="index_button").click().run()
        assert not at.exception
        assert any("not found, or you do not have access" in e.value for e in at.sidebar.error)
        assert chat_disabled(at) and at.sidebar.button(key="index_button").label == "Clone and index"

    @pytest.mark.parametrize("bad", ["http://github.com/acme/shop", "git://github.com/acme/shop", "https://ghp_secret123@github.com/acme/shop"])
    def test_unsupported_or_credentialed_urls_are_explained_and_offer_no_download(self, monkeypatch, llm, bad):
        monkeypatch.setenv(REPO_ENV, bad)
        at = start()
        assert at.sidebar.error and not at.sidebar.button and chat_disabled(at)
        assert "ghp_secret123" not in page_text(at)  # an error never repeats a token that was typed by mistake

    def test_updating_an_existing_clone_fetches_but_rebuilding_does_not(self, remote):
        at = start()
        at.sidebar.button(key="index_button").click().run()
        remote.calls.clear()
        at.sidebar.expander[0].button(key="rebuild_button").click().run()
        assert not at.exception and remote.calls == []
        at.sidebar.button(key="index_button").click().run()
        assert remote.calls == [self.URL]
