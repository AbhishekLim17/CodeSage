from __future__ import annotations

import pytest

from codebase_ai.config import Settings
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.llm.base import ProviderError
from codebase_ai.rag.answerer import (
    CODE_TOKEN_SAFETY,
    PROMPT_OVERHEAD_TOKENS,
    Answer,
    Answerer,
    context_budget_for,
    create_answerer,
)
from codebase_ai.rag.prompts import NO_CONTEXT_ANSWER, REFUSAL_ANSWER, SYSTEM_PROMPT
from codebase_ai.retrieval.query import CONDENSE_SYSTEM, Turn
from codebase_ai.retrieval.retriever import Retriever
from helpers import ScriptedProvider

QUESTION = "shopping cart total price"


@pytest.fixture
def built(sample_repo, tmp_path, fake_embedder):
    idx = RepoIndex(sample_repo, tmp_path / "idx")
    Indexer(idx, fake_embedder).run()
    yield idx
    idx.close()


def answerer_for(built, embedder, provider, **kwargs) -> Answerer:
    return Answerer(Retriever(built, embedder), provider, **kwargs)


class TestContextBudget:
    def test_a_provider_with_no_window_limit_gets_the_requested_budget(self):
        assert context_budget_for(ScriptedProvider(), 12_000, 16_000) == 12_000

    def test_a_small_window_shrinks_the_budget_so_prompt_and_answer_fit(self):
        provider = ScriptedProvider(context_window=16_384, max_output_tokens=4096)
        expected = int((16_384 - 4096 - PROMPT_OVERHEAD_TOKENS) / CODE_TOKEN_SAFETY)
        assert context_budget_for(provider, 12_000, 16_000) == expected
        assert expected < 12_000

    def test_the_answer_reserve_is_the_smaller_of_max_tokens_and_the_provider_cap(self):
        capped = ScriptedProvider(context_window=16_384, max_output_tokens=4096)
        uncapped = ScriptedProvider(context_window=16_384)
        assert context_budget_for(capped, 12_000, 1000) > context_budget_for(capped, 12_000, 4096)
        # Asking for more output than the cap allows reserves only the cap, the same as asking for exactly the cap.
        assert context_budget_for(capped, 12_000, 16_000) == context_budget_for(capped, 12_000, 4096)
        assert context_budget_for(uncapped, 12_000, 16_000) < context_budget_for(capped, 12_000, 16_000)

    def test_a_big_window_never_raises_the_budget_above_what_was_asked(self):
        assert context_budget_for(ScriptedProvider(context_window=1_000_000), 12_000, 16_000) == 12_000

    def test_a_tiny_window_still_leaves_room_for_some_context(self):
        assert context_budget_for(ScriptedProvider(context_window=2048, max_output_tokens=2048), 12_000, 16_000) == 1000


class TestStreaming:
    def test_sources_are_known_before_the_model_is_called(self, built, fake_embedder):
        provider = ScriptedProvider("cartTotal adds up the prices [1].")
        stream = answerer_for(built, fake_embedder, provider).stream(QUESTION)
        assert provider.calls == []
        assert stream.sources and stream.sources[0].path == "web/cart.js"

    def test_text_arrives_in_pieces_and_joins_to_the_reply(self, built, fake_embedder):
        provider = ScriptedProvider("cartTotal adds up the prices [1].", chunk_size=5)
        stream = answerer_for(built, fake_embedder, provider).stream(QUESTION)
        pieces = list(stream)
        assert len(pieces) > 3
        assert "".join(pieces) == "cartTotal adds up the prices [1]."
        assert stream.text_so_far == "cartTotal adds up the prices [1]."

    def test_the_answer_is_not_available_until_the_stream_is_finished(self, built, fake_embedder):
        stream = answerer_for(built, fake_embedder, ScriptedProvider("x [1].")).stream(QUESTION)
        with pytest.raises(RuntimeError, match="not finished"):
            _ = stream.answer
        iterator = iter(stream)
        next(iterator)
        with pytest.raises(RuntimeError, match="not finished"):
            _ = stream.answer

    def test_iterating_twice_does_not_call_the_model_again(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        stream = answerer_for(built, fake_embedder, provider).stream(QUESTION)
        first = list(stream)
        assert first and list(stream) == []
        assert len(provider.calls) == 1
        assert stream.answer.raw_text == "x [1]."

    def test_the_prompt_carries_the_numbered_sources_and_the_question(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        answerer_for(built, fake_embedder, provider).ask(QUESTION)
        (call,) = provider.calls
        assert call["system"] == SYSTEM_PROMPT
        assert "<sources>" in call["prompt"] and "[1] web/cart.js:" in call["prompt"]
        assert call["prompt"].endswith(f"Question: {QUESTION}")
        assert len(call["messages"]) == 1 and call["messages"][0].role == "user"

    def test_generation_settings_are_passed_through(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        answerer_for(built, fake_embedder, provider, max_tokens=1234, temperature=0.3, system="be brief").ask(QUESTION)
        (call,) = provider.calls
        assert (call["max_tokens"], call["temperature"], call["system"]) == (1234, 0.3, "be brief")

    def test_temperature_is_left_unset_by_default(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        answerer_for(built, fake_embedder, provider).ask(QUESTION)
        assert provider.calls[0]["temperature"] is None
        assert provider.calls[0]["max_tokens"] == 16_000


class TestFinishedAnswer:
    def test_a_grounded_answer(self, built, fake_embedder):
        provider = ScriptedProvider("cartTotal adds up the prices [1].", model="m-1", name="scripted")
        answer = answerer_for(built, fake_embedder, provider).ask(QUESTION)
        assert isinstance(answer, Answer)
        assert answer.question == QUESTION
        assert answer.text == answer.raw_text == "cartTotal adds up the prices [1]."
        assert answer.grounded and answer.warnings == []
        assert (answer.provider, answer.model, answer.finish) == ("scripted", "m-1", "stop")
        assert answer.usage is not None and answer.usage.output_tokens is not None
        assert answer.retrieval_seconds >= 0 and answer.generation_seconds >= 0
        assert not answer.no_context and not answer.refused and not answer.truncated

    def test_cited_and_uncited_sources_are_matched_to_their_numbers(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("Only this [1].")).ask(QUESTION)
        assert [n for n, _ in answer.cited_sources] == [1]
        assert answer.cited_sources[0][1] is answer.sources[0]
        assert [n for n, _ in answer.uncited_sources] == list(range(2, len(answer.sources) + 1))
        assert len(answer.cited_sources) + len(answer.uncited_sources) == len(answer.sources)

    def test_an_uncited_answer_is_flagged_as_unverified(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("It just works.")).ask(QUESTION)
        assert not answer.grounded
        assert any("cites no retrieved code" in w for w in answer.warnings)

    def test_citations_to_missing_sources_are_removed_and_warned_about(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("It sums [1] and taxes [99].")).ask(QUESTION)
        assert answer.raw_text == "It sums [1] and taxes [99]."
        assert answer.text == "It sums [1] and taxes."
        assert answer.citations.invalid == (99,)
        assert any("[99]" in w and "do not exist" in w for w in answer.warnings)

    def test_a_location_the_model_never_saw_is_warned_about(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("See web/checkout.js:10-20 [1].")).ask(QUESTION)
        assert any("web/checkout.js:10-20" in w for w in answer.warnings)

    def test_a_location_copied_from_a_source_header_is_not_warned_about(self, built, fake_embedder):
        def reply(prompt: str) -> str:
            header = next(line for line in prompt.splitlines() if line.startswith("[1] "))
            location = header.split()[1]
            return f"It is at {location} [1]."

        answer = answerer_for(built, fake_embedder, ScriptedProvider(reply)).ask(QUESTION)
        assert answer.citations.unverified_locations == ()
        assert answer.warnings == []

    def test_an_answer_cut_off_at_the_limit_is_flagged(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("It starts [1] and then", finish="length")).ask(QUESTION)
        assert answer.truncated
        assert any("ANSWER_MAX_TOKENS" in w for w in answer.warnings)

    def test_a_refusal_with_no_text_gets_a_plain_message(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("", finish="refusal")).ask(QUESTION)
        assert answer.refused
        assert answer.text == REFUSAL_ANSWER
        assert any("declined" in w for w in answer.warnings)
        assert not any("cites no retrieved code" in w for w in answer.warnings)

    def test_a_refusal_after_some_text_keeps_what_was_written(self, built, fake_embedder):
        answer = answerer_for(built, fake_embedder, ScriptedProvider("Partly [1]", finish="refusal")).ask(QUESTION)
        assert answer.refused and answer.text == "Partly [1]"

    def test_the_model_that_answered_is_reported_even_if_it_differs_from_the_requested_one(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].", model="requested", reported_model="fallback-model")
        assert answerer_for(built, fake_embedder, provider).ask(QUESTION).model == "fallback-model"

    def test_the_requested_model_is_used_when_the_provider_reports_none(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].", model="requested", reported_model=None)
        assert answerer_for(built, fake_embedder, provider).ask(QUESTION).model == "requested"


class TestNoContext:
    def test_nothing_retrieved_means_the_model_is_never_called(self, built):
        provider = ScriptedProvider("should never be used")
        answerer = Answerer(Retriever(built, None, mode="keyword"), provider)
        answer = answerer.ask("zzzqqq")
        assert provider.calls == []
        assert answer.no_context and answer.sources == ()
        assert answer.text == NO_CONTEXT_ANSWER
        assert answer.warnings == ["No relevant code was found in the index, so no answer was generated."]
        assert not answer.grounded

    def test_the_stream_still_yields_the_message(self, built):
        provider = ScriptedProvider("unused")
        stream = Answerer(Retriever(built, None, mode="keyword"), provider).stream("zzzqqq")
        assert stream.sources == []
        assert "".join(stream) == NO_CONTEXT_ANSWER
        assert stream.answer.finish == "stop"


class TestProviderFailures:
    def test_an_error_before_any_text_propagates(self, built, fake_embedder):
        error = ProviderError("Key rejected.", "auth", provider="scripted")
        stream = answerer_for(built, fake_embedder, ScriptedProvider(error=error)).stream(QUESTION)
        with pytest.raises(ProviderError, match="Key rejected"):
            list(stream)
        assert stream.text_so_far == ""
        with pytest.raises(RuntimeError, match="not finished"):
            _ = stream.answer

    def test_an_error_mid_stream_keeps_what_had_arrived(self, built, fake_embedder):
        error = ProviderError("Connection dropped.", "connection", provider="scripted", retryable=True)
        provider = ScriptedProvider("cartTotal adds up the prices [1].", chunk_size=4, fail_after_chars=12, error=error)
        stream = answerer_for(built, fake_embedder, provider).stream(QUESTION)
        iterator = iter(stream)
        first = next(iterator)
        with pytest.raises(ProviderError) as caught:
            list(iterator)
        assert caught.value.retryable
        assert stream.text_so_far.startswith(first)
        assert stream.text_so_far == "cartTotal ad"
        with pytest.raises(RuntimeError):
            _ = stream.answer

    def test_retrieval_errors_surface_when_the_stream_is_created_and_the_model_is_not_called(self):
        from codebase_ai.retrieval.retriever import RetrievalError

        class BrokenRetriever:
            def retrieve(self, question):
                raise RetrievalError("The index was built with another embedding model.")

        provider = ScriptedProvider()
        answerer = Answerer(BrokenRetriever(), provider)  # type: ignore[arg-type]
        with pytest.raises(RetrievalError, match="another embedding model"):
            answerer.stream(QUESTION)
        assert provider.calls == []


class TestCreateAnswerer:
    def test_settings_shape_the_retriever_and_generation(self, built, fake_embedder, monkeypatch):
        monkeypatch.setenv("RETRIEVE_TOP_K", "7")
        monkeypatch.setenv("TEST_PENALTY", "0.25")
        monkeypatch.setenv("KEYWORD_WEIGHT", "2")
        monkeypatch.setenv("ANSWER_MAX_TOKENS", "999")
        monkeypatch.setenv("ANSWER_TEMPERATURE", "0.4")
        settings = Settings(_env_file=None)
        answerer = create_answerer(settings, built, fake_embedder, ScriptedProvider())
        retriever = answerer.retriever.base  # wrapped so overview questions can get the repository map
        assert (retriever.top_k, retriever.test_penalty, retriever.keyword_weight) == (7, 0.25, 2.0)
        assert retriever.mode == "vector"
        assert (answerer.max_tokens, answerer.temperature) == (999, 0.4)

    def test_the_mode_can_be_overridden(self, built, fake_embedder):
        answerer = create_answerer(Settings(_env_file=None), built, fake_embedder, ScriptedProvider(), mode="hybrid")
        assert answerer.retriever.base.mode == "hybrid"

    def test_the_budget_follows_the_providers_context_window(self, built, fake_embedder):
        settings = Settings(_env_file=None)
        roomy = create_answerer(settings, built, fake_embedder, ScriptedProvider())
        cramped = create_answerer(
            settings, built, fake_embedder, ScriptedProvider(context_window=16_384, max_output_tokens=4096)
        )
        assert roomy.retriever.base.budget_tokens == settings.context_token_budget
        assert cramped.retriever.base.budget_tokens < roomy.retriever.base.budget_tokens

    def test_the_provider_is_used_as_given(self, built, fake_embedder):
        provider = ScriptedProvider("Yes [1].")
        answer = create_answerer(Settings(_env_file=None), built, fake_embedder, provider).ask(QUESTION)
        assert answer.text == "Yes [1]." and len(provider.calls) == 1


class TestFollowUps:
    HISTORY = (Turn("How is the cart total computed?", "cartTotal adds up the prices [1]."),)

    @staticmethod
    def two_step(standalone: str, answer: str = "It applies the tax rate [1]."):
        """A model that rewrites when asked to (the prompt has a conversation block) and answers otherwise."""

        def reply(prompt: str) -> str:
            return standalone if "<conversation>" in prompt else answer

        return ScriptedProvider(reply)

    def test_a_follow_up_is_rewritten_then_that_question_is_searched_and_answered(self, built, fake_embedder):
        provider = self.two_step("How is tax applied when computing the shopping cart total?")
        answer = answerer_for(built, fake_embedder, provider).ask("and the tax?", self.HISTORY)
        condense_call, answer_call = provider.calls
        assert condense_call["system"] == CONDENSE_SYSTEM
        assert answer_call["prompt"].endswith("Question: How is tax applied when computing the shopping cart total?")
        assert answer.question == "and the tax?"
        assert answer.searched_for == "How is tax applied when computing the shopping cart total?"
        assert answer.retrieval.query == answer.searched_for
        assert answer.condensed is not None and answer.condensed.rewritten

    def test_the_rewrite_happens_before_anything_is_retrieved_or_streamed(self, built, fake_embedder):
        provider = self.two_step("Standalone question about cart tax?")
        stream = answerer_for(built, fake_embedder, provider).stream("and the tax?", self.HISTORY)
        assert len(provider.calls) == 1  # the rewrite only; the answer is generated as the stream is read
        assert stream.searched_for == "Standalone question about cart tax?"
        list(stream)
        assert len(provider.calls) == 2

    def test_a_first_question_needs_no_rewrite(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        answer = answerer_for(built, fake_embedder, provider).ask(QUESTION)
        assert len(provider.calls) == 1 and answer.condensed is None
        assert answer.searched_for == QUESTION

    def test_rewriting_can_be_turned_off(self, built, fake_embedder):
        provider = ScriptedProvider("x [1].")
        answer = answerer_for(built, fake_embedder, provider, condense=False).ask("and the tax?", self.HISTORY)
        assert len(provider.calls) == 1 and answer.searched_for == "and the tax?"

    def test_only_the_configured_number_of_turns_reach_the_rewrite(self, built, fake_embedder):
        history = tuple(Turn(f"question number {n}", "an answer") for n in range(5))
        provider = self.two_step("Standalone?")
        answerer_for(built, fake_embedder, provider, history_turns=2).ask("next?", history)
        prompt = provider.calls[0]["prompt"]
        assert "question number 4" in prompt and "question number 3" in prompt and "question number 2" not in prompt

    def test_a_failed_rewrite_searches_the_original_and_says_so(self, built, fake_embedder):
        def reply(prompt: str) -> str:
            return "   " if "<conversation>" in prompt else "It works [1]."

        answer = answerer_for(built, fake_embedder, ScriptedProvider(reply)).ask("cart total", self.HISTORY)
        assert answer.searched_for == "cart total"
        assert any("could not be turned into a standalone question" in w for w in answer.warnings)

    def test_a_provider_error_while_rewriting_stops_before_retrieval(self, built, fake_embedder):
        provider = ScriptedProvider(error=ProviderError("Key rejected.", "auth", provider="scripted"))
        with pytest.raises(ProviderError, match="Key rejected"):
            answerer_for(built, fake_embedder, provider).stream("and the tax?", self.HISTORY)


class TestOverviewQuestions:
    def test_an_overview_question_is_answered_from_the_map_and_the_readme(self, built, fake_embedder):
        provider = ScriptedProvider("It is a small sample project [1], described in its README [2].")
        answerer = create_answerer(Settings(_env_file=None), built, fake_embedder, provider)
        answer = answerer.ask("What does this project do?")
        assert answer.retrieval.overview
        assert answer.sources[0].role == "map" and answer.sources[1].path == "README.md"
        (call,) = provider.calls
        assert "[1] (repository map):" in call["prompt"] and "[2] README.md:" in call["prompt"]
        assert answer.grounded and answer.citations.cited == (1, 2) and answer.warnings == []

    def test_the_map_can_be_turned_off(self, built, fake_embedder, monkeypatch):
        monkeypatch.setenv("USE_REPO_MAP", "false")
        answerer = create_answerer(Settings(_env_file=None), built, fake_embedder, ScriptedProvider("x [1]."))
        assert isinstance(answerer.retriever, Retriever)
        assert not answerer.ask("What does this project do?").retrieval.overview

    def test_a_narrow_question_never_gets_the_map(self, built, fake_embedder):
        answerer = create_answerer(Settings(_env_file=None), built, fake_embedder, ScriptedProvider("x [1]."))
        answer = answerer.ask(QUESTION)
        assert not answer.retrieval.overview and all(s.role != "map" for s in answer.sources)

    def test_the_settings_reach_the_answerer(self, built, fake_embedder, monkeypatch):
        monkeypatch.setenv("CONDENSE_FOLLOWUPS", "false")
        monkeypatch.setenv("HISTORY_TURNS", "5")
        monkeypatch.setenv("REPO_MAP_TOKENS", "321")
        answerer = create_answerer(Settings(_env_file=None), built, fake_embedder, ScriptedProvider())
        assert (answerer.condense, answerer.history_turns, answerer.retriever.map_tokens) == (False, 5, 321)
