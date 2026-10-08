from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from codebase_ai.answer_eval import (
    JUDGE_SYSTEM,
    PROMPTS,
    WORKED_EXAMPLE,
    AnswerReport,
    AnswerResult,
    Judgement,
    agreement,
    apply_prompt,
    build_judge_message,
    export_sample,
    freeze_retrieval,
    judge_faithfulness,
    parse_judgement,
    run_answer_eval,
    sample_for_review,
    score_answer,
)
from codebase_ai.evaluation import EvalQuestion
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.llm.base import ProviderError
from codebase_ai.rag.answerer import Answerer
from codebase_ai.rag.prompts import CITATION_REMINDER, SYSTEM_PROMPT
from codebase_ai.retrieval.overview import OverviewRetriever
from codebase_ai.retrieval.retriever import RetrievalError, Retriever
from helpers import ScriptedProvider

QUESTION = EvalQuestion(
    id="q1",
    question="shopping cart total price",
    type="explain",
    gold_files=("web/cart.js",),
    gold_symbols=("cartTotal",),
)


@pytest.fixture
def built(sample_repo, tmp_path, fake_embedder):
    index = RepoIndex(sample_repo, tmp_path / "idx")
    Indexer(index, fake_embedder).run()
    yield index
    index.close()


def answerer(built, fake_embedder, reply="cartTotal adds up the prices [1].", **kwargs) -> Answerer:
    return Answerer(Retriever(built, fake_embedder), ScriptedProvider(reply, **kwargs))


def judge_reply(verdict="supported", claims=(), reason="Backed by the code."):
    return json.dumps({"verdict": verdict, "unsupported_claims": list(claims), "reason": reason})


class TestParseJudgement:
    def test_a_clean_reply(self):
        judgement = parse_judgement(judge_reply("partially_supported", ["it retries three times"], "Only partly."))
        assert judgement.verdict == "partially_supported"
        assert judgement.unsupported_claims == ("it retries three times",)
        assert judgement.reason == "Only partly." and judgement.score == 0.5 and judgement.error is None

    def test_json_wrapped_in_prose_or_a_code_fence_is_still_read(self):
        text = "Sure! Here is my verdict:\n```json\n" + judge_reply("supported") + "\n```\nHope that helps."
        assert parse_judgement(text).verdict == "supported"

    @pytest.mark.parametrize(
        ("text", "error"),
        [
            ("I think it is fine.", "did not reply with JSON"),
            ("{not json", "could not be read"),
            (json.dumps({"verdict": "great"}), "unknown verdict"),
            (json.dumps(["supported"]), "did not reply with JSON"),
            (json.dumps({"reason": "no verdict"}), "unknown verdict"),
        ],
    )
    def test_unusable_replies_become_an_error_not_an_exception(self, text, error):
        judgement = parse_judgement(text)
        assert judgement.verdict is None and judgement.score is None and error in judgement.error

    def test_odd_claim_lists_are_tolerated(self):
        assert parse_judgement(json.dumps({"verdict": "supported", "unsupported_claims": "none"})).unsupported_claims == ()


class TestJudging:
    def answered(self, built, fake_embedder, reply="cartTotal adds up the prices [1]."):
        return answerer(built, fake_embedder, reply).ask(QUESTION.question)

    def test_the_judge_sees_only_the_cited_code_under_its_original_numbers(self, built, fake_embedder):
        answer = self.answered(built, fake_embedder, "It uses the tax rate [2] and the total [1].")
        assert answer.citations.cited == (2, 1)
        judge = ScriptedProvider(judge_reply())
        judge_faithfulness(judge, answer)
        (call,) = judge.calls
        assert call["system"] == JUDGE_SYSTEM
        assert "<question>\nshopping cart total price\n</question>" in call["prompt"]
        assert call["prompt"].count("\n[") >= 2 and f"[1] {answer.sources[0].location}" in call["prompt"]
        assert f"[2] {answer.sources[1].location}" in call["prompt"]
        assert answer.sources[2].location not in call["prompt"]  # supplied but not cited: not part of the judgement
        assert call["prompt"].rstrip().endswith("</answer>")

    def test_a_verdict_comes_back(self, built, fake_embedder):
        answer = self.answered(built, fake_embedder)
        judgement = judge_faithfulness(ScriptedProvider(judge_reply("unsupported", ["all of it"])), answer)
        assert judgement.verdict == "unsupported" and judgement.unsupported_claims == ("all of it",)

    def test_an_answer_with_no_citations_is_not_sent_to_the_judge(self, built, fake_embedder):
        answer = self.answered(built, fake_embedder, "It just works.")
        judge = ScriptedProvider(judge_reply())
        judgement = judge_faithfulness(judge, answer)
        assert judge.calls == [] and judgement.verdict is None and "no sources" in judgement.reason

    def test_a_cut_off_reply_is_reported(self, built, fake_embedder):
        answer = self.answered(built, fake_embedder)
        judgement = judge_faithfulness(ScriptedProvider('{"verdict": "sup', finish="length"), answer)
        assert judgement.verdict is None and "cut off" in judgement.error

    def test_hostile_code_in_a_source_cannot_break_out_of_the_excerpt_block(self, built, fake_embedder):
        answer = self.answered(built, fake_embedder)
        message = build_judge_message("q?", answer.text, [(1, answer.sources[0])])
        assert message.count("</cited_code>") == 1 and message.index("</cited_code>") < message.index("<answer>")


class TestScoring:
    def test_a_grounded_answer_that_cites_the_gold_file(self, built, fake_embedder):
        answer = answerer(built, fake_embedder).ask(QUESTION.question)
        result = score_answer(QUESTION, answer)
        assert result.grounded and result.gold_cited and result.validity == 1.0 and result.precision == 1.0
        assert result.uncited_share == pytest.approx(1 - 1 / len(answer.sources))
        assert result.cited == (1,) and not result.refused and not result.truncated
        assert result.tokens_in is not None and result.seconds >= 0

    def test_citing_the_wrong_file_lowers_precision_and_misses_the_gold_file(self, built, fake_embedder):
        answer = answerer(built, fake_embedder, "It is in the server [1].").ask("start the server")
        result = score_answer(QUESTION, answer)  # scored against the cart question's labels
        assert result.grounded and not result.gold_cited and result.precision == 0.0

    def test_a_gold_symbol_in_the_cited_code_counts_even_from_another_file(self, built, fake_embedder):
        question = EvalQuestion("q", "start the server", "explain", ("nowhere.go",), gold_symbols=("NewServer",))
        probe = answerer(built, fake_embedder).ask("start the server")
        number = next(n for n, s in enumerate(probe.sources, start=1) if "NewServer" in s.text)
        answer = answerer(built, fake_embedder, f"It is built by NewServer [{number}].").ask("start the server")
        assert score_answer(question, answer).precision == 1.0
        other = next(n for n, s in enumerate(probe.sources, start=1) if "NewServer" not in s.text)
        wrong = answerer(built, fake_embedder, f"Not here [{other}].").ask("start the server")
        assert score_answer(question, wrong).precision == 0.0

    def test_an_invented_citation_lowers_validity(self, built, fake_embedder):
        answer = answerer(built, fake_embedder, "Sum [1] and tax [99].").ask(QUESTION.question)
        result = score_answer(QUESTION, answer)
        assert result.validity == 0.5 and result.cited == (1,)

    def test_an_uncited_answer_has_no_precision_or_validity(self, built, fake_embedder):
        result = score_answer(QUESTION, answerer(built, fake_embedder, "Trust me.").ask(QUESTION.question))
        assert not result.grounded and result.precision is None and result.validity is None and not result.gold_cited

    def test_an_unanswerable_question_has_no_precision_and_no_gold_file_to_cite(self, built, fake_embedder):
        unanswerable = EvalQuestion("u1", "How are failed payments retried?", "unanswerable", ())
        result = score_answer(unanswerable, answerer(built, fake_embedder).ask(unanswerable.question))
        assert result.grounded and result.precision is None and result.gold_cited is None  # undefined, not 0
        cart = score_answer(QUESTION, answerer(built, fake_embedder).ask(QUESTION.question))
        assert AnswerReport("r", [result, cart]).summary["gold_cited_rate"] == 1.0  # the unanswerable one is left out

    def test_a_location_the_model_never_saw_is_counted(self, built, fake_embedder):
        answer = answerer(built, fake_embedder, "See web/checkout.js:1-9 [1].").ask(QUESTION.question)
        assert score_answer(QUESTION, answer).unverified_locations == 1

    def test_refusals_and_truncation_are_flagged(self, built, fake_embedder):
        refused = score_answer(QUESTION, answerer(built, fake_embedder, "", finish="refusal").ask(QUESTION.question))
        cut = score_answer(QUESTION, answerer(built, fake_embedder, "It begins [1] and", finish="length").ask(QUESTION.question))
        assert refused.refused and not refused.grounded and cut.truncated and cut.grounded

    def test_directories_and_the_map_are_relevant_to_an_overview_question(self, built, fake_embedder):
        overview = EvalQuestion("o", "What does this project do?", "overview", ("README.md",), gold_dirs=("web",))
        base = Retriever(built, fake_embedder)
        provider = ScriptedProvider("It is a shop [1], see the readme [2] and the web code [3].")
        answer = Answerer(OverviewRetriever(base), provider).ask(overview.question)
        assert [s.role for s in answer.sources[:1]] == ["map"]
        result = score_answer(overview, answer)
        assert result.grounded and result.precision is not None and result.precision >= 2 / 3
        assert result.gold_cited  # the README is a gold file

    def test_the_judgement_is_carried_into_the_result(self, built, fake_embedder):
        answer = answerer(built, fake_embedder).ask(QUESTION.question)
        result = score_answer(QUESTION, answer, Judgement("supported", reason="ok"))
        assert result.judgement.verdict == "supported"


class TestRun:
    QUESTIONS = (
        QUESTION,
        EvalQuestion("q2", "start the server", "explain", ("go/server.go",)),
    )

    def test_answers_and_judges_every_question(self, built, fake_embedder):
        judge = ScriptedProvider(judge_reply("supported"))
        seen = []
        report = run_answer_eval(
            answerer(built, fake_embedder), self.QUESTIONS, judge=judge, name="demo",
            progress=lambda n, total, result: seen.append((n, total, result.id)),
        )
        assert report.name == "demo" and [r.id for r in report.results] == ["q1", "q2"]
        assert seen == [(1, 2, "q1"), (2, 2, "q2")]
        assert len(judge.calls) == 2
        summary = report.summary
        assert summary["questions"] == 2 and summary["errors"] == 0 and summary["judged"] == 2
        assert summary["grounded_rate"] == 1.0 and summary["faithfulness"] == 1.0 and summary["supported_rate"] == 1.0

    def test_without_a_judge_faithfulness_is_simply_absent(self, built, fake_embedder):
        summary = run_answer_eval(answerer(built, fake_embedder), self.QUESTIONS).summary
        assert summary["judged"] == 0 and summary["faithfulness"] is None and summary["grounded_rate"] == 1.0

    def test_a_question_with_no_context_is_not_judged(self, built):
        provider = ScriptedProvider("unused")
        quiet = Answerer(Retriever(built, None, mode="keyword"), provider)
        judge = ScriptedProvider(judge_reply())
        report = run_answer_eval(quiet, [EvalQuestion("n", "zzzqqq", "explain", ("a.py",))], judge=judge)
        assert judge.calls == [] and report.results[0].no_context and report.results[0].judgement.verdict is None

    def test_a_transient_failure_is_recorded_and_the_run_continues(self, built, fake_embedder):
        calls = {"n": 0}

        provider = ScriptedProvider("It works [1].")
        original = provider.stream

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderError("Rate limited.", "rate_limit", provider="scripted", retryable=True)
            return original(*args, **kwargs)

        provider.stream = flaky  # type: ignore[method-assign]
        report = run_answer_eval(Answerer(Retriever(built, fake_embedder), provider), self.QUESTIONS)
        first, second = report.results
        assert first.error == "Rate limited." and second.error is None and second.grounded
        assert report.summary["errors"] == 1 and report.summary["questions"] == 2
        assert report.summary["grounded_rate"] == 1.0  # errors are left out of the rates, not counted as failures

    def test_a_rejected_key_stops_the_run_at_once(self, built, fake_embedder):
        provider = ScriptedProvider(error=ProviderError("Key rejected.", "auth", provider="scripted"))
        with pytest.raises(ProviderError, match="Key rejected"):
            run_answer_eval(Answerer(Retriever(built, fake_embedder), provider), self.QUESTIONS)
        assert len(provider.calls) == 1  # it did not go on to the second question

    def test_a_transient_judge_failure_is_recorded_on_that_answer_only(self, built, fake_embedder):
        judge = ScriptedProvider(error=ProviderError("Overloaded.", "server", provider="judge", retryable=True))
        report = run_answer_eval(answerer(built, fake_embedder), self.QUESTIONS, judge=judge)
        assert all(r.error is None and r.grounded for r in report.results)
        assert all(r.judgement.verdict is None and "Overloaded." in (r.judgement.error or "") for r in report.results)

    def test_a_rejected_judge_key_stops_the_run(self, built, fake_embedder):
        judge = ScriptedProvider(error=ProviderError("Judge key rejected.", "auth", provider="judge"))
        with pytest.raises(ProviderError, match="Judge key"):
            run_answer_eval(answerer(built, fake_embedder), self.QUESTIONS, judge=judge)


class TestSummary:
    def result(self, id, **kwargs):
        return AnswerResult(id=id, type="explain", question="q", **kwargs)

    def test_rates_and_means_skip_what_was_not_measured(self):
        report = AnswerReport(
            "r",
            [
                self.result("a", grounded=True, validity=1.0, precision=1.0, gold_cited=True, uncited_share=0.5,
                            judgement=Judgement("supported")),
                self.result("b", grounded=True, validity=0.5, precision=0.0, uncited_share=0.0,
                            judgement=Judgement("partially_supported")),
                self.result("c", grounded=False, judgement=Judgement(None, reason="no citations")),
                self.result("d", error="boom"),
            ],
        )
        s = report.summary
        assert (s["questions"], s["errors"], s["judged"]) == (4, 1, 2)
        assert s["grounded_rate"] == pytest.approx(2 / 3)
        assert s["citation_validity"] == 0.75 and s["citation_precision"] == 0.5
        assert s["gold_cited_rate"] == pytest.approx(1 / 3)
        assert s["faithfulness"] == 0.75 and s["supported_rate"] == 0.5

    def test_an_empty_report_has_no_rates(self):
        s = AnswerReport("r").summary
        assert s["questions"] == 0 and s["grounded_rate"] is None and s["faithfulness"] is None


class TestHandCheck:
    RESULTS: ClassVar[list[AnswerResult]] = [
        AnswerResult("a", "explain", "qa", answer="Answer A [1].", cited=(1,), judgement=Judgement("supported")),
        AnswerResult("b", "explain", "qb", answer="Answer B [2].", cited=(2,), judgement=Judgement("unsupported")),
        AnswerResult("c", "explain", "qc", answer="Answer C.", judgement=Judgement(None)),
        AnswerResult("d", "explain", "qd", answer="Answer D [1].", cited=(1,), judgement=Judgement("partially_supported")),
    ]

    def test_the_sample_is_reproducible_ordered_and_only_judged_answers(self):
        first = sample_for_review(self.RESULTS, 2, seed=7)
        assert [r.id for r in first] == [r.id for r in sample_for_review(self.RESULTS, 2, seed=7)]
        assert len(first) == 2 and [r.id for r in first] == sorted(r.id for r in first)
        assert all(r.judgement.verdict is not None for r in sample_for_review(self.RESULTS, 10))
        assert len(sample_for_review(self.RESULTS, 10)) == 3

    def test_the_export_hides_the_judges_verdict(self):
        text = export_sample(self.RESULTS[:2])
        assert "## a" in text and "**Question:** qa" in text and "Answer A [1]." in text and "**Cited:** [1]" in text
        for verdict in ("supported", "unsupported"):
            assert f"verdict: {verdict}" not in text
        assert "Judge" not in text.split("\n\n", 2)[2]  # the body never mentions the judge

    def test_the_export_shows_the_cited_code_so_it_can_be_checked_on_its_own(self):
        # Found by doing the hand check: "[14]" means nothing without the run's context, which the reader never sees.
        code = "def f():\n    return '```'"
        result = AnswerResult("e", "explain", "qe", answer="It returns a fence [3].", cited=(3,), cited_code=((3, "a.py:1-2", code),))
        text = export_sample([result])
        assert "**[3] a.py:1-2**" in text and code in text
        assert "````\n" + code + "\n````" in text  # a fence longer than the backticks inside the code

    def test_scoring_keeps_the_cited_code(self, built, fake_embedder):
        result = score_answer(QUESTION, answerer(built, fake_embedder).ask(QUESTION.question))
        ((number, location, code),) = result.cited_code
        assert number == 1 and location.startswith("web/cart.js:") and "cartTotal" in code

    def test_agreement_counts_exact_and_one_step_matches_for_graded_questions_only(self):
        human = {"a": "supported", "b": "partially_supported", "d": "supported", "c": "supported", "zzz": "supported"}
        result = agreement(self.RESULTS, human)
        assert (result.compared, result.matching, result.within_one_step) == (3, 1, 3)
        assert result.rate == pytest.approx(1 / 3)

    def test_agreement_ignores_invalid_human_verdicts_and_empty_input(self):
        assert agreement(self.RESULTS, {"a": "yes"}).compared == 0
        assert agreement(self.RESULTS, {}).rate is None


class TestOracle:
    """The oracle context: the same retrieval, limited to each question's gold files (research plan, step 3.5)."""

    SERVER = EvalQuestion(id="q2", question="shopping cart total price", type="explain", gold_files=("go/server.go",))

    def test_only_the_gold_files_are_searched_with_the_usual_ranking_and_budget(self, built, fake_embedder):
        retriever = Retriever(built, fake_embedder)
        assert retriever.retrieve(self.SERVER.question).sources[0].path == "web/cart.js"  # what retrieval would show
        payload = freeze_retrieval(retriever, [self.SERVER], {}, oracle=True)
        stored = payload["questions"][self.SERVER.question]["sources"]
        assert stored and {s["path"] for s in stored} == {"go/server.go"}
        assert payload["meta"]["oracle"] is True

    def test_an_unanswerable_question_has_no_oracle(self, built, fake_embedder):
        unanswerable = EvalQuestion("u1", "How are failed payments retried?", "unanswerable", ())
        payload = freeze_retrieval(Retriever(built, fake_embedder), [self.SERVER, unanswerable], {}, oracle=True)
        assert list(payload["questions"]) == [self.SERVER.question]

    def test_gold_files_missing_from_the_index_are_an_error_not_an_empty_context(self, built, fake_embedder):
        typo = EvalQuestion(id="q3", question="anything", type="explain", gold_files=("go/no_such_file.go",))
        with pytest.raises(RetrievalError, match="q3"):
            freeze_retrieval(Retriever(built, fake_embedder), [typo], {}, oracle=True)

    def test_only_vector_search_can_be_limited_to_some_files(self, built):
        with pytest.raises(ValueError, match="vector"):
            Retriever(built, None, mode="keyword").retrieve("anything", only_paths=["go/server.go"])


class TestPromptConditions:
    """The study's prompts P0 to P4 (research/STEP2_PROTOCOL_DRAFT.md section 4)."""

    def test_the_prompt_texts_are_word_for_word_the_protocols(self):
        protocol = (Path(__file__).parent.parent / "research" / "STEP2_PROTOCOL_DRAFT.md").read_text(encoding="utf-8")
        for text in (SYSTEM_PROMPT, CITATION_REMINDER, WORKED_EXAMPLE):
            assert text in protocol

    @pytest.mark.parametrize("name", PROMPTS)
    def test_each_prompt_changes_only_what_it_says(self, built, fake_embedder, name):
        a = answerer(built, fake_embedder)
        shown = a.retriever.retrieve(QUESTION.question).sources
        assert len(shown) > 3  # otherwise P3 would change nothing here
        record = apply_prompt(a, name)
        answer = a.ask(QUESTION.question)
        call = a.provider.calls[0]
        assert call["system"] == record["system"] and call["system"].startswith(SYSTEM_PROMPT)
        assert (WORKED_EXAMPLE in call["system"]) == (name == "P2")
        assert call["prompt"].endswith(CITATION_REMINDER) == (name != "P0")
        expected = {"P3": shown[:3], "P4": shown[::-1]}.get(name, shown)
        assert [s.location for s in answer.sources] == [s.location for s in expected]
        # "[1]" is checked against the first source the model was shown, which for P4 is the worst-ranked one.
        assert answer.cited_sources[0][1].location == expected[0].location

    def test_an_unknown_prompt_is_refused(self, built, fake_embedder):
        with pytest.raises(ValueError, match="P0, P1"):
            apply_prompt(answerer(built, fake_embedder), "P9")
