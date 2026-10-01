from __future__ import annotations

import sys
import types

import pytest

from codebase_ai.config import Settings
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.retrieval.reranker import CrossEncoderReranker, RerankError, create_reranker
from codebase_ai.retrieval.retriever import RetrievalError, Retriever


class FakeCrossEncoder:
    """Stands in for sentence-transformers' CrossEncoder: ``predict`` gets ``(query, text)`` pairs."""

    def __init__(self, scorer=None, error: Exception | None = None):
        self.scorer = scorer or (lambda query, text: float(len(text)))
        self.error = error
        self.calls: list[dict] = []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        if self.error is not None:
            raise self.error
        return [self.scorer(query, text) for query, text in pairs]


class ScriptedReranker:
    """A reranker that scores by a function of the chunk text, and records what it was asked."""

    model_id = "scripted-reranker"

    def __init__(self, scorer, *, wrong_length: bool = False, error: Exception | None = None):
        self.scorer = scorer
        self.wrong_length = wrong_length
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        if self.error is not None:
            raise self.error
        scores = [self.scorer(text) for text in texts]
        return scores[:-1] if self.wrong_length else scores


@pytest.fixture
def built(sample_repo, tmp_path, fake_embedder):
    index = RepoIndex(sample_repo, tmp_path / "idx")
    Indexer(index, fake_embedder).run()
    yield index
    index.close()


class TestCrossEncoderReranker:
    def test_scores_each_text_against_the_query(self):
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker("some/model", model=model)
        assert reranker.score("q", ["a", "abc"]) == [1.0, 3.0]
        assert model.calls[0]["pairs"] == [("q", "a"), ("q", "abc")]
        assert reranker.model_id == "cross-encoder:some/model"

    def test_no_texts_need_no_model(self):
        reranker = CrossEncoderReranker("some/model", model=FakeCrossEncoder(error=RuntimeError("must not run")))
        assert reranker.score("q", []) == []

    def test_scores_are_plain_floats(self):
        import numpy as np

        reranker = CrossEncoderReranker("m", model=FakeCrossEncoder(lambda q, t: np.float32(0.5)))
        (score,) = reranker.score("q", ["x"])
        assert type(score) is float and score == 0.5

    def test_a_failing_model_becomes_a_rerank_error(self):
        reranker = CrossEncoderReranker("some/model", model=FakeCrossEncoder(error=RuntimeError("out of memory")))
        with pytest.raises(RerankError, match="some/model.*out of memory"):
            reranker.score("q", ["x"])

    def test_a_model_that_cannot_be_loaded_says_which_one(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("no such model")

        monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=refuse))
        with pytest.raises(RerankError, match="Could not load reranker model 'nope/none': no such model"):
            CrossEncoderReranker("nope/none").score("q", ["x"])

    def test_remote_code_is_never_trusted(self, monkeypatch):
        seen = {}

        def fake(name, **kwargs):
            seen.update(kwargs)
            return FakeCrossEncoder()

        monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=fake))
        CrossEncoderReranker("m", max_length=256).score("q", ["x"])
        assert seen == {"max_length": 256, "trust_remote_code": False}


class TestCreateReranker:
    def test_off_by_default(self):
        settings = Settings(_env_file=None)
        assert settings.reranker_model is None and settings.rerank_top_n == 30
        assert create_reranker(settings) is None

    def test_configured_by_the_environment(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL", "org/reranker")
        monkeypatch.setenv("RERANK_TOP_N", "12")
        settings = Settings(_env_file=None)
        reranker = create_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker) and reranker.model_id == "cross-encoder:org/reranker"
        assert settings.rerank_top_n == 12


class TestRetrieverWithReranker:
    QUERY = "shopping cart total price"

    def plain(self, built, fake_embedder, **kwargs) -> Retriever:
        return Retriever(built, fake_embedder, **kwargs)

    def test_without_a_reranker_nothing_changes(self, built, fake_embedder):
        a = self.plain(built, fake_embedder).rank(self.QUERY)
        b = self.plain(built, fake_embedder, reranker=None, rerank_top=5).rank(self.QUERY)
        assert [(s.chunk.id, s.score) for s in a] == [(s.chunk.id, s.score) for s in b]

    def test_the_top_candidates_are_reordered_by_the_reranker(self, built, fake_embedder):
        base = self.plain(built, fake_embedder).rank(self.QUERY)
        reranker = ScriptedReranker(lambda text: 1.0 if "def make_service" in text else 0.0)
        ranked = self.plain(built, fake_embedder, reranker=reranker).rank(self.QUERY)
        assert ranked[0].chunk.symbol == "make_service"
        assert [s.rank for s in ranked] == list(range(1, len(ranked) + 1))
        assert {s.chunk.id for s in ranked} == {s.chunk.id for s in base}  # reordered, nothing added or lost

    def test_it_reads_the_same_text_that_was_embedded_and_the_query(self, built, fake_embedder):
        reranker = ScriptedReranker(lambda text: 0.0)
        retriever = self.plain(built, fake_embedder, reranker=reranker)
        retriever.rank(self.QUERY)
        query, texts = reranker.calls[0]
        assert query == self.QUERY
        assert all(" • " in text.split("\n", 1)[0] for text in texts)  # embed_text starts with path • symbol • kind

    def test_only_the_top_n_are_reranked_and_the_rest_follow_in_their_old_order(self, built, fake_embedder):
        base = self.plain(built, fake_embedder).rank(self.QUERY)
        assert len(base) > 4
        reranker = ScriptedReranker(lambda text: -float(len(text)))  # prefers short chunks
        ranked = self.plain(built, fake_embedder, reranker=reranker, rerank_top=3).rank(self.QUERY)
        assert len(reranker.calls[0][1]) == 3
        assert {s.chunk.id for s in ranked[:3]} == {s.chunk.id for s in base[:3]}
        assert [s.chunk.id for s in ranked[3:]] == [s.chunk.id for s in base[3:]]

    def test_scores_never_increase_down_the_ranking(self, built, fake_embedder):
        reranker = ScriptedReranker(lambda text: float(len(text) % 7) - 3.0)  # mixed signs, like real logits
        ranked = self.plain(built, fake_embedder, reranker=reranker, rerank_top=6).rank(self.QUERY)
        scores = [s.score for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_test_files_stay_demoted_even_if_the_reranker_loves_them(self, tmp_path, fake_embedder):
        repo = tmp_path / "proj"
        (repo / "tests").mkdir(parents=True)
        (repo / "src").mkdir()
        (repo / "src" / "cart.py").write_text("def total_price(items):\n    return sum(items)\n", encoding="utf-8")
        (repo / "tests" / "test_cart.py").write_text("def test_total_price():\n    assert total_price([1]) == 1\n", encoding="utf-8")
        index = RepoIndex(repo, tmp_path / "idx2")
        try:
            Indexer(index, fake_embedder).run()
            loves_tests = ScriptedReranker(lambda text: 10.0 if "test_total_price" in text else 9.0)
            demoted = Retriever(index, fake_embedder, reranker=loves_tests).rank("total price of the cart")
            assert demoted[0].chunk.path == "src/cart.py"
            about_tests = Retriever(index, fake_embedder, reranker=loves_tests).rank("tests for the total price")
            assert about_tests[0].chunk.path == "tests/test_cart.py"
            off = Retriever(index, fake_embedder, reranker=loves_tests, test_penalty=1.0).rank("total price of the cart")
            assert off[0].chunk.path == "tests/test_cart.py"
        finally:
            index.close()

    def test_a_single_candidate_is_not_worth_a_model_call(self, tmp_path, fake_embedder):
        repo = tmp_path / "one"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
        index = RepoIndex(repo, tmp_path / "idx3")
        try:
            Indexer(index, fake_embedder).run()
            reranker = ScriptedReranker(lambda text: 0.0)
            assert len(Retriever(index, fake_embedder, reranker=reranker).rank("x")) == 1
            assert reranker.calls == []
        finally:
            index.close()

    def test_a_reranker_failure_is_a_retrieval_error_with_its_message(self, built, fake_embedder):
        reranker = ScriptedReranker(lambda text: 0.0, error=RerankError("Could not load reranker model 'm': offline"))
        with pytest.raises(RetrievalError, match="offline"):
            self.plain(built, fake_embedder, reranker=reranker).rank(self.QUERY)

    def test_a_reranker_that_returns_the_wrong_number_of_scores_is_rejected(self, built, fake_embedder):
        reranker = ScriptedReranker(lambda text: 0.0, wrong_length=True)
        with pytest.raises(RetrievalError, match="scores for"):
            self.plain(built, fake_embedder, reranker=reranker).rank(self.QUERY)

    def test_rerank_top_must_be_positive(self, built, fake_embedder):
        with pytest.raises(ValueError, match="rerank_top"):
            self.plain(built, fake_embedder, rerank_top=0)

    def test_the_reordering_carries_through_to_the_sources_the_model_sees(self, built, fake_embedder):
        reranker = ScriptedReranker(lambda text: 1.0 if "def make_service" in text else 0.0)
        result = self.plain(built, fake_embedder, reranker=reranker).retrieve(self.QUERY)
        assert result.sources[0].path == "app/user_service.py"
