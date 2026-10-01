"""Retrieve, build prompt, call the LLM (stream or complete), validate citations, return answer plus sources."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codebase_ai.llm.base import FinishReason, LLMProvider, Message, StreamDone, TextDelta, Usage
from codebase_ai.rag.citations import CitationReport, validate_citations
from codebase_ai.rag.prompts import (
    NO_CONTEXT_ANSWER,
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_user_message,
)
from codebase_ai.retrieval.overview import OverviewRetriever
from codebase_ai.retrieval.query import Condensed, Turn, condense_question
from codebase_ai.retrieval.reranker import Reranker
from codebase_ai.retrieval.retriever import RetrievalResult, Retriever, Source

if TYPE_CHECKING:
    from codebase_ai.config import Settings
    from codebase_ai.index.embedder import Embedder
    from codebase_ai.index.indexer import RepoIndex

# The retriever estimates 4 characters per token, but source code usually tokenizes closer to 3.
CODE_TOKEN_SAFETY = 1.4
PROMPT_OVERHEAD_TOKENS = 1500  # system prompt, question and framing


def context_budget_for(provider: LLMProvider, requested: int, max_tokens: int) -> int:
    """How much retrieved code to put in the prompt.

    Hosted models have room for the requested budget. A provider that limits its own context window (Ollama) gets a
    smaller budget, so the prompt plus the answer fit and nothing is silently cut off.
    """
    window = provider.context_window
    if window is None:
        return requested
    reserve = min(max_tokens, provider.max_output_tokens or max_tokens)
    room = window - reserve - PROMPT_OVERHEAD_TOKENS
    return max(1000, min(requested, int(room / CODE_TOKEN_SAFETY)))


@dataclass(frozen=True)
class Answer:
    """A finished answer with everything needed to check it: the sources it was given and how it cited them."""

    question: str
    text: str  # with citation markers to non-existent sources removed
    raw_text: str  # exactly what the model wrote
    sources: tuple[Source, ...]  # numbered from 1; ``sources[n - 1]`` is what ``[n]`` refers to
    citations: CitationReport
    provider: str
    model: str
    finish: FinishReason
    usage: Usage | None
    no_context: bool  # nothing relevant was retrieved, so the model was not called
    retrieval: RetrievalResult
    retrieval_seconds: float
    generation_seconds: float
    condensed: Condensed | None = None  # how a follow-up was turned into the question that was searched for

    @property
    def searched_for(self) -> str:
        """The question that was searched for and put to the model (the typed one unless a follow-up was rewritten)."""
        return self.condensed.standalone if self.condensed else self.question

    @property
    def grounded(self) -> bool:
        return self.citations.grounded

    @property
    def refused(self) -> bool:
        return self.finish == "refusal"

    @property
    def truncated(self) -> bool:
        return self.finish == "length"

    @property
    def cited_sources(self) -> list[tuple[int, Source]]:
        return [(n, self.sources[n - 1]) for n in self.citations.cited]

    @property
    def uncited_sources(self) -> list[tuple[int, Source]]:
        return [(n, self.sources[n - 1]) for n in self.citations.uncited]

    @property
    def warnings(self) -> list[str]:
        """Things a reader should know before trusting this answer, in plain language."""
        notes: list[str] = []
        if self.no_context:
            return ["No relevant code was found in the index, so no answer was generated."]
        if self.condensed is not None and self.condensed.note:
            notes.append(
                f"The follow-up could not be turned into a standalone question ({self.condensed.note}); "
                "it was searched for as typed."
            )
        if self.refused:
            notes.append("The model declined to answer this request (a provider safety check).")
        elif not self.grounded:
            notes.append("This answer cites no retrieved code, so treat it as unverified.")
        if self.citations.invalid:
            markers = ", ".join(f"[{n}]" for n in self.citations.invalid)
            notes.append(f"The answer cited sources that do not exist ({markers}); those citations were removed.")
        if self.citations.unverified_locations:
            listed = ", ".join(self.citations.unverified_locations)
            notes.append(f"The answer mentions locations that do not match the retrieved code: {listed}.")
        if self.truncated:
            notes.append("The answer was cut off at the output limit; raise ANSWER_MAX_TOKENS for a longer one.")
        return notes


class AnswerStream:
    """A live answer: ``sources`` are known at once, iterating yields text as it arrives, then ``answer`` is ready.

    If the provider fails part-way, iteration raises its ``ProviderError``; whatever arrived is in ``text_so_far``.
    """

    def __init__(
        self,
        answerer: Answerer,
        question: str,
        retrieval: RetrievalResult,
        retrieval_seconds: float,
        condensed: Condensed | None = None,
    ) -> None:
        self._answerer = answerer
        self.question = question
        self.condensed = condensed
        self.searched_for = condensed.standalone if condensed else question
        self.retrieval = retrieval
        self.sources: list[Source] = retrieval.sources
        self._retrieval_seconds = retrieval_seconds
        self._parts: list[str] = []
        self._answer: Answer | None = None
        self._started = False

    @property
    def text_so_far(self) -> str:
        return "".join(self._parts)

    @property
    def answer(self) -> Answer:
        if self._answer is None:
            raise RuntimeError("The answer is not finished; iterate the stream to the end first.")
        return self._answer

    def __iter__(self) -> Iterator[str]:
        if self._started:
            return
        self._started = True
        provider = self._answerer.provider
        started = time.perf_counter()
        done = StreamDone()

        if not self.sources:
            self._parts.append(NO_CONTEXT_ANSWER)
            yield NO_CONTEXT_ANSWER
        else:
            message = Message("user", build_user_message(self.searched_for, self.sources))
            for event in provider.stream(
                self._answerer.system,
                [message],
                max_tokens=self._answerer.max_tokens,
                temperature=self._answerer.temperature,
            ):
                if isinstance(event, TextDelta):
                    self._parts.append(event.text)
                    yield event.text
                else:
                    done = event
            if done.finish == "refusal" and not self.text_so_far.strip():
                self._parts.append(REFUSAL_ANSWER)
                yield REFUSAL_ANSWER

        raw = self.text_so_far
        report = validate_citations(raw, self.sources)
        self._answer = Answer(
            question=self.question,
            text=report.clean_text,
            raw_text=raw,
            sources=tuple(self.sources),
            citations=report,
            provider=provider.name,
            model=done.model or provider.model,
            finish=done.finish,
            usage=done.usage,
            no_context=not self.sources,
            retrieval=self.retrieval,
            retrieval_seconds=self._retrieval_seconds,
            generation_seconds=time.perf_counter() - started,
            condensed=self.condensed,
        )


class Answerer:
    """Answers questions about an indexed repository, citing the code it was shown."""

    def __init__(
        self,
        retriever: Retriever | OverviewRetriever,
        provider: LLMProvider,
        *,
        max_tokens: int = 16_000,
        temperature: float | None = None,
        system: str = SYSTEM_PROMPT,
        condense: bool = True,
        history_turns: int = 3,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self.condense = condense
        self.history_turns = history_turns

    def stream(self, question: str, history: Sequence[Turn] = ()) -> AnswerStream:
        """Retrieve now, generate as the returned stream is iterated. Raises ``RetrievalError`` like the retriever.

        With ``history`` (earlier exchanges, oldest first) a follow-up is first rewritten into a standalone question
        with one short model call, and that question is what is searched for and answered. No history, no call.
        A ``ProviderError`` from that call propagates from here, before anything is retrieved.
        """
        started = time.perf_counter()
        condensed = None
        if self.condense and history:
            condensed = condense_question(self.provider, history, question, turns=self.history_turns)
        retrieval = self.retriever.retrieve(condensed.standalone if condensed else question)
        return AnswerStream(self, question, retrieval, time.perf_counter() - started, condensed)

    def ask(self, question: str, history: Sequence[Turn] = ()) -> Answer:
        stream = self.stream(question, history)
        for _ in stream:
            pass
        return stream.answer


def create_answerer(
    settings: Settings,
    index: RepoIndex,
    embedder: Embedder | None,
    provider: LLMProvider,
    *,
    mode: str | None = None,
    reranker: Reranker | None = None,
) -> Answerer:
    """Wire a retriever and a provider together from settings, sizing the context to what the provider can take."""
    budget = context_budget_for(provider, settings.context_token_budget, settings.answer_max_tokens)
    retriever = Retriever(
        index,
        embedder,
        mode=mode or settings.retrieval_mode,  # type: ignore[arg-type]
        top_k=settings.retrieve_top_k,
        keyword_weight=settings.keyword_weight,
        budget_tokens=budget,
        test_penalty=settings.test_penalty,
        reranker=reranker,
        rerank_top=settings.rerank_top_n,
    )
    wrapped = OverviewRetriever(retriever, map_tokens=settings.repo_map_tokens) if settings.use_repo_map else retriever
    return Answerer(
        wrapped,
        provider,
        max_tokens=settings.answer_max_tokens,
        temperature=settings.answer_temperature,
        condense=settings.condense_followups,
        history_turns=settings.history_turns,
    )
