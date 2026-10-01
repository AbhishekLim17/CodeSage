"""Retrieval over a RepoIndex: hybrid search, selection under a token budget, and merging into cited sources.

Pipeline for one query (``Retriever.retrieve``):

1. **Rank** chunks: vector search and/or keyword (BM25) search, fused with Reciprocal Rank Fusion.
2. **Select** under a token budget in two passes: the best chunk of each distinct file first (so one large file
   cannot crowd out the rest), then more chunks in rank order, at most ``max_chunks_per_file`` per file.
3. **Add context**: a retrieved method whose class was split into member chunks gets the class header chunk, so
   the reader sees which class it belongs to.
4. **Merge** overlapping or directly adjacent chunks of a file into one span, with exact text and line range.

An optional cross-encoder (``reranker``) re-orders the best candidates between steps 1 and 2 (see ``reranker.py``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from codebase_ai.chunking import Chunk, SearchHit
from codebase_ai.chunking.models import KIND_CLASS, KIND_METHOD
from codebase_ai.index.embedder import Embedder
from codebase_ai.index.indexer import RepoIndex
from codebase_ai.metrics import unique_in_order
from codebase_ai.retrieval.fusion import reciprocal_rank_fusion
from codebase_ai.retrieval.reranker import Reranker, RerankError

Mode = Literal["vector", "keyword", "hybrid"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid")
CHARS_PER_TOKEN = 4  # rough; providers count differently, so the budget is approximate


class RetrievalError(RuntimeError):
    """Retrieval cannot run as configured; the message is safe to show to the user."""


def estimate_tokens(text: str) -> int:
    return max(1, -(-len(text) // CHARS_PER_TOKEN))


_TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|specs?|e2e|integration|fixtures|__mocks__)/"
    r"|(^|/)(test_[^/]*|conftest\.py)$"
    r"|_test\.[a-z]+$"
    r"|\.(test|spec|itest)\.[a-z]+$",
    re.IGNORECASE,
)
_TEST_WORDS = frozenset(
    [
        *("test", "tests", "testing", "tested", "spec", "specs"),
        *("mock", "mocks", "fixture", "fixtures"),
        *("pytest", "vitest", "jest", "unittest"),
    ]
)


def is_test_path(path: str) -> bool:
    """True for files that are tests, specs, mocks or fixtures rather than the code under test."""
    return _TEST_PATH.search(path) is not None


def mentions_tests(query: str) -> bool:
    """True if the query is about tests, in which case test files must not be demoted."""
    return any(word in _TEST_WORDS for word in re.findall(r"[a-z]+", query.lower()))


def apply_test_penalty(scored: list[tuple[Chunk, float]], penalty: float, query: str) -> list[tuple[Chunk, float]]:
    """Lower the score of chunks from test files (best-first order is kept for everything else).

    ``penalty`` is a multiplier below 1 (0.5 halves a positive score). Nothing changes when it is 1 or when the
    query itself is about tests. Tests are demoted, not dropped, so they can still be the answer.
    """
    if penalty >= 1.0 or mentions_tests(query):
        return scored
    adjusted = [
        (chunk, score - abs(score) * (1.0 - penalty) if is_test_path(chunk.path) else score) for chunk, score in scored
    ]
    return sorted(adjusted, key=lambda pair: -pair[1])  # stable: ties keep their earlier order


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its position in the ranking (1 = best)."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Source:
    """A span of one file to show to a reader (or an LLM) and cite as ``path:start-end``."""

    path: str
    language: str
    start_line: int
    end_line: int
    text: str
    symbols: tuple[str, ...]
    kinds: tuple[str, ...]
    score: float
    rank: int  # best rank among the chunks merged into this span
    chunk_ids: tuple[str, ...]
    role: Literal["match", "context", "map"] = "match"  # "map": the generated repository map, not a file

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class RetrievalResult:
    query: str
    mode: str
    candidates: list[ScoredChunk] = field(default_factory=list)  # full ranking, before budget and merging
    sources: list[Source] = field(default_factory=list)  # what fits in the budget, best first
    overview: bool = False  # the question was about the whole repository, so the repository map was added

    @property
    def ranked_files(self) -> list[str]:
        """Files in the order their best chunk ranked; independent of the token budget."""
        return unique_in_order(c.chunk.path for c in self.candidates)

    @property
    def context_files(self) -> list[str]:
        return unique_in_order(s.path for s in self.sources)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sources)


class Retriever:
    """Finds the code that answers a question. Cheap to construct; holds no state between queries."""

    def __init__(
        self,
        index: RepoIndex,
        embedder: Embedder | None = None,
        *,
        mode: Mode = "vector",
        top_k: int = 30,
        keyword_weight: float = 1.0,
        budget_tokens: int = 12_000,
        max_chunks: int = 12,
        max_chunks_per_file: int = 4,
        add_context: bool = True,
        test_penalty: float = 0.5,
        reranker: Reranker | None = None,
        rerank_top: int = 30,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if top_k <= 0 or budget_tokens <= 0 or max_chunks <= 0 or max_chunks_per_file <= 0 or keyword_weight < 0:
            raise ValueError("top_k, budget_tokens, max_chunks and max_chunks_per_file must be positive")
        if not 0 < test_penalty <= 1:
            raise ValueError("test_penalty must be in (0, 1]; 1 turns the demotion of test files off")
        if rerank_top <= 0:
            raise ValueError("rerank_top must be positive")
        if mode != "keyword":
            if embedder is None:
                raise RetrievalError(f"mode '{mode}' needs an embedder (only 'keyword' works without one).")
            stored = index.info()["embedding_model_id"]
            if stored != embedder.model_id:
                raise RetrievalError(
                    f"The index was built with {stored}, but {embedder.model_id} is configured. "
                    f"Re-run: codebase-ai index {index.repo_root} --full"
                )
        self.index = index
        self.embedder = embedder
        self.mode = mode
        self.top_k = top_k
        self.keyword_weight = keyword_weight
        self.budget_tokens = budget_tokens
        self.max_chunks = max_chunks
        self.max_chunks_per_file = max_chunks_per_file
        self.add_context = add_context
        self.test_penalty = test_penalty
        self.reranker = reranker
        self.rerank_top = rerank_top

    # -- ranking ----------------------------------------------------------------------------------------

    def rank(self, query: str) -> list[ScoredChunk]:
        """Chunks best-first for ``query`` using the configured mode."""
        vector_hits: list[SearchHit] = []
        keyword_hits: list[SearchHit] = []
        if self.mode != "keyword":
            assert self.embedder is not None
            vector_hits = self.index.vectors.query(self.embedder.embed_query(query), self.top_k)
        if self.mode != "vector":
            keyword_hits = self.index.keyword.search(query, self.top_k)

        if self.mode == "vector":
            ordered = [(h.chunk_id, h.score) for h in vector_hits]
        elif self.mode == "keyword":
            ordered = [(h.chunk_id, h.score) for h in keyword_hits]
        else:
            ordered = reciprocal_rank_fusion(
                [[h.chunk_id for h in vector_hits], [h.chunk_id for h in keyword_hits]],
                weights=[1.0, self.keyword_weight],
            )
        chunks = {c.id: c for c in self.index.keyword.get_chunks([chunk_id for chunk_id, _ in ordered])}
        known = [(chunks[chunk_id], score) for chunk_id, score in ordered if chunk_id in chunks]
        known = apply_test_penalty(known, self.test_penalty, query)
        if self.reranker is not None:
            known = self._rerank(query, known)
        return [ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(known, start=1)]

    def _rerank(self, query: str, known: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        """Re-order the best ``rerank_top`` candidates by the cross-encoder; the rest follow in their old order.

        Test files are demoted again on the new scores (the reranker knows nothing of that policy), and the tail is
        scored just below the last re-ordered chunk so scores still never increase down the ranking.
        """
        assert self.reranker is not None
        head, tail = known[: self.rerank_top], known[self.rerank_top :]
        if len(head) < 2:
            return known
        try:
            scores = self.reranker.score(query, [chunk.embed_text for chunk, _ in head])
        except RerankError as exc:
            raise RetrievalError(str(exc)) from exc
        if len(scores) != len(head):
            raise RetrievalError(f"The reranker returned {len(scores)} scores for {len(head)} chunks.")
        reordered = sorted(
            apply_test_penalty([(chunk, score) for (chunk, _), score in zip(head, scores, strict=True)], self.test_penalty, query),
            key=lambda pair: -pair[1],
        )
        floor = min(score for _, score in reordered)
        return reordered + [(chunk, floor - (offset + 1) * 1e-6) for offset, (chunk, _) in enumerate(tail)]

    # -- selection --------------------------------------------------------------------------------------

    def _select(self, ranked: list[ScoredChunk]) -> list[ScoredChunk]:
        chosen: list[ScoredChunk] = []
        taken: set[str] = set()
        per_file: Counter[str] = Counter()
        used = 0
        for first_pass in (True, False):
            for scored in ranked:
                path = scored.chunk.path
                if len(chosen) >= self.max_chunks:
                    return chosen
                if scored.chunk.id in taken or per_file[path] >= self.max_chunks_per_file:
                    continue
                if first_pass and per_file[path] > 0:
                    continue
                cost = estimate_tokens(scored.chunk.text)
                if chosen and used + cost > self.budget_tokens:
                    continue  # the top-ranked chunk is always kept, so a result is never empty
                chosen.append(scored)
                taken.add(scored.chunk.id)
                per_file[path] += 1
                used += cost
        return chosen

    def _add_class_headers(self, chosen: list[ScoredChunk]) -> tuple[list[ScoredChunk], set[str]]:
        """Add each retrieved method's class header chunk (when the class was split) if the budget allows."""
        used = sum(estimate_tokens(s.chunk.text) for s in chosen)
        present = {s.chunk.id for s in chosen}
        extra: list[ScoredChunk] = []
        context_ids: set[str] = set()
        for scored in chosen:
            chunk = scored.chunk
            if chunk.kind != KIND_METHOD or not chunk.symbol or "." not in chunk.symbol:
                continue
            parent = chunk.symbol.rsplit(".", 1)[0]
            headers = self.index.keyword.find_chunks(chunk.path, symbol=parent, kind=KIND_CLASS)
            if not headers or headers[0].id in present:
                continue
            header = headers[0]
            cost = estimate_tokens(header.text)
            if used + cost > self.budget_tokens:
                continue
            extra.append(ScoredChunk(header, scored.score, scored.rank))
            present.add(header.id)
            context_ids.add(header.id)
            used += cost
        return chosen + extra, context_ids

    # -- merging ----------------------------------------------------------------------------------------

    @staticmethod
    def _merge(selected: list[ScoredChunk], context_ids: set[str]) -> list[Source]:
        by_path: dict[str, list[ScoredChunk]] = {}
        for scored in selected:
            by_path.setdefault(scored.chunk.path, []).append(scored)

        sources: list[Source] = []
        for group in by_path.values():
            group.sort(key=lambda s: (s.chunk.start_line, s.chunk.end_line))
            cluster = [group[0]]
            cluster_end = group[0].chunk.end_line
            for scored in group[1:]:
                if scored.chunk.start_line <= cluster_end + 1:  # overlapping or directly adjacent
                    cluster.append(scored)
                    cluster_end = max(cluster_end, scored.chunk.end_line)
                else:
                    sources.append(Retriever._make_source(cluster, context_ids))
                    cluster, cluster_end = [scored], scored.chunk.end_line
            sources.append(Retriever._make_source(cluster, context_ids))
        sources.sort(key=lambda s: (s.rank, s.role == "context", s.path, s.start_line))
        return sources

    @staticmethod
    def _make_source(cluster: list[ScoredChunk], context_ids: set[str]) -> Source:
        lines: dict[int, str] = {}
        for scored in cluster:
            for offset, line in enumerate(scored.chunk.text.split("\n")):
                lines[scored.chunk.start_line + offset] = line
        start, end = min(lines), max(lines)
        chunks = [s.chunk for s in cluster]
        return Source(
            path=chunks[0].path,
            language=chunks[0].language,
            start_line=start,
            end_line=end,
            text="\n".join(lines[n] for n in range(start, end + 1)),
            symbols=tuple(unique_in_order(c.symbol for c in chunks if c.symbol)),
            kinds=tuple(unique_in_order(c.kind for c in chunks)),
            score=max(s.score for s in cluster),
            rank=min(s.rank for s in cluster),
            chunk_ids=tuple(c.id for c in chunks),
            role="context" if all(c.id in context_ids for c in chunks) else "match",
        )

    # -- public -----------------------------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """Rank, select under the budget, add class context, and merge into sources."""
        ranked = self.rank(query)
        selected = self._select(ranked)
        context_ids: set[str] = set()
        if self.add_context:
            selected, context_ids = self._add_class_headers(selected)
        return RetrievalResult(
            query=query,
            mode=self.mode,
            candidates=ranked,
            sources=self._merge(selected, context_ids),
        )


def _hydrate(index: RepoIndex, hits: list[SearchHit]) -> list[tuple[Chunk, float]]:
    chunks = {c.id: c for c in index.keyword.get_chunks([h.chunk_id for h in hits])}
    return [(chunks[h.chunk_id], h.score) for h in hits if h.chunk_id in chunks]


def search_vector(index: RepoIndex, embedder: Embedder, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
    """Top ``k`` chunks by embedding similarity (cosine, higher is better)."""
    return _hydrate(index, index.vectors.query(embedder.embed_query(query), k))


def search_keyword(index: RepoIndex, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
    """Top ``k`` chunks by BM25 keyword match (identifier-aware; higher is better)."""
    return _hydrate(index, index.keyword.search(query, k))
