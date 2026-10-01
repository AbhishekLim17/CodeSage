"""A retriever that answers questions about the whole repository with the repository map as well.

It wraps an ordinary ``Retriever`` and behaves identically for every question that is about one piece of the code.
For an overview question ("what does this project do?") it puts two extra sources first: the generated repository
map and the top of the README. The total stays inside the same token budget: retrieved sources are dropped from
the end (lowest ranked first) to make room, but at least one is always kept.
"""

from __future__ import annotations

from collections.abc import Callable

from codebase_ai.retrieval.query import is_overview_question
from codebase_ai.retrieval.repo_map import build_repo_map, map_source, readme_source
from codebase_ai.retrieval.retriever import RetrievalResult, Retriever, Source


class OverviewRetriever:
    """Same interface as ``Retriever.retrieve``; adds the repository map for overview questions."""

    def __init__(
        self,
        base: Retriever,
        *,
        map_tokens: int = 1500,
        is_overview: Callable[[str], bool] = is_overview_question,
    ) -> None:
        self.base = base
        self.map_tokens = map_tokens
        self.is_overview = is_overview

    @property
    def mode(self) -> str:
        return self.base.mode

    @property
    def index(self):
        return self.base.index

    def retrieve(self, query: str) -> RetrievalResult:
        result = self.base.retrieve(query)
        if not self.is_overview(query):
            return result

        repo_map = build_repo_map(self.base.index, budget_tokens=self.map_tokens)
        added: list[Source] = [map_source(repo_map)]
        readme = readme_source(self.base.index, repo_map)
        if readme is not None and not any(s.path == readme.path and s.start_line <= 1 for s in result.sources):
            added.append(readme)

        room = self.base.budget_tokens - sum(s.tokens for s in added)
        kept: list[Source] = []
        used = 0
        for source in result.sources:
            if kept and used + source.tokens > room:
                break
            kept.append(source)
            used += source.tokens
        return RetrievalResult(
            query=result.query,
            mode=result.mode,
            candidates=result.candidates,
            sources=[*added, *kept],
            overview=True,
        )
